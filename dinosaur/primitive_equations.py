# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The primitive equations written for a semi-implicit solver."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Mapping, Sequence

from dinosaur import coordinate_systems
from dinosaur import hybrid_coordinates
from dinosaur import jax_numpy_utils
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import units
from dinosaur import vertical_interpolation
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tree_math


Array = typing.Array
Numeric = typing.Numeric
Quantity = typing.Quantity

OrographyInitFn = Callable[..., Array]


# All `einsum`s should be done at highest available precision.
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)

# For consistency with commonly accepted notation, we use Greek letters within
# some of the functions below.
# pylint: disable=invalid-name

#  =============================================================================
#  Data Structures
#
#  Data classes that describe the state, scale and parameters of the system.
#  =============================================================================


@tree_math.struct
class State:
  """Records the state of a system described by the primitive equations."""

  vorticity: Array
  divergence: Array
  temperature_variation: Array
  log_surface_pressure: Array
  tracers: Mapping[str, Array] = dataclasses.field(default_factory=dict)
  sim_time: float | None = None


def _asdict(state: State) -> dict[str, Any]:
  # Exclude sim_time if it is None (the default value), to avoid breaking
  # backwards compatibility with State.asdict() and State.astuple() before
  # sim_time was added.
  return {
      field.name: getattr(state, field.name)
      for field in state.fields
      if field.name != 'sim_time' or state.sim_time is not None
  }


State.asdict = _asdict


class StateShapeError(Exception):
  """Exceptions for unexpected state shapes."""


def validate_state_shape(
    state: State, coords: coordinate_systems.CoordinateSystem
):
  """Validates that values in `state` have appropriate shapes."""
  if state.vorticity.shape != coords.modal_shape:
    raise StateShapeError(
        f'Expected vorticity shape {coords.modal_shape}; '
        f'got shape {state.vorticity.shape}.'
    )
  if state.divergence.shape != coords.modal_shape:
    raise StateShapeError(
        f'Expected divergence shape {coords.modal_shape}; '
        f'got shape {state.divergence.shape}.'
    )
  if state.temperature_variation.shape != coords.modal_shape:
    raise StateShapeError(
        f'Expected temperature_variation shape {coords.modal_shape}; '
        f'got shape {state.temperature_variation.shape}.'
    )
  if state.log_surface_pressure.shape != coords.surface_modal_shape:
    raise StateShapeError(
        f'Expected log_surface_pressure shape {coords.surface_modal_shape}; '
        f'got shape {state.log_surface_pressure.shape}.'
    )
  for tracer_name, array in state.tracers.items():
    if array.shape[-3:] != coords.modal_shape:
      raise StateShapeError(
          f'Expected tracer {tracer_name} shape {coords.modal_shape}; '
          f'got shape {array.shape}.'
      )


@jax.named_call
def _vertical_matvec(a: Array, x: Array) -> jax.Array:
  return einsum('gh,...hml->...gml', a, x)


@jax.named_call
def _vertical_matvec_per_wavenumber(a: Array, x: Array) -> jax.Array:
  return einsum('lgh,...hml->...gml', a, x)


@tree_math.struct
class DiagnosticStateSigma:
  """Stores nodal diagnostic values used to compute explicit tendencies.

  The expected shapes of the state are described in terms of # of layers `h`,
  # of longitude quadrature points `q` and # of latitude quadrature points `t`.

  Attributes:
    vorticity: nodal values of the vorticity field of shape [h, q, t].
    divergence: nodal values of the divergence field of shape [h, q, t].
    temperature_variation: nodal values of the T' field of shape [h, q, t].
    cos_lat_u: tuple of nodal values of cosθ * velocity_vector, each of shape
      [h, q, t].
    sigma_dot_explicit: nodal values of d𝜎/dt due to pressure gradient terms `u
      · ∇(log(ps))` of shape [h, q, t].
    sigma_dot_full: nodal values of d𝜎/dt due to all terms of shape [h, q, t].
    cos_lat_grad_log_sp: (2,) nodal values of cosθ · ∇(log(surface_pressure)) of
      shape [1, q, t].
    u_dot_grad_log_sp: nodal values of `u · ∇(log(surface_pressure))` of shape
      [h, q, t].
    tracers: mapping from tracer names to correspondong nodal values of shape
      [h, q, t].
  """

  vorticity: Array
  divergence: Array
  temperature_variation: Array
  cos_lat_u: tuple[Array, Array]
  sigma_dot_explicit: Array
  sigma_dot_full: Array
  cos_lat_grad_log_sp: Array
  u_dot_grad_log_sp: Array
  tracers: Mapping[str, Array]


@jax.named_call
def compute_diagnostic_state_sigma(
    state: State,
    coords: coordinate_systems.CoordinateSystem,
) -> DiagnosticStateSigma:
  """Computes DiagnosticState in nodal basis based on the modal `state`."""

  # TODO(dkochkov) Investigate clipping hyperparameters.
  # when converting to nodal, we need to clip wavenumbers.
  def to_nodal_fn(x):
    return coords.horizontal.to_nodal(x)

  nodal_vorticity = to_nodal_fn(state.vorticity)
  nodal_divergence = to_nodal_fn(state.divergence)
  nodal_temperature_variation = to_nodal_fn(state.temperature_variation)
  tracers = to_nodal_fn(state.tracers)
  nodal_cos_lat_u = jax.tree_util.tree_map(
      to_nodal_fn,
      spherical_harmonic.get_cos_lat_vector(
          state.vorticity, state.divergence, coords.horizontal, clip=False
      ),
  )
  cos_lat_grad_log_sp = coords.horizontal.cos_lat_grad(
      state.log_surface_pressure, clip=False
  )
  nodal_cos_lat_grad_log_sp = to_nodal_fn(cos_lat_grad_log_sp)
  nodal_u_dot_grad_log_sp = sum(
      jax.tree_util.tree_map(
          lambda x, y: x * y * coords.horizontal.sec2_lat,
          nodal_cos_lat_u,
          nodal_cos_lat_grad_log_sp,
      )
  )
  f_explicit = sigma_coordinates.cumulative_sigma_integral(
      nodal_u_dot_grad_log_sp, coords.vertical
  )
  f_full = sigma_coordinates.cumulative_sigma_integral(
      nodal_divergence + nodal_u_dot_grad_log_sp, coords.vertical
  )
  # note: we only need velocities at the inner boundaries of coords.vertical.
  sum_𝜎 = np.cumsum(coords.vertical.layer_thickness)[:, np.newaxis, np.newaxis]
  sigma_dot_explicit = lax.slice_in_dim(
      sum_𝜎 * lax.slice_in_dim(f_explicit, -1, None) - f_explicit, 0, -1
  )
  sigma_dot_full = lax.slice_in_dim(
      sum_𝜎 * lax.slice_in_dim(f_full, -1, None) - f_full, 0, -1
  )
  return DiagnosticStateSigma(
      vorticity=nodal_vorticity,
      divergence=nodal_divergence,
      temperature_variation=nodal_temperature_variation,
      cos_lat_u=nodal_cos_lat_u,
      sigma_dot_explicit=sigma_dot_explicit,
      sigma_dot_full=sigma_dot_full,
      cos_lat_grad_log_sp=nodal_cos_lat_grad_log_sp,
      u_dot_grad_log_sp=nodal_u_dot_grad_log_sp,
      tracers=tracers,
  )


def _vertical_interp(x, xp, fp):
  # interp defaults to constant extrapolation, which matches default boundary
  # conditions for advected fields (dx_dsigma_boundary_values = 0) from
  # from sigma_coordinates.centered_vertical_advection.
  assert x.ndim in {1, 3} and xp.ndim in {1, 3}
  interpolate_fn = vertical_interpolation.interp
  in_axes = (-1 if x.ndim == 3 else None, -1 if xp.ndim == 3 else None, -1)
  interpolate_fn = jax.vmap(interpolate_fn, in_axes, out_axes=-1)  # y
  interpolate_fn = jax.vmap(interpolate_fn, in_axes, out_axes=-1)  # x
  interpolate_fn = jax.vmap(interpolate_fn, (0, None, None), out_axes=0)
  return interpolate_fn(x, xp, fp)


def compute_vertical_velocity_sigma(
    state: State, coords: coordinate_systems.CoordinateSystem
) -> jax.Array:
  """Calculate vertical velocity at the center of each layer."""
  sigma_dot_boundaries = compute_diagnostic_state_sigma(
      state, coords
  ).sigma_dot_full
  assert sigma_dot_boundaries.ndim == 3
  # This matches the default boundary conditions for vertical velocity
  # from sigma_coordinates.centered_vertical_advection
  sigma_dot_padded = jnp.pad(sigma_dot_boundaries, [(1, 1), (0, 0), (0, 0)])
  return 0.5 * (sigma_dot_padded[1:] + sigma_dot_padded[:-1])


def semi_lagrangian_vertical_advection_step_sigma(
    state: State, coords: coordinate_systems.CoordinateSystem, dt: float
) -> State:
  """Take a first-order step for semi-Lagrangian vertical advection."""
  velocity = compute_vertical_velocity_sigma(state, coords)
  target = coords.vertical.centers
  source = target[:, jnp.newaxis, jnp.newaxis] - dt * velocity

  def interpolate(x):
    if x.ndim < 3 or x.shape[0] == 1:
      return x  # not a 3D variable
    # TODO(shoyer): avoid unnecessary transformations to nodal space.
    x = coords.horizontal.to_nodal(x)
    x = _vertical_interp(target, source, x)
    x = coords.horizontal.to_modal(x)
    return x

  return jax.tree_util.tree_map(interpolate, state)


# ==============================================================================
#  Helper Functions
#
#  Functions used to compute individual terms and intermediate values for the
#  primitive equations.
#  =============================================================================


def get_sigma_ratios(
    coordinates: sigma_coordinates.SigmaCoordinates,
) -> np.ndarray:
  """Returns the log ratios of the sigma values for the given coordinates.

  These values are used as weights when computing geopotentials. In
  'Numerical Methods for Fluid Dynamics', Durran refers to these values as
  `𝜎[j]`.

  Args:
    coordinates: the `SigmaCoordinates` object describing the spacing of layers
      in 𝜎 coordinates.

  Returns:
    A vector 𝜶 with length `coordinates.layers` such that, for `n + 1` layers,
                 𝜶[n] = -log(𝜎[n])
                 𝜶[j] = log(𝜎[j + 1] / 𝜎[j]) / 2    for j < n
  """
  alpha = np.diff(np.log(coordinates.centers), append=0) / 2
  alpha[-1] = -np.log(coordinates.centers[-1])
  return alpha


def get_geopotential_weights_sigma(
    coordinates: sigma_coordinates.SigmaCoordinates,
    ideal_gas_constant: float,
) -> np.ndarray:
  """Returns a matrix of weights used to compute the geopotential.

  In 'Numerical Methods for Fluid Dynamics' §8.6.5, Durran refers to this matrix
  as `G `.

  Args:
    coordinates: the `SigmaCoordinates` object describing the spacing of layers
      in 𝜎 coordinates.
    ideal_gas_constant: the ideal gas constant `R`

  Returns:
    A matrix `G` with shape `[coordinates.layers, coordinates.layers]` such that

               𝜶[0]    𝜶[0] + 𝜶[1]    𝜶[1] + 𝜶[2]    𝜶[2] + 𝜶[3]    ᠁
    G / R  =   0       𝜶[1]           𝜶[1] + 𝜶[2]    𝜶[2] + 𝜶[3]    ᠁
               0       0              𝜶[2]           𝜶[2] + 𝜶[3]    ᠁
               ⋮       ⋮               ⋮              ⋮              ⋱

    where 𝜶 is the vector returned by `sigma_ratios`.
  """
  # Since this matrix is computed only once, we favor readability over
  # efficiency in its construction.
  alpha = get_sigma_ratios(coordinates)
  weights = np.zeros([coordinates.layers, coordinates.layers])
  for j in range(coordinates.layers):
    weights[j, j] = alpha[j]
    for k in range(j + 1, coordinates.layers):
      weights[j, k] = alpha[k] + alpha[k - 1]
  return ideal_gas_constant * weights


def get_geopotential_diff_sigma(
    temperature: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    ideal_gas_constant: float,
    method: str = 'dense',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Calculate the implicit geopotential term."""
  if method == 'dense':
    weights = get_geopotential_weights_sigma(coordinates, ideal_gas_constant)
    return _vertical_matvec(weights, temperature)
  elif method == 'sparse':
    alpha = ideal_gas_constant * get_sigma_ratios(coordinates)
    alpha2 = np.concatenate([[0], alpha[1:] + alpha[:-1]])
    return (
        jax_numpy_utils.reverse_cumsum(
            alpha2[:, np.newaxis, np.newaxis] * temperature,
            axis=0,
            sharding=sharding,
        )
        + (alpha - alpha2)[:, np.newaxis, np.newaxis] * temperature
    )
  else:
    raise ValueError(f'unknown {method=} for get_geopotential_diff')


def get_geopotential_on_sigma(
    temperature: typing.Array,
    specific_humidity: typing.Array | None = None,
    clouds: typing.Array | None = None,
    *,
    nodal_orography: typing.Array,
    sigma: sigma_coordinates.SigmaCoordinates,
    gravity_acceleration: float,
    ideal_gas_constant: float,
    water_vapor_gas_constant: float | None = None,
    sharding: jax.sharding.NamedSharding | None = None,
) -> jnp.ndarray:
  """Computes geopotential in nodal space using nodal temperature and moisture.

  If `specific_humidity` is None, computes dry geopotential. If clouds are
  provided, the cloud condensation is subtracted from the virtual temperature.

  Args:
    temperature: nodal values of temperature.
    specific_humidity: nodal values of specific humidity. If provided, moisture
      effects are included in geopotential calculation.
    clouds: nodal values of cloud condensate.
    nodal_orography: nodal values of orography.
    sigma: sigma coordinates.
    gravity_acceleration: gravity.
    ideal_gas_constant: ideal gas constant for dry air.
    water_vapor_gas_constant: ideal gas constant for water vapor. Must be
      provided if `specific_humidity` is provided.
    sharding: optional sharding.

  Returns:
    Nodal geopotential.
  """
  surface_geopotential = nodal_orography * gravity_acceleration
  if specific_humidity is not None:
    if water_vapor_gas_constant is None:
      raise ValueError(
          'Must provide `water_vapor_gas_constant` with `specific_humidity`.'
      )
    gas_const_ratio = water_vapor_gas_constant / ideal_gas_constant
    cloud_effect = 0.0 if clouds is None else clouds
    virtual_temp = temperature * (
        1 + (gas_const_ratio - 1) * specific_humidity - cloud_effect
    )
  else:
    virtual_temp = temperature
  geopotential_diff = get_geopotential_diff_sigma(
      virtual_temp, sigma, ideal_gas_constant, sharding=sharding
  )
  return surface_geopotential + geopotential_diff


def get_temperature_implicit_weights_sigma(
    coordinates: sigma_coordinates.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
) -> np.ndarray:
  """Returns weights used to compute implicit terms for the temperature.

  In 'Numerical Methods for Fluid Dynamics' §8.6.5, Durran refers to this matrix
  as `H`.

  Args:
    coordinates: the `SigmaCoordinates` object describing the spacing of layers
      in 𝜎 coordinates.
    reference_temperature: a vector of reference temperatures, indexed by layer.
      Temperature in each layer is described as a deviation from this reference
      value.
    kappa: the ratio of the ideal gas constant R to the isobaric heat capacity.
      This value is often denoted 𝜅 in the literature. For dry air and the
      temperature range observed on earth, this value is roughly 0.2857.

  Returns:
    A matrix `H` with shape `[coordinates.layers, coordinates.layers]` whose
    entry in row `r` and column `s` is given by

    H[r, s] / Δ𝜎[s] = 𝜅T[r] · (P(r - s) 𝛼[r] + P(r - s - 1) 𝛼[r - 1]) / Δ𝜎[r]
                      - ̇K[r, s]
                      - K[r - 1, s]

    with

    K[r, s] = (T[r + 1] - T[r]) / (Δ𝜎[r + 1] + Δ𝜎[r])
              · (P(r - s) - sum(Δ𝜎[:r + 1]))

    K[r, s] = 0  if r < 0

    K[r, s] = 0  when `r = coordinates.layers - 1`

    where`T` is the reference temperature and `P` is an indicator function that
    takes the value 0 on negative numbers and 1 on non-negative numbers.
  """
  if (
      reference_temperature.ndim != 1
      or reference_temperature.shape[-1] != coordinates.layers
  ):
    raise ValueError(
        '`reference_temp` must be a vector of length `coordinates.layers`; '
        f'got shape {reference_temperature.shape} and '
        f'{coordinates.layers} layers.'
    )

  # The function P in matrix form, where `p[r, s] = p(r - s)`
  p = np.tril(np.ones([coordinates.layers, coordinates.layers]))

  # Compute the first term in the sum above.
  alpha = get_sigma_ratios(coordinates)[..., np.newaxis]
  p_alpha = p * alpha
  p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
  p_alpha_shifted[0] = 0
  h0 = (
      kappa
      * reference_temperature[..., np.newaxis]
      * (p_alpha + p_alpha_shifted)
      / coordinates.layer_thickness[..., np.newaxis]
  )

  # Constructing the values k[r, s].
  temp_diff = np.diff(reference_temperature)
  thickness_sum = (
      coordinates.layer_thickness[:-1] + coordinates.layer_thickness[1:]
  )
  # (T[r + 1] - T[r]) / (Δ𝜎[r + 1] + Δ𝜎[r])
  k0 = np.concatenate((temp_diff / thickness_sum, [0]), axis=0)[..., np.newaxis]

  thickness_cumulative = np.cumsum(coordinates.layer_thickness)[..., np.newaxis]
  # P(r - s) - sum(Δ𝜎[:r + 1])
  k1 = p - thickness_cumulative

  k = k0 * k1

  # `k_shifted[r, s] = k[r - 1, s]`, padded with zeros at `r = 0`.
  k_shifted = np.roll(k, 1, axis=0)
  k_shifted[0] = 0

  return (h0 - k - k_shifted) * coordinates.layer_thickness


def get_temperature_implicit_sigma(
    divergence: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
    method: str = 'dense',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Calculate the implicit temperature term."""
  weights = -get_temperature_implicit_weights_sigma(
      coordinates, reference_temperature, kappa
  )

  if method == 'dense':
    return _vertical_matvec(weights, divergence)
  elif method == 'sparse':
    diag_weights = np.diag(weights)
    up_weights = np.concatenate([[0], weights[1:, 0]])
    down_weights = np.concatenate([weights[:-1, -1], [0]])
    up_divergence = (
        jax_numpy_utils.cumsum(divergence, axis=0, sharding=sharding)
        - divergence
    )
    result = (
        up_weights[:, np.newaxis, np.newaxis] * up_divergence
        + diag_weights[:, np.newaxis, np.newaxis] * divergence
    )
    if (down_weights != 0).any():
      # down_weights is only non-zero for non-constant reference temperature
      down_divergence = (
          jax_numpy_utils.reverse_cumsum(divergence, axis=0, sharding=sharding)
          - divergence
      )
      result += down_weights[:, np.newaxis, np.newaxis] * down_divergence
    return result
  else:
    raise ValueError(f'unknown {method=} for get_temperature_implicit')


def _get_implicit_term_matrix_sigma(
    eta, coords, reference_temperature, kappa, ideal_gas_constant
) -> np.ndarray:
  """Returns a matrix corresponding to `PrimitiveEquations.implicit_terms`."""

  # First we construct matrices that will be building blocks for the larger
  # implicit term matrix.
  eye = np.eye(coords.vertical.layers)[np.newaxis]
  lam = coords.horizontal.laplacian_eigenvalues
  g = get_geopotential_weights_sigma(coords.vertical, ideal_gas_constant)
  r = ideal_gas_constant
  h = get_temperature_implicit_weights_sigma(
      coords.vertical, reference_temperature, kappa
  )
  t = reference_temperature[:, np.newaxis]
  thickness = coords.vertical.layer_thickness[np.newaxis, np.newaxis, :]

  # In the einsums, broadcasts and reshapes below, letters are assigned to
  # axes as follows:
  #  l: the 'total wavenumber' axis.
  #  k: the height axis, indexing layers in sigma coordinates. `k` is used to
  #     index layers in the "input".
  #  j: the height axis, indexing layers in sigma coordinates. `k` is used to
  #     index layers in the "output".
  #  o: an axis with size 1.

  # Renaming for the dimensions of each 'block' of the matrix for brevity.
  l = coords.horizontal.modal_shape[1]
  j = k = coords.vertical.layers

  row0 = np.concatenate(
      [
          np.broadcast_to(eye, [l, j, k]),
          eta * np.einsum('l,jk->ljk', lam, g),
          eta * r * np.einsum('l,jo->ljo', lam, t),
      ],
      axis=2,
  )
  row1 = np.concatenate(
      [
          eta * np.broadcast_to(h[np.newaxis], [l, j, k]),
          np.broadcast_to(eye, [l, j, k]),
          np.zeros([l, j, 1]),
      ],
      axis=2,
  )
  row2 = np.concatenate(
      [
          np.broadcast_to(eta * thickness, [l, 1, k]),
          np.zeros([l, 1, k]),
          np.ones([l, 1, 1]),
      ],
      axis=2,
  )
  return np.concatenate((row0, row1, row2), axis=1)


def div_sec_lat(
    m_component: Array, n_component: Array, grid: spherical_harmonic.Grid
) -> Array:
  """Computes div_sec_lat (aka H operator in Durran) in modal basis.

  Computes divergences of sec(θ) * (m, n) vector (equivalently operator H):

    H(M, N) = ((1 / cos²θ) * ∂M/∂λ + ∂N/∂(sinθ) / R)

  Which captures some explicit tendencies in primitive equations.

  Args:
    m_component: the value of the `M` input field in nodal representation.
    n_component: the value of the `N` input field in nodal representation.
    grid: the object describing the basis used in the horizontal direction.

  Returns:
    Value of H(M, N) in modal representation.
  """
  # Note: this operator does not include the 1/a scaling factor.
  m_component = grid.to_modal(m_component * grid.sec2_lat)
  n_component = grid.to_modal(n_component * grid.sec2_lat)
  return grid.div_cos_lat((m_component, n_component), clip=False)


def truncated_modal_orography(
    orography: Array,
    coords: coordinate_systems.CoordinateSystem,
    wavenumbers_to_clip: int = 1,
) -> Array:
  """Returns modal orography with `n` highest wavenumbers truncated."""
  grid = coords.horizontal
  expected_shape = grid.nodal_shape
  if orography.shape != expected_shape:
    raise ValueError(f'Expected nodal orography with shape={expected_shape}')
  return grid.clip_wavenumbers(grid.to_modal(orography), n=wavenumbers_to_clip)


def filtered_modal_orography(
    orography: Array,
    coords: coordinate_systems.CoordinateSystem,
    input_coords: coordinate_systems.CoordinateSystem | None = None,
    filter_fns: Sequence[typing.PostProcessFn] = tuple(),
) -> Array:
  """Returns modal `orography` interpolated to `coords` and filtered."""
  if input_coords is None:
    input_coords = coords
  expected_shape = input_coords.horizontal.nodal_shape
  if orography.shape != expected_shape:
    raise ValueError(f'Expected nodal orography with shape={expected_shape}')
  interpolate_fn = coordinate_systems.get_spectral_interpolate_fn(
      input_coords, coords, expect_same_vertical=False
  )
  modal_orography = interpolate_fn(input_coords.horizontal.to_modal(orography))
  for filter_fn in filter_fns:
    modal_orography = filter_fn(modal_orography)
  return modal_orography


#  =============================================================================
#  The `PrimitiveEquations` Classes
#
#  The `PrimitiveEquationsBase` class expresses the general structure of the
#  primitive equations organized to be solved via semi-implicit time stepping.
#  Subclasses `PrimitiveEquationsSigma` and `PrimitiveEquationsHybrid` then
#  complete the implementation for sigma- and hybrid-coordinate systems.
#  =============================================================================


@dataclasses.dataclass
class PrimitiveEquationsBase(time_integration.ImplicitExplicitODE):
  """Base class for semi-implicit primitive equations.

  Attributes:
    reference_temperature: An array of shape [layers]. All temperature values
      will be expressed as their difference from this value.
    orography: An array of shape `coords.horizontal.modal_shape` describing the
      topography in modal representation.
    coords: horizontal and vertical descritization.
    physics_specs: object holding physical constants and schema for converting
      between dimensional and nondimensional quantities.
    vertical_matmul_method: 'dense' or 'sparse', indicating the method to use
      for vertical matrix multiplications inside calculations of implicit
      geopotential and temperature terms. 'sparse' uses a matrix-free
      calculation involving `cumsum`, and is faster only when the calculation
      uses vertical sharding.
    implicit_inverse_method: 'split', 'stacked' or 'blockwise' method to use for
      the implicit inverse calculation.
    include_vertical_advection: whether to include tendencies from vertical
      advection or to drop it.
    humidity_key: Key for specific humidity in tracers dict. If provided,
      moisture effects are included in the dynamics.
    cloud_keys: Keys for cloud water species in tracers dict. If provided, cloud
      effects are included in virtual temperature calculation.
  """

  reference_temperature: np.ndarray
  orography: Array
  coords: coordinate_systems.CoordinateSystem
  physics_specs: units.SimUnitsProtocol
  vertical_matmul_method: str | None = dataclasses.field(
      default=None, kw_only=True
  )
  implicit_inverse_method: str = dataclasses.field(
      default='split', kw_only=True
  )
  include_vertical_advection: bool = dataclasses.field(
      default=True, kw_only=True
  )
  humidity_key: str | None = dataclasses.field(default=None, kw_only=True)
  cloud_keys: tuple[str, ...] | None = dataclasses.field(
      default=None, kw_only=True
  )

  def __post_init__(self):
    if not np.allclose(
        self.coords.horizontal.radius, self.physics_specs.radius, rtol=1e-5
    ):
      raise ValueError(
          'inconsistent radius between coordinates and constants: '
          f'{self.coords.horizontal.radius=} != {self.physics_specs.radius=}'
      )
    if self.cloud_keys is not None and self.humidity_key is None:
      raise ValueError('cloud_keys requires humidity_key to be set.')

  def _get_tracer(self, state_or_aux_state: Any, key: str) -> Array:
    if key not in state_or_aux_state.tracers:
      raise ValueError(
          f'`{key}` is not found in tracers: '
          f'{state_or_aux_state.tracers.keys()}.'
      )
    return state_or_aux_state.tracers[key]

  def _get_specific_humidity(self, state_or_aux_state: Any) -> Array:
    """Extracts `speicific_humidity` from tracers."""
    if self.humidity_key is None:
      raise ValueError('humidity_key is not set.')
    return self._get_tracer(state_or_aux_state, self.humidity_key)

  def _cloud_virtual_t_adjustment(self, aux_state: Any) -> Array:
    """Extracts adjustment to the virtual temperature due to clouds."""
    adjustment = jnp.asarray(0.0)
    if self.cloud_keys is not None:
      for key in self.cloud_keys:
        # clouds reduce the virtual temperature, hence the negative sign.
        adjustment -= self._get_tracer(aux_state, key)
    return adjustment

  def _virtual_temperature_adjustment(self, aux_state: Any) -> Array:
    """Computes the factor (1 + 0.61q - q_cloud) for virtual temperature."""
    if self.humidity_key is None:
      return jnp.asarray(1.0)
    q = self._get_specific_humidity(aux_state)
    gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
    moisture_contribution = (gas_const_ratio - 1) * q
    adjustment = 1 + moisture_contribution
    # _cloud_virtual_t_adjustment returns a negative value (-cloud_water).
    adjustment += self._cloud_virtual_t_adjustment(aux_state)
    return adjustment

  @property
  def coriolis_parameter(self) -> Array:
    """Returns the value `2Ω sin(θ)` associated with Coriolis force."""
    _, sin_lat = self.coords.horizontal.nodal_mesh
    return 2 * self.physics_specs.angular_velocity * sin_lat

  @property
  def T_ref(self) -> Array:
    """Returns `reference_temperature` with spatial dimensions appended."""
    return self.reference_temperature[..., np.newaxis, np.newaxis]

  @jax.named_call
  def _vertical_tendency(self, w: Array, x: Array) -> Array:
    """Computes vertical nodal tendency of `x` due to vertical_velocity `w`."""
    # subclasses must define `vertical_advection` attribute.
    return self.vertical_advection(w, x, self.coords.vertical)  # pytype: disable=attribute-error

  @jax.named_call
  def kinetic_energy_tendency(self, aux_state: Any) -> Array:
    """Computes explicit tendency of divergence due to kinetic energy term."""
    nodal_cos_lat_u2 = jnp.stack(aux_state.cos_lat_u) ** 2
    kinetic = nodal_cos_lat_u2.sum(0) * self.coords.horizontal.sec2_lat / 2
    return -self.coords.horizontal.laplacian(
        self.coords.horizontal.to_modal(kinetic)
    )

  @jax.named_call
  def orography_tendency(self) -> Array:
    """Computes orography contribution to div tendency due to geopotential."""
    # this term should broadcast correctly as layers are leading indices.
    return -self.physics_specs.g * self.coords.horizontal.laplacian(
        self.orography
    )

  @jax.named_call
  def horizontal_scalar_advection(
      self,
      scalar: Array,
      aux_state: Any,
  ) -> tuple[Array, Array]:
    """Computes explicit tendency of `scalar` due to horizontal advection."""
    u, v = aux_state.cos_lat_u
    nodal_terms = scalar * aux_state.divergence
    modal_terms = -div_sec_lat(u * scalar, v * scalar, self.coords.horizontal)
    return nodal_terms, modal_terms

  def _get_geopotential_diff(
      self,
      temperature_diff: Array,
      method: str,
      sharding: jax.sharding.NamedSharding | None,
      surface_pressure: Array | None = None,
  ) -> Array:
    """Returns geopotential difference."""
    # Must be implemented by subclasses. Depends on the vertical discretization.
    raise NotImplementedError()

  def divergence_tendency_due_to_humidity(
      self,
      state: State,
      aux_state: Any,
  ) -> Array:
    """Computes divergence tendencies from geopotential and pressure terms."""
    raise NotImplementedError()

  def vorticity_tendency_due_to_humidity(
      self,
      state: State,
      aux_state: Any,
  ) -> Array:
    """Computes vorticity tendencies due to humidity."""
    raise NotImplementedError()


@dataclasses.dataclass
class PrimitiveEquationsSigma(PrimitiveEquationsBase):
  """Primitive Equations solved on terrain following sigma coordinates."""

  vertical_advection: Callable[..., jax.Array] = dataclasses.field(
      default=sigma_coordinates.centered_vertical_advection, kw_only=True
  )

  def _get_geopotential_diff(
      self,
      temperature_diff: Array,
      method: str,
      sharding: jax.sharding.NamedSharding | None,
      surface_pressure: Array | None = None,

  ) -> Array:
    """Returns geopotential difference in sigma coordinates."""
    del surface_pressure  # Unused in sigma coordinates.
    return get_geopotential_diff_sigma(
        temperature_diff,
        self.coords.vertical,
        self.physics_specs.R,
        method=method,
        sharding=sharding,
    )

  @jax.named_call
  def divergence_tendency_due_to_humidity(
      self,
      state: State,
      aux_state: Any,
  ) -> Array:
    """Computes divergence tendencies from geopotential and pressure terms.

    These tendencies account for moisture-induced terms in the dycore that
    need to be accounted for explicitly. The terms computer here specifically
    correspond to laplacian of moist part of:
      1: ∆(R * (Tv - T) * log(surface_pressure))
      2: ∆(Φ(Tv) - Φ(T))

    Args:
      state: spectral state of the system for which tendencies are computed.
      aux_state: diagnostic state with pre-computed nodal values.

    Returns:
      Divergence tendencies induced by moisture in geopotential and pressure
      terms in spectral representation.
    """
    method = self.vertical_matmul_method
    if method is None:
      mesh = self.coords.spmd_mesh
      method = (
          'sparse'
          if (mesh is not None and mesh.shape['z'] > 1)
          else 'dense'
      )

    q = self._get_specific_humidity(aux_state)
    physics_specs = self.physics_specs
    # corresponds to the contribution of the difference of (virtual - normal)
    # temperature times laplacian of log surface pressure.
    nodal_laplacian_lsp = self.coords.horizontal.to_nodal(
        self.coords.horizontal.laplacian(state.log_surface_pressure)
    )
    nodal_laplacian_correction_term = (
        q
        * nodal_laplacian_lsp
        * self.T_ref
        * (physics_specs.R_vapor - physics_specs.R)
    )
    # corresponds to the term that differentiates the spatially dependent part
    # of reference virtual temperature.
    q_modal = self._get_specific_humidity(state)
    cos_lat_grad_q = self.coords.horizontal.cos_lat_grad(q_modal, clip=False)
    nodal_cos_lat_grad_q = self.coords.horizontal.to_nodal(cos_lat_grad_q)
    coefficient = self.T_ref * (physics_specs.R_vapor - physics_specs.R)
    nodal_dot_term = (
        coefficient
        * self.coords.horizontal.sec2_lat
        * (
            nodal_cos_lat_grad_q[0] * aux_state.cos_lat_grad_log_sp[0]
            + nodal_cos_lat_grad_q[1] * aux_state.cos_lat_grad_log_sp[1]
        )
    )

    # TODO(dkochkov) Consider computing T_ref * q part implicitly.
    temperature = aux_state.temperature_variation + self.T_ref
    temperature_diff = (
        q * temperature * (physics_specs.R_vapor / physics_specs.R - 1)
    )
    nodal_surface_pressure = jnp.exp(
        self.coords.horizontal.to_nodal(state.log_surface_pressure)
    )

    geopotential_diff = self._get_geopotential_diff(
        temperature_diff,
        method=method,
        sharding=self.coords.dycore_sharding,
        surface_pressure=nodal_surface_pressure,
    )

    return -self.coords.horizontal.laplacian(
        self.coords.horizontal.to_modal(geopotential_diff)
    ) - self.coords.horizontal.to_modal(
        nodal_dot_term + nodal_laplacian_correction_term
    )

  @jax.named_call
  def vorticity_tendency_due_to_humidity(
      self,
      state: State,
      aux_state: Any,
  ) -> Array:
    """Computes vorticity tendencies due to humidity."""
    physics_specs = self.physics_specs
    q_modal = self._get_specific_humidity(state)
    cos_lat_grad_q = self.coords.horizontal.cos_lat_grad(q_modal, clip=False)
    nodal_cos_lat_grad_q = self.coords.horizontal.to_nodal(cos_lat_grad_q)
    nodal_cos_lat_grad_log_sp = aux_state.cos_lat_grad_log_sp
    coefficient = self.T_ref * (physics_specs.R_vapor - physics_specs.R)
    nodal_curl_term = (
        coefficient
        * self.coords.horizontal.sec2_lat
        * (
            nodal_cos_lat_grad_log_sp[0] * nodal_cos_lat_grad_q[1]
            - nodal_cos_lat_grad_log_sp[1] * nodal_cos_lat_grad_q[0]
        )
    )
    return self.coords.horizontal.to_modal(nodal_curl_term)

  @jax.named_call
  def _t_omega_over_sigma_sp(
      self, temperature_field: Array, g_term: Array, v_dot_grad_log_sp: Array
  ) -> Array:
    """Computes nodal terms of the form `T * omega / p` in temperature tendency.

    A helper function for evaluation of the terms in temperature tendency
    equation of the form:

      ∂T/∂t[n] ~ (T * ⍵/p)[n], where ⍵=dp/dt

    It uses the numerical scheme described in 'Numerical Methods for Fluid
    Dynamics' §8.6.3, eq. 8.124 which approximates ⍵/p as:

      ⍵/p[n] = v·∇(ln(ps))[n] - (1 / Δ𝜎[n]) * (𝛼[n] * sum(G[:n] * Δ𝜎[:n]) +
                                               𝛼[n-1] * sum(G[:n-1] * Δ𝜎[:n-1]))

    Args:
      temperature_field: the temperature (T) in nodal representation.
      g_term: the value `G` in nodal representation.
      v_dot_grad_log_sp: the inner product of velocity and gradient of surface
        pressure in nodal representation.

    Returns:
      Values of (T * ⍵/p) due to the provided T, G, v·∇(ln(ps)).
    """
    f = sigma_coordinates.cumulative_sigma_integral(
        g_term, self.coords.vertical, sharding=self.coords.dycore_sharding
    )
    alpha = get_sigma_ratios(self.coords.vertical)
    alpha = alpha[:, np.newaxis, np.newaxis]  # make alpha broadcast to `f`.
    del_𝜎 = self.coords.vertical.layer_thickness[:, np.newaxis, np.newaxis]
    padding = [(1, 0), (0, 0), (0, 0)]
    g_part = (alpha * f + jnp.pad(alpha * f, padding)[:-1, ...]) / del_𝜎
    return temperature_field * (v_dot_grad_log_sp - g_part)

  @jax.named_call
  def curl_and_div_tendencies(
      self,
      aux_state: DiagnosticStateSigma,
  ) -> tuple[Array, Array]:
    """Computes curl and divergence tendencies for vorticity ζ and divergence 𝛅.

    Computes to explicit tendencies (dζ_dt, d𝛅_dt) to due to curl and divergence
    terms in the primitive equations, as described in primitive equations notes:
    g3doc/primitive_equations.md Eq. (1)-(4).
    Specifically, the terms computed correspond to:

      dζ_dt = -k · ∇ ✕ ((ζ + f)(k ✕ v) + d𝜎_dt · ∂v/∂𝜎 + RT'∇(ln(p_s)))
      d𝛅_dt = - ∇ · ((ζ + f)(k ✕ v) + d𝜎_dt · ∂v/∂𝜎 + RT'∇(ln(p_s)))

    Args:
      aux_state: diagnostic state with pre-computed nodal values.

    Returns:
      Tuple of divergence and vorticity tendencies due to curl and divergence
      terms in the primitive equations.
    """
    sec2_lat = self.coords.horizontal.sec2_lat
    # note the cos_lat cancels out with sec2_lat and cos in derivative ops.
    u, v = aux_state.cos_lat_u
    total_vorticity = aux_state.vorticity + self.coriolis_parameter
    # note that u, v are switched to correspond to `k ✕ v = (-v, u)`.
    nodal_vorticity_u = -v * total_vorticity * sec2_lat
    nodal_vorticity_v = u * total_vorticity * sec2_lat
    # vertical and pressure gradient terms
    d𝜎_dt = aux_state.sigma_dot_full
    if self.include_vertical_advection:
      # vertical tendency is equal to `-1 * dot{sigma} * u`, hence negation here
      sigma_dot_u = -self._vertical_tendency(d𝜎_dt, u)
      sigma_dot_v = -self._vertical_tendency(d𝜎_dt, v)
    else:
      sigma_dot_u = 0
      sigma_dot_v = 0

    adjustment = self._virtual_temperature_adjustment(aux_state)
    rt = self.physics_specs.R * aux_state.temperature_variation * adjustment

    grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
    vertical_term_u = (sigma_dot_u + rt * grad_log_ps_u) * sec2_lat
    vertical_term_v = (sigma_dot_v + rt * grad_log_ps_v) * sec2_lat
    combined_u = self.coords.horizontal.to_modal(
        nodal_vorticity_u + vertical_term_u
    )
    combined_v = self.coords.horizontal.to_modal(
        nodal_vorticity_v + vertical_term_v
    )
    # computing tendencies
    dζ_dt = -self.coords.horizontal.curl_cos_lat(
        (combined_u, combined_v), clip=False
    )
    d𝛅_dt = -self.coords.horizontal.div_cos_lat(
        (combined_u, combined_v), clip=False
    )
    return (dζ_dt, d𝛅_dt)

  @jax.named_call
  def nodal_temperature_vertical_tendency(
      self,
      aux_state: DiagnosticStateSigma,
  ) -> Array | float:
    """Computes explicit vertical tendency of the temperature."""
    # two types of terms of sigma_dot * ∂T/∂𝜎
    # second term is zero if T_ref does not depend on layer_id.
    sigma_dot_explicit = aux_state.sigma_dot_explicit
    sigma_dot_full = aux_state.sigma_dot_full
    temperature_variation = aux_state.temperature_variation
    if self.include_vertical_advection:
      tendency = self._vertical_tendency(sigma_dot_full, temperature_variation)
    else:
      tendency = 0
    if np.unique(self.T_ref.ravel()).size > 1:
      # only non-zero if T_ref is not a constant
      tendency += self._vertical_tendency(sigma_dot_explicit, self.T_ref)
    return tendency

  @jax.named_call
  def nodal_temperature_adiabatic_tendency(
      self, aux_state: DiagnosticStateSigma
  ) -> Array:
    """Computes explicit temperature tendency due to adiabatic processes."""
    g_explicit = aux_state.u_dot_grad_log_sp
    g_full = g_explicit + aux_state.divergence
    mean_t_part = self._t_omega_over_sigma_sp(
        self.T_ref, g_explicit, aux_state.u_dot_grad_log_sp
    )
    if self.humidity_key is None:
      variation_t_part = self._t_omega_over_sigma_sp(
          aux_state.temperature_variation, g_full, aux_state.u_dot_grad_log_sp
      )
      return self.physics_specs.kappa * (mean_t_part + variation_t_part)
    else:
      gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
      heat_capacity_ratio = self.physics_specs.Cp_vapor / self.physics_specs.Cp
      q = self._get_specific_humidity(aux_state)
      # Here Tv refers to virtual temperature. The terms below capture
      # tendencies from full temperature variation and moist T_ref terms.
      variation_temperature_component = aux_state.temperature_variation * (
          (1 + (gas_const_ratio - 1) * q) / (1 + (heat_capacity_ratio - 1) * q)
      )
      humidity_reference_component = self.T_ref * (
          ((gas_const_ratio - heat_capacity_ratio) * q)
          / (1 + (heat_capacity_ratio - 1) * q)
      )
      variation_and_humidity_terms = (
          variation_temperature_component + humidity_reference_component
      )
      variation_and_Tv_part = self._t_omega_over_sigma_sp(
          variation_and_humidity_terms, g_full, aux_state.u_dot_grad_log_sp
      )
      return self.physics_specs.kappa * (mean_t_part + variation_and_Tv_part)

  @jax.named_call
  def nodal_log_pressure_tendency(
      self, aux_state: DiagnosticStateSigma
  ) -> Array:
    """Computes explicit tendency of the log_surface_pressure."""
    # computes -∑G[i] * ∆𝜎[i] where G[i] = u[i] · ∇(log(ps)).
    g = aux_state.u_dot_grad_log_sp
    return -sigma_coordinates.sigma_integral(g, self.coords.vertical)

  @jax.named_call
  def explicit_terms(self, state: State) -> State:
    """Computes explicit tendencies of the primitive equations."""
    aux_state = compute_diagnostic_state_sigma(state, self.coords)
    # tendencies that are computed in modal representation
    vorticity_dot, divergence_dot = self.curl_and_div_tendencies(aux_state)
    kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
    orography_tendency = self.orography_tendency()

    if self.humidity_key is not None:
      humidity_vort_correction_tendency = (
          self.vorticity_tendency_due_to_humidity(state, aux_state)
      )
      humidity_div_correction_tendency = (
          self.divergence_tendency_due_to_humidity(state, aux_state)
      )
      vorticity_dot += humidity_vort_correction_tendency
      divergence_dot += humidity_div_correction_tendency

    horizontal_tendency_fn = functools.partial(
        self.horizontal_scalar_advection, aux_state=aux_state
    )
    dT_dt_horizontal_nodal, dT_dt_horizontal_modal = horizontal_tendency_fn(
        aux_state.temperature_variation
    )
    tracers_horizontal_nodal_and_modal = jax.tree_util.tree_map(
        horizontal_tendency_fn, aux_state.tracers
    )
    # tendencies in nodal domain
    dT_dt_vertical = self.nodal_temperature_vertical_tendency(aux_state)
    dT_dt_adiabatic = self.nodal_temperature_adiabatic_tendency(aux_state)
    log_sp_tendency = self.nodal_log_pressure_tendency(aux_state)
    sigma_dot_full = aux_state.sigma_dot_full
    if self.include_vertical_advection:
      vertical_tendency_fn = functools.partial(
          self._vertical_tendency, sigma_dot_full
      )
    else:
      vertical_tendency_fn = lambda x: 0
    tracers_vertical_nodal = jax.tree_util.tree_map(
        vertical_tendency_fn, aux_state.tracers
    )
    # combining tendencies
    to_modal_fn = self.coords.horizontal.to_modal
    vorticity_tendency = vorticity_dot
    divergence_tendency = (
        divergence_dot + kinetic_energy_tendency + orography_tendency
    )
    temperature_tendency = (
        to_modal_fn(dT_dt_horizontal_nodal + dT_dt_vertical + dT_dt_adiabatic)
        + dT_dt_horizontal_modal
    )
    log_surface_pressure_tendency = to_modal_fn(log_sp_tendency)
    tracers_tendency = jax.tree_util.tree_map(
        lambda x, y_z: to_modal_fn(x + y_z[0]) + y_z[1],
        tracers_vertical_nodal,
        tracers_horizontal_nodal_and_modal,
    )
    tendency = State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
        tracers=tracers_tendency,
        sim_time=None if state.sim_time is None else 1.0,
    )
    # Note: clipping the final total wavenumber from the explicit tendencies
    # matches SPEEDY.
    return self.coords.horizontal.clip_wavenumbers(tendency)

  @jax.named_call
  def implicit_terms(self, state: State) -> State:
    """Returns the implicit terms of the primitive equations.

    See go/primitive-equations for more details on the implicit/explicit
    partitioning of the terms in the primitive equations.

    Args:
      state: the `State` from which to compute the implicit terms.

    Returns:
      A `State` containing the explicit terms of the primitive equations.
    """
    method = self.vertical_matmul_method
    if method is None:
      mesh = self.coords.spmd_mesh
      method = 'sparse' if mesh is not None and mesh.shape['z'] > 1 else 'dense'

    geopotential_diff = get_geopotential_diff_sigma(
        state.temperature_variation,
        self.coords.vertical,
        self.physics_specs.R,
        method=method,
        sharding=self.coords.dycore_sharding,
    )
    rt_log_p = (
        self.physics_specs.ideal_gas_constant
        * self.T_ref
        * state.log_surface_pressure
    )
    vorticity_implicit = jnp.zeros_like(state.vorticity)
    divergence_implicit = -self.coords.horizontal.laplacian(
        geopotential_diff + rt_log_p
    )
    temperature_variation_implicit = get_temperature_implicit_sigma(
        state.divergence,
        self.coords.vertical,
        self.reference_temperature,
        self.physics_specs.kappa,
        method=method,
        sharding=self.coords.dycore_sharding,
    )
    log_surface_pressure_implicit = -_vertical_matvec(
        self.coords.vertical.layer_thickness[np.newaxis], state.divergence
    )
    tracers_implicit = jax.tree_util.tree_map(jnp.zeros_like, state.tracers)
    return State(
        vorticity=vorticity_implicit,
        divergence=divergence_implicit,
        temperature_variation=temperature_variation_implicit,
        log_surface_pressure=log_surface_pressure_implicit,
        tracers=tracers_implicit,
        sim_time=None if state.sim_time is None else 0.0,
    )

  @jax.named_call
  def implicit_inverse(self, state: State, step_size: float) -> State:
    """Computes the inverse `(1 - step_size * implicit_terms)⁻¹.

    Args:
      state: the `State` to which the inverse will be applied.
      step_size: a value that depends on the choice of time integration method.

    Returns:
      The result of applying `(1 - step_size * implicit_terms)⁻¹.
    """
    if isinstance(step_size, jax.core.Tracer):
      # We require a static value for `eta` so that we can compute the inverse
      # in numpy. This allows us to use high precision and to precompute these
      # values for efficiency.
      raise TypeError(
          f'`step_size` must be concrete but a Tracer was passed: {step_size}. '
          'This error is likely caused by '
          '`jax.jit(primitive.inverse_terms)(state, eta). Instead, do '
          '`jax.jit(lambda s: primitive.inverse_terms(s, eta=eta))(state)`.'
      )

    implicit_matrix = _get_implicit_term_matrix_sigma(
        step_size,
        self.coords,
        self.reference_temperature,
        self.physics_specs.kappa,
        self.physics_specs.R,
    )
    assert implicit_matrix.dtype == np.float64

    # We can assign a set of indices to each quantity div, temp and logp
    layers = self.coords.vertical.layers
    div = slice(0, layers)
    temp = slice(layers, 2 * layers)
    logp = slice(2 * layers, 2 * layers + 1)
    temp_logp = slice(layers, 2 * layers + 1)

    def named_vertical_matvec(name):
      return jax.named_call(_vertical_matvec_per_wavenumber, name=name)

    if self.implicit_inverse_method == 'split':
      # Directly invert the implicit matrix, and apply vertical matrix-vector
      # products to each term. This is the fastest method in the unsharded case.
      inverse = np.linalg.inv(implicit_matrix)
      assert not np.isnan(inverse).any()

      inverted_divergence = (
          named_vertical_matvec('div_from_div')(
              inverse[:, div, div], state.divergence
          )
          + named_vertical_matvec('div_from_temp')(
              inverse[:, div, temp], state.temperature_variation
          )
          + named_vertical_matvec('div_from_logp')(
              inverse[:, div, logp], state.log_surface_pressure
          )
      )
      inverted_temperature_variation = (
          named_vertical_matvec('temp_from_div')(
              inverse[:, temp, div], state.divergence
          )
          + named_vertical_matvec('temp_from_temp')(
              inverse[:, temp, temp], state.temperature_variation
          )
          + named_vertical_matvec('temp_from_logp')(
              inverse[:, temp, logp], state.log_surface_pressure
          )
      )
      inverted_log_surface_pressure = (
          named_vertical_matvec('logp_from_div')(
              inverse[:, logp, div], state.divergence
          )
          + named_vertical_matvec('logp_from_temp')(
              inverse[:, logp, temp], state.temperature_variation
          )
          + named_vertical_matvec('logp_from_logp')(
              inverse[:, logp, logp], state.log_surface_pressure
          )
      )

    elif self.implicit_inverse_method == 'stacked':
      # Apply the matrix inverse once to concatenated inputs, then split.
      # This version exists mostly for pedagogical reasons. Numerically it is
      # doing the same calculation as 'split', but it turns out to be slower on
      # on TPUs.
      inverse = np.linalg.inv(implicit_matrix)
      assert not np.isnan(inverse).any()
      stacked_state = jnp.concatenate([
          state.divergence,
          state.temperature_variation,
          state.log_surface_pressure,
      ])
      stacked_inverse = named_vertical_matvec('inverse')(inverse, stacked_state)
      inverted_divergence = stacked_inverse[div]
      inverted_temperature_variation = stacked_inverse[temp]
      inverted_log_surface_pressure = stacked_inverse[logp]

    elif self.implicit_inverse_method == 'blockwise':
      # Use blockwise matrix inversion to reduce the number of matrix-vector
      # products. This is potentially faster in cases where matrix-vector
      # products are expensive, such as in the case of sharding across vertical
      # levels.
      #
      # Note that `implicit_matrix`` has a block-sparse form, where `I` denotes
      # the identity matrix:
      #   [  I   ηλg  ηx ]
      #   [ ηh    I    0 ]
      #   [ ηy    0    1 ]
      #
      # Setting G = [[ηλg, ηx]] and H = [[ηh], [ηy]], we can write this in
      # 2x2 block form as:
      #   [ I G ]
      #   [ H I ]
      #
      # Rather than divertly inverting the full matrix, we can use block-wise
      # matrix inversion:
      # https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
      #   [ I  G ]⁻¹ = [ (I - G H)⁻¹            0 ] @ [ I   -G ]
      #   [ H  I ]     [ 0            (I - H G)⁻¹ ]   [ -H   I ]
      #
      # Now we multiply by these two matrices sequentially. The action of the
      # first matrix is given by:
      #   [  I   -ηλg  -ηx ]   [ d ]     [ d - ηλgt - ηxσ ]
      #   [ -ηh    I     0 ] @ [ t ]  =  [    -ηhd + t    ]
      #   [ -ηy    0     1 ]   [ σ ]     [    -ηyd + σ    ]
      #
      # Because we can calculate matrix-vector products with `g` and `h` in a
      # matrix-free fashion with cumulative sums, this identity allows us to
      # only apply two matrix inverses with square matrices of size equal to the
      # number of vertical layers, versus four such inverses with the 'split'
      # implementation.
      GH = (
          implicit_matrix[:, div, temp_logp]
          @ implicit_matrix[:, temp_logp, div]
      )
      div_inverse = np.linalg.inv(np.eye(layers) - GH)

      η = step_size
      λ = self.coords.horizontal.laplacian_eigenvalues

      gt = get_geopotential_diff_sigma(
          state.temperature_variation,
          self.coords.vertical,
          self.physics_specs.R,
          method='sparse',
          sharding=self.coords.dycore_sharding,
      )
      div_from_temp = η * λ[np.newaxis, np.newaxis, :] * gt

      div_from_logp = named_vertical_matvec('div_from_logp')(
          implicit_matrix[:, div, logp], state.log_surface_pressure
      )
      inverted_divergence = named_vertical_matvec('div_solve')(
          div_inverse, state.divergence - div_from_temp - div_from_logp
      )
      HG = (
          implicit_matrix[:, temp_logp, div]
          @ implicit_matrix[:, div, temp_logp]
      )
      temp_logp_inverse = np.linalg.inv(np.eye(layers + 1) - HG)

      hd = -get_temperature_implicit_sigma(
          state.divergence,
          self.coords.vertical,
          self.reference_temperature,
          self.physics_specs.kappa,
          method='sparse',
          sharding=self.coords.dycore_sharding,
      )
      temp_from_div = η * hd
      temp_part = state.temperature_variation - temp_from_div

      logp_from_div = named_vertical_matvec('logp_from_div')(
          implicit_matrix[:, logp, div], state.divergence
      )
      logp_part = state.log_surface_pressure - logp_from_div

      inverted_temperature_variation = named_vertical_matvec(
          'temp_solve_from_temp'
      )(temp_logp_inverse[:, :-1, :-1], temp_part) + named_vertical_matvec(
          'temp_solve_from_logp'
      )(
          temp_logp_inverse[:, :-1:, -1:], logp_part
      )
      inverted_log_surface_pressure = named_vertical_matvec(
          'logp_solve_from_temp'
      )(temp_logp_inverse[:, -1:, :-1], temp_part) + named_vertical_matvec(
          'logp_solve_from_temp'
      )(
          temp_logp_inverse[:, -1:, -1:], logp_part
      )
    else:
      raise ValueError(
          f'invalid implicit_inverse_method {self.implicit_inverse_method}'
      )

    inverted_vorticity = state.vorticity
    inverted_tracers = state.tracers

    return State(
        inverted_vorticity,
        inverted_divergence,
        inverted_temperature_variation,
        inverted_log_surface_pressure,
        inverted_tracers,
        sim_time=state.sim_time,
    )


################################################################################
# Primitive equations on Hybrid coordinates implementation.
################################################################################


@tree_math.struct
class DiagnosticStateHybrid:
  """Stores nodal diagnostic values used to compute explicit tendencies."""

  vorticity: Array
  divergence: Array
  temperature_variation: Array
  cos_lat_u: tuple[Array, Array]
  mass_flux_explicit: Array
  mass_flux_full: Array
  cos_lat_grad_log_sp: Array
  u_dot_grad_log_sp: Array
  tracers: dict[str, Array]
  layer_pressure_thickness: Array


@jax.named_call
def _get_vertical_discretization_coeffs_numpy(
    coordinates: hybrid_coordinates.HybridCoordinates,
    p_s_ref: float,
):
  """Computes coefficients for vertical discretization in Simmons & Burridge."""
  # Pressure at interfaces at reference surface pressure
  p_half_ref = coordinates.a_boundaries + coordinates.b_boundaries * p_s_ref
  dp_ref = np.diff(p_half_ref)

  if np.any(dp_ref <= 0):
    raise ValueError('Pressure must decrease monotonically with height.')

  # Alpha coefficient from S&B 1981, for geopotential on full levels
  p_k_minus_half = p_half_ref[:-1]
  p_k_plus_half = p_half_ref[1:]
  # Avoid log(0) at the model top
  safe_p_k_minus_half = np.maximum(p_k_minus_half, 1e-6 * p_s_ref)
  safe_p_k_plus_half = np.maximum(p_k_plus_half, 1e-6 * p_s_ref)
  log_p_interface_ratio = np.log(safe_p_k_plus_half / safe_p_k_minus_half)
  alpha_k = 1 - (p_k_minus_half / dp_ref) * log_p_interface_ratio
  return alpha_k, log_p_interface_ratio


@jax.named_call
def _get_vertical_discretization_coeffs(
    coordinates: hybrid_coordinates.HybridCoordinates,
    p_surface: Array,
):
  """Computes coefficients for vertical discretization in Simmons & Burridge."""
  # Pressure at interfaces at reference surface pressure
  a_boundaries = coordinates.a_boundaries
  b_boundaries = coordinates.b_boundaries

  if np.ndim(p_surface) > 0:
    assert p_surface.ndim == 3
    # Make a, b broadcastable to p_surface with lon-lat dimensions.
    a_boundaries = a_boundaries[:, np.newaxis, np.newaxis]
    b_boundaries = b_boundaries[:, np.newaxis, np.newaxis]

  p_half_ref = a_boundaries + b_boundaries * p_surface
  dp_ref = jax_numpy_utils.diff(p_half_ref, axis=0)
  # Alpha coefficient from S&B 1981, for geopotential on full levels
  p_k_minus_half = lax.slice_in_dim(p_half_ref, 0, -1)
  p_k_plus_half = lax.slice_in_dim(p_half_ref, 1, None)
  # Avoid log(0) at the model top
  safe_p_k_minus_half = jnp.maximum(p_k_minus_half, 1e-6 * p_surface)
  safe_p_k_plus_half = jnp.maximum(p_k_plus_half, 1e-6 * p_surface)
  log_p_interface_ratio = jnp.log(safe_p_k_plus_half / safe_p_k_minus_half)
  alpha_k = 1 - (p_k_minus_half / dp_ref) * log_p_interface_ratio
  return alpha_k, log_p_interface_ratio


@jax.named_call
def compute_diagnostic_state_hybrid(
    state: State,
    coords: coordinate_systems.CoordinateSystem,
) -> DiagnosticStateHybrid:
  """Computes DiagnosticState in nodal basis based on the modal `state`."""

  def to_nodal_fn(x):
    return coords.horizontal.to_nodal(x)

  nodal_vorticity = to_nodal_fn(state.vorticity)
  nodal_divergence = to_nodal_fn(state.divergence)
  nodal_temperature_variation = to_nodal_fn(state.temperature_variation)
  tracers = jax.tree_util.tree_map(to_nodal_fn, state.tracers)
  nodal_cos_lat_u = jax.tree_util.tree_map(
      to_nodal_fn,
      spherical_harmonic.get_cos_lat_vector(
          state.vorticity, state.divergence, coords.horizontal, clip=False
      ),
  )
  cos_lat_grad_log_sp = coords.horizontal.cos_lat_grad(
      state.log_surface_pressure, clip=False
  )
  nodal_cos_lat_grad_log_sp = to_nodal_fn(cos_lat_grad_log_sp)
  nodal_u_dot_grad_log_sp = sum(
      jax.tree_util.tree_map(
          lambda x, y: x * y * coords.horizontal.sec2_lat,
          nodal_cos_lat_u,
          nodal_cos_lat_grad_log_sp,
      )
  )

  # Hybrid vertical velocity / mass flux calculation.
  nodal_surface_pressure = jnp.exp(to_nodal_fn(state.log_surface_pressure))
  delta_p = coords.vertical.layer_thickness(nodal_surface_pressure)
  delta_b = coords.vertical.sigma_thickness[:, np.newaxis, np.newaxis]

  # D_k = div(v * dp) = dp * div(v) + v . grad(dp)
  # grad(dp) = grad(da + db * ps) = db * ps * grad(ln ps)
  d_k_term1 = delta_p * nodal_divergence
  d_k_term2 = delta_b * nodal_surface_pressure * nodal_u_dot_grad_log_sp
  d_k_full = d_k_term1 + d_k_term2
  d_k_explicit = d_k_term2  # Explicit part (advection)

  # M_{k+1/2} = - sum_{r=1}^k D_r + B_{k+1/2} sum_{r=1}^N D_r
  def compute_mass_flux(d_k):
    sum_d = jnp.sum(d_k, axis=0, keepdims=True)
    cumsum_d = jax_numpy_utils.cumsum(d_k, axis=0)
    # pad top with 0
    cumsum_d_padded = jnp.pad(cumsum_d, ((1, 0), (0, 0), (0, 0)))
    b_boundaries = coords.vertical.b_boundaries[:, np.newaxis, np.newaxis]
    return -cumsum_d_padded + b_boundaries * sum_d

  mass_flux_full = compute_mass_flux(d_k_full)
  mass_flux_explicit = compute_mass_flux(d_k_explicit)

  return DiagnosticStateHybrid(
      vorticity=nodal_vorticity,
      divergence=nodal_divergence,
      temperature_variation=nodal_temperature_variation,
      cos_lat_u=nodal_cos_lat_u,
      mass_flux_explicit=mass_flux_explicit,
      mass_flux_full=mass_flux_full,
      cos_lat_grad_log_sp=nodal_cos_lat_grad_log_sp,
      u_dot_grad_log_sp=nodal_u_dot_grad_log_sp,
      tracers=tracers,
      layer_pressure_thickness=delta_p,
  )


def get_geopotential_weights_hybrid(
    coordinates: hybrid_coordinates.HybridCoordinates,
    ideal_gas_constant: float,
    p_s_ref: float,
) -> np.ndarray:
  """Returns a matrix of weights used to compute the geopotential."""
  alpha_k, log_p_ratio = _get_vertical_discretization_coeffs_numpy(
      coordinates, p_s_ref
  )
  layers = coordinates.layers
  weights = np.zeros([layers, layers])
  # Per S&B 1981, Eq 2.10, geopotential is found by integrating up from surface.
  # Φ'_k = R * [ Σ_{j=k+1 to N} T'_j Δ(ln p)_j + α_k T'_k ].
  # This corresponds to an upper triangular matrix operation on T'.
  for k in range(layers):
    # Term for temperature at the current level k.
    weights[k, k] = alpha_k[k]
    # Terms for temperatures at levels j below k (j > k).
    for j in range(k + 1, layers):
      weights[k, j] = log_p_ratio[j]
  return ideal_gas_constant * weights


def get_geopotential_diff_hybrid(
    temperature: Array,
    coordinates: hybrid_coordinates.HybridCoordinates,
    ideal_gas_constant: float,
    p_surface: float | Array,
    method: str = 'dense',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Calculate the implicit geopotential term."""
  # If p_surface is spatially varying, we cannot use the dense matrix method
  # because the matrix would be spatially dependent.
  is_variable_ps = np.ndim(p_surface) > 0
  if is_variable_ps and method == 'dense':
    raise ValueError(
        '`dense` method is not supported for spatially varying p_surface, '
        'use `sparse` method instead.'
    )

  if method == 'dense':
    weights = get_geopotential_weights_hybrid(
        coordinates, ideal_gas_constant, p_s_ref=p_surface
    )
    return _vertical_matvec(weights, temperature)
  elif method == 'sparse':
    if is_variable_ps:
      alpha_k, log_p_ratio = _get_vertical_discretization_coeffs(
          coordinates, p_surface
      )
    else:
      alpha_k, log_p_ratio = _get_vertical_discretization_coeffs_numpy(
          coordinates, p_surface
      )
    if np.ndim(log_p_ratio) == 1:
      log_p_ratio = log_p_ratio[:, np.newaxis, np.newaxis]
      alpha_k = alpha_k[:, np.newaxis, np.newaxis]

    # Φ'_k = R * [ Σ_{j=k+1 to N} T'_j Δ(ln p)_j + α_k T'_k ]
    weighted_temp = temperature * log_p_ratio
    # Sum from j=k+1 to N-1 (0-indexed)
    full_integral = jax_numpy_utils.reverse_cumsum(
        weighted_temp, axis=0, sharding=sharding
    )
    integral_term = full_integral - weighted_temp  # Removes the j=k term
    diagonal_term = alpha_k * temperature
    return ideal_gas_constant * (integral_term + diagonal_term)
  else:
    raise ValueError(f'unknown {method=} for get_geopotential_diff')


def get_geopotential_on_hybrid(
    temperature: typing.Array,
    surface_pressure: typing.Array | float,
    specific_humidity: typing.Array | None = None,
    clouds: typing.Array | None = None,
    *,
    nodal_orography: typing.Array,
    coordinates: hybrid_coordinates.HybridCoordinates,
    gravity_acceleration: float,
    ideal_gas_constant: float,
    water_vapor_gas_constant: float | None = None,
    sharding: jax.sharding.NamedSharding | None = None,
) -> jnp.ndarray:
  """Computes geopotential in nodal space using nodal temperature and moisture.

  If `specific_humidity` is None, computes dry geopotential. If clouds are
  provided, the cloud condensation is subtracted from the virtual temperature.

  Args:
    temperature: nodal values of temperature.
    surface_pressure: nodal values of surface pressure.
    specific_humidity: nodal values of specific humidity. If provided, moisture
      effects are included in geopotential calculation.
    clouds: nodal values of cloud condensate.
    nodal_orography: nodal values of orography.
    coordinates: hybrid coordinates.
    gravity_acceleration: gravity.
    ideal_gas_constant: ideal gas constant for dry air.
    water_vapor_gas_constant: ideal gas constant for water vapor. Must be
      provided if `specific_humidity` is provided.
    sharding: optional sharding.

  Returns:
    Nodal geopotential.
  """
  surface_geopotential = nodal_orography * gravity_acceleration
  if specific_humidity is not None:
    if water_vapor_gas_constant is None:
      raise ValueError(
          'Must provide `water_vapor_gas_constant` with `specific_humidity`.'
      )
    gas_const_ratio = water_vapor_gas_constant / ideal_gas_constant
    cloud_effect = 0.0 if clouds is None else clouds
    virtual_temp = temperature * (
        1 + (gas_const_ratio - 1) * specific_humidity - cloud_effect
    )
  else:
    virtual_temp = temperature
  geopotential_diff = get_geopotential_diff_hybrid(
      virtual_temp,
      coordinates,
      ideal_gas_constant,
      surface_pressure,
      method='sparse',
      sharding=sharding,
  )
  return surface_geopotential + geopotential_diff


def get_temperature_implicit_weights_hybrid(
    coordinates: hybrid_coordinates.HybridCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
    p_s_ref: float,
) -> np.ndarray:
  """Returns weights used to compute implicit terms for the temperature."""
  if (
      reference_temperature.ndim != 1
      or reference_temperature.shape[-1] != coordinates.layers
  ):
    raise ValueError(
        '`reference_temp` must be a vector of length `coordinates.layers`; '
        f'got shape {reference_temperature.shape} and '
        f'{coordinates.layers} layers.'
    )
  p = np.tril(np.ones([coordinates.layers, coordinates.layers]))
  alpha_k, _ = _get_vertical_discretization_coeffs_numpy(coordinates, p_s_ref)
  alpha = alpha_k[..., np.newaxis]

  p_alpha = p * alpha
  p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
  p_alpha_shifted[0] = 0

  delta_p_ref = (
      coordinates.pressure_thickness + coordinates.sigma_thickness * p_s_ref
  )

  h0 = (
      kappa
      * reference_temperature[..., np.newaxis]
      * (p_alpha + p_alpha_shifted)
      / delta_p_ref[..., np.newaxis]
  )
  temp_diff = np.diff(reference_temperature)
  thickness_sum = delta_p_ref[:-1] + delta_p_ref[1:]
  k0 = np.concatenate((temp_diff / thickness_sum, [0]), axis=0)[..., np.newaxis]
  # This part of the vertical advection scheme for T_ref should depend on sigma
  # for the definition of vertical velocity.
  # The implicit vertical velocity (eta-dot) is proportional to the vertical
  # integral of divergence weighted by d(sigma), not d(p)/p_s_ref.
  # Therefore, we use `sigma_thickness` for the cumulative sum.
  thickness_cumulative = np.cumsum(coordinates.sigma_thickness)[..., np.newaxis]
  k1 = p - thickness_cumulative
  k = k0 * k1
  k_shifted = np.roll(k, 1, axis=0)
  k_shifted[0] = 0
  return (h0 - k - k_shifted) * delta_p_ref


def get_temperature_implicit_hybrid(
    divergence: Array,
    coordinates: hybrid_coordinates.HybridCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
    p_s_ref: float,
    method: str = 'dense',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Calculate the implicit temperature term."""
  weights = -get_temperature_implicit_weights_hybrid(
      coordinates, reference_temperature, kappa, p_s_ref
  )

  if method == 'dense':
    return _vertical_matvec(weights, divergence)
  elif method == 'sparse':
    diag_weights = np.diag(weights)
    up_weights = np.concatenate([[0], weights[1:, 0]])
    down_weights = np.concatenate([weights[:-1, -1], [0]])
    up_divergence = (
        jax_numpy_utils.cumsum(divergence, axis=0, sharding=sharding)
        - divergence
    )
    result = (
        up_weights[:, np.newaxis, np.newaxis] * up_divergence
        + diag_weights[:, np.newaxis, np.newaxis] * divergence
    )
    if (down_weights != 0).any():
      down_divergence = (
          jax_numpy_utils.reverse_cumsum(divergence, axis=0, sharding=sharding)
          - divergence
      )
      result += down_weights[:, np.newaxis, np.newaxis] * down_divergence
    return result
  else:
    raise ValueError(f'unknown {method=} for get_temperature_implicit')


@jax.named_call
def _get_pgf_lps_coefficient(
    temperature: Array,
    surface_pressure: Array,
    coords: hybrid_coordinates.HybridCoordinates,
    R: float,
) -> Array:
  """Computes PGF coefficient from R*T*∇ln(p) term."""

  # Discretized term from Simmons & Burridge (1981) to ensure conservation.
  # The term corresponds to `C.'II_k` in `C_k · ∇ln(p_s)`. Defining:
  # vertical_weight_psg = [ ln(p_{k+1/2}/p_{k-1/2}) * B_{k-1/2} + α_k * ΔB_k ]
  # C_k = (R T_v / Δp_k) * vertical_weight_psg * p_s
  alpha_k, log_p_interface_ratio = _get_vertical_discretization_coeffs(
      coords, surface_pressure
  )
  b_boundaries = coords.b_boundaries
  # b_minus is B_{k-1/2} (top of layer k)
  b_minus = b_boundaries[:-1]
  db = jax_numpy_utils.diff(b_boundaries, axis=0)

  # Broadcast to match shape (layers, lat, lon)
  if np.ndim(surface_pressure) > 0:
    assert surface_pressure.ndim == 3
    b_minus = b_minus[:, np.newaxis, np.newaxis]
    db = db[:, np.newaxis, np.newaxis]

  dp = coords.layer_thickness(surface_pressure)
  # Avoid division by zero for empty layers
  dp = jnp.maximum(dp, 1e-6 * surface_pressure)
  vertical_weight_psg = log_p_interface_ratio * b_minus + alpha_k * db
  return (R * temperature / dp) * vertical_weight_psg * surface_pressure


def _get_pgf_lps_coefficient_numpy(
    temperature: np.ndarray,
    surface_pressure: float,
    coords: hybrid_coordinates.HybridCoordinates,
    R: float,
) -> np.ndarray:
  """NumPy version of PGF coefficient for the implicit matrix."""
  # Discretized term from Simmons & Burridge (1981) to ensure conservation.
  # The term corresponds to `C.'II_k` in `C_k · ∇ln(p_s)`. Defining:
  # vertical_weight_psg = [ ln(p_{k+1/2}/p_{k-1/2}) * B_{k-1/2} + α_k * ΔB_k ]
  # C_k = (R T_v / Δp_k) * vertical_weight_psg * p_s
  alpha_k, log_p_interface_ratio = _get_vertical_discretization_coeffs_numpy(
      coords, surface_pressure
  )
  b_boundaries = coords.b_boundaries
  # b_minus is B_{k-1/2} (top of layer k)
  b_minus = b_boundaries[:-1]
  db = np.diff(b_boundaries, axis=0)

  # Pressure thickness at full levels
  # dp_k = p_{k+1/2} - p_{k-1/2} = (A_{k+1/2} - A_{k-1/2}) + (B_{k+1/2} - B_{k-1/2}) * p_s
  # Since p_s is scalar here (reference pressure), this is just the layer thickness
  # computed by the coordinates object for that pressure.
  p_half = coords.a_boundaries + coords.b_boundaries * surface_pressure
  dp = np.diff(p_half, axis=0)
  dp = np.maximum(dp, 1e-6 * surface_pressure)

  vertical_weight_psg = log_p_interface_ratio * b_minus + alpha_k * db
  return (R * temperature / dp) * vertical_weight_psg * surface_pressure


def _get_implicit_term_matrix_hybrid(
    eta, coords, reference_temperature, kappa, ideal_gas_constant, p_s_ref
) -> np.ndarray:
  """Returns a matrix corresponding to `PrimitiveEquations.implicit_terms`."""
  eye = np.eye(coords.vertical.layers)[np.newaxis]
  lam = coords.horizontal.laplacian_eigenvalues
  g = get_geopotential_weights_hybrid(
      coords.vertical, ideal_gas_constant, p_s_ref=p_s_ref
  )
  h = get_temperature_implicit_weights_hybrid(
      coords.vertical, reference_temperature, kappa, p_s_ref=p_s_ref
  )
  pgf_lps_coeff = _get_pgf_lps_coefficient_numpy(
      reference_temperature, p_s_ref, coords.vertical, ideal_gas_constant
  )
  levels = coords.vertical
  effective_sigma_thickness = (
      levels.pressure_thickness / p_s_ref + levels.sigma_thickness
  )
  l = coords.horizontal.modal_shape[1]
  j = k = coords.vertical.layers
  row0 = np.concatenate(
      [
          np.broadcast_to(eye, [l, j, k]),
          eta * np.einsum('l,jk->ljk', lam, g),
          eta * np.einsum('l,j->lj', lam, pgf_lps_coeff)[..., np.newaxis],
      ],
      axis=2,
  )
  row1 = np.concatenate(
      [
          eta * np.broadcast_to(h[np.newaxis], [l, j, k]),
          np.broadcast_to(eye, [l, j, k]),
          np.zeros([l, j, 1]),
      ],
      axis=2,
  )
  row2 = np.concatenate(
      [
          np.broadcast_to(
              eta * effective_sigma_thickness[np.newaxis, :], [l, 1, k]
          ),
          np.zeros([l, 1, k]),
          np.ones([l, 1, 1]),
      ],
      axis=2,
  )
  return np.concatenate((row0, row1, row2), axis=1)


@jax.named_call
def hybrid_vertical_advection(
    mass_flux: Array,
    x: Array,
    pressure_thickness: Array,
    axis: int = -3,
) -> jax.Array:
  """Computes vertical advection on hybrid coordinates."""
  x_diff = jax_numpy_utils.diff(x, axis=axis)
  mass_flux_internal = lax.slice_in_dim(mass_flux, 1, -1, axis=axis)
  flux = mass_flux_internal * x_diff
  padding = [(0, 0)] * x.ndim
  padding[axis] = (1, 1)
  flux_padded = jnp.pad(flux, padding)
  flux_sum = lax.slice_in_dim(
      flux_padded, 1, None, axis=axis
  ) + lax.slice_in_dim(flux_padded, 0, -1, axis=axis)
  return -0.5 * flux_sum / pressure_thickness


@dataclasses.dataclass
class PrimitiveEquationsHybrid(PrimitiveEquationsBase):
  """Primitive Equations solved on terrain following hybrid coordinates.

  Note: this implementation has not been thoroughly verified and is added for
  research purposes. If you notice any inconsistencies or issues with this
  solver, please open an issues.
  """

  vertical_advection: Callable[..., jax.Array] = dataclasses.field(
      default=hybrid_coordinates.centered_vertical_advection, kw_only=True
  )
  reference_surface_pressure: typing.Quantity = dataclasses.field(
      default=(101325.0 * scales.units.pascal), kw_only=True
  )
  hpa_quantity: typing.Quantity = dataclasses.field(
      default=scales.units.hPa, kw_only=True
  )

  def __post_init__(self):
    super().__post_init__()
    nondim_reference_surface_pressure = self.physics_specs.nondimensionalize(
        self.reference_surface_pressure
    )
    if nondim_reference_surface_pressure <= 0:
      raise ValueError('`reference_surface_pressure` must be positive.')
    self._nondim_reference_surface_pressure = nondim_reference_surface_pressure
    nondim_a_boundaries = self.physics_specs.nondimensionalize(
        self.coords.vertical.a_boundaries * self.hpa_quantity
    )
    self.nondim_levels = hybrid_coordinates.HybridCoordinates(
        a_boundaries=nondim_a_boundaries,
        b_boundaries=self.coords.vertical.b_boundaries,
    )
    nondim_coords = dataclasses.replace(
        self.coords,
        vertical=self.nondim_levels,
    )
    self.nondim_coords = nondim_coords

  @property
  def p_s_ref(self) -> float:
    return self._nondim_reference_surface_pressure

  @jax.named_call
  def _vertical_tendency(
      self, w: Array, x: Array, pressure_thickness: Array | None = None
  ) -> Array:
    """Computes vertical nodal tendency of `x`."""
    if pressure_thickness is not None:
      return hybrid_vertical_advection(w, x, pressure_thickness)
    else:
      return self.vertical_advection(w, x, self.coords.vertical)

  def _get_geopotential_diff(
      self,
      temperature_diff: Array,
      method: str,
      sharding: jax.sharding.NamedSharding | None,
      surface_pressure: Array | None = None,
  ) -> Array:
    """Returns geopotential difference in hybrid coordinates."""
    return get_geopotential_diff_hybrid(
        temperature_diff,
        self.nondim_levels,
        self.physics_specs.R,
        surface_pressure,
        method=method,
        sharding=sharding,
    )

  @jax.named_call
  def _t_omega_over_p_hybrid(
      self,
      temperature_field: Array,
      g_term: Array,
      v_dot_grad_ln_p: Array,
      nodal_surface_pressure: Array,
  ) -> Array:
    """Computes nodal terms of the form `T*omega/p` in temperature tendency."""
    # integrand is divergence of mass flux: g_term * dp
    levels = self.nondim_levels
    integrand = g_term * levels.layer_thickness(nodal_surface_pressure)
    f = jax_numpy_utils.cumsum(
        integrand, axis=0, sharding=self.coords.dycore_sharding
    )
    alpha_k, _ = _get_vertical_discretization_coeffs_numpy(
        self.nondim_levels, self.p_s_ref
    )
    alpha = alpha_k[:, np.newaxis, np.newaxis]

    dp = levels.layer_thickness(nodal_surface_pressure)
    padding = [(1, 0), (0, 0), (0, 0)]
    # Discretization of the vertical integral term in omega/p equation
    g_part = (alpha * f + jnp.pad(alpha * f, padding)[:-1, ...]) / dp
    return temperature_field * (v_dot_grad_ln_p - g_part)

  @jax.named_call
  def curl_and_div_tendencies(
      self,
      aux_state: DiagnosticStateHybrid,
      nodal_surface_pressure: Array,
  ) -> tuple[Array, Array]:
    """Computes tendencies for vorticity ζ and divergence 𝛅."""
    sec2_lat = self.nondim_coords.horizontal.sec2_lat
    u, v = aux_state.cos_lat_u
    total_vorticity = aux_state.vorticity + self.coriolis_parameter
    nodal_vorticity_u = -v * total_vorticity * sec2_lat
    nodal_vorticity_v = u * total_vorticity * sec2_lat
    mass_flux = aux_state.mass_flux_full
    delta_p = aux_state.layer_pressure_thickness
    if self.include_vertical_advection:
      sigma_dot_u = -self._vertical_tendency(
          mass_flux, u, pressure_thickness=delta_p
      )
      sigma_dot_v = -self._vertical_tendency(
          mass_flux, v, pressure_thickness=delta_p
      )
    else:
      sigma_dot_u = 0
      sigma_dot_v = 0

    # Explicit part of PGF is the term associated with temperature.
    # The implicit solver uses a linearized PGF coefficient based on p_s_ref.
    # We must add the residual (full - linear) for the reference temperature
    # to the explicit tendencies to correctly capture PGF over topography.
    pgf_coeff = _get_pgf_lps_coefficient(
        aux_state.temperature_variation + self.T_ref,
        nodal_surface_pressure,
        self.nondim_levels,
        self.physics_specs.R,
    )
    pgf_coeff_ref_linear = _get_pgf_lps_coefficient_numpy(
        self.reference_temperature,
        self.p_s_ref,
        self.nondim_levels,
        self.physics_specs.R,
    )
    pgf_coeff_ref_linear = jnp.array(pgf_coeff_ref_linear)[
        ..., np.newaxis, np.newaxis
    ]
    pgf_coeff = pgf_coeff * self._virtual_temperature_adjustment(aux_state)

    # We subtract the dry linear term because the implicit solver adds it
    # (and does not account for moisture).
    pgf_coeff -= pgf_coeff_ref_linear
    grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
    vertical_term_u = (sigma_dot_u + pgf_coeff * grad_log_ps_u) * sec2_lat
    vertical_term_v = (sigma_dot_v + pgf_coeff * grad_log_ps_v) * sec2_lat
    combined_u = self.coords.horizontal.to_modal(
        nodal_vorticity_u + vertical_term_u
    )
    combined_v = self.coords.horizontal.to_modal(
        nodal_vorticity_v + vertical_term_v
    )
    dζ_dt = -self.coords.horizontal.curl_cos_lat(
        (combined_u, combined_v), clip=False
    )
    d𝛅_dt = -self.coords.horizontal.div_cos_lat(
        (combined_u, combined_v), clip=False
    )
    return (dζ_dt, d𝛅_dt)

  @jax.named_call
  def nodal_temperature_vertical_tendency(
      self,
      aux_state: DiagnosticStateHybrid,
  ) -> Array | float:
    """Computes explicit vertical tendency of the temperature."""
    mass_flux_explicit = aux_state.mass_flux_explicit
    mass_flux_full = aux_state.mass_flux_full
    temperature_variation = aux_state.temperature_variation
    delta_p = aux_state.layer_pressure_thickness
    if self.include_vertical_advection:
      tendency = self._vertical_tendency(
          mass_flux_full, temperature_variation, pressure_thickness=delta_p
      )
    else:
      tendency = 0
    if np.unique(self.T_ref.ravel()).size > 1:
      tendency += self._vertical_tendency(
          mass_flux_explicit, self.T_ref, pressure_thickness=delta_p
      )
    return tendency

  @jax.named_call
  def nodal_temperature_adiabatic_tendency(
      self, aux_state: DiagnosticStateHybrid, nodal_surface_pressure: Array
  ) -> Array:
    """Computes explicit temperature tendency due to adiabatic processes."""
    levels = self.nondim_levels
    # Consistent full-level pressure definition for energy conservation.
    # We use the same weights as in the PGF calculation
    # (Simmons & Burridge 1981).
    alpha_k, log_p_ratio = _get_vertical_discretization_coeffs(
        levels, nodal_surface_pressure
    )
    b_boundaries = levels.b_boundaries[:, np.newaxis, np.newaxis]
    b_minus = b_boundaries[:-1]
    db = jax_numpy_utils.diff(b_boundaries, axis=0)

    # Vertical weight Wₖ = ln(p_upper/p_lower) * B_minus + αₖ * ΔB
    vertical_weight_psg = log_p_ratio * b_minus + alpha_k * db

    # Compute v·∇ln(p) = (v·∇ln pₛ) * (pₛ / Δp) * Wₖ
    dp = levels.layer_thickness(nodal_surface_pressure)
    # Avoid division by zero for empty layers
    dp = jnp.maximum(dp, 1e-6 * nodal_surface_pressure)
    scaled_v_dot_grad_log_sp = (
        aux_state.u_dot_grad_log_sp
        * nodal_surface_pressure
        * vertical_weight_psg
        / dp
    )
    # `g_term` for `_t_omega_over_p_hybrid` is `∇·(vΔp) / Δp`.
    # `∇·(vΔp) = Δp(∇·v) + v·(∇Δp)`.
    # `v·(∇Δp) = v·(∇(ΔB p_s)) = (v·∇ln(p_s)) * p_s * ΔB`.
    dp_k = levels.layer_thickness(nodal_surface_pressure)
    # This term `integrand_g` is the `g_term` passed to the helper.
    integrand_g = aux_state.divergence + (
        aux_state.u_dot_grad_log_sp
        * nodal_surface_pressure
        * levels.sigma_thickness[:, np.newaxis, np.newaxis]
        / dp_k
    )
    # Split g into explicit (udg part) and implicit (divergence part)
    g_explicit = integrand_g - aux_state.divergence
    g_full = integrand_g

    mean_t_part = self._t_omega_over_p_hybrid(
        self.T_ref,
        g_explicit,
        scaled_v_dot_grad_log_sp,
        nodal_surface_pressure,
    )
    if self.humidity_key is None:
      variation_t_part = self._t_omega_over_p_hybrid(
          aux_state.temperature_variation,
          g_full,
          scaled_v_dot_grad_log_sp,
          nodal_surface_pressure,
      )
      return self.physics_specs.kappa * (mean_t_part + variation_t_part)
    else:
      gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
      heat_capacity_ratio = self.physics_specs.Cp_vapor / self.physics_specs.Cp
      q = self._get_specific_humidity(aux_state)
      variation_temperature_component = aux_state.temperature_variation * (
          (1 + (gas_const_ratio - 1) * q) / (1 + (heat_capacity_ratio - 1) * q)
      )
      humidity_reference_component = self.T_ref * (
          ((gas_const_ratio - heat_capacity_ratio) * q)
          / (1 + (heat_capacity_ratio - 1) * q)
      )
      variation_and_humidity_terms = (
          variation_temperature_component + humidity_reference_component
      )
      variation_and_Tv_part = self._t_omega_over_p_hybrid(
          variation_and_humidity_terms,
          g_full,
          scaled_v_dot_grad_log_sp,
          nodal_surface_pressure,
      )
      return self.physics_specs.kappa * (mean_t_part + variation_and_Tv_part)

  @jax.named_call
  def nodal_log_pressure_tendency(
      self, aux_state: DiagnosticStateHybrid, nodal_surface_pressure: Array
  ) -> Array:
    """Computes explicit tendency of the log_surface_pressure."""
    levels = self.nondim_levels
    # Explicit tendency = Total tendency - Implicit tendency.
    # Total d(ln ps)/dt = - Σ [ (ΔA/ps + ΔB)div + ΔB*udg ]
    # Implicit d(ln ps)/dt = - Σ [ (ΔA/ps_ref + ΔB)div ]
    # The explicit part is the difference.
    delta_a = levels.pressure_thickness[:, np.newaxis, np.newaxis]
    delta_b = levels.sigma_thickness[:, np.newaxis, np.newaxis]
    # Linearization error from the divergence term is treated explicitly
    div_a_term = jnp.sum(
        aux_state.divergence
        * delta_a
        * (1 / nodal_surface_pressure - 1 / self.p_s_ref),
        axis=0,
    )
    # Advective term is fully explicit.
    udg_b_term = jnp.sum(aux_state.u_dot_grad_log_sp * delta_b, axis=0)
    return -(div_a_term + udg_b_term)

  @jax.named_call
  def explicit_terms(self, state: State) -> State:
    """Computes explicit tendencies of the primitive equations."""
    aux_state = compute_diagnostic_state_hybrid(state, self.nondim_coords)
    nodal_surface_pressure = jnp.exp(
        self.coords.horizontal.to_nodal(state.log_surface_pressure)
    )

    vorticity_dot, divergence_dot = self.curl_and_div_tendencies(
        aux_state, nodal_surface_pressure
    )
    kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
    orography_tendency = self.orography_tendency()

    # To maintain hydrostatic balance, we must include the PGF term associated
    # with the reference state T_ref.
    # The total reference PGF is: ∇Φ(T_ref) + α(T_ref)∇p.
    # We need to add the residual (full - linear) for the temperature
    # variation T' to the explicit tendencies.
    # The implicit solver uses a linearized geopotential based on p_s_ref.
    # We combine T_ref and T' here for efficiency.
    temperature = self.T_ref + aux_state.temperature_variation
    adjustment = self._virtual_temperature_adjustment(aux_state)
    geopotential_diff_full = get_geopotential_diff_hybrid(
        temperature * adjustment,
        self.nondim_levels,
        self.physics_specs.R,
        nodal_surface_pressure,
        method='sparse',
        sharding=self.coords.dycore_sharding,
    )
    geopotential_diff_linear = get_geopotential_diff_hybrid(
        aux_state.temperature_variation,
        self.nondim_levels,
        self.physics_specs.R,
        self.p_s_ref,
        method='dense',
        sharding=self.coords.dycore_sharding,
    )
    geopotential_tendency_residual = -self.coords.horizontal.laplacian(
        self.coords.horizontal.to_modal(
            geopotential_diff_full - geopotential_diff_linear
        )
    )

    if self.humidity_key is not None:
      humidity_vort_correction_tendency = (
          self.vorticity_tendency_due_to_humidity(state, aux_state)
      )
      humidity_div_correction_tendency = (
          self.divergence_tendency_due_to_humidity(state, aux_state)
      )
      vorticity_dot += humidity_vort_correction_tendency
      divergence_dot += humidity_div_correction_tendency

    horizontal_tendency_fn = functools.partial(
        self.horizontal_scalar_advection, aux_state=aux_state
    )
    dT_dt_horizontal_nodal, dT_dt_horizontal_modal = horizontal_tendency_fn(
        aux_state.temperature_variation
    )
    tracers_horizontal_nodal_and_modal = jax.tree_util.tree_map(
        horizontal_tendency_fn, aux_state.tracers
    )
    dT_dt_vertical = self.nodal_temperature_vertical_tendency(aux_state)
    dT_dt_adiabatic = self.nodal_temperature_adiabatic_tendency(
        aux_state, nodal_surface_pressure
    )
    log_sp_tendency = self.nodal_log_pressure_tendency(
        aux_state, nodal_surface_pressure
    )
    mass_flux_full = aux_state.mass_flux_full
    if self.include_vertical_advection:
      vertical_tendency_fn = functools.partial(
          self._vertical_tendency,
          mass_flux_full,
          pressure_thickness=aux_state.layer_pressure_thickness,
      )
    else:
      vertical_tendency_fn = lambda x: 0
    tracers_vertical_nodal = jax.tree_util.tree_map(
        vertical_tendency_fn, aux_state.tracers
    )
    to_modal_fn = self.coords.horizontal.to_modal
    vorticity_tendency = vorticity_dot
    divergence_tendency = (
        divergence_dot
        + kinetic_energy_tendency
        + orography_tendency
        + geopotential_tendency_residual
    )
    temperature_tendency = (
        to_modal_fn(dT_dt_horizontal_nodal + dT_dt_vertical + dT_dt_adiabatic)
        + dT_dt_horizontal_modal
    )
    log_surface_pressure_tendency = to_modal_fn(log_sp_tendency)[
        jnp.newaxis, ...
    ]
    tracers_tendency = jax.tree_util.tree_map(
        lambda x, y_z: to_modal_fn(x + y_z[0]) + y_z[1],
        tracers_vertical_nodal,
        tracers_horizontal_nodal_and_modal,
    )
    tendency = State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
        tracers=tracers_tendency,
        sim_time=None if state.sim_time is None else 1.0,
    )
    return self.coords.horizontal.clip_wavenumbers(tendency)

  @jax.named_call
  def implicit_terms(self, state: State) -> State:
    """Returns the implicit terms of the primitive equations."""
    method = self.vertical_matmul_method
    if method is None:
      mesh = self.coords.spmd_mesh
      method = 'sparse' if mesh is not None and mesh.shape['z'] > 1 else 'dense'

    geopotential_diff = get_geopotential_diff_hybrid(
        state.temperature_variation,
        self.nondim_levels,
        self.physics_specs.R,
        self.p_s_ref,
        method=method,
        sharding=self.coords.dycore_sharding,
    )
    pgf_lps_coeff = _get_pgf_lps_coefficient_numpy(
        self.reference_temperature,
        self.p_s_ref,
        self.nondim_levels,
        self.physics_specs.R,
    )
    rt_log_p = (
        pgf_lps_coeff[:, np.newaxis, np.newaxis] * state.log_surface_pressure
    )

    vorticity_implicit = jnp.zeros_like(state.vorticity)
    divergence_implicit = -self.coords.horizontal.laplacian(
        geopotential_diff + rt_log_p
    )
    temperature_variation_implicit = get_temperature_implicit_hybrid(
        state.divergence,
        self.nondim_levels,
        self.reference_temperature,
        self.physics_specs.kappa,
        self.p_s_ref,
        method=method,
        sharding=self.coords.dycore_sharding,
    )
    levels = self.nondim_levels
    effective_sigma_thickness = (
        levels.pressure_thickness / self.p_s_ref + levels.sigma_thickness
    )
    log_surface_pressure_implicit = -_vertical_matvec(
        effective_sigma_thickness[np.newaxis], state.divergence
    )
    tracers_implicit = jax.tree_util.tree_map(jnp.zeros_like, state.tracers)
    return State(
        vorticity=vorticity_implicit,
        divergence=divergence_implicit,
        temperature_variation=temperature_variation_implicit,
        log_surface_pressure=log_surface_pressure_implicit,
        tracers=tracers_implicit,
        sim_time=None if state.sim_time is None else 0.0,
    )

  @jax.named_call
  def implicit_inverse(self, state: State, step_size: float) -> State:
    """Computes the inverse `(1 - step_size * implicit_terms)⁻¹."""
    if isinstance(step_size, jax.core.Tracer):
      raise TypeError(
          f'`step_size` must be concrete but a Tracer was passed: {step_size}. '
      )
    implicit_matrix = _get_implicit_term_matrix_hybrid(
        step_size,
        self.nondim_coords,
        self.reference_temperature,
        self.physics_specs.kappa,
        self.physics_specs.R,
        self.p_s_ref,
    )
    assert implicit_matrix.dtype == np.float64
    layers = self.nondim_levels.layers
    div = slice(0, layers)
    temp = slice(layers, 2 * layers)
    logp = slice(2 * layers, 2 * layers + 1)

    def named_vertical_matvec(name):
      return jax.named_call(_vertical_matvec_per_wavenumber, name=name)

    inverse = np.linalg.inv(implicit_matrix)
    assert not np.isnan(inverse).any()
    inverted_divergence = (
        named_vertical_matvec('div_from_div')(
            inverse[:, div, div], state.divergence
        )
        + named_vertical_matvec('div_from_temp')(
            inverse[:, div, temp], state.temperature_variation
        )
        + named_vertical_matvec('div_from_logp')(
            inverse[:, div, logp], state.log_surface_pressure
        )
    )
    inverted_temperature_variation = (
        named_vertical_matvec('temp_from_div')(
            inverse[:, temp, div], state.divergence
        )
        + named_vertical_matvec('temp_from_temp')(
            inverse[:, temp, temp], state.temperature_variation
        )
        + named_vertical_matvec('temp_from_logp')(
            inverse[:, temp, logp], state.log_surface_pressure
        )
    )
    inverted_log_surface_pressure = (
        named_vertical_matvec('logp_from_div')(
            inverse[:, logp, div], state.divergence
        )
        + named_vertical_matvec('logp_from_temp')(
            inverse[:, logp, temp], state.temperature_variation
        )
        + named_vertical_matvec('logp_from_logp')(
            inverse[:, logp, logp], state.log_surface_pressure
        )
    )
    inverted_vorticity = state.vorticity
    inverted_tracers = state.tracers
    return State(
        inverted_vorticity,
        inverted_divergence,
        inverted_temperature_variation,
        inverted_log_surface_pressure,
        inverted_tracers,
        sim_time=state.sim_time,
    )

  @jax.named_call
  def divergence_tendency_due_to_humidity(
      self,
      state: State,
      aux_state: Any,
  ) -> Array:
    """Computes divergence tendencies due to humidity for Hybrid coordinates.

    Returns zero because:
    1. The PGF terms are handled in `explicit_terms` by adjusting `pgf_coeff`
      `in curl_and_div_tendencies`.
    2. The Geopotential terms are handled in `explicit_terms` by adjusting the
       temperature passed to `get_geopotential_diff_hybrid`.
    """
    return jnp.zeros_like(state.divergence)

  @jax.named_call
  def vorticity_tendency_due_to_humidity(
      self,
      state: State,
      aux_state: Any,
  ) -> Array:
    """Computes vorticity tendencies due to humidity.

    In Hybrid, we sum T' and T_{ref} before multiplying by moisture, so the
    interaction term is generated automatically within the main PGF calculation
    in `curl_and_div_tendencies`.
    In Sigma, the PGF calculation handles T' and T_{ref} separately (one
    explicit, one implicit), so the interaction term q * T_{ref} is missed
    and must be added back explicitly.
    """
    return jnp.zeros_like(state.vorticity)


###############################################################################
# Deprecated aliases for backwards compatibility.
################################################################################

DiagnosticState = DiagnosticStateSigma
semi_lagrangian_vertical_advection_step = (
    semi_lagrangian_vertical_advection_step_sigma
)
compute_diagnostic_state = compute_diagnostic_state_sigma
PrimitiveEquationsSpecs = units.SimUnits
StateWithTime = State


class PrimitiveEquations(PrimitiveEquationsSigma):
  """Deprecated alias for backwards compatibility."""

  def __init__(
      self,
      reference_temperature: np.ndarray,
      orography: Array,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: units.SimUnitsProtocol,
      *,
      vertical_matmul_method: str | None = None,
      implicit_inverse_method: str = 'split',
      vertical_advection: Callable[
          ..., jax.Array
      ] = sigma_coordinates.centered_vertical_advection,
      include_vertical_advection: bool = True,
  ):
    super().__init__(
        reference_temperature,
        orography,
        coords,
        physics_specs,
        vertical_matmul_method=vertical_matmul_method,
        implicit_inverse_method=implicit_inverse_method,
        vertical_advection=vertical_advection,
        include_vertical_advection=include_vertical_advection,
        humidity_key=None,
        cloud_keys=None,
    )


PrimitiveEquationsWithTime = PrimitiveEquations


class MoistPrimitiveEquations(PrimitiveEquationsSigma):
  """Deprecated alias for backwards compatibility."""

  def __init__(
      self,
      reference_temperature: np.ndarray,
      orography: Array,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: units.SimUnitsProtocol,
      *,
      vertical_matmul_method: str | None = None,
      implicit_inverse_method: str = 'split',
      vertical_advection: Callable[
          ..., jax.Array
      ] = sigma_coordinates.centered_vertical_advection,
      include_vertical_advection: bool = True,
  ):
    super().__init__(
        reference_temperature,
        orography,
        coords,
        physics_specs,
        vertical_matmul_method=vertical_matmul_method,
        implicit_inverse_method=implicit_inverse_method,
        vertical_advection=vertical_advection,
        include_vertical_advection=include_vertical_advection,
        humidity_key='specific_humidity',
        cloud_keys=None,
    )


class MoistPrimitiveEquationsWithCloudMoisture(PrimitiveEquationsSigma):
  """Deprecated alias for backwards compatibility."""

  def __init__(
      self,
      reference_temperature: np.ndarray,
      orography: Array,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: units.SimUnitsProtocol,
      *,
      vertical_matmul_method: str | None = None,
      implicit_inverse_method: str = 'split',
      vertical_advection: Callable[
          ..., jax.Array
      ] = sigma_coordinates.centered_vertical_advection,
      include_vertical_advection: bool = True,
  ):
    super().__init__(
        reference_temperature,
        orography,
        coords,
        physics_specs,
        vertical_matmul_method=vertical_matmul_method,
        implicit_inverse_method=implicit_inverse_method,
        vertical_advection=vertical_advection,
        include_vertical_advection=include_vertical_advection,
        humidity_key='specific_humidity',
        cloud_keys=(
            'specific_cloud_liquid_water_content',
            'specific_cloud_ice_water_content',
        ),
    )


def get_geopotential_with_moisture(
    temperature: typing.Array,
    specific_humidity: typing.Array,
    nodal_orography: typing.Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    gravity_acceleration: float,
    ideal_gas_constant: float,
    water_vapor_gas_constant: float,
    sharding: jax.sharding.NamedSharding | None = None,
    clouds: typing.Array | None = None,
) -> jnp.ndarray:
  """Computes geopotential in nodal space using nodal temperature and `q`."""
  return get_geopotential_on_sigma(
      temperature,
      specific_humidity,
      clouds=clouds,
      nodal_orography=nodal_orography,
      sigma=coordinates,
      gravity_acceleration=gravity_acceleration,
      ideal_gas_constant=ideal_gas_constant,
      water_vapor_gas_constant=water_vapor_gas_constant,
      sharding=sharding,
  )


# pylint: enable=invalid-name
