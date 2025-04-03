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
from dinosaur import jax_numpy_utils
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import vertical_interpolation
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tree_math

units = scales.units

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


StateWithTime = State  # deprecated alias


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


@tree_math.struct
class DiagnosticState:
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
def compute_diagnostic_state(
    state: State,
    coords: coordinate_systems.CoordinateSystem,
) -> DiagnosticState:
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
  return DiagnosticState(
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


def compute_vertical_velocity(
    state: State, coords: coordinate_systems.CoordinateSystem
) -> jax.Array:
  """Calculate vertical velocity at the center of each layer."""
  sigma_dot_boundaries = compute_diagnostic_state(state, coords).sigma_dot_full
  assert sigma_dot_boundaries.ndim == 3
  # This matches the default boundary conditions for vertical velocity
  # from sigma_coordinates.centered_vertical_advection
  sigma_dot_padded = jnp.pad(sigma_dot_boundaries, [(1, 1), (0, 0), (0, 0)])
  return 0.5 * (sigma_dot_padded[1:] + sigma_dot_padded[:-1])


def semi_lagrangian_vertical_advection_step(
    state: State, coords: coordinate_systems.CoordinateSystem, dt: float
) -> State:
  """Take a first-order step for semi-Lagrangian vertical advection."""
  velocity = compute_vertical_velocity(state, coords)
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


@dataclasses.dataclass(frozen=True)
class PrimitiveEquationsSpecs:
  """Physical constants and scale used in the primitive equations.

  Attributes:
    radius: the non-dimensionalized radius of the domain.
    angular_velocity: the non-dimensionalized angular velocity of the rotating
      domain.
    gravity_acceleration: the non-dimensionalized value of gravitational
      acceleration.
    ideal_gas_constant: the non-dimensionalized gas constant.
    water_vapor_gas_constant: the non-dimensionalized gas constant for vapor.
    water_vapor_isobaric_heat_capacity: isobaric heat capacity of vapor.
    kappa: `ideal_gas_constant / Cp` where  Cp is the isobaric heat capacity.
    scale: an instance implementing `ScaleProtocol` that will be used to
      (non-)dimensionalize quantities.
  """

  radius: float
  angular_velocity: float
  gravity_acceleration: float
  ideal_gas_constant: float
  water_vapor_gas_constant: float
  water_vapor_isobaric_heat_capacity: float
  kappa: float
  scale: scales.ScaleProtocol

  @property
  def R(self) -> float:
    """Alias for `ideal_gas_constant`."""
    return self.ideal_gas_constant

  @property
  def R_vapor(self) -> float:
    """Alias for `ideal_gas_constant`."""
    return self.water_vapor_gas_constant

  @property
  def g(self) -> float:
    """Alias for `gravity_acceleration`."""
    return self.gravity_acceleration

  @property
  def Cp(self) -> float:
    """Isobaric heat capacity."""
    return self.ideal_gas_constant / self.kappa

  @property
  def Cp_vapor(self) -> float:
    """Alias for `water_vapor_isobaric_heat_capacity`."""
    return self.water_vapor_isobaric_heat_capacity

  def nondimensionalize(self, quantity: Quantity) -> Numeric:
    """Non-dimensionalizes and rescales `quantity`."""
    return self.scale.nondimensionalize(quantity)

  def nondimensionalize_timedelta64(self, timedelta: np.timedelta64) -> Numeric:
    """Non-dimensionalizes and rescales a numpy timedelta."""
    base_unit = 's'
    return self.scale.nondimensionalize(
        timedelta / np.timedelta64(1, base_unit) * units(base_unit)
    )

  def dimensionalize(self, value: Numeric, unit: units.Unit) -> Quantity:
    """Rescales and adds units to the given non-dimensional value."""
    return self.scale.dimensionalize(value, unit)

  def dimensionalize_timedelta64(self, value: Numeric) -> np.timedelta64:
    """Rescales and casts the given non-dimensional value to timedelta64."""
    base_unit = 's'  # return value is rounded down to nearest base_unit
    dt = self.scale.dimensionalize(value, units(base_unit)).m
    if isinstance(dt, np.ndarray):
      return dt.astype(f'timedelta64[{base_unit}]')
    else:
      return np.timedelta64(int(dt), base_unit)

  @classmethod
  def from_si(
      cls,
      radius_si: Quantity = scales.RADIUS,
      angular_velocity_si: Quantity = scales.ANGULAR_VELOCITY,
      gravity_acceleration_si: Quantity = scales.GRAVITY_ACCELERATION,
      ideal_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT,
      water_vapor_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT_H20,
      water_vapor_isobaric_heat_capacity_si: Quantity = scales.WATER_VAPOR_CP,
      kappa_si: Quantity = scales.KAPPA,
      scale: scales.ScaleProtocol = scales.DEFAULT_SCALE,
  ) -> PrimitiveEquationsSpecs:
    # pylint: disable=g-doc-args,g-doc-return-or-yield
    """Constructs `PrimitiveEquantionSpecs` from constants with units.

    By default uses units in which the radius and angular_velocity are set to
    one.
    """
    return cls(
        scale.nondimensionalize(radius_si),
        scale.nondimensionalize(angular_velocity_si),
        scale.nondimensionalize(gravity_acceleration_si),
        scale.nondimensionalize(ideal_gas_constant_si),
        scale.nondimensionalize(water_vapor_gas_constant_si),
        scale.nondimensionalize(water_vapor_isobaric_heat_capacity_si),
        scale.nondimensionalize(kappa_si),
        scale,
    )


#  =============================================================================
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


def get_geopotential_weights(
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


def get_geopotential_diff(
    temperature: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    ideal_gas_constant: float,
    method: str = 'dense',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Calculate the implicit geopotential term."""
  if method == 'dense':
    weights = get_geopotential_weights(coordinates, ideal_gas_constant)
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


def get_geopotential(
    temperature_variation: Array,
    reference_temperature: Array,
    orography: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    gravity_acceleration: float,
    ideal_gas_constant: float,
    sharding: jax.sharding.NamedSharding | None = None,
) -> jnp.ndarray:
  """Computes geopotential at sigma values determined by `coordinates`.

  Args:
    temperature_variation: temperature variation in spectral basis.
    reference_temperature: a vector of reference temperatures, indexed by layer.
      Temperature in each layer is described as a deviation from this reference
      value.
    orography: the topography of the surface of the planet, in the spectral
      basis.
    coordinates: the `SigmaCoordinates` object describing the spacing of layers
      in 𝜎 coordinates.
    gravity_acceleration: the non-dimensionalized value of gravitational
      acceleration.
    ideal_gas_constant: the non-dimensionalized gas constant.
    sharding: JAX sharding for temperature_variation and outputs.

  Returns:
    An array containing the geopotential in the spectral basis. Note that the
    geopotential is computed relative to the radius of the planet, not relative
    to the center of the planet.
  """
  surface_geopotential = orography * gravity_acceleration
  temperature = spherical_harmonic.add_constant(
      temperature_variation, reference_temperature
  )
  geopotential_diff = get_geopotential_diff(
      temperature, coordinates, ideal_gas_constant, sharding=sharding
  )
  return surface_geopotential + geopotential_diff


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
  gas_const_ratio = water_vapor_gas_constant / ideal_gas_constant
  surface_geopotential = nodal_orography * gravity_acceleration
  if clouds is None:
    clouds = 0.0
  virtual_temp = temperature * (
      1 + (gas_const_ratio - 1) * specific_humidity - clouds
  )
  geopotential_diff = get_geopotential_diff(
      virtual_temp, coordinates, ideal_gas_constant, sharding=sharding
  )
  return surface_geopotential + geopotential_diff


def get_temperature_implicit_weights(
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


def get_temperature_implicit(
    divergence: Array,
    coordinates: sigma_coordinates.SigmaCoordinates,
    reference_temperature: np.ndarray,
    kappa: float,
    method: str = 'dense',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Calculate the implicit temperature term."""
  weights = -get_temperature_implicit_weights(
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


@jax.named_call
def _vertical_matvec(a: Array, x: Array) -> jax.Array:
  """Matrix-vector product `a.x` taken along the "vertical" axis, -3."""
  return einsum('gh,...hml->...gml', a, x)


@jax.named_call
def _vertical_matvec_per_wavenumber(a: Array, x: Array) -> jax.Array:
  """Same as `_vertical_matvec` but maps across `total_wavenumber`."""
  # It turns out it's faster to use regular jnp.einsum here instead of
  # jax_numpy_utils.sharded_einsum even in the vertical sharding case, because
  # we use relatively few vertical layers so the dominant cost of the einsum
  # is inter-core communication rather than MXU operations. XLA SPMD ends up
  # implementing this einsum by replicating inputs across all shards, which it
  # can reuse for each of a handful calls to `_vertical_matvec_per_wavenumber`.
  return einsum('lgh,...hml->...gml', a, x)


def _get_implicit_term_matrix(
    eta, coords, reference_temperature, kappa, ideal_gas_constant
) -> np.ndarray:
  """Returns a matrix corresponding to `PrimitiveEquations.implicit_terms`."""

  # First we construct matrices that will be building blocks for the larger
  # implicit term matrix.
  eye = np.eye(coords.vertical.layers)[np.newaxis]
  lam = coords.horizontal.laplacian_eigenvalues
  g = get_geopotential_weights(coords.vertical, ideal_gas_constant)
  r = ideal_gas_constant
  h = get_temperature_implicit_weights(
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
#  The `PrimitiveEquations` Class
#
#  The `PrimitiveEquations` class expresses the primitive equations in a form
#  that is appropriate for semi-implicit time stepping.
#  =============================================================================


@dataclasses.dataclass
class PrimitiveEquations(time_integration.ImplicitExplicitODE):
  """A semi-implicit description of the primitive equations.

  Attributes:
    reference_temperature: An array of shape [layers]. All temperature values
      will be expressed as their difference from this value.
    orography: An array of shape `coords.horizontal.modal_shape` describing the
      topography in modal representation.
    coords: horizontal and vertical descritization.
    physics_specs: an `PrimitiveEquationSpecs` object describing the scales and
      physical constants used in the primitive equations.
    vertical_matmul_method: 'dense' or 'sparse', indicating the method to use
      for vertical matrix multiplications inside calculations of implicit
      geopotential and temperature terms. 'sparse' uses a matrix-free
      calculation involving `cumsum`, and is faster only when the calculation
      uses vertical sharding.
    implicit_inverse_method: 'split', 'stacked' or 'blockwise' method to use for
      the implicit inverse calculation.
    vertical_advection: function to use for calculating tendencies from vertical
      advection.
    include_vertical_advection: whether to include tendencies from vertical
      advection or to drop it.
  """

  reference_temperature: np.ndarray
  orography: Array
  coords: coordinate_systems.CoordinateSystem
  physics_specs: PrimitiveEquationsSpecs

  vertical_matmul_method: str | None = dataclasses.field(
      default=None, kw_only=True
  )
  implicit_inverse_method: str = dataclasses.field(
      default='split', kw_only=True
  )
  vertical_advection: Callable[..., jax.Array] = dataclasses.field(
      default=sigma_coordinates.centered_vertical_advection, kw_only=True
  )
  include_vertical_advection: bool = dataclasses.field(
      default=True, kw_only=True
  )

  def __post_init__(self):
    if not np.allclose(
        self.coords.horizontal.radius, self.physics_specs.radius, rtol=1e-5
    ):
      raise ValueError(
          'inconsistent radius between coordinates and constants: '
          f'{self.coords.horizontal.radius=} != {self.physics_specs.radius=}'
      )

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
    return self.vertical_advection(w, x, self.coords.vertical)

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
  def kinetic_energy_tendency(self, aux_state: DiagnosticState) -> Array:
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
  def curl_and_div_tendencies(
      self,
      aux_state: DiagnosticState,
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
    rt = self.physics_specs.R * aux_state.temperature_variation
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
      aux_state: DiagnosticState,
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
  def horizontal_scalar_advection(
      self,
      scalar: Array,
      aux_state: DiagnosticState,
  ) -> tuple[Array, Array]:
    """Computes explicit tendency of `scalar` due to horizontal advection."""
    u, v = aux_state.cos_lat_u
    nodal_terms = scalar * aux_state.divergence
    modal_terms = -div_sec_lat(u * scalar, v * scalar, self.coords.horizontal)
    return nodal_terms, modal_terms

  @jax.named_call
  def nodal_temperature_adiabatic_tendency(
      self, aux_state: DiagnosticState
  ) -> Array:
    """Computes explicit temperature tendency due to adiabatic processes."""
    g_explicit = aux_state.u_dot_grad_log_sp
    g_full = g_explicit + aux_state.divergence
    mean_t_part = self._t_omega_over_sigma_sp(
        self.T_ref, g_explicit, aux_state.u_dot_grad_log_sp
    )
    variation_t_part = self._t_omega_over_sigma_sp(
        aux_state.temperature_variation, g_full, aux_state.u_dot_grad_log_sp
    )
    return self.physics_specs.kappa * (mean_t_part + variation_t_part)

  @jax.named_call
  def nodal_log_pressure_tendency(self, aux_state: DiagnosticState) -> Array:
    """Computes explicit tendency of the log_surface_pressure."""
    # computes -∑G[i] * ∆𝜎[i] where G[i] = u[i] · ∇(log(ps)).
    g = aux_state.u_dot_grad_log_sp
    return -sigma_coordinates.sigma_integral(g, self.coords.vertical)

  @jax.named_call
  def explicit_terms(self, state: State) -> State:
    """Computes explicit tendencies of the primitive equations."""
    aux_state = compute_diagnostic_state(state, self.coords)
    # tendencies that are computed in modal representation
    vorticity_tendency, divergence_dot = self.curl_and_div_tendencies(aux_state)
    kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
    orography_tendency = self.orography_tendency()
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

    geopotential_diff = get_geopotential_diff(
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
    temperature_variation_implicit = get_temperature_implicit(
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

    implicit_matrix = _get_implicit_term_matrix(
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

      gt = get_geopotential_diff(
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

      hd = -get_temperature_implicit(
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


PrimitiveEquationsWithTime = PrimitiveEquations  # deprecated alias


@dataclasses.dataclass
class MoistPrimitiveEquations(PrimitiveEquations):
  """Primitive equations that take into account humidity and advance time."""

  def _get_specific_humidity(self, aux_state: DiagnosticState) -> Array:
    """Extracts `speicific_humidity` from tracers in DiagnosticState."""
    if 'specific_humidity' not in aux_state.tracers:
      raise ValueError(
          '`specific_humidity` is not found in tracers: '
          f'{aux_state.tracers.keys()}.'
      )
    return aux_state.tracers['specific_humidity']

  def _virtual_temperature(self, aux_state, moisture_contribution):
    """Calculates the virtual temperature without the contribution of clouds."""
    return (
        self.physics_specs.R
        * aux_state.temperature_variation
        * (1 + moisture_contribution)
    )

  @jax.named_call
  def curl_and_div_tendencies(
      self,
      aux_state: DiagnosticState,
  ) -> tuple[Array, Array]:
    """Computes curl and divergence tendencies for vorticity ζ and divergence 𝛅.

    Computes to explicit tendencies (dζ_dt, d𝛅_dt) to due to curl and divergence
    terms in the primitive equations, with account for humidity by using virtual
    temperature instead of thermodynamic temperature.
    See `curl_and_div_tendencies` in PrimitiveEquations for details or ECMWFs
    notes http://shortn/_CDj2woFwzv.

    Args:
      aux_state: diagnostic state with pre-computed nodal values.

    Returns:
      Tuple of divergence and vorticity tendencies due to curl and divergence
      terms in the primitive equations.
    """
    gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
    q = self._get_specific_humidity(aux_state)
    moisture_contribution = (gas_const_ratio - 1) * q
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
    # we use virtual temperature Tv in these tendencies to accound for humidity.
    rTv = self._virtual_temperature(aux_state, moisture_contribution)
    grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
    vertical_term_u = (sigma_dot_u + rTv * grad_log_ps_u) * sec2_lat
    vertical_term_v = (sigma_dot_v + rTv * grad_log_ps_v) * sec2_lat
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
  def nodal_temperature_adiabatic_tendency(
      self, aux_state: DiagnosticState
  ) -> Array:
    """Computes explicit temperature tendency due to adiabatic processes."""
    gas_const_ratio = self.physics_specs.R_vapor / self.physics_specs.R
    heat_capacity_ratio = self.physics_specs.Cp_vapor / self.physics_specs.Cp
    g_explicit = aux_state.u_dot_grad_log_sp
    g_full = g_explicit + aux_state.divergence
    q = self._get_specific_humidity(aux_state)
    mean_t_part = self._t_omega_over_sigma_sp(
        self.T_ref, g_explicit, aux_state.u_dot_grad_log_sp
    )
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
  def divergence_tendency_due_to_humidity(
      self,
      state: State,
      aux_state: DiagnosticState,
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
      method = 'sparse' if mesh is not None and mesh.shape['z'] > 1 else 'dense'

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
    geopotential_diff = get_geopotential_diff(
        temperature_diff,
        self.coords.vertical,
        physics_specs.R,
        method=method,
        sharding=self.coords.dycore_sharding,
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
      aux_state: DiagnosticState,
  ) -> Array:
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
  def explicit_terms(self, state: StateWithTime) -> StateWithTime:
    """Evaluates explicit terms in the ODE."""
    aux_state = compute_diagnostic_state(state, self.coords)
    # tendencies that are computed in modal representation
    vorticity_dot, divergence_dot = self.curl_and_div_tendencies(aux_state)
    humidity_vort_correction_tendency = self.vorticity_tendency_due_to_humidity(
        state, aux_state
    )
    kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
    orography_tendency = self.orography_tendency()
    humidity_div_correction_tendency = self.divergence_tendency_due_to_humidity(
        state, aux_state
    )
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
    vorticity_tendency = vorticity_dot + humidity_vort_correction_tendency
    divergence_tendency = (
        divergence_dot
        + kinetic_energy_tendency
        + orography_tendency
        + humidity_div_correction_tendency
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
    explicit_terms = State(
        vorticity=vorticity_tendency,
        divergence=divergence_tendency,
        temperature_variation=temperature_tendency,
        log_surface_pressure=log_surface_pressure_tendency,
        tracers=tracers_tendency,
        sim_time=None if state.sim_time is None else 1.0,
    )
    # Note: clipping the final total wavenumber from the explicit tendencies
    # matches SPEEDY.
    return self.coords.horizontal.clip_wavenumbers(explicit_terms)


@dataclasses.dataclass
class MoistPrimitiveEquationsWithCloudMoisture(MoistPrimitiveEquations):
  """Primitive equations that calculate virtual temperature with clouds."""

  def _virtual_temperature(self, aux_state, moisture_contribution):
    """Calculates virtual temperature with clouds."""
    return (
        self.physics_specs.R
        * aux_state.temperature_variation
        * (
            1
            + moisture_contribution
            - self._get_cloud_water(aux_state)
            - self._get_cloud_ice(aux_state)
        )
    )

  def _get_cloud_water(self, aux_state: DiagnosticState) -> Array:
    """Extracts `specific_cloud_liquid_water_content` from tracers in DiagnosticState."""
    if 'specific_cloud_liquid_water_content' not in aux_state.tracers:
      raise ValueError(
          '`specific_cloud_liquid_water_content` is not found in tracers: '
          f'{aux_state.tracers.keys()}.'
      )
    return aux_state.tracers['specific_cloud_liquid_water_content']

  def _get_cloud_ice(self, aux_state: DiagnosticState) -> Array:
    """Extracts `specific_cloud_ice_water_content` from tracers in DiagnosticState."""
    if 'specific_cloud_ice_water_content' not in aux_state.tracers:
      raise ValueError(
          '`specific_cloud_ice_water_content` is not found in tracers: '
          f'{aux_state.tracers.keys()}.'
      )
    return aux_state.tracers['specific_cloud_ice_water_content']


# pylint: enable=invalid-name
