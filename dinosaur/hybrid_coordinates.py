# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A vertical coordinate system that interpolates between sigma and pressure."""

from __future__ import annotations

import dataclasses
import functools
import importlib

import dinosaur
from dinosaur import jax_numpy_utils
from dinosaur import sigma_coordinates
from dinosaur import typing
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


Array = typing.Array
einsum = functools.partial(jnp.einsum, precision=lax.Precision.HIGHEST)
# For consistency with commonly accepted notation, we use Greek letters within
# some of the functions below.
# pylint: disable=invalid-name


def _slice_shape_along_axis(
    x: np.ndarray,
    axis: int,
    slice_width: int = 1,
) -> tuple[int, ...]:
  """Returns a shape of `x` sliced along `axis` with width `slice_width`."""
  x_shape = list(x.shape)
  x_shape[axis] = slice_width
  return tuple(x_shape)


@jax.named_call
def cumulative_integral_over_pressure(
    x: Array,
    surface_pressure: Array,
    coordinates: HybridCoordinates,
    axis: int = -3,
    downward: bool = True,
    cumsum_method: str = 'dot',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Approximates cumulative integral of `x` with respect to pressure."""
  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )
  a_thickness = coordinates.pressure_thickness
  b_thickness = coordinates.sigma_thickness
  a_thickness = a_thickness.astype(x.dtype)[:, np.newaxis, np.newaxis]
  b_thickness = b_thickness.astype(x.dtype)[:, np.newaxis, np.newaxis]
  dp = a_thickness + b_thickness * surface_pressure
  xdp = x * dp
  if downward:
    return jax_numpy_utils.cumsum(
        xdp, axis, method=cumsum_method, sharding=sharding
    )
  else:
    return jax_numpy_utils.reverse_cumsum(
        xdp, axis, method=cumsum_method, sharding=sharding
    )


@jax.named_call
def integral_over_pressure(
    x: Array,
    surface_pressure: Array,
    coordinates: HybridCoordinates,
    axis: int = -3,
    keepdims: bool = True,
    ) -> jax.Array:
  """Approximates definite integral of `x` over pressure."""
  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )
  x_axes = range(x.ndim)
  da = coordinates.pressure_thickness.astype(x.dtype)
  db = coordinates.sigma_thickness.astype(x.dtype)
  thickness_axes = [x_axes[axis]]
  x_da = einsum(x, x_axes, da, thickness_axes, x_axes)
  int_x_da = x_da.sum(axis=axis, keepdims=keepdims)
  x_db = einsum(x, x_axes, db, thickness_axes, x_axes)
  int_x_db = x_db.sum(axis=axis, keepdims=keepdims)
  return int_x_da + surface_pressure * int_x_db


@jax.named_call
def cumulative_integral_over_sigma(
    x: Array,
    coordinates: HybridCoordinates,
    axis: int = -3,
    downward: bool = True,
    cumsum_method: str = 'dot',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Approximates cumulative integral of `x` over `dσ` variables."""
  # Note: This integral only includes the sigma and not the pressure component.
  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )
  x_axes = range(x.ndim)
  d_sigma = coordinates.sigma_thickness
  d_sigma_axes = [x_axes[axis]]
  x_d_sigma = einsum(x, x_axes, d_sigma, d_sigma_axes, x_axes)
  if downward:
    return jax_numpy_utils.cumsum(
        x_d_sigma, axis, method=cumsum_method, sharding=sharding
    )
  else:
    return jax_numpy_utils.reverse_cumsum(
        x_d_sigma, axis, method=cumsum_method, sharding=sharding
    )


@jax.named_call
def integral_over_sigma(
    x: Array,
    coordinates: HybridCoordinates,
    axis: int = -3,
    keepdims: bool = True,
) -> jax.Array:
  """Approximates definite integral of `x` over `sigma` coordinate."""
  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )
  x_axes = range(x.ndim)
  d_sigma = coordinates.sigma_thickness
  d_sigma_axes = [x_axes[axis]]
  x_d_sigma = einsum(x, x_axes, d_sigma, d_sigma_axes, x_axes)
  return x_d_sigma.sum(axis=axis, keepdims=keepdims)


def centered_difference(
    x: Array,
    coordinates: HybridCoordinates,
    axis: int = -3,
) -> Array:
  """Derivative of `x` with respect to `η` along specified `axis`.

  The derivative is approximated as

  (∂x / ∂η)[n + ½] ≈ (x[n + 1] - x[n]) / (η[n + 1] - η[n])

  So, the derivatives will be located on the 'boundaries' between layers and
  will consist of one fewer values than `x`.

  Args:
    x: an array of values with `x.shape[axis] == coordinates.layers`. These
      values are "located" at `coordinates` layer centers.
    coordinates: a `HybridCoordinates` object describing the vertical
      coordinates to differentiate against. Must satisfy `coordinates.layers ==
      x.shape[axis]`.
    axis: the axis along which to approximate the derivative. Defaults to -3
      since this is the axis we typically use to index layers.

  Returns:
    An array `x_η` with `x_η.shape[axis] == x.shape[axis] - 1`,
    containing approximate values of the derivative at of `x` at
    `coordinates.internal_boundaries`.
  """
  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`; '
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )

  dx = jax_numpy_utils.diff(x, axis=axis)
  dx_axes = range(dx.ndim)
  inv_dη = 1 / coordinates.center_to_center
  inv_dη_axes = [dx_axes[axis]]
  return einsum(dx, dx_axes, inv_dη, inv_dη_axes, dx_axes, precision='float32')


@jax.named_call
def centered_vertical_advection(
    w: Array,
    x: Array,
    coordinates: HybridCoordinates,
    axis: int = -3,
    w_boundary_values: tuple[Array, Array] | None = None,
    dx_dη_boundary_values: tuple[Array, Array] | None = None,
) -> jnp.ndarray:
  """Compute vertical advection using 2nd order finite differences."""
  if w_boundary_values is None:
    w_slc_shape = _slice_shape_along_axis(w, axis)
    w_boundary_values = (
        jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
        jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
    )
  if dx_dη_boundary_values is None:
    x_slc_shape = _slice_shape_along_axis(x, axis)
    dx_dη_boundary_values = (
        jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
        jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
    )

  w_boundary_top, w_boundary_bot = w_boundary_values
  w = jnp.concatenate([w_boundary_top, w, w_boundary_bot], axis=axis)

  x_diff = centered_difference(x, coordinates, axis)
  x_diff_boundary_top, x_diff_boundary_bot = dx_dη_boundary_values
  x_diff = jnp.concatenate(
      [x_diff_boundary_top, x_diff, x_diff_boundary_bot], axis=axis
  )

  w_times_x_diff = w * x_diff
  return -0.5 * (
      lax.slice_in_dim(w_times_x_diff, 1, None, axis=axis)
      + lax.slice_in_dim(w_times_x_diff, 0, -1, axis=axis)
  )


@dataclasses.dataclass(frozen=True)
class HybridCoordinates:
  """Specifies the vertical coordinate with hybrid sigma-pressure levels.

  This allows for matching the vertical coordinate system used by ECMWF and most
  other operational forecasting systems.

  The pressure corresponding to a level is given by the formula `a + b * sp`
  where `sp` is surface pressure.

  Attributes:
    a_boundaries: offset coefficient for the boundaries of each level, starting
      at the level closest to the top of the atmosphere.
    b_boundaries: slope coefficient for the boundaries of each level, starting
      at the level closest to the top of the atmosphere.
    layers: number of vertical layers.
  """

  a_boundaries: np.ndarray
  b_boundaries: np.ndarray

  def __post_init__(self):
    object.__setattr__(self, 'a_boundaries', np.asarray(self.a_boundaries))
    object.__setattr__(self, 'b_boundaries', np.asarray(self.b_boundaries))
    if len(self.a_boundaries) != len(self.b_boundaries):
      raise ValueError(
          'Expected `a_boundaries` and `b_boundaries` to have the same length, '
          f'got {len(self.a_boundaries)} and {len(self.b_boundaries)}.'
      )

  @classmethod
  def from_coefficients(
      cls,
      a_coeffs: list[float] | np.ndarray,
      b_coeffs: list[float] | np.ndarray,
      p0: float = 1000.0,
  ) -> HybridCoordinates:
    """Creates coordinates from A, B coefficients and reference pressure.

    Implements the formulation: P(k) = A(k) * p0 + B(k) * Ps

    This is common in NCAR/CAM literature.

    Args:
        a_coeffs: Dimensionless A coefficients.
        b_coeffs: Dimensionless B coefficients (sigma component).
        p0: Reference pressure in hPa (default 1000 hPa).
    """
    a_boundaries = np.array(a_coeffs) * p0
    return cls(a_boundaries=a_boundaries, b_boundaries=np.array(b_coeffs))

  @classmethod
  def analytic_levels(
      cls,
      n_levels: int,
      p_top: float = 0.0,
      p0: float = 1000.0,
      sigma_exponent: float = 3.0,
      stretch_exponent: float = 2.0,
  ) -> HybridCoordinates:
    """Generates analytically smooth hybrid coordinates.

    This uses a power-law strategy to blend from pressure to sigma.

    The vertical domain is defined by a master coordinate `η` in [0, 1].
    The distribution of `η` is stretched to concentrate levels near the
    surface.

    The pressure is then split into A and B coefficients such that:
        P(η) = A(η) * p_ref + B(η) * p_s

    When p_s == p_ref, the total pressure is exactly P(η).

    Args:
        n_levels: Number of vertical layers (resulting in n_levels + 1
          interfaces).
        p_top: The pressure at the top of the model (in Pascals).
        p0: Reference surface pressure (P0) for defining A coefficients.
        sigma_exponent: Controls the "hybridization". 1.0 = Pure Sigma (terrain
          following everywhere). >1.0 = Hybrid (becomes more isobaric aloft).
          Typical values are 3.0 to 5.0 for Earth-like atmospheres.
        stretch_exponent: Controls vertical resolution spacing. 1.0 = Linear
          spacing. >1.0 = Concentrates levels near the surface (PBL).
    """
    # We use a power law to concentrate resolution at the surface.
    k = np.linspace(0, 1, n_levels + 1)
    eta = k**stretch_exponent

    # This connects p_top (at eta=0) to p_ref (at eta=1)
    p_profile = p_top + eta * (p0 - p_top)

    # B must be 0 at top and 1 at surface.
    # We use a power law: B = eta^r.
    # Higher `sigma_exponent` means B decays faster, making the grid
    # "pure pressure" (flat) more quickly as you go up.
    b_boundaries = eta**sigma_exponent

    # derived from the constraint: P_profile = A * p_ref + B * p_ref
    # Therefore: A = (P_profile / p_ref) - B
    a_boundaries = (p_profile / p0) - b_boundaries

    # Enforce machine precision consistency at boundaries
    a_boundaries[0] = p_top / p0  # Top is pure pressure (if B[0]=0)
    b_boundaries[0] = 0.0
    a_boundaries[-1] = 0.0  # Surface is pure sigma
    b_boundaries[-1] = 1.0
    return cls.from_coefficients(a_boundaries, b_boundaries, p0)

  @classmethod
  def analytic_ecmwf_like(
      cls,
      n_levels: int = 137,
      p0: float = 1013.25,
      p_top: float = 0.0,
      p_transition: float = 80.0,
      alpha_strat: float = 4.2,
      alpha_surf: float = 0.25,
      b_exponent: float = 1.3,
  ) -> HybridCoordinates:
    """Generates hybrid coefficients with higher resolution at TOA and surface.
    we note that this is an empirically-derived set of parameters that allows
    to produce concentration of levels in the lower and upper parts of the
    atmosphere.

    Args:
      n_levels: The desired number of vertical layers.
      p0: Reference surface pressure in hPa.
      p_top: Pressure at the top of the model in hPa.
      p_transition: Pressure level in hPa. Above this pressure (i.e., closer
        to surface), coordinates transition from pure pressure to hybrid.
      alpha_strat: Power law exponent used near TOA. Higher values (>1) result
        in thinner layers (higher resolution) near TOA.
      alpha_surf: Power law exponent used near the surface. Lower values (<1)
        result in thinner layers (higher resolution) near the surface.
      b_exponent: Exponent applied to normalized pressure in the hybrid region
        to calculate the `b` coefficient. Controls how rapidly levels
        transition from pressure-following to terrain-following.

    Returns:
      A HybridCoordinates object with coefficients defined in hPa.
    """

    k = np.linspace(0, 1, n_levels + 1)
    exponents = np.linspace(alpha_strat, alpha_surf, n_levels + 1)
    p_norm = k**exponents
    p_target = p_top + p_norm * (p0 - p_top)
    a_coeffs = np.zeros_like(p_target)
    b_coeffs = np.zeros_like(p_target)

    for i, p in enumerate(p_target):
      if p <= p_transition:
        # REGIME 1: Pure Pressure
        b_coeffs[i] = 0.0
        a_coeffs[i] = p
      else:
        # REGIME 2: Hybrid
        # Calculate coordinate tau (0 at transition, 1 at surface)
        tau = (p - p_transition) / (p0 - p_transition)
        b_val = tau**b_exponent

        # Enforce boundary
        if i == n_levels:
          b_val = 1.0

        b_coeffs[i] = b_val

        # A is the residual
        a_val = p - (b_val * p0)
        a_coeffs[i] = max(0.0, a_val)
    a_coeffs[0] = 0.0
    b_coeffs[0] = 0.0
    return cls(a_boundaries=a_coeffs, b_boundaries=b_coeffs)

  def get_eta(self, p_surface: float) -> np.ndarray:
    """Returns the eta values for a given pressure."""
    etas = self.a_centers / p_surface + self.b_centers
    return etas

  @property
  def pressure_thickness(self) -> np.ndarray:
    """Returns thickness of pressure part of hybrid coordinates."""
    return np.diff(self.a_boundaries)

  @property
  def sigma_thickness(self) -> np.ndarray:
    """Returns thickness of sigma part of hybrid coordinates."""
    return np.diff(self.b_boundaries)

  @property
  def a_centers(self) -> np.ndarray:
    """Returns center values for pressure part of hybrid coordinates."""
    return (self.a_boundaries[1:] + self.a_boundaries[:-1]) / 2

  @property
  def b_centers(self) -> np.ndarray:
    """Returns center values for sigma part of hybrid coordinates."""
    return (self.b_boundaries[1:] + self.b_boundaries[:-1]) / 2

  @property
  def center_to_center(self) -> np.ndarray:
    """Returns center-to-center distance in sigma part of coordinates."""
    return np.diff(self.b_centers)

  def pressure_boundaries(self, surface_pressure: typing.Numeric) -> Array:
    """Returns boundaries of each layer in pressure units."""
    ndim = len([s for s in jnp.shape(surface_pressure) if s > 1])
    new_dims = tuple(range(1, ndim + 1))
    a_boundaries = np.expand_dims(self.a_boundaries, new_dims)
    b_boundaries = np.expand_dims(self.b_boundaries, new_dims)
    return a_boundaries + b_boundaries * surface_pressure

  def pressure_centers(self, surface_pressure: typing.Numeric) -> Array:
    """Returns centers of each layer in pressure units."""
    boundaries = self.pressure_boundaries(surface_pressure)
    return (boundaries[1:] + boundaries[:-1]) / 2

  def layer_thickness(self, surface_pressure: typing.Numeric) -> Array:
    """Returns thickness of each layer in pressure units."""
    ndim = len([s for s in jnp.shape(surface_pressure) if s > 1])
    new_dims = tuple(range(1, ndim + 1))
    p_thickness = np.expand_dims(self.pressure_thickness, new_dims)
    sigma_thickness = np.expand_dims(self.sigma_thickness, new_dims)
    return p_thickness + sigma_thickness * surface_pressure

  @classmethod
  def from_sigma_levels(
      cls, sigma_levels: sigma_coordinates.SigmaCoordinates
  ) -> HybridCoordinates:
    """Creates hybrid coordinates that effectively are sigma coordinates."""
    b_boundaries = np.array(sigma_levels.boundaries)
    a_boundaries = np.zeros_like(b_boundaries)
    return cls(a_boundaries=a_boundaries, b_boundaries=b_boundaries)

  @classmethod
  def _from_resource_csv(cls, path: str) -> HybridCoordinates:
    levels_csv = importlib.resources.files(dinosaur).joinpath(path)
    with levels_csv.open() as f:
      a_in_pa, b = np.loadtxt(f, skiprows=1, usecols=(1, 2), delimiter='\t').T
    a = a_in_pa / 100  # convert from Pa to hPa
    # any reasonable hybrid coordinate system falls in this range (certainly
    # including UFS and ECMWF)
    assert 100 < a.max() < 1000
    return cls(a_boundaries=a, b_boundaries=b)

  @classmethod
  def ecmwf137_interpolated(
      cls,
      n_levels: int,
  ) -> HybridCoordinates:
    """Returns hybrid coordinates interpolated from ECMWF 137 levels."""
    base = cls.ECMWF137()

    x_old = np.linspace(0, 1, base.layers + 1)
    x_new = np.linspace(0, 1, n_levels + 1)

    a_new = np.interp(x_new, x_old, base.a_boundaries)
    b_new = np.interp(x_new, x_old, base.b_boundaries)

    return cls(a_boundaries=a_new, b_boundaries=b_new)

  @classmethod
  def ECMWF137(cls) -> HybridCoordinates:  # pylint: disable=invalid-name
    """Returns the 137 model levels used by ECMWF's IFS (e.g., in ERA5).

    Pressure is returned in units of hPa.

    For details, see the ECMWF wiki:
    https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
    """
    return cls._from_resource_csv('data/ecmwf137_hybrid_levels.csv')

  @classmethod
  def UFS127(cls) -> HybridCoordinates:  # pylint: disable=invalid-name
    """Returns the 127 model levels used by NOAA's UFS (GFS v16).

    Pressure is returned in units of hPa.

    For details, see the documentation for UFS Replay:
    https://ufs2arco.readthedocs.io/en/latest/example_pressure_interpolation.html

    Source data:
    https://github.com/NOAA-PSL/ufs2arco/blob/v0.1.2/ufs2arco/replay_vertical_levels.yaml
    """
    return cls._from_resource_csv('data/ufs127_hybrid_levels.csv')

  @property
  def layers(self) -> int:
    return len(self.a_boundaries) - 1

  def __hash__(self):
    return hash(
        (tuple(self.a_boundaries.tolist()), tuple(self.b_boundaries.tolist()))
    )

  def __eq__(self, other):
    return (
        isinstance(other, HybridCoordinates)
        and np.array_equal(self.a_boundaries, other.a_boundaries)
        and np.array_equal(self.b_boundaries, other.b_boundaries)
    )

  def get_sigma_boundaries(
      self, surface_pressure: typing.Numeric
  ) -> typing.Array:
    """Returns centers of sigma levels for a given surface pressure.

    Args:
      surface_pressure: scalar surface pressure, in the same units as
        `a_boundaries`.

    Returns:
      Array with shape `(layers + 1,)`.
    """
    return self.a_boundaries / surface_pressure + self.b_boundaries

  def get_sigma_centers(self, surface_pressure: typing.Numeric) -> typing.Array:
    """Returns centers of sigma levels for a given surface pressure.

    Args:
      surface_pressure: scalar surface pressure, in the same units as
        `a_boundaries`.

    Returns:
      Array with shape `(layers,)`.
    """
    boundaries = self.get_sigma_boundaries(surface_pressure)
    return (boundaries[1:] + boundaries[:-1]) / 2

  def to_approx_sigma_coords(
      self, layers: int, surface_pressure: float = 1013.25
  ) -> sigma_coordinates.SigmaCoordinates:
    """Interpolate these hybrid coordinates to approximate sigma levels.

    The resulting coordinates will typically not be equidistant.

    Args:
      layers: number of sigma layers to return.
      surface_pressure: reference surface pressure to use for interpolation. The
        default value is 1013.25, which is one standard atmosphere in hPa.

    Returns:
      New SigmaCoordinates object wih the requested number of layers.
    """
    original_bounds = self.get_sigma_boundaries(surface_pressure)
    interpolated_bounds = jax.vmap(jnp.interp, (0, None, None))(
        jnp.linspace(0, 1, num=layers + 1),
        jnp.linspace(0, 1, num=self.layers + 1),
        original_bounds,
    )
    interpolated_bounds = np.array(interpolated_bounds)
    # Some hybrid coordinates (e.g., from UFS) start at a non-zero pressure
    # level. It is not clear that this makes sense for Dinosaur, so to be safe,
    # we set the first level to 0 (zero pressure) and the last level to 1
    # (surface pressure).
    interpolated_bounds[0] = 0.0
    interpolated_bounds[-1] = 1.0
    return sigma_coordinates.SigmaCoordinates(interpolated_bounds)
