# Copyright 2026 Google LLC
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

"""Semi-Lagrangian transport on the sphere.

This module provides the building blocks for semi-Lagrangian advection on
full Gaussian grids: spherical geometry helpers (Cartesian coordinates,
tangent bases, and parallel transport of tangent vectors), a fixed-iteration
departure-point solver, and gather-based linear/cubic Lagrange interpolation
with longitude wrapping, cross-pole halos, and an optional quasi-monotone
limiter (Bermejo & Staniforth 1992).

All operations use static shapes and a fixed number of iterations, so they
are reverse-mode differentiable and compatible with `jax.jit`/`jax.vmap`.

Conventions:

- Nodal fields have trailing dimensions [longitude, latitude], matching
  `spherical_harmonic.Grid.nodal_shape`.
- Winds are true velocities (u, v) in the local (east, north) directions,
  *not* the cosθ-scaled velocities used by dinosaur's spectral diagnostics.
- Cartesian vectors and positions place the (x, y, z) components along a
  leading axis of size 3, with z pointing at the north pole.

References:
  Bermejo, R. & Staniforth, A., 1992: The conversion of semi-Lagrangian
    advection schemes to quasi-monotone schemes. Mon. Wea. Rev., 120.
  Diamantakis, M., 2014: The semi-Lagrangian technique in atmospheric
    modelling. ECMWF Seminar on Numerical Methods.
  Diamantakis, M. & Váňa, F., 2022: A fast converging and concise algorithm
    for computing the departure points in semi-Lagrangian weather and
    climate models. Q. J. R. Meteorol. Soc., 148, 670-684. (Introduced in
    the IFS as "a new way of computing semi-Lagrangian advection", ECMWF
    Newsletter 173, 2022.)
  Shapiro, R., 1970: Smoothing, filtering, and boundary effects. Rev.
    Geophys., 8, 359-387.
  Staniforth, A. & Côté, J., 1991: Semi-Lagrangian integration schemes for
    atmospheric models — a review. Mon. Wea. Rev., 119.
  Váňa, F., Bénard, P., Geleyn, J.-F., Simon, A. & Seity, Y., 2008:
    Semi-Lagrangian advection scheme with controlled damping: an
    alternative to nonlinear horizontal diffusion in a numerical weather
    prediction model. Q. J. R. Meteorol. Soc., 134, 523-537.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Callable, NamedTuple

from dinosaur import coordinate_systems
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np
import tree_math


Array = typing.Array

# All `einsum`s should be done at highest available precision.
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


#  =============================================================================
#  Spherical geometry helpers.
#  =============================================================================


def lon_lat_to_cartesian(lon: Array, sin_lat: Array) -> jnp.ndarray:
  """Computes Cartesian coordinates of unit vectors on the sphere.

  Args:
    lon: longitudes in radians.
    sin_lat: sine of latitudes.

  Returns:
    Array of shape [3, *lon.shape] holding (x, y, z) coordinates of the unit
    vectors pointing at (lon, lat).
  """
  cos_lat = jnp.sqrt(1 - jnp.square(sin_lat))
  return jnp.stack(
      [cos_lat * jnp.cos(lon), cos_lat * jnp.sin(lon), sin_lat], axis=0
  )


def cartesian_to_lon_sin_lat(r: Array) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes (longitude, sine of latitude) from Cartesian unit vectors.

  Args:
    r: array of shape [3, ...] holding unit vectors on the sphere.

  Returns:
    Tuple of longitudes in [0, 2π) and sines of latitude.
  """
  x, y, z = r
  lon = jnp.arctan2(y, x) % (2 * np.pi)
  sin_lat = jnp.clip(z, -1, 1)
  return lon, sin_lat


def cartesian_wind(u: Array, v: Array, lon: Array, sin_lat: Array) -> Array:
  """Converts horizontal winds to Cartesian components.

  The Cartesian components of a tangent vector field are three smooth scalar
  fields on the sphere (unlike (u, v), which are only piecewise smooth at the
  poles), which makes them suitable for componentwise interpolation.

  Args:
    u: zonal wind (towards the east).
    v: meridional wind (towards the north).
    lon: longitudes in radians, broadcastable against `u`.
    sin_lat: sine of latitudes, broadcastable against `u`.

  Returns:
    Array of shape [3, ...] holding the wind vector `u ê_lon + v ê_lat` in
    Cartesian components.
  """
  cos_lat = jnp.sqrt(1 - jnp.square(sin_lat))
  sin_lon = jnp.sin(lon)
  cos_lon = jnp.cos(lon)
  return jnp.stack([
      -u * sin_lon - v * sin_lat * cos_lon,
      u * cos_lon - v * sin_lat * sin_lon,
      v * cos_lat,
  ])


def tangent_wind(
    w: Array, lon: Array, sin_lat: Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Projects a Cartesian vector field onto the local (east, north) basis.

  Args:
    w: array of shape [3, ...] holding Cartesian vector components.
    lon: longitudes in radians, broadcastable against `w[0]`.
    sin_lat: sine of latitudes, broadcastable against `w[0]`.

  Returns:
    Tuple (u, v) of zonal and meridional components. Any radial component of
    `w` is discarded.
  """
  cos_lat = jnp.sqrt(1 - jnp.square(sin_lat))
  sin_lon = jnp.sin(lon)
  cos_lon = jnp.cos(lon)
  w_x, w_y, w_z = w
  u = -w_x * sin_lon + w_y * cos_lon
  v = -w_x * sin_lat * cos_lon - w_y * sin_lat * sin_lon + w_z * cos_lat
  return u, v


def parallel_transport(w: Array, r_from: Array, r_to: Array) -> jnp.ndarray:
  """Transports tangent vectors along great circles.

  Applies the rotation about `r_from ✕ r_to` that takes `r_from` to `r_to`,
  which is exact parallel transport of tangent vectors along the connecting
  great circle. Uses a closed form of the Rodrigues rotation that is free of
  0/0 for `r_from == r_to` (it is singular only for antipodal points, far
  outside the displacements of a time step).

  Args:
    w: array of shape [3, ...] of tangent vectors at `r_from`.
    r_from: array of shape [3, ...] of unit vectors.
    r_to: array of shape [3, ...] of unit vectors.

  Returns:
    Array of shape [3, ...] holding `w` transported to the tangent plane at
    `r_to`.
  """
  cos_angle = (r_from * r_to).sum(axis=0)
  chord = r_from + r_to
  return (
      w
      - ((chord * w).sum(axis=0) / (1 + cos_angle)) * chord
      + 2 * (r_from * w).sum(axis=0) * r_to
  )


def _normalize(r: Array) -> jnp.ndarray:
  """Normalizes vectors of shape [3, ...] to unit length."""
  return r / jnp.sqrt(jnp.square(r).sum(axis=0, keepdims=True))


#  =============================================================================
#  Gather-based interpolation on the full Gaussian grid.
#  =============================================================================


def _lagrange_weights(x: Array, nodes: Array) -> jnp.ndarray:
  """Computes Lagrange interpolation weights.

  Args:
    x: coordinates at which to interpolate, of shape [...].
    nodes: stencil node coordinates, of shape [..., stencil_size]. Nodes need
      not be uniformly spaced.

  Returns:
    Weights of shape [..., stencil_size] that are exact for polynomials of
    degree `stencil_size - 1`.
  """
  stencil_size = nodes.shape[-1]
  diff = x[..., jnp.newaxis] - nodes
  denom = nodes[..., :, jnp.newaxis] - nodes[..., jnp.newaxis, :]
  eye = np.eye(stencil_size, dtype=bool)
  num = jnp.where(eye, 1.0, diff[..., jnp.newaxis, :])
  den = jnp.where(eye, 1.0, denom)
  return jnp.prod(num / den, axis=-1)


_HALO_WIDTH = 2  # supports stencils up to 4 points wide in latitude.


def _latitude_halo_axis(grid: spherical_harmonic.Grid) -> np.ndarray:
  """Returns latitudes (radians) extended with `_HALO_WIDTH` rows per pole.

  A point beyond the pole at "latitude" φ > π/2 corresponds to the physical
  location (π - φ, λ + π), so halo rows take coordinates mirrored about the
  pole. The extended axis is strictly increasing, which makes latitude (not
  its sine, which folds at the poles) the coordinate used for interpolation.
  """
  phi = grid.latitudes
  if not np.all(np.diff(phi) > 0):
    raise ValueError('expected latitudes in increasing order')
  south = -np.pi - phi[_HALO_WIDTH - 1 :: -1]
  north = np.pi - phi[: -_HALO_WIDTH - 1 : -1]
  extended = np.concatenate([south, phi, north])
  if not np.all(np.diff(extended) > 0):
    # e.g. 'equiangular_with_poles' grids, whose mirrored halo coordinates
    # duplicate the pole nodes and would produce 0/0 Lagrange weights.
    raise ValueError(
        'halo-extended latitudes are not strictly increasing; grids with '
        'nodes at the poles are not supported'
    )
  return extended


def _extend_with_pole_halo(field: Array, grid: spherical_harmonic.Grid) -> Array:
  """Appends `_HALO_WIDTH` cross-pole halo rows on both latitude ends.

  Halo values are taken from the antipodal longitude (a roll by half the
  longitude nodes), which is exact for even numbers of longitude nodes. Only
  scalar fields (including Cartesian components of tangent vectors) may be
  extended this way; (u, v) wind components would need sign flips.

  Args:
    field: array of shape [..., longitude_nodes, latitude_nodes].
    grid: the horizontal grid.

  Returns:
    Array of shape [..., longitude_nodes, latitude_nodes + 2 * _HALO_WIDTH].
  """
  num_lon = grid.longitude_nodes
  if num_lon % 2:
    raise ValueError(
        'cross-pole halos require an even number of longitude nodes; '
        f'got {num_lon}. Note that `Grid.with_wavenumbers` may produce odd '
        'longitude counts; the `Grid.T*` and `Grid.TL*` constructors do not.'
    )
  rolled = jnp.roll(field, num_lon // 2, axis=-2)
  south = rolled[..., _HALO_WIDTH - 1 :: -1]
  north = rolled[..., : -_HALO_WIDTH - 1 : -1]
  return jnp.concatenate([south, field, north], axis=-1)


def _stencil_offsets(order: str) -> np.ndarray:
  """Returns stencil offsets relative to the lower bracketing node."""
  if order == 'linear':
    return np.arange(2)
  elif order == 'cubic':
    return np.arange(-1, 3)
  else:
    raise ValueError(f'unknown interpolation {order=}')


@dataclasses.dataclass
class _HorizontalStencil:
  """Precomputed gather indices and weights for horizontal interpolation.

  Attributes:
    lon_index: [..., stencil] longitude gather indices (wrapped).
    lat_index: [..., stencil] latitude gather indices into the halo-extended
      latitude axis.
    lon_weights: [..., stencil] interpolation weights along longitude.
    lat_weights: [..., stencil] interpolation weights along latitude.
    extended_latitude_size: size of the halo-extended latitude axis.
  """

  lon_index: Array
  lat_index: Array
  lon_weights: Array
  lat_weights: Array
  extended_latitude_size: int


def _horizontal_stencil(
    grid: spherical_harmonic.Grid,
    lon: Array,
    sin_lat: Array,
    order: str,
) -> _HorizontalStencil:
  """Computes gather indices and weights for points (lon, sin_lat)."""
  offsets = _stencil_offsets(order)
  lon = jnp.asarray(lon)
  # Latitude via arctan2 rather than arcsin: for |sin_lat| within float
  # rounding of 1 (a trajectory point on a pole), arcsin has an infinite
  # derivative and its reverse-mode gradient becomes NaN. The epsilon keeps
  # the gradient finite while perturbing the pole latitude by less than
  # 1e-6 radians.
  sin_lat = jnp.clip(jnp.asarray(sin_lat), -1, 1)
  cos_lat = jnp.sqrt(jnp.maximum(1 - jnp.square(sin_lat), 1e-12))
  phi = jnp.arctan2(sin_lat, cos_lat)

  # Longitude nodes are uniform, so the bracketing index is a floor divide.
  # Weights are computed with unwrapped (real line) node coordinates; only
  # gather indices are wrapped.
  longitudes = grid.nodal_axes[0]
  lon_spacing = 2 * np.pi / grid.longitude_nodes
  lon_cell = jnp.floor((lon - longitudes[0]) / lon_spacing).astype(jnp.int32)
  lon_stencil = lon_cell[..., jnp.newaxis] + offsets
  lon_nodes = lon_stencil * lon_spacing + longitudes[0]
  lon_weights = _lagrange_weights(lon, lon_nodes)
  lon_index = lon_stencil % grid.longitude_nodes

  # Latitude nodes are non-uniform; searchsorted locates the bracketing ring
  # on the halo-extended axis and Lagrange weights use actual node latitudes.
  phi_extended = _latitude_halo_axis(grid)
  lat_cell = (
      jnp.searchsorted(jnp.asarray(phi_extended), phi, side='right') - 1
  ).astype(jnp.int32)
  # Physical points always satisfy |φ| <= π/2, which keeps cubic stencils
  # within the extended axis; clip for numerical safety only.
  lat_cell = jnp.clip(
      lat_cell, -offsets[0], phi_extended.size - 1 - offsets[-1]
  )
  lat_stencil = lat_cell[..., jnp.newaxis] + offsets
  lat_nodes = jnp.asarray(phi_extended)[lat_stencil]
  lat_weights = _lagrange_weights(phi, lat_nodes)

  return _HorizontalStencil(
      lon_index=lon_index,
      lat_index=lat_stencil,
      lon_weights=lon_weights,
      lat_weights=lat_weights,
      extended_latitude_size=phi_extended.size,
  )


_LIMITERS = ('quasi_monotone',)


def _validate_limiter(limiter: str | None) -> None:
  if limiter is not None and limiter not in _LIMITERS:
    raise ValueError(
        f'unknown interpolation {limiter=}; valid values are None (no '
        f'limiting) or one of {_LIMITERS}'
    )


def _linear_cell_slice(order: str) -> slice:
  """Slice selecting the 2-point bracketing cell within a stencil."""
  start = -int(_stencil_offsets(order)[0])
  return slice(start, start + 2)


def _gather_2d(field_extended: Array, stencil: _HorizontalStencil) -> Array:
  """Gathers stencil values [..., stencil, stencil] from [lon, lat_ext].

  Uses a single 1-D gather with precomputed linear indices; the `ravel` of
  the contiguous field compiles to a zero-cost bitcast (see the analogous
  gather in `interpolate_3d`).
  """
  flat_index = (
      stencil.lon_index[..., :, jnp.newaxis] * stencil.extended_latitude_size
      + stencil.lat_index[..., jnp.newaxis, :]
  )
  return jnp.take(field_extended.ravel(), flat_index, mode='clip')


def _interpolate_horizontal(
    field: Array,
    grid: spherical_harmonic.Grid,
    stencil: _HorizontalStencil,
    order: str,
    limiter: str | None,
) -> jnp.ndarray:
  """Interpolates a single [lon, lat] field with a precomputed stencil."""
  values = _gather_2d(_extend_with_pole_halo(field, grid), stencil)
  result = einsum(
      '...i,...j,...ij->...', stencil.lon_weights, stencil.lat_weights, values
  )
  if limiter == 'quasi_monotone':
    cell = _linear_cell_slice(order)
    corners = values[..., cell, cell]
    result = jnp.clip(
        result,
        corners.min(axis=(-2, -1)),
        corners.max(axis=(-2, -1)),
    )
  return result


@dataclasses.dataclass(frozen=True)
class GridInterpolator:
  """Interpolates nodal fields on the sphere at arbitrary points.

  Interpolation is a tensor-product Lagrange rule: uniform in longitude
  (with periodic wrapping) and on the actual Gaussian node latitudes (with
  cross-pole halo rows), so it is exact for polynomials in (longitude,
  latitude) up to the stencil order.

  Attributes:
    grid: the horizontal grid holding the source data.
    order: 'linear' (2✕2 stencil) or 'cubic' (4✕4 stencil).
    limiter: None (the default) for unlimited interpolation, or
      'quasi_monotone' for the limiter of Bermejo & Staniforth (1992):
      interpolated values are clipped to the range of the 2✕2 bracketing
      cell corners, which prevents new extrema (and preserves positivity)
      at the cost of formal accuracy at extrema.
  """

  grid: spherical_harmonic.Grid
  order: str = 'cubic'
  limiter: str | None = None

  def __post_init__(self):
    _validate_limiter(self.limiter)

  def __call__(self, field: Array, lon: Array, sin_lat: Array) -> jnp.ndarray:
    """Interpolates `field` at the points (lon, sin_lat).

    Args:
      field: nodal values of shape [longitude_nodes, latitude_nodes] (points
        may then have any shape), or [levels, longitude_nodes,
        latitude_nodes], in which case each level is interpolated at its own
        set of points via `jax.vmap` over the leading axis.
      lon: longitudes in radians, of shape [*points] or [levels, *points].
      sin_lat: sine of latitudes, same shape as `lon`.

    Returns:
      Interpolated values of shape `lon.shape`.
    """
    field = jnp.asarray(field)
    interpolate = functools.partial(
        self._interpolate_single, order=self.order, limiter=self.limiter
    )
    if field.ndim == 2:
      return interpolate(field, lon, sin_lat)
    if field.ndim != 3:
      raise ValueError(
          f'expected a field with 2 or 3 dimensions; got {field.shape=}'
      )
    if lon.shape[0] != field.shape[0]:
      raise ValueError(
          'batched interpolation requires points with a leading levels axis '
          f'matching the field: {field.shape=}, {lon.shape=}'
      )
    return jax.vmap(interpolate)(field, lon, sin_lat)

  def _interpolate_single(
      self,
      field: Array,
      lon: Array,
      sin_lat: Array,
      order: str,
      limiter: str | None,
  ) -> jnp.ndarray:
    return _interpolate_horizontal_single(
        field, lon, sin_lat, grid=self.grid, order=order, limiter=limiter
    )


# jitting the per-field interpolation entry points means repeated call sites
# with the same signature (e.g. the two stages of an RK2 step, or several
# tracers with the same limiter) share a single lowered computation instead
# of inlining a copy each (~30% fewer lowered ops on a full step). Measured
# GPU compile time is unchanged — XLA's cost is dominated elsewhere — so
# this is a tracing/lowering cleanup, not a compile-time fix. `grid` is
# hashable (frozen dataclass), following the
# `spherical_harmonic.vor_div_to_uv_nodal` precedent.
@functools.partial(jax.jit, static_argnames=('grid', 'order', 'limiter'))
def _interpolate_horizontal_single(
    field: Array,
    lon: Array,
    sin_lat: Array,
    *,
    grid: spherical_harmonic.Grid,
    order: str,
    limiter: str | None,
) -> jnp.ndarray:
  stencil = _horizontal_stencil(grid, lon, sin_lat, order)
  return _interpolate_horizontal(field, grid, stencil, order, limiter)


@functools.partial(jax.jit, static_argnames=('grid', 'order', 'limiter'))
def interpolate_3d(
    field: Array,
    grid: spherical_harmonic.Grid,
    lon: Array,
    sin_lat: Array,
    sigma: Array,
    sigma_nodes: np.ndarray,
    *,
    order: str = 'cubic',
    limiter: str | None = None,
) -> jnp.ndarray:
  """Interpolates a 3-D field at points (lon, sin_lat, sigma).

  Horizontal interpolation follows `GridInterpolator` (tensor-product
  Lagrange of the given order); vertical interpolation is linear in σ between
  `sigma_nodes` with constant extrapolation beyond the first and last nodes
  (matching `vertical_interpolation.interp` and the zero-gradient boundary
  conditions of `sigma_coordinates.centered_vertical_advection`).

  Args:
    field: nodal values of shape [len(sigma_nodes), longitude_nodes,
      latitude_nodes].
    grid: the horizontal grid holding the source data.
    lon: longitudes in radians, of any shape (typically [layers, longitude
      nodes, latitude_nodes] for departure points of full-level arrivals).
    sin_lat: sine of latitudes, same shape as `lon`.
    sigma: σ coordinates of the interpolation points, same shape as `lon`.
    sigma_nodes: increasing σ values at which `field` levels live (layer
      centers for prognostic fields, layer boundaries for vertical velocity).
    order: horizontal interpolation order, 'linear' or 'cubic'.
    limiter: None for unlimited interpolation, or 'quasi_monotone' to clip
      interpolated values to the range of the 2✕2✕2 bracketing cell corners
      (the Bermejo & Staniforth 1992 limiter).

  Returns:
    Interpolated values of shape `lon.shape`.
  """
  _validate_limiter(limiter)
  field = jnp.asarray(field)
  if len(sigma_nodes) < 2:
    raise ValueError('vertical interpolation requires at least 2 levels')
  if field.ndim != 3 or field.shape[0] != len(sigma_nodes):
    raise ValueError(
        f'expected field of shape [{len(sigma_nodes)}, lon, lat]; '
        f'got {field.shape}'
    )
  stencil = _horizontal_stencil(grid, lon, sin_lat, order)
  sigma_nodes = jnp.asarray(sigma_nodes)
  num_levels = sigma_nodes.shape[0]
  # Linear vertical interpolation with constant extrapolation: clipping σ to
  # the node range makes the boundary node weight saturate at 1.
  sigma = jnp.clip(jnp.asarray(sigma), sigma_nodes[0], sigma_nodes[-1])
  level_cell = (
      jnp.searchsorted(sigma_nodes, sigma, side='right') - 1
  ).astype(jnp.int32)
  level_cell = jnp.clip(level_cell, 0, num_levels - 2)
  below = sigma_nodes[level_cell]
  above = sigma_nodes[level_cell + 1]
  fraction = (sigma - below) / (above - below)
  level_weights = jnp.stack([1 - fraction, fraction], axis=-1)
  level_index = level_cell[..., jnp.newaxis] + np.arange(2)

  extended = _extend_with_pole_halo(field, grid)
  extended_latitude_size = extended.shape[-1]
  # One 1-D gather on the flattened field with precomputed linear indices:
  # `ravel` of the contiguous field compiles to a zero-cost bitcast, and the
  # single gather lowers more efficiently than the equivalent multi-array
  # advanced indexing, with defined out-of-bounds semantics via mode='clip'.
  # int32 indices limit the flattened size to 2**31 elements: ample for any
  # realistic grid (the limit is roughly T1700 at 137 levels).
  flat_index = (
      level_index[..., :, jnp.newaxis, jnp.newaxis] * grid.longitude_nodes
      + stencil.lon_index[..., jnp.newaxis, :, jnp.newaxis]
  ) * extended_latitude_size + stencil.lat_index[
      ..., jnp.newaxis, jnp.newaxis, :
  ]
  values = jnp.take(extended.ravel(), flat_index, mode='clip')
  result = einsum(
      '...v,...i,...j,...vij->...',
      level_weights,
      stencil.lon_weights,
      stencil.lat_weights,
      values,
  )
  if limiter == 'quasi_monotone':
    cell = _linear_cell_slice(order)
    corners = values[..., :, cell, cell]
    result = jnp.clip(
        result,
        corners.min(axis=(-3, -2, -1)),
        corners.max(axis=(-3, -2, -1)),
    )
  return result


#  =============================================================================
#  Departure points.
#  =============================================================================


class VerticalNodes(NamedTuple):
  """Fixed vertical interpolation nodes for semi-Lagrangian transport.

  Any vertical coordinate whose per-level node positions do not vary in
  space or time works: σ layer centers/boundaries, or hybrid-coordinate
  reference-σ nodes (the hybrid *level* coordinate is fixed even though the
  pressure of each level depends on surface pressure). Provides the same
  `.centers`/`.boundaries` attributes as `sigma_coordinates.SigmaCoordinates`
  consumed by the transport functions.

  Attributes:
    centers: increasing node values at layer centers, shape [layers].
    boundaries: increasing node values at layer boundaries, [layers + 1].
  """

  centers: np.ndarray
  boundaries: np.ndarray


@tree_math.struct
class DeparturePoints:
  """Departure points of trajectories arriving at grid points.

  Attributes:
    lon: longitudes in radians.
    sin_lat: sine of latitudes.
    cartesian: unit vectors on the sphere of shape [3, *lon.shape].
    sigma: σ coordinates of departure points, or None for horizontal-only
      trajectories.
  """

  lon: Array
  sin_lat: Array
  cartesian: Array
  sigma: Array | None = None


def horizontal_departure_points(
    u: Array,
    v: Array,
    grid: spherical_harmonic.Grid,
    dt: float,
    *,
    iterations: int = 2,
    initial_guess: DeparturePoints | None = None,
) -> DeparturePoints:
  """Solves for departure points of horizontal-only trajectories.

  Uses the standard fixed-point iteration on the trajectory midpoint
  (Robert-style; see Diamantakis 2014, Eq. 5), performed in Cartesian
  coordinates on the unit sphere so there is no polar singularity. Winds at
  the midpoint are interpolated bilinearly, which is standard and sufficient
  for the trajectory solve (Staniforth & Côté 1991). The iteration count is
  fixed (no tolerances) for reverse-mode differentiability; convergence
  requires `dt * max‖∇V‖ < 1`. Two iterations give second-order accurate
  departure points.

  This geocentric-Cartesian formulation — interpolating Cartesian wind
  components, with no per-iteration rotation matrices, no spherical-polar
  trigonometric solve, and no double-precision requirement near the poles —
  is the same reformulation ECMWF adopted for the IFS in Cycle 48r1
  (Diamantakis & Váňa 2022; ECMWF Newsletter 173). That work's warm-start
  refinement — starting the iteration from previously computed departure
  points instead of the arrival points, which let the IFS cut its iteration
  count from 5 to 3 at equal accuracy — is available via `initial_guess`:
  the SETTLS stepper carries the previous step's departure points
  (`warm_start_departures`), and the one-step RK2 stepper seeds its
  corrector stage's solve from the predictor's (`warm_start_corrector`,
  no multistep memory required). One warm-started iteration matches the
  residual of two cold ones on baroclinic flows.

  Args:
    u: nodal zonal wind (true winds, not cosθ-scaled) of shape
      [longitude_nodes, latitude_nodes] or [levels, longitude_nodes,
      latitude_nodes]. Each level is advected by its own wind field.
    v: nodal meridional wind, same shape as `u`.
    grid: the horizontal grid.
    dt: time step. Trajectories are integrated backwards over `dt`.
    iterations: number of fixed-point iterations.
    initial_guess: optional departure points to warm-start the iteration
      (e.g. a predictor stage's or the previous step's solution, following
      the IFS practice noted above); defaults to the arrival points.

  Returns:
    DeparturePoints with fields of shape [*batch, longitude_nodes,
    latitude_nodes] and `sigma=None`.
  """
  u = jnp.asarray(u)
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  arrival = lon_lat_to_cartesian(
      jnp.broadcast_to(lon_mesh, u.shape),
      jnp.broadcast_to(sin_lat_mesh, u.shape),
  )
  wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
  interpolator = GridInterpolator(grid, order='linear')
  angular_dt = dt / grid.radius

  departure = arrival if initial_guess is None else initial_guess.cartesian
  for _ in range(iterations):
    midpoint = _normalize((arrival + departure) / 2)
    lon_mid, sin_lat_mid = cartesian_to_lon_sin_lat(midpoint)
    # vmap over the three Cartesian components: the stencil indices and
    # weights are computed once and the gathers batch into one.
    wind_mid = jax.vmap(
        lambda component: interpolator(component, lon_mid, sin_lat_mid)
    )(wind)
    departure = _normalize(arrival - angular_dt * wind_mid)

  lon, sin_lat = cartesian_to_lon_sin_lat(departure)
  return DeparturePoints(
      lon=lon, sin_lat=sin_lat, cartesian=departure, sigma=None
  )


def departure_points_3d(
    u: Array,
    v: Array,
    sigma_dot: Array,
    coords: coordinate_systems.CoordinateSystem,
    dt: float,
    *,
    iterations: int = 2,
    vertical_nodes: VerticalNodes | None = None,
    initial_guess: DeparturePoints | None = None,
) -> DeparturePoints:
  """Solves for 3-D departure points of trajectories arriving at layer centers.

  The horizontal update matches `horizontal_departure_points`; the vertical
  position is updated jointly in the same fixed-point iteration using σ̇
  interpolated (tri)linearly at the trajectory midpoint. Departure σ values
  are clipped to the range of layer centers, consistent with constant
  extrapolation of the advected fields beyond the first and last layers.

  Args:
    u: nodal zonal wind (true winds, not cosθ-scaled) at layer centers, of
      shape [layers, longitude_nodes, latitude_nodes].
    v: nodal meridional wind, same shape as `u`.
    sigma_dot: vertical velocity dσ/dt at all layer boundaries, of shape
      [layers + 1, longitude_nodes, latitude_nodes] (zero at σ = 0 and
      σ = 1), as returned by e.g. `PrimitiveEquationsSigma.nodal_velocities`.
    coords: horizontal and vertical coordinate system.
    dt: time step. Trajectories are integrated backwards over `dt`.
    iterations: number of fixed-point iterations.
    vertical_nodes: optional fixed vertical nodes to use instead of
      `coords.vertical` (e.g. hybrid-coordinate reference-σ nodes, with
      `sigma_dot` the rate of change of that node coordinate).
    initial_guess: optional departure points to warm-start the iteration;
      defaults to the arrival points. See `horizontal_departure_points`.

  Returns:
    DeparturePoints with fields of shape [layers, longitude_nodes,
    latitude_nodes], including departure σ coordinates.
  """
  grid = coords.horizontal
  if vertical_nodes is None:
    vertical_nodes = VerticalNodes(
        centers=coords.vertical.centers, boundaries=coords.vertical.boundaries
    )
  centers = vertical_nodes.centers
  boundaries = vertical_nodes.boundaries
  u = jnp.asarray(u)

  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  arrival = lon_lat_to_cartesian(
      jnp.broadcast_to(lon_mesh, u.shape),
      jnp.broadcast_to(sin_lat_mesh, u.shape),
  )
  sigma_arrival = jnp.broadcast_to(
      jnp.asarray(centers, dtype=u.dtype)[:, jnp.newaxis, jnp.newaxis], u.shape
  )
  wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
  angular_dt = dt / grid.radius
  interpolate = functools.partial(
      interpolate_3d, grid=grid, order='linear'
  )

  if initial_guess is None:
    departure = arrival
    sigma_departure = sigma_arrival
  else:
    departure = initial_guess.cartesian
    sigma_departure = initial_guess.sigma
  for _ in range(iterations):
    midpoint = _normalize((arrival + departure) / 2)
    sigma_mid = (sigma_arrival + sigma_departure) / 2
    lon_mid, sin_lat_mid = cartesian_to_lon_sin_lat(midpoint)
    point = dict(lon=lon_mid, sin_lat=sin_lat_mid, sigma=sigma_mid)
    # vmap over the three Cartesian components: the stencil indices and
    # weights are computed once and the gathers batch into one.
    wind_mid = jax.vmap(
        functools.partial(interpolate, sigma_nodes=centers, **point)
    )(wind)
    sigma_dot_mid = interpolate(sigma_dot, sigma_nodes=boundaries, **point)
    departure = _normalize(arrival - angular_dt * wind_mid)
    sigma_departure = jnp.clip(
        sigma_arrival - dt * sigma_dot_mid, float(centers[0]), float(centers[-1])
    )

  lon, sin_lat = cartesian_to_lon_sin_lat(departure)
  return DeparturePoints(
      lon=lon, sin_lat=sin_lat, cartesian=departure, sigma=sigma_departure
  )


#  =============================================================================
#  Transport.
#  =============================================================================


def transport_scalar_2d(
    field: Array,
    departure: DeparturePoints,
    interpolator: GridInterpolator,
) -> jnp.ndarray:
  """Remaps a scalar field to arrival points along 2-D trajectories.

  Args:
    field: nodal values of shape [longitude_nodes, latitude_nodes] or
      [levels, longitude_nodes, latitude_nodes].
    departure: horizontal departure points with fields shaped like `field`.
    interpolator: horizontal interpolation rule.

  Returns:
    The transported field at arrival points, shaped like `departure.lon`.
  """
  return interpolator(field, departure.lon, departure.sin_lat)


def transport_scalar(
    field: Array,
    departure: DeparturePoints,
    vertical: sigma_coordinates.SigmaCoordinates,
    interpolator: GridInterpolator,
) -> jnp.ndarray:
  """Remaps a scalar field to arrival points along 3-D trajectories.

  Args:
    field: nodal values at layer centers, of shape [layers, longitude_nodes,
      latitude_nodes].
    departure: 3-D departure points (with σ coordinates).
    vertical: vertical coordinates on which `field` lives: anything with
      increasing `.centers` node values (`SigmaCoordinates` or
      `VerticalNodes`).
    interpolator: interpolation rule (its grid, order and limiter settings
      are used).

  Returns:
    The transported field at arrival points, shaped like `departure.lon`.
  """
  if departure.sigma is None:
    raise ValueError('transport_scalar requires 3-D departure points')
  return interpolate_3d(
      field,
      interpolator.grid,
      departure.lon,
      departure.sin_lat,
      departure.sigma,
      sigma_nodes=vertical.centers,
      order=interpolator.order,
      limiter=interpolator.limiter,
  )


def planetary_velocity(r: Array, rotation_rate: float, radius: float) -> Array:
  """Computes the planetary momentum field 2Ω ✕ R at unit vectors `r`.

  This is twice the velocity of solid-body planetary rotation: the quantity
  `v + 2Ω ✕ R` obeys a momentum equation with no Coriolis force, since the
  horizontal projection of `2Ω ✕ v` is exactly `f k ✕ v` (IFS's LADVF
  option; Temperton, Hortal & Simmons 2001).

  Args:
    r: array of shape [3, ...] of unit vectors on the sphere.
    rotation_rate: planetary angular velocity Ω.
    radius: radius of the sphere.

  Returns:
    Array of shape [3, ...]: `2 Ω a (ẑ ✕ r)`, a tangent vector field pointing
    east with magnitude `2Ωa cosθ`.
  """
  scale = 2 * rotation_rate * radius
  return jnp.stack([-scale * r[1], scale * r[0], jnp.zeros_like(r[0])])


def _finish_wind_transport(
    wind_departure: Array,
    departure: DeparturePoints,
    grid: spherical_harmonic.Grid,
    rotate: bool,
    planetary_rotation_rate: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Rotates interpolated Cartesian winds to arrival and projects to (u, v)."""
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  arrival = lon_lat_to_cartesian(
      jnp.broadcast_to(lon_mesh, departure.lon.shape),
      jnp.broadcast_to(sin_lat_mesh, departure.lon.shape),
  )
  if planetary_rotation_rate is not None:
    # add the analytic planetary momentum at the departure point (only the
    # relative wind is ever interpolated) and subtract it at arrival.
    wind_departure = wind_departure + planetary_velocity(
        departure.cartesian, planetary_rotation_rate, grid.radius
    )
  if rotate:
    wind_arrival = parallel_transport(
        wind_departure, departure.cartesian, arrival
    )
  else:
    wind_arrival = wind_departure
  if planetary_rotation_rate is not None:
    wind_arrival = wind_arrival - planetary_velocity(
        arrival, planetary_rotation_rate, grid.radius
    )
  return tangent_wind(wind_arrival, lon_mesh, sin_lat_mesh)


def transport_wind(
    u: Array,
    v: Array,
    departure: DeparturePoints,
    vertical: sigma_coordinates.SigmaCoordinates,
    interpolator: GridInterpolator,
    *,
    rotate: bool = True,
    planetary_rotation_rate: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Remaps horizontal winds to arrival points along 3-D trajectories.

  Winds are interpolated as the three Cartesian components of the tangent
  vector field (smooth everywhere, including across the poles), then rotated
  from the departure-point tangent plane to the arrival-point tangent plane
  by exact parallel transport along the great circle, and finally projected
  onto the local (east, north) basis. Skipping the rotation commits an
  O(|Δr|²) directional error, so it defaults to on.

  Args:
    u: nodal zonal wind (true winds, not cosθ-scaled) at layer centers, of
      shape [layers, longitude_nodes, latitude_nodes].
    v: nodal meridional wind, same shape as `u`.
    departure: 3-D departure points (with σ coordinates).
    vertical: vertical coordinates on which the winds live (anything with
      increasing `.centers` node values).
    interpolator: interpolation rule. Limiters are never applied to winds.
    rotate: whether to parallel-transport interpolated vectors from the
      departure to the arrival tangent plane.
    planetary_rotation_rate: if not None, transport the planetary momentum
      `v + 2Ω ✕ R` instead of the plain wind `v`, with the analytic planetary
      component added at departure and removed at arrival (see
      `planetary_velocity`). Along-trajectory transport of this quantity has
      no Coriolis force, so equations using it drop `f k ✕ v` from their
      explicit terms entirely — the standard configuration for long time
      steps. Note this makes transport affine rather than linear: the
      caller's (u, v) must carry the state winds with coefficient exactly
      one (see `SemiLagrangianImplicitExplicitODE.semi_lagrangian_transport`).

  Returns:
    Tuple (u, v) of transported winds at arrival points.
  """
  if departure.sigma is None:
    raise ValueError('transport_wind requires 3-D departure points')
  grid = interpolator.grid
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
  interpolate = functools.partial(
      interpolate_3d,
      grid=grid,
      lon=departure.lon,
      sin_lat=departure.sin_lat,
      sigma=departure.sigma,
      sigma_nodes=vertical.centers,
      order=interpolator.order,
      limiter=None,
  )
  # vmap over the three Cartesian components: the stencil indices and
  # weights are computed once and the gathers batch into one.
  wind_departure = jax.vmap(interpolate)(wind)
  return _finish_wind_transport(
      wind_departure, departure, grid, rotate, planetary_rotation_rate
  )


def transport_wind_2d(
    u: Array,
    v: Array,
    departure: DeparturePoints,
    interpolator: GridInterpolator,
    *,
    rotate: bool = True,
    planetary_rotation_rate: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Remaps horizontal winds along 2-D trajectories (see `transport_wind`).

  Args:
    u: nodal zonal wind (true winds, not cosθ-scaled) of shape
      [longitude_nodes, latitude_nodes] or [levels, longitude_nodes,
      latitude_nodes].
    v: nodal meridional wind, same shape as `u`.
    departure: horizontal departure points with fields shaped like `u`.
    interpolator: interpolation rule. Limiters are never applied to winds.
    rotate: whether to parallel-transport interpolated vectors from the
      departure to the arrival tangent plane.
    planetary_rotation_rate: if not None, transport the planetary momentum
      `v + 2Ω ✕ R` (see `transport_wind`).

  Returns:
    Tuple (u, v) of transported winds at arrival points.
  """
  grid = interpolator.grid
  unlimited = dataclasses.replace(interpolator, limiter=None)
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
  # vmap over the three Cartesian components: the stencil indices and
  # weights are computed once and the gathers batch into one.
  wind_departure = jax.vmap(
      lambda component: unlimited(component, departure.lon, departure.sin_lat)
  )(wind)
  return _finish_wind_transport(
      wind_departure, departure, grid, rotate, planetary_rotation_rate
  )


def nodal_diffusion_filter(
    grid: spherical_harmonic.Grid,
    dt: float,
    tau: float,
    order: int = 2,
) -> Callable[[Array], jnp.ndarray]:
  """Returns a grid-space smoothing filter for nodal fields.

  The nodal analogue of `time_integration.horizontal_diffusion_step_filter`:
  nodal tracers never enter the spectral basis, so they receive no spectral
  hyperdiffusion, and with purely advective dynamics nothing removes the
  grid-scale variance that the deformation of the flow generates — cubic
  interpolation's implicit damping vanishes for near-stationary features.
  This filter supplies that missing dissipation with a separable
  index-space smoother: periodic in longitude and using cross-pole halo
  rows in latitude (the same halo machinery as transport, so the pole rows
  are handled exactly).

  Per step and per direction the response on a wave of index wavenumber k
  is `1 - mu·sin²(kΔ/2)` for `order=1` (the 1-2-1 kernel) or
  `1 - mu·sin⁴(kΔ/2)` for `order=2` (a Shapiro-type δ⁴ kernel), with
  `mu = 1 - exp(-dt/tau)` so that `tau` is the e-folding time of the
  two-gridpoint mode — the same convention as the spectral filter, whose
  `tau` is a natural default here. `order=2` matches the scale selectivity
  of the modal ∇⁴ hyperdiffusion (at equal `tau` it damps an 8Δ wave ~7x
  more slowly than `order=1`) and is the default.

  Shape preservation: the `order=1` kernel is a convex combination, so it
  is positivity-preserving and creates no new extrema unconditionally. The
  `order=2` kernel can overshoot, so its result is clipped to the local
  3✕3 range of the unfiltered field (Bermejo & Staniforth-style), which
  restores the same guarantee.

  Caveats: the filter acts in index space (like operational Shapiro
  filtering), so near the poles the longitude pass acts on physically
  small scales, and the two rings nearest each pole receive slightly
  stronger smoothing because the latitude spacing jumps across the pole
  (a ~1% one-pass effect at mu = 1 for smooth O(1) fields, bounded by the
  order-2 range clip); and like semi-Lagrangian transport itself it is
  not exactly mass-conserving (the latitude pass ignores Gaussian
  quadrature weights). For flow-adaptive damping applied through the transport
  interpolation itself, see the SLHD scheme of Váňa et al. (2008), noted
  in the module references.

  Args:
    grid: the horizontal grid.
    dt: (nondimensional) time step between filter applications.
    tau: e-folding time of the two-gridpoint mode, in the units of `dt`.
    order: 1 for the monotone 1-2-1 kernel, 2 (default) for the
      scale-selective Shapiro δ⁴ kernel with a local-range clip.

  Returns:
    A function mapping a nodal field of shape [..., longitude_nodes,
    latitude_nodes] to the filtered field of the same shape.
  """
  if order not in (1, 2):
    raise ValueError(f'unsupported {order=}; expected 1 or 2')
  mu = float(1 - np.exp(-dt / tau))
  h = _HALO_WIDTH

  def second_difference_lon(x):
    return (jnp.roll(x, 1, axis=-2) - 2 * x + jnp.roll(x, -1, axis=-2)) / 4

  def second_difference_lat(x):
    n = x.shape[-1]
    extended = _extend_with_pole_halo(x, grid)
    return (
        extended[..., h - 1 : h - 1 + n]
        - 2 * x
        + extended[..., h + 1 : h + 1 + n]
    ) / 4

  def local_range(x):
    n = x.shape[-1]
    extended = _extend_with_pole_halo(x, grid)
    low = high = None
    for lat_slice in (
        slice(h - 1, h - 1 + n),
        slice(h, h + n),
        slice(h + 1, h + 1 + n),
    ):
      band = extended[..., lat_slice]
      for shift in (-1, 0, 1):
        rolled = jnp.roll(band, shift, axis=-2) if shift else band
        low = rolled if low is None else jnp.minimum(low, rolled)
        high = rolled if high is None else jnp.maximum(high, rolled)
    return low, high

  def apply_filter(field: Array) -> jnp.ndarray:
    field = jnp.asarray(field)
    result = field
    for second_difference in (second_difference_lon, second_difference_lat):
      increment = second_difference(result)
      if order == 2:
        increment = -second_difference(increment)
      result = result + mu * increment
    if order == 2:
      low, high = local_range(field)
      result = jnp.clip(result, low, high)
    return result

  return apply_filter
