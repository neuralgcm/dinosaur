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
departure-point solver, and linear/cubic Lagrange interpolation (both
gather-based and TPU/SPMD-friendly MXU matmul-based) with longitude wrapping,
cross-pole halos, and an optional quasi-monotone limiter (Bermejo & Staniforth
1992).

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
  Staniforth, A. & Côté, J., 1991: Semi-Lagrangian integration schemes for
    atmospheric models — a review. Mon. Wea. Rev., 119.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Literal, NamedTuple

from dinosaur import coordinate_systems
from dinosaur import jax_numpy_utils
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tree_math


Array = typing.Array
InterpolationMethod = Literal['gather', 'matmul']
_INTERPOLATION_METHODS = ('gather', 'matmul')
P = jax.sharding.PartitionSpec

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
#  Interpolation on the full Gaussian grid (gather and TPU MXU matmul).
#  =============================================================================


def _round_to_multiple(x: int, multiple: int) -> int:
  return multiple * math.ceil(x / multiple)


def _validate_interpolation_method(method: str | None) -> None:
  if method is not None and method not in _INTERPOLATION_METHODS:
    raise ValueError(
        f'unknown interpolation {method=}; valid values are None (auto) or '
        f'one of {_INTERPOLATION_METHODS}'
    )


def _resolve_interpolation_method(
    method: str | None, grid: spherical_harmonic.Grid
) -> str:
  """Resolves `method=None` lazily at trace time (never at init/import time)."""
  _validate_interpolation_method(method)
  if method is not None:
    return method
  if grid.spmd_mesh is not None or jax.default_backend() == 'tpu':
    return 'matmul'
  return 'gather'


def _lagrange_weights_tuple(
    x: Array, nodes: tuple[Array, ...]
) -> tuple[jnp.ndarray, ...]:
  """Computes Lagrange weights as a tuple, avoiding [..., S, S] tensors."""
  s = len(nodes)
  return tuple(
      math.prod(
          (x - nodes[j]) / (nodes[i] - nodes[j]) for j in range(s) if j != i
      )
      for i in range(s)
  )


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
  node_tuple = tuple(nodes[..., i] for i in range(stencil_size))
  return jnp.stack(_lagrange_weights_tuple(x, node_tuple), axis=-1)


def _contract_stencil(values: Array, *weights: Array) -> jnp.ndarray:
  """Contracts gathered stencil values with per-dimension weights.

  `values` has one trailing axis per weight vector (the tensor-product
  stencil); each `weights[k]` has shape [..., stencil_size_k]. Production
  SL codes evaluate this contraction as a cascade of 1-D interpolations so
  the full stencil tensor never exists in memory; the XLA equivalent is
  fusion. An einsum lowers to dot_general, whose operands must be
  materialized — the gathered stencil tensor is written to memory — which
  is the fastest form on CPU (0.55✕ the fused form) but measurably slower
  on GPU, where a broadcast-multiply + reduce fuses with the producing
  gather and skips the materialization (1.15–1.20✕ isolated on A100 at
  T170/L32, and considerably more in a full model step, where the skipped
  materializations compound: 32.0 → 26.3 ms/step with linear vertical and
  58.4 → 33.3 ms/step with cubic on the 2-day ERA5 benchmark). The
  backend is known at trace time, so pick per platform.
  """
  if jax.default_backend() == 'cpu':
    n = len(weights)
    letters = 'ijkl'[:n]
    operands = ','.join(f'...{c}' for c in letters)
    return einsum(f'{operands},...{letters}->...', *weights, values)
  return _contract_stencil_fused(values, *weights)


def _contract_stencil_fused(values: Array, *weights: Array) -> jnp.ndarray:
  """The gather-fusable form of `_contract_stencil` (any backend)."""
  n = len(weights)
  result = values
  for axis, w in enumerate(weights):
    # place the weight's stencil axis at the matching values axis, with
    # explicit size-1 axes on both sides so batch dimensions stay aligned.
    shape = w.shape[:-1] + (1,) * axis + (w.shape[-1],) + (1,) * (n - 1 - axis)
    result = result * w.reshape(shape)
  return result.sum(axis=tuple(range(-n, 0)))


_HALO_WIDTH = 2  # supports stencils up to 4 points wide in latitude.


def _latitude_halo_axis(grid: spherical_harmonic.Grid) -> np.ndarray:
  """Returns latitudes (radians) extended with `_HALO_WIDTH` rows per pole.

  A point beyond the pole at "latitude" φ > π/2 corresponds to the physical
  location (π - φ, λ + π), so halo rows take coordinates mirrored about the
  pole. The extended axis is strictly increasing, which makes latitude (not
  its sine, which folds at the poles) the coordinate used for interpolation.
  """
  phi = grid.latitudes[: grid.latitude_nodes]
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
  extended this way; (u, v) wind components would need sign flips. Any nodal
  padding beyond `(longitude_nodes, latitude_nodes)` is stripped first.

  Args:
    field: array of shape [..., *grid.nodal_shape].
    grid: the horizontal grid.

  Returns:
    Array of shape [..., longitude_nodes, latitude_nodes + 2 * _HALO_WIDTH].
  """
  num_lon = grid.longitude_nodes
  num_lat = grid.latitude_nodes
  if num_lon % 2:
    raise ValueError(
        'cross-pole halos require an even number of longitude nodes; '
        f'got {num_lon}. Note that `Grid.with_wavenumbers` may produce odd '
        'longitude counts; the `Grid.T*` and `Grid.TL*` constructors do not.'
    )
  phys = field[..., :num_lon, :num_lat]
  rolled = jnp.roll(phys, num_lon // 2, axis=-2)
  south = jnp.flip(rolled[..., :_HALO_WIDTH], axis=-1)
  north = jnp.flip(rolled[..., -_HALO_WIDTH:], axis=-1)
  return jnp.concatenate([south, phys, north], axis=-1)


def _zero_nodal_padding(
    field: Array, grid: spherical_harmonic.Grid
) -> jnp.ndarray:
  """Zeros out padded nodal entries when `grid.nodal_padding != (0, 0)`."""
  field = jnp.asarray(field)
  pad_x, pad_y = grid.nodal_padding
  if (pad_x == 0 and pad_y == 0) or field.shape[-2:] != grid.nodal_shape:
    return field
  nx, ny = grid.nodal_shape
  valid = jnp.ones((nx, ny), dtype=bool)
  if pad_x:
    valid = valid & (jnp.arange(nx)[:, jnp.newaxis] < grid.longitude_nodes)
  if pad_y:
    valid = valid & (jnp.arange(ny)[jnp.newaxis, :] < grid.latitude_nodes)
  return jnp.where(valid, field, 0)


def _stencil_offsets(order: str) -> np.ndarray:
  """Returns stencil offsets relative to the lower bracketing node."""
  if order == 'linear':
    return np.arange(2)
  elif order == 'cubic':
    return np.arange(-1, 3)
  else:
    raise ValueError(f'unknown interpolation {order=}')


def _horizontal_stencil_tuple(
    grid: spherical_harmonic.Grid,
    lon: Array,
    sin_lat: Array,
    order: str,
) -> tuple[
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
    jnp.ndarray,
    jnp.ndarray,
    int,
]:
  """Computes tuple-unrolled horizontal stencil indices, weights, and cells."""
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
  n_lon = grid.longitude_nodes
  lon0 = float(grid.nodal_axes[0][0])
  lon_spacing = 2 * np.pi / n_lon
  lon_cell = jnp.floor((lon - lon0) / lon_spacing).astype(jnp.int32)
  lon_nodes = tuple(
      (lon_cell + int(off)) * lon_spacing + lon0 for off in offsets
  )
  lon_weights = _lagrange_weights_tuple(lon, lon_nodes)
  lon_indices = tuple((lon_cell + int(off)) % n_lon for off in offsets)

  # Latitude nodes are non-uniform; searchsorted locates the bracketing ring
  # on the halo-extended axis and Lagrange weights use actual node latitudes.
  phi_extended = _latitude_halo_axis(grid)
  phi_ext_jax = jnp.asarray(phi_extended, dtype=phi.dtype)
  lat_cell = (
      jnp.searchsorted(phi_ext_jax, phi, side='right') - 1
  ).astype(jnp.int32)
  # Physical points always satisfy |φ| <= π/2, which keeps cubic stencils
  # within the extended axis; clip for numerical safety only.
  lat_cell = jnp.clip(
      lat_cell, -int(offsets[0]), phi_extended.size - 1 - int(offsets[-1])
  )
  lat_indices = tuple(lat_cell + int(off) for off in offsets)
  lat_nodes = tuple(phi_ext_jax[idx] for idx in lat_indices)
  lat_weights = _lagrange_weights_tuple(phi, lat_nodes)

  return (
      lon_indices,
      lon_weights,
      lat_indices,
      lat_weights,
      lon_cell % n_lon,
      lat_cell,
      phi_extended.size,
  )


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
  (
      lon_indices,
      lon_weights,
      lat_indices,
      lat_weights,
      _,
      _,
      ext_size,
  ) = _horizontal_stencil_tuple(grid, lon, sin_lat, order)
  return _HorizontalStencil(
      lon_index=jnp.stack(lon_indices, axis=-1),
      lat_index=jnp.stack(lat_indices, axis=-1),
      lon_weights=jnp.stack(lon_weights, axis=-1),
      lat_weights=jnp.stack(lat_weights, axis=-1),
      extended_latitude_size=ext_size,
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


def _cell_bounds_2d(f_buf: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Precomputes 2✕2 bracketing-cell (min, max) on [..., N_lon, N_lat_buf]."""
  f_next_lon = jnp.roll(f_buf, shift=-1, axis=-2)
  lo = jnp.minimum(f_buf, f_next_lon)
  hi = jnp.maximum(f_buf, f_next_lon)
  lo = jnp.minimum(lo[..., :-1], lo[..., 1:])
  hi = jnp.maximum(hi[..., :-1], hi[..., 1:])
  return lo, hi


def _cell_bounds_3d(f_buf: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Precomputes 2✕2✕2 bracketing-cell (min, max) on [C, L_buf, N_lon, N_lat_buf]."""
  lo, hi = _cell_bounds_2d(f_buf)
  lo = jnp.minimum(lo[:, :-1, :, :], lo[:, 1:, :, :])
  hi = jnp.maximum(hi[:, :-1, :, :], hi[:, 1:, :, :])
  return lo, hi


def _gather_lat_with_halo(
    f_x: jax.Array,
    grid: spherical_harmonic.Grid,
    max_lat_shift: int,
) -> tuple[jax.Array, int | jax.Array, int | jax.Array]:
  """Builds local latitude buffer [..., n_lon, n_lat_buf] with pole halos."""
  n_lon = grid.longitude_nodes
  n_lat = grid.latitude_nodes
  if n_lon % 2:
    raise ValueError(
        'cross-pole halos require an even number of longitude nodes; '
        f'got {n_lon}. Note that `Grid.with_wavenumbers` may produce odd '
        'longitude counts; the `Grid.T*` and `Grid.TL*` constructors do not.'
    )
  half_turn = n_lon // 2
  mesh = grid.spmd_mesh
  y_shards = mesh.shape['y'] if mesh is not None else 1
  if y_shards == 1:
    phys = f_x[..., :n_lat]
    rolled = jnp.roll(phys, shift=half_turn, axis=-2)
    south = jnp.flip(rolled[..., :_HALO_WIDTH], axis=-1)
    north = jnp.flip(rolled[..., -_HALO_WIDTH:], axis=-1)
    return jnp.concatenate([south, phys, north], axis=-1), 0, 0

  iy = lax.axis_index('y')
  ny_loc = f_x.shape[-1]
  pad_y = grid.nodal_padding[1]
  h_lat = max_lat_shift + _HALO_WIDTH
  if pad_y == 0 and ny_loc >= h_lat:
    perm_to_north = [(s, (s + 1) % y_shards) for s in range(y_shards)]
    perm_to_south = [(s, (s - 1) % y_shards) for s in range(y_shards)]
    recv_left = lax.ppermute(f_x[..., -h_lat:], 'y', perm=perm_to_north)
    recv_right = lax.ppermute(f_x[..., :h_lat], 'y', perm=perm_to_south)

    south_pole = jnp.roll(
        jnp.flip(f_x[..., :_HALO_WIDTH], axis=-1), shift=half_turn, axis=-2
    )
    north_pole = jnp.roll(
        jnp.flip(f_x[..., -_HALO_WIDTH:], axis=-1), shift=half_turn, axis=-2
    )
    zeros_pad = jnp.zeros(
        f_x.shape[:-1] + (h_lat - _HALO_WIDTH,), dtype=f_x.dtype
    )
    pole_left = jnp.concatenate([zeros_pad, south_pole], axis=-1)
    pole_right = jnp.concatenate([north_pole, zeros_pad], axis=-1)

    left_halo = jnp.where(iy == 0, pole_left, recv_left)
    right_halo = jnp.where(iy == y_shards - 1, pole_right, recv_right)
    f_xy_buf = jnp.concatenate([left_halo, f_x, right_halo], axis=-1)
    j_offset = iy * ny_loc
    buf_lat_origin = _HALO_WIDTH + iy * ny_loc - h_lat
    return f_xy_buf, j_offset, buf_lat_origin

  f_all_y = lax.all_gather(f_x, 'y', axis=-1, tiled=True)
  phys = f_all_y[..., :n_lat]
  rolled = jnp.roll(phys, shift=half_turn, axis=-2)
  south = jnp.flip(rolled[..., :_HALO_WIDTH], axis=-1)
  north = jnp.flip(rolled[..., -_HALO_WIDTH:], axis=-1)
  return jnp.concatenate([south, phys, north], axis=-1), iy * ny_loc, 0


def _gather_2d_flat_index(stencil: _HorizontalStencil) -> jax.Array:
  return (
      stencil.lon_index[..., :, jnp.newaxis] * stencil.extended_latitude_size
      + stencil.lat_index[..., jnp.newaxis, :]
  )


def _interpolate_horizontal_from_flat_index(
    field_extended: Array,
    flat_index: Array,
    stencil: _HorizontalStencil,
    order: str,
    limiter: str | None,
) -> jnp.ndarray:
  """Interpolates one [lon, lat_ext] field given precomputed flat indices."""
  values = jnp.take(field_extended.ravel(), flat_index, mode='clip')
  result = _contract_stencil(
      values, stencil.lon_weights, stencil.lat_weights
  )
  if limiter is not None:
    if limiter != 'quasi_monotone':
      raise ValueError(f'unimplemented interpolation {limiter=}')
    cell = _linear_cell_slice(order)
    corners = values[..., cell, cell]
    result = jnp.clip(
        result,
        corners.min(axis=(-2, -1)),
        corners.max(axis=(-2, -1)),
    )
  return result


def _interpolate_horizontal(
    field: Array,
    grid: spherical_harmonic.Grid,
    stencil: _HorizontalStencil,
    order: str,
    limiter: str | None,
) -> jnp.ndarray:
  """Interpolates a single [lon, lat] field with a precomputed stencil."""
  extended = _extend_with_pole_halo(field, grid)
  flat_index = _gather_2d_flat_index(stencil)
  return _interpolate_horizontal_from_flat_index(
      extended, flat_index, stencil, order, limiter
  )


def _matmul_2d_points(
    fields: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    *,
    order: str,
    limiter: str | None,
) -> jnp.ndarray:
  """MXU matmul 2-D interpolation of [C, nx, ny] at arbitrary [*points]."""
  f_ext = _extend_with_pole_halo(fields, grid)  # [C, n_lon, n_lat_ext]
  dtype = f_ext.dtype
  c_total, n_lon, n_lat_ext = f_ext.shape
  point_shape = lon.shape
  q_total = lon.size
  if q_total == 0 or c_total == 0:
    return jnp.zeros((c_total, *point_shape), dtype=dtype)

  lon_flat = jnp.asarray(lon).reshape(-1)
  sin_lat_flat = jnp.asarray(sin_lat).reshape(-1)
  q_chunk = min(q_total, 1024)
  q_padded = _round_to_multiple(q_total, q_chunk)
  if q_padded > q_total:
    lon_flat = jnp.pad(lon_flat, [(0, q_padded - q_total)], mode='edge')
    sin_lat_flat = jnp.pad(sin_lat_flat, [(0, q_padded - q_total)], mode='edge')

  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, f_ext, f_ext)
  dot_prec_bf16 = jax_numpy_utils.resolve_dot_precision(
      None, f_ext, f_ext, lhs_exact_in_bf16=True
  )
  f_2d = f_ext.reshape(c_total * n_lon, n_lat_ext)
  bounds_2d = None
  if limiter is not None:
    lo_2d, hi_2d = _cell_bounds_2d(f_ext)
    bounds_2d = jnp.concatenate([lo_2d, hi_2d], axis=0).reshape(
        2 * c_total * n_lon, n_lat_ext - 1
    )

  def _eval_q_chunk(lon_q, sin_lat_q):
    (
        lon_indices,
        lon_weights,
        lat_indices,
        lat_weights,
        lon_cell0,
        lat_cell0,
        _,
    ) = _horizontal_stencil_tuple(grid, lon_q, sin_lat_q, order)
    w_lat_mat = sum(
        w[:, jnp.newaxis] * jax.nn.one_hot(idx, n_lat_ext, dtype=dtype)
        for idx, w in zip(lat_indices, lat_weights)
    )  # [q, n_lat_ext]
    f_at_lat = jnp.dot(
        w_lat_mat, f_2d.T, precision=dot_prec_f32
    ).reshape(lon_q.shape[0], c_total, n_lon)
    w_lon_mat = sum(
        w[:, jnp.newaxis] * jax.nn.one_hot(idx, n_lon, dtype=dtype)
        for idx, w in zip(lon_indices, lon_weights)
    )  # [q, n_lon]
    val = (f_at_lat * w_lon_mat[:, jnp.newaxis, :]).sum(axis=-1).T  # [C, q]

    if limiter is not None:
      p_lat_mat = jax.nn.one_hot(lat_cell0, n_lat_ext - 1, dtype=dtype)
      b_at_lat = jnp.dot(
          p_lat_mat, bounds_2d.T, precision=dot_prec_bf16
      ).reshape(lon_q.shape[0], 2 * c_total, n_lon)
      p_lon_mat = jax.nn.one_hot(lon_cell0, n_lon, dtype=dtype)
      b_val = (b_at_lat * p_lon_mat[:, jnp.newaxis, :]).sum(axis=-1).T
      val = jnp.clip(val, b_val[:c_total], b_val[c_total:])
    return val

  n_chunks = q_padded // q_chunk
  if n_chunks == 1:
    out_flat = _eval_q_chunk(lon_flat, sin_lat_flat)
  else:
    _, out_chunks = lax.scan(
        lambda _, qs: (None, _eval_q_chunk(qs[0], qs[1])),
        None,
        (lon_flat.reshape(n_chunks, q_chunk), sin_lat_flat.reshape(n_chunks, q_chunk)),
    )
    out_flat = out_chunks.transpose(1, 0, 2).reshape(c_total, q_padded)
  return out_flat[:, :q_total].reshape((c_total, *point_shape))


def _matmul_2d_local(
    f_buf: jax.Array,
    lon_q: jax.Array,
    sin_lat_q: jax.Array,
    *,
    grid: spherical_harmonic.Grid,
    order: str,
    limiter: str | None,
    j_offset: int | jax.Array,
    buf_lat_origin: int | jax.Array,
    max_lat_shift: int,
) -> jax.Array:
  """Windowed MXU matmul 2-D interpolation on [C, n_lon, n_lat_buf]."""
  dtype = f_buf.dtype
  c_cur, n_lon, n_lat_buf = f_buf.shape
  n_lat = grid.latitude_nodes
  n_lat_ext = n_lat + 2 * _HALO_WIDTH
  nx_loc, ny_loc = lon_q.shape
  w_lat_size = min(n_lat_ext, 2 * max_lat_shift + 4)

  (
      lon_indices,
      lon_weights,
      lat_indices,
      lat_weights,
      lon_cell0,
      lat_cell0,
      _,
  ) = _horizontal_stencil_tuple(grid, lon_q, sin_lat_q, order)

  j_global = jnp.minimum(
      j_offset + jnp.arange(ny_loc, dtype=jnp.int32), n_lat - 1
  )
  if w_lat_size == n_lat_ext:
    win_lat_start = jnp.zeros_like(j_global)
  else:
    win_lat_start = jnp.clip(
        j_global + 1 - max_lat_shift, 0, n_lat_ext - w_lat_size
    )
  buf_lat_start = win_lat_start - buf_lat_origin
  m_lat_win = jax.nn.one_hot(
      buf_lat_start[:, jnp.newaxis]
      + jnp.arange(w_lat_size, dtype=jnp.int32)[jnp.newaxis, :],
      n_lat_buf,
      dtype=dtype,
  )  # [ny_loc, w_lat_size, n_lat_buf]

  dot_prec_bf16 = jax_numpy_utils.resolve_dot_precision(
      None, m_lat_win, f_buf, lhs_exact_in_bf16=True
  )
  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, f_buf, f_buf)

  f_2d = f_buf.transpose(1, 0, 2).reshape(n_lon * c_cur, n_lat_buf)
  f_lat_win = (
      jnp.dot(f_2d, m_lat_win.reshape(ny_loc * w_lat_size, n_lat_buf).T, precision=dot_prec_bf16)
      .reshape(n_lon, c_cur, ny_loc, w_lat_size)
      .transpose(2, 0, 3, 1)
      .reshape(ny_loc, n_lon, w_lat_size * c_cur)
  )

  w_lon_mat = sum(
      w.T[:, :, jnp.newaxis] * jax.nn.one_hot(idx.T, n_lon, dtype=dtype)
      for idx, w in zip(lon_indices, lon_weights)
  )  # [ny_loc, nx_loc, n_lon]
  gathered_lon = jnp.einsum(
      'jim,jmk->jik', w_lon_mat, f_lat_win, precision=dot_prec_f32
  )  # [ny_loc, nx_loc, w_lat_size * c_cur]

  w_lat_mat = sum(
      w.T[:, :, jnp.newaxis]
      * jax.nn.one_hot(
          idx.T - win_lat_start[:, jnp.newaxis], w_lat_size, dtype=dtype
      )
      for idx, w in zip(lat_indices, lat_weights)
  )  # [ny_loc, nx_loc, w_lat_size]

  val = (
      gathered_lon.reshape(ny_loc, nx_loc, w_lat_size, c_cur)
      * w_lat_mat[:, :, :, jnp.newaxis]
  ).sum(axis=2).transpose(2, 1, 0)

  if limiter is not None:
    lo_buf, hi_buf = _cell_bounds_2d(f_buf)
    bounds_buf = jnp.concatenate([lo_buf, hi_buf], axis=0)
    w_lat_cell_size = min(n_lat_ext - 1, max(1, w_lat_size - 1))
    if w_lat_cell_size == n_lat_ext - 1:
      win_lat_cell_start = jnp.zeros_like(j_global)
    else:
      win_lat_cell_start = jnp.clip(
          win_lat_start, 0, (n_lat_ext - 1) - w_lat_cell_size
      )
    buf_lat_cell_start = win_lat_cell_start - buf_lat_origin
    m_lat_cell_win = jax.nn.one_hot(
        buf_lat_cell_start[:, jnp.newaxis]
        + jnp.arange(w_lat_cell_size, dtype=jnp.int32)[jnp.newaxis, :],
        n_lat_buf - 1,
        dtype=dtype,
    )
    b_2d = bounds_buf.transpose(1, 0, 2).reshape(
        n_lon * 2 * c_cur, n_lat_buf - 1
    )
    b_lat_win = (
        jnp.dot(
            b_2d,
            m_lat_cell_win.reshape(ny_loc * w_lat_cell_size, n_lat_buf - 1).T,
            precision=dot_prec_bf16,
        )
        .reshape(n_lon, 2 * c_cur, ny_loc, w_lat_cell_size)
        .transpose(2, 0, 3, 1)
        .reshape(ny_loc, n_lon, w_lat_cell_size * 2 * c_cur)
    )
    p_lon_mat = jax.nn.one_hot(lon_cell0.T, n_lon, dtype=dtype)
    b_lon = jnp.einsum(
        'jim,jmk->jik', p_lon_mat, b_lat_win, precision=dot_prec_bf16
    )
    p_lat_mat = jax.nn.one_hot(
        lat_cell0.T - win_lat_cell_start[:, jnp.newaxis],
        w_lat_cell_size,
        dtype=dtype,
    )
    b_val = (
        b_lon.reshape(ny_loc, nx_loc, w_lat_cell_size, 2 * c_cur)
        * p_lat_mat[:, :, :, jnp.newaxis]
    ).sum(axis=2).transpose(2, 1, 0)
    val = jnp.clip(val, b_val[:c_cur], b_val[c_cur:])
  return val


def _interpolate_2d_grid_matmul(
    fields: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    *,
    order: str,
    limiter: str | None,
    max_lat_shift: int,
) -> jax.Array:
  """Grid-aligned 2-D matmul interpolation (single-device and SPMD)."""
  mesh = grid.spmd_mesh
  if mesh is None:
    f_ext = _extend_with_pole_halo(fields, grid)
    return _matmul_2d_local(
        f_ext,
        lon,
        sin_lat,
        grid=grid,
        order=order,
        limiter=limiter,
        j_offset=0,
        buf_lat_origin=0,
        max_lat_shift=max_lat_shift,
    )

  x_shards = mesh.shape['x']
  f_spec = P(None, 'x', 'y')
  q_spec = P('x', 'y')

  def _sharded(f_loc, lon_loc, sin_lat_loc):
    if x_shards > 1:
      f_x = lax.all_gather(f_loc, 'x', axis=1, tiled=True)
    else:
      f_x = f_loc
    f_x = f_x[:, : grid.longitude_nodes, :]
    f_xy_buf, j_offset, buf_lat_origin = _gather_lat_with_halo(
        f_x, grid, max_lat_shift
    )
    return _matmul_2d_local(
        f_xy_buf,
        lon_loc,
        sin_lat_loc,
        grid=grid,
        order=order,
        limiter=limiter,
        j_offset=j_offset,
        buf_lat_origin=buf_lat_origin,
        max_lat_shift=max_lat_shift,
    )

  return jax.shard_map(
      _sharded,
      mesh=mesh,
      in_specs=(f_spec, q_spec, q_spec),
      out_specs=f_spec,
  )(fields, lon, sin_lat)


def _interpolate_2d_multi(
    fields: Array,
    grid: spherical_harmonic.Grid,
    lon: Array,
    sin_lat: Array,
    *,
    order: str = 'cubic',
    limiter: str | None = None,
    method: InterpolationMethod | None = None,
    max_lat_shift: int | None = 6,
) -> jnp.ndarray:
  """Interpolates [C, nx, ny] or [C, levels, nx, ny] at shared (lon, sin_lat)."""
  _validate_limiter(limiter)
  resolved = _resolve_interpolation_method(method, grid)
  fields = jnp.asarray(fields)
  lon = jnp.asarray(lon)
  sin_lat = jnp.asarray(sin_lat)
  if fields.shape[-2:] != grid.nodal_shape:
    raise ValueError(
        f'expected field with trailing shape {grid.nodal_shape}; got'
        f' {fields.shape}'
    )

  if fields.ndim == 3:
    # [C, nx, ny] with lon [*points]
    if resolved == 'gather':
      stencil = _horizontal_stencil(grid, lon, sin_lat, order)
      flat_index = _gather_2d_flat_index(stencil)
      extended = _extend_with_pole_halo(fields, grid)
      return jax.vmap(
          lambda ext: _interpolate_horizontal_from_flat_index(
              ext, flat_index, stencil, order, limiter
          )
      )(extended)
    if lon.shape == grid.nodal_shape and max_lat_shift is not None:
      return _interpolate_2d_grid_matmul(
          fields,
          grid,
          lon,
          sin_lat,
          order=order,
          limiter=limiter,
          max_lat_shift=max_lat_shift,
      )
    return _matmul_2d_points(
        fields, grid, lon, sin_lat, order=order, limiter=limiter
    )

  if fields.ndim == 4:
    # [C, levels, nx, ny] with lon [levels, *points]
    levels = fields.shape[1]
    if lon.shape[0] != levels:
      raise ValueError(
          'batched interpolation requires points with a leading levels axis '
          f'matching the field: {fields.shape=}, {lon.shape=}'
      )
    if levels == 1:
      out = _interpolate_2d_multi(
          fields[:, 0],
          grid,
          lon[0],
          sin_lat[0],
          order=order,
          limiter=limiter,
          method=resolved,
          max_lat_shift=max_lat_shift,
      )
      return out[:, jnp.newaxis, ...]

    if (
        resolved == 'matmul'
        and lon.shape == (levels, *grid.nodal_shape)
        and max_lat_shift is not None
        and grid.spmd_mesh is not None
    ):
      mesh = grid.spmd_mesh
      z_shards = mesh.shape['z'] if 'z' in mesh.axis_names else 1
      x_shards = mesh.shape['x']
      l_pad = _round_to_multiple(levels, z_shards)
      if l_pad > levels:
        fields_pad = jnp.pad(
            fields, [(0, 0), (0, l_pad - levels), (0, 0), (0, 0)], mode='edge'
        )
        lon_pad = jnp.pad(
            lon, [(0, l_pad - levels), (0, 0), (0, 0)], mode='edge'
        )
        sin_lat_pad = jnp.pad(
            sin_lat, [(0, l_pad - levels), (0, 0), (0, 0)], mode='edge'
        )
      else:
        fields_pad, lon_pad, sin_lat_pad = fields, lon, sin_lat
      z_spec = 'z' if z_shards > 1 else None
      f_spec = P(None, z_spec, 'x', 'y')
      q_spec = P(z_spec, 'x', 'y')

      def _sharded_levels(f_loc, lon_loc, sin_lat_loc):
        if x_shards > 1:
          f_x = lax.all_gather(f_loc, 'x', axis=2, tiled=True)
        else:
          f_x = f_loc
        f_x = f_x[:, :, : grid.longitude_nodes, :]
        f_xy_buf, j_offset, buf_lat_origin = _gather_lat_with_halo(
            f_x, grid, max_lat_shift
        )
        # Transpose level to axis 0 and scan across local levels
        f_by_l = jnp.swapaxes(f_xy_buf, 0, 1)
        _, out_by_l = lax.scan(
            lambda _, inp: (
                None,
                _matmul_2d_local(
                    inp[0],
                    inp[1],
                    inp[2],
                    grid=grid,
                    order=order,
                    limiter=limiter,
                    j_offset=j_offset,
                    buf_lat_origin=buf_lat_origin,
                    max_lat_shift=max_lat_shift,
                ),
            ),
            None,
            (f_by_l, lon_loc, sin_lat_loc),
        )
        return jnp.swapaxes(out_by_l, 0, 1)

      out_pad = jax.shard_map(
          _sharded_levels,
          mesh=mesh,
          in_specs=(f_spec, q_spec, q_spec),
          out_specs=f_spec,
      )(fields_pad, lon_pad, sin_lat_pad)
      return out_pad[:, :levels, :, :]

    return jax.vmap(
        lambda f_l, lon_l, sl_l: _interpolate_2d_multi(
            f_l,
            grid,
            lon_l,
            sl_l,
            order=order,
            limiter=limiter,
            method=resolved,
            max_lat_shift=max_lat_shift,
        ),
        in_axes=(1, 0, 0),
        out_axes=1,
    )(fields, lon, sin_lat)

  raise ValueError(f'expected 3-D or 4-D fields; got {fields.shape=}')


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
    method: 'gather' (1-D gather + fused contraction, fastest on single GPU),
      'matmul' (MXU matrix multiplication without unstructured gather/scatter,
      designed for TPU and SPMD sharding), or None (auto-selects 'matmul' on
      TPU or when `grid.spmd_mesh` is set, and 'gather' otherwise).
    max_lat_shift: maximum meridional displacement in latitude grid cells
      around arrival nodes for windowed 'matmul' interpolation on grid-aligned
      queries (None disables latitude windowing).
  """

  grid: spherical_harmonic.Grid
  order: str = 'cubic'
  limiter: str | None = None
  method: InterpolationMethod | None = dataclasses.field(
      default=None, kw_only=True
  )
  max_lat_shift: int | None = dataclasses.field(default=6, kw_only=True)

  def __post_init__(self):
    _validate_limiter(self.limiter)
    _validate_interpolation_method(self.method)

  def __call__(self, field: Array, lon: Array, sin_lat: Array) -> jnp.ndarray:
    """Interpolates `field` at the points (lon, sin_lat).

    Args:
      field: nodal values of shape [longitude_nodes, latitude_nodes] (points
        may then have any shape), or [levels, longitude_nodes,
        latitude_nodes], in which case each level is interpolated at its own
        set of points.
      lon: longitudes in radians, of shape [*points] or [levels, *points].
      sin_lat: sine of latitudes, same shape as `lon`.

    Returns:
      Interpolated values of shape `lon.shape`.
    """
    field = jnp.asarray(field)
    if field.ndim not in (2, 3):
      raise ValueError(
          f'expected a field with 2 or 3 dimensions; got {field.shape=}'
      )
    if field.ndim == 3 and lon.shape[0] != field.shape[0]:
      raise ValueError(
          'batched interpolation requires points with a leading levels axis '
          f'matching the field: {field.shape=}, {lon.shape=}'
      )
    return _interpolate_horizontal_dispatched(
        field,
        lon,
        sin_lat,
        grid=self.grid,
        order=self.order,
        limiter=self.limiter,
        method=self.method,
        max_lat_shift=self.max_lat_shift,
    )


@functools.partial(
    jax.jit,
    static_argnames=('grid', 'order', 'limiter', 'method', 'max_lat_shift'),
)
def _interpolate_horizontal_dispatched(
    field: Array,
    lon: Array,
    sin_lat: Array,
    *,
    grid: spherical_harmonic.Grid,
    order: str,
    limiter: str | None,
    method: InterpolationMethod | None = None,
    max_lat_shift: int | None = 6,
) -> jnp.ndarray:
  return _interpolate_2d_multi(
      field[jnp.newaxis, ...],
      grid,
      lon,
      sin_lat,
      order=order,
      limiter=limiter,
      method=method,
      max_lat_shift=max_lat_shift,
  )[0]


def _vertical_stencil_tuple(
    sigma: Array,
    sigma_nodes: Array,
    vertical_order: Literal['linear', 'cubic'],
) -> tuple[
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
    jnp.ndarray,
    jnp.ndarray,
]:
  """Returns tuple-unrolled (level_indices, level_weights, bracket_offset, level_cell)."""
  sigma = jnp.asarray(sigma)
  sigma_nodes = jnp.asarray(sigma_nodes, dtype=sigma.dtype)
  num_levels = sigma_nodes.shape[0]
  sigma = jnp.clip(sigma, sigma_nodes[0], sigma_nodes[-1])
  level_cell = (
      jnp.searchsorted(sigma_nodes, sigma, side='right') - 1
  ).astype(jnp.int32)
  level_cell = jnp.clip(level_cell, 0, num_levels - 2)
  below = sigma_nodes[level_cell]
  above = sigma_nodes[level_cell + 1]
  fraction = (sigma - below) / (above - below)
  if vertical_order == 'linear':
    return (
        (level_cell, level_cell + 1),
        (1 - fraction, fraction),
        jnp.zeros_like(level_cell),
        level_cell,
    )
  if vertical_order != 'cubic':
    raise ValueError(f'unknown {vertical_order=}')
  start = jnp.clip(level_cell - 1, 0, num_levels - 4)
  level_indices = tuple(start + a for a in range(4))
  level_nodes = tuple(sigma_nodes[idx] for idx in level_indices)
  cubic_weights = _lagrange_weights_tuple(sigma, level_nodes)
  bracket_offset = level_cell - start
  boundary_cell = (level_cell == 0) | (level_cell == num_levels - 2)
  level_weights = tuple(
      jnp.where(
          boundary_cell,
          jnp.where(bracket_offset == a, 1 - fraction, 0.0)
          + jnp.where(bracket_offset + 1 == a, fraction, 0.0),
          cubic_weights[a],
      )
      for a in range(4)
  )
  return level_indices, level_weights, bracket_offset, level_cell


def _vertical_stencil(
    sigma: Array,
    sigma_nodes: Array,
    vertical_order: Literal['linear', 'cubic'],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Returns (level_index, level_weights, bracket_offset) for the vertical.

  σ is clipped to the node range first, so boundary weights saturate and
  interpolation never extrapolates (constant extrapolation beyond the first
  and last nodes). 'linear' uses the two bracketing nodes. 'cubic' uses a
  4-point Lagrange stencil on the (possibly non-uniform) nodes, degraded to
  linear within the first and last cells — the standard operational rule
  (e.g. IFS), which also keeps the no-extrapolation property exact.
  `bracket_offset` is the position of the lower bracketing node within the
  stencil (needed by the quasi-monotone limiter, which clips against the
  bracketing cell only).
  """
  level_indices, level_weights, bracket_offset, _ = _vertical_stencil_tuple(
      sigma, sigma_nodes, vertical_order
  )
  return (
      jnp.stack(level_indices, axis=-1),
      jnp.stack(level_weights, axis=-1),
      bracket_offset,
  )


def _bracketing_level_corners(
    corners: Array, bracket_offset: Array
) -> jnp.ndarray:
  """Restricts corner values to the two levels bracketing the departure σ.

  The quasi-monotone limiter must clip against the bracketing cell only:
  including outer stencil levels would admit values the departure point
  does not lie between. For the linear vertical stencil the offset is zero
  and this selection is the identity.
  """
  index = (
      bracket_offset[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
      + np.arange(2)[:, np.newaxis, np.newaxis]
  )
  return jnp.take_along_axis(corners, index, axis=-3)


def _interpolate_3d_gather(
    fields_4d: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    sigma: jax.Array,
    sigma_nodes: np.ndarray,
    *,
    order: str,
    limiter: str | None,
    vertical_order: Literal['linear', 'cubic'],
    limiter_bound_reference: Array | None,
) -> jnp.ndarray:
  """Gather-based 3-D interpolation of [C, L_src, nx, ny] sharing one stencil."""
  stencil = _horizontal_stencil(grid, lon, sin_lat, order)
  level_index, level_weights, bracket_offset = _vertical_stencil(
      sigma, jnp.asarray(sigma_nodes), vertical_order
  )
  extended = _extend_with_pole_halo(fields_4d, grid)
  extended_latitude_size = extended.shape[-1]
  # One 1-D gather per channel on the flattened field with precomputed linear
  # indices shared across all channels.
  flat_index = (
      level_index[..., :, jnp.newaxis, jnp.newaxis] * grid.longitude_nodes
      + stencil.lon_index[..., jnp.newaxis, :, jnp.newaxis]
  ) * extended_latitude_size + stencil.lat_index[
      ..., jnp.newaxis, jnp.newaxis, :
  ]

  reference_corners = None
  reference_at_departure = 0.0
  if limiter is not None and limiter_bound_reference is not None:
    ref_arr = jnp.asarray(limiter_bound_reference, dtype=extended.dtype)
    ref_centered = ref_arr - jnp.mean(ref_arr)
    reference_corners = ref_centered[level_index]
    reference_at_departure = (level_weights * reference_corners).sum(-1)

  def _eval_one(ext_one):
    values = jnp.take(ext_one.ravel(), flat_index, mode='clip')
    result = _contract_stencil(
        values, level_weights, stencil.lon_weights, stencil.lat_weights
    )
    if limiter is not None:
      if limiter != 'quasi_monotone':
        raise ValueError(f'unimplemented interpolation {limiter=}')
      cell = _linear_cell_slice(order)
      corners = values[..., :, cell, cell]
      if reference_corners is not None:
        corners = corners + reference_corners[..., :, jnp.newaxis, jnp.newaxis]
        result = result + reference_at_departure
      corners = _bracketing_level_corners(corners, bracket_offset)
      result = jnp.clip(
          result,
          corners.min(axis=(-3, -2, -1)),
          corners.max(axis=(-3, -2, -1)),
      )
      result = result - reference_at_departure
    return result

  c_total = extended.shape[0]
  if c_total == 1:
    return _eval_one(extended[0])[jnp.newaxis, ...]
  if c_total <= 4:
    return jax.vmap(_eval_one)(extended)
  chunk = 4
  c_pad = _round_to_multiple(c_total, chunk)
  ext_pad = (
      jnp.pad(extended, [(0, c_pad - c_total), (0, 0), (0, 0), (0, 0)])
      if c_pad > c_total
      else extended
  )
  ext_grouped = ext_pad.reshape((c_pad // chunk, chunk) + extended.shape[1:])
  _, out_grouped = lax.scan(
      lambda _, grp: (None, jax.vmap(_eval_one)(grp)), None, ext_grouped
  )
  return out_grouped.reshape((c_pad,) + lon.shape)[:c_total]


def _matmul_3d_points(
    fields_4d: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    sigma: jax.Array,
    sigma_nodes: np.ndarray,
    *,
    order: str,
    limiter: str | None,
    vertical_order: Literal['linear', 'cubic'],
    limiter_bound_reference: Array | None,
) -> jnp.ndarray:
  """Unwindowed MXU matmul 3-D interpolation for arbitrary query shapes."""
  f_ext = _extend_with_pole_halo(fields_4d, grid)  # [C, L_src, n_lon, n_lat_ext]
  dtype = f_ext.dtype
  c_total, l_src, n_lon, n_lat_ext = f_ext.shape
  point_shape = lon.shape
  q_total = lon.size
  if q_total == 0 or c_total == 0:
    return jnp.zeros((c_total, *point_shape), dtype=dtype)

  lon_flat = jnp.asarray(lon).reshape(-1)
  sin_lat_flat = jnp.asarray(sin_lat).reshape(-1)
  sigma_flat = jnp.asarray(sigma).reshape(-1)
  q_chunk = min(q_total, 1024)
  q_padded = _round_to_multiple(q_total, q_chunk)
  if q_padded > q_total:
    pad_w = [(0, q_padded - q_total)]
    lon_flat = jnp.pad(lon_flat, pad_w, mode='edge')
    sin_lat_flat = jnp.pad(sin_lat_flat, pad_w, mode='edge')
    sigma_flat = jnp.pad(sigma_flat, pad_w, mode='edge')

  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, f_ext, f_ext)
  dot_prec_bf16 = jax_numpy_utils.resolve_dot_precision(
      None, f_ext, f_ext, lhs_exact_in_bf16=True
  )

  ref_centered = None
  if limiter is not None and limiter_bound_reference is not None:
    ref_arr = jnp.asarray(limiter_bound_reference, dtype=dtype)
    ref_centered = ref_arr - jnp.mean(ref_arr)

  bounds_by_l = None
  if limiter is not None:
    f_for_bounds = (
        f_ext + ref_centered[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
        if ref_centered is not None
        else f_ext
    )
    lo_b, hi_b = _cell_bounds_3d(f_for_bounds)
    bounds_by_l = jnp.swapaxes(
        jnp.concatenate([lo_b, hi_b], axis=0), 0, 1
    ).reshape(l_src - 1, 2 * c_total * n_lon, n_lat_ext - 1)

  f_by_l = jnp.swapaxes(f_ext, 0, 1).reshape(l_src, c_total * n_lon, n_lat_ext)

  def _eval_q_chunk(lon_q, sin_lat_q, sigma_q):
    q_len = lon_q.shape[0]
    (
        lon_indices,
        lon_weights,
        lat_indices,
        lat_weights,
        lon_cell0,
        lat_cell0,
        _,
    ) = _horizontal_stencil_tuple(grid, lon_q, sin_lat_q, order)
    level_indices, level_weights, _, level_cell0 = _vertical_stencil_tuple(
        sigma_q, sigma_nodes, vertical_order
    )

    w_lat_mat = sum(
        w[:, jnp.newaxis] * jax.nn.one_hot(idx, n_lat_ext, dtype=dtype)
        for idx, w in zip(lat_indices, lat_weights)
    )  # [q, n_lat_ext]
    w_lon_mat = sum(
        w[:, jnp.newaxis] * jax.nn.one_hot(idx, n_lon, dtype=dtype)
        for idx, w in zip(lon_indices, lon_weights)
    )  # [q, n_lon]
    w_z_mat = sum(
        w[:, jnp.newaxis] * jax.nn.one_hot(idx, l_src, dtype=dtype)
        for idx, w in zip(level_indices, level_weights)
    )  # [q, l_src]

    def _accum_level(acc, inp):
      f_l, w_z_l = inp
      f_at_lat = jnp.dot(w_lat_mat, f_l.T, precision=dot_prec_f32).reshape(
          q_len, c_total, n_lon
      )
      val_l = (f_at_lat * w_lon_mat[:, jnp.newaxis, :]).sum(axis=-1)
      return acc + val_l * w_z_l[:, jnp.newaxis], None

    val_qc, _ = lax.scan(
        _accum_level,
        jnp.zeros((q_len, c_total), dtype=dtype),
        (f_by_l, w_z_mat.T),
    )
    val = val_qc.T  # [C, q]

    if limiter is not None:
      p_lat_mat = jax.nn.one_hot(lat_cell0, n_lat_ext - 1, dtype=dtype)
      p_lon_mat = jax.nn.one_hot(lon_cell0, n_lon, dtype=dtype)
      p_z_mat = jax.nn.one_hot(level_cell0, l_src - 1, dtype=dtype)

      def _accum_bounds(acc, inp):
        b_l, p_z_l = inp
        b_at_lat = jnp.dot(p_lat_mat, b_l.T, precision=dot_prec_bf16).reshape(
            q_len, 2 * c_total, n_lon
        )
        b_val_l = (b_at_lat * p_lon_mat[:, jnp.newaxis, :]).sum(axis=-1)
        return acc + b_val_l * p_z_l[:, jnp.newaxis], None

      b_qc, _ = lax.scan(
          _accum_bounds,
          jnp.zeros((q_len, 2 * c_total), dtype=dtype),
          (bounds_by_l, p_z_mat.T),
      )
      b_val = b_qc.T
      lo_q, hi_q = b_val[:c_total], b_val[c_total:]
      if ref_centered is not None:
        ref_q = sum(
            w * ref_centered[idx]
            for idx, w in zip(level_indices, level_weights)
        )
        val = (
            jnp.clip(val + ref_q[jnp.newaxis, :], lo_q, hi_q)
            - ref_q[jnp.newaxis, :]
        )
      else:
        val = jnp.clip(val, lo_q, hi_q)
    return val

  n_chunks = q_padded // q_chunk
  if n_chunks == 1:
    out_flat = _eval_q_chunk(lon_flat, sin_lat_flat, sigma_flat)
  else:
    _, out_chunks = lax.scan(
        lambda _, qs: (None, _eval_q_chunk(qs[0], qs[1], qs[2])),
        None,
        (
            lon_flat.reshape(n_chunks, q_chunk),
            sin_lat_flat.reshape(n_chunks, q_chunk),
            sigma_flat.reshape(n_chunks, q_chunk),
        ),
    )
    out_flat = out_chunks.transpose(1, 0, 2).reshape(c_total, q_padded)
  return out_flat[:, :q_total].reshape((c_total, *point_shape))


def _matmul_3d_local(
    f_buf: jax.Array,
    lon_q: jax.Array,
    sin_lat_q: jax.Array,
    sigma_q: jax.Array,
    *,
    grid: spherical_harmonic.Grid,
    sigma_nodes: np.ndarray,
    order: str,
    limiter: str | None,
    vertical_order: Literal['linear', 'cubic'],
    limiter_bound_reference: jax.Array | None,
    k_offset: int | jax.Array,
    buf_z_origin: int | jax.Array,
    j_offset: int | jax.Array,
    buf_lat_origin: int | jax.Array,
    max_lat_shift: int,
    max_vertical_shift: int,
) -> jax.Array:
  """Windowed MXU matmul 3-D interpolation on [C, L_buf, n_lon, n_lat_buf]."""
  dtype = f_buf.dtype
  n_lon = grid.longitude_nodes
  n_lat = grid.latitude_nodes
  n_lat_ext = n_lat + 2 * _HALO_WIDTH
  l_src = sigma_nodes.shape[0]
  num_channels, l_buf, _, n_lat_buf = f_buf.shape
  l_q_loc, nx_loc, ny_loc = lon_q.shape

  w_lat_size = min(n_lat_ext, 2 * max_lat_shift + 4)
  w_z_size = min(l_src, 2 * max_vertical_shift + 4)

  (
      lon_indices,
      lon_weights,
      lat_indices,
      lat_weights,
      lon_cell0,
      lat_cell0,
      _,
  ) = _horizontal_stencil_tuple(grid, lon_q, sin_lat_q, order)
  level_indices, level_weights, _, level_cell0 = _vertical_stencil_tuple(
      sigma_q, sigma_nodes, vertical_order
  )

  j_global = jnp.minimum(
      j_offset + jnp.arange(ny_loc, dtype=jnp.int32), n_lat - 1
  )
  if w_lat_size == n_lat_ext:
    win_lat_start = jnp.zeros_like(j_global)
  else:
    win_lat_start = jnp.clip(
        j_global + 1 - max_lat_shift, 0, n_lat_ext - w_lat_size
    )
  buf_lat_start = win_lat_start - buf_lat_origin
  m_lat_win = jax.nn.one_hot(
      buf_lat_start[:, jnp.newaxis]
      + jnp.arange(w_lat_size, dtype=jnp.int32)[jnp.newaxis, :],
      n_lat_buf,
      dtype=dtype,
  )  # [ny_loc, w_lat_size, n_lat_buf]
  m_lat_2d_t = m_lat_win.reshape(ny_loc * w_lat_size, n_lat_buf).T

  dot_prec_bf16 = jax_numpy_utils.resolve_dot_precision(
      None, m_lat_win, f_buf, lhs_exact_in_bf16=True
  )
  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, f_buf, f_buf)

  ref_at_dep = None
  ref_buf = None
  if limiter is not None and limiter_bound_reference is not None:
    ref_arr = jnp.asarray(limiter_bound_reference, dtype=dtype)
    ref_centered = ref_arr - jnp.mean(ref_arr)
    ref_at_dep = sum(
        w * ref_centered[idx] for idx, w in zip(level_indices, level_weights)
    )
    k_buf_levels = jnp.clip(
        buf_z_origin + jnp.arange(l_buf, dtype=jnp.int32), 0, l_src - 1
    )
    ref_buf = ref_centered[k_buf_levels]

  if limiter is not None:
    w_lat_cell_size = min(n_lat_ext - 1, max(1, w_lat_size - 1))
    if w_lat_cell_size == n_lat_ext - 1:
      win_lat_cell_start = jnp.zeros_like(j_global)
    else:
      win_lat_cell_start = jnp.clip(
          win_lat_start, 0, (n_lat_ext - 1) - w_lat_cell_size
      )
    buf_lat_cell_start = win_lat_cell_start - buf_lat_origin
    m_lat_cell_win = jax.nn.one_hot(
        buf_lat_cell_start[:, jnp.newaxis]
        + jnp.arange(w_lat_cell_size, dtype=jnp.int32)[jnp.newaxis, :],
        n_lat_buf - 1,
        dtype=dtype,
    )
    m_lat_cell_2d_t = m_lat_cell_win.reshape(
        ny_loc * w_lat_cell_size, n_lat_buf - 1
    ).T
    w_z_cell_size = min(l_src - 1, max(1, w_z_size - 1))

  def _eval_channel_chunk(f_chunk: jax.Array) -> jax.Array:
    c_cur = f_chunk.shape[0]
    k_yc = w_lat_size * c_cur

    # Extract latitude windows once across all L_buf levels with >=128 minor dims
    f_chunk_2d = f_chunk.transpose(1, 2, 0, 3).reshape(
        l_buf * n_lon * c_cur, n_lat_buf
    )
    f_lat_win = (
        jnp.dot(f_chunk_2d, m_lat_2d_t, precision=dot_prec_bf16)
        .reshape(l_buf, n_lon, c_cur, ny_loc, w_lat_size)
        .transpose(0, 3, 1, 4, 2)
        .reshape(l_buf, ny_loc, n_lon, k_yc)
    )

    if limiter is not None:
      f_for_bounds = (
          f_chunk + ref_buf[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
          if ref_buf is not None
          else f_chunk
      )
      lo_buf, hi_buf = _cell_bounds_3d(f_for_bounds)
      bounds_buf = jnp.concatenate([lo_buf, hi_buf], axis=0)
      k_yc_cell = w_lat_cell_size * (2 * c_cur)
      b_chunk_2d = bounds_buf.transpose(1, 2, 0, 3).reshape(
          (l_buf - 1) * n_lon * 2 * c_cur, n_lat_buf - 1
      )
      bounds_lat_win = (
          jnp.dot(b_chunk_2d, m_lat_cell_2d_t, precision=dot_prec_bf16)
          .reshape(l_buf - 1, n_lon, 2 * c_cur, ny_loc, w_lat_cell_size)
          .transpose(0, 3, 1, 4, 2)
          .reshape(l_buf - 1, ny_loc, n_lon, k_yc_cell)
      )

    def _scan_level(_, inputs):
      (
          k_loc,
          lon_idx_k,
          lon_w_k,
          lat_idx_k,
          lat_w_k,
          lvl_idx_k,
          lvl_w_k,
          lon_c0_k,
          lat_c0_k,
          lvl_c0_k,
      ) = inputs

      k_glob = jnp.minimum(k_offset + k_loc, l_src - 1)
      if w_z_size == l_src:
        win_z_start = jnp.int32(0)
      else:
        win_z_start = jnp.clip(
            k_glob + 1 - max_vertical_shift, 0, l_src - w_z_size
        )
      buf_z_start = win_z_start - buf_z_origin

      m_z_k = jax.nn.one_hot(
          buf_z_start + jnp.arange(w_z_size, dtype=jnp.int32),
          l_buf,
          dtype=dtype,
      )  # [w_z_size, l_buf]
      f_patch_flat = (
          jnp.dot(m_z_k, f_lat_win.reshape(l_buf, -1), precision=dot_prec_bf16)
          .reshape(w_z_size, ny_loc, n_lon, k_yc)
          .transpose(1, 2, 0, 3)
          .reshape(ny_loc, n_lon, w_z_size * k_yc)
      )

      w_lon_mat = sum(
          w.T[:, :, jnp.newaxis] * jax.nn.one_hot(idx.T, n_lon, dtype=dtype)
          for idx, w in zip(lon_idx_k, lon_w_k)
      )
      gathered_lon = jnp.einsum(
          'jim,jmk->jik', w_lon_mat, f_patch_flat, precision=dot_prec_f32
      )

      w_z_mat = sum(
          w.T[:, :, jnp.newaxis]
          * jax.nn.one_hot(idx.T - win_z_start, w_z_size, dtype=dtype)
          for idx, w in zip(lvl_idx_k, lvl_w_k)
      )
      w_lat_mat = sum(
          w.T[:, :, jnp.newaxis]
          * jax.nn.one_hot(
              idx.T - win_lat_start[:, jnp.newaxis], w_lat_size, dtype=dtype
          )
          for idx, w in zip(lat_idx_k, lat_w_k)
      )
      w_zy = (
          w_z_mat[:, :, :, jnp.newaxis] * w_lat_mat[:, :, jnp.newaxis, :]
      ).reshape(ny_loc, nx_loc, w_z_size * w_lat_size)

      if c_cur == 1:
        val = (gathered_lon * w_zy).sum(axis=-1).T[jnp.newaxis, :, :]
      else:
        gathered_zy_c = gathered_lon.reshape(
            ny_loc, nx_loc, w_z_size * w_lat_size, c_cur
        )
        val = (
            (gathered_zy_c * w_zy[:, :, :, jnp.newaxis])
            .sum(axis=2)
            .transpose(2, 1, 0)
        )

      if limiter is not None:
        if w_z_cell_size == l_src - 1:
          win_z_cell_start = jnp.int32(0)
        else:
          win_z_cell_start = jnp.clip(
              win_z_start, 0, (l_src - 1) - w_z_cell_size
          )
        buf_z_cell_start = win_z_cell_start - buf_z_origin
        m_z_cell_k = jax.nn.one_hot(
            buf_z_cell_start + jnp.arange(w_z_cell_size, dtype=jnp.int32),
            l_buf - 1,
            dtype=dtype,
        )
        b_patch_flat = (
            jnp.dot(
                m_z_cell_k,
                bounds_lat_win.reshape(l_buf - 1, -1),
                precision=dot_prec_bf16,
            )
            .reshape(w_z_cell_size, ny_loc, n_lon, k_yc_cell)
            .transpose(1, 2, 0, 3)
            .reshape(ny_loc, n_lon, w_z_cell_size * k_yc_cell)
        )
        p_lon_mat = jax.nn.one_hot(lon_c0_k.T, n_lon, dtype=dtype)
        b_lon = jnp.einsum(
            'jim,jmk->jik', p_lon_mat, b_patch_flat, precision=dot_prec_bf16
        )
        p_z_mat = jax.nn.one_hot(
            lvl_c0_k.T - win_z_cell_start, w_z_cell_size, dtype=dtype
        )
        p_lat_mat = jax.nn.one_hot(
            lat_c0_k.T - win_lat_cell_start[:, jnp.newaxis],
            w_lat_cell_size,
            dtype=dtype,
        )
        p_zy = (
            p_z_mat[:, :, :, jnp.newaxis] * p_lat_mat[:, :, jnp.newaxis, :]
        ).reshape(ny_loc, nx_loc, w_z_cell_size * w_lat_cell_size)
        b_zy_c = b_lon.reshape(
            ny_loc, nx_loc, w_z_cell_size * w_lat_cell_size, 2 * c_cur
        )
        b_val = (
            (b_zy_c * p_zy[:, :, :, jnp.newaxis])
            .sum(axis=2)
            .transpose(2, 1, 0)
        )
        lo_k, hi_k = b_val[:c_cur], b_val[c_cur:]
        if ref_at_dep is not None:
          ref_k = lax.dynamic_index_in_dim(
              ref_at_dep, k_loc, axis=0, keepdims=False
          )
          val = (
              jnp.clip(val + ref_k[jnp.newaxis, :, :], lo_k, hi_k)
              - ref_k[jnp.newaxis, :, :]
          )
        else:
          val = jnp.clip(val, lo_k, hi_k)

      return None, val

    scan_inputs = (
        jnp.arange(l_q_loc, dtype=jnp.int32),
        lon_indices,
        lon_weights,
        lat_indices,
        lat_weights,
        level_indices,
        level_weights,
        lon_cell0,
        lat_cell0,
        level_cell0,
    )
    _, chunk_out = lax.scan(_scan_level, None, scan_inputs)
    return chunk_out.transpose(1, 0, 2, 3)

  chunk_size = 8
  if num_channels <= chunk_size:
    return _eval_channel_chunk(f_buf)

  c_pad = _round_to_multiple(num_channels, chunk_size)
  f_pad = (
      jnp.pad(f_buf, [(0, c_pad - num_channels), (0, 0), (0, 0), (0, 0)])
      if c_pad > num_channels
      else f_buf
  )
  f_grouped = f_pad.reshape(
      (c_pad // chunk_size, chunk_size) + f_buf.shape[1:]
  )
  _, out_grouped = lax.scan(
      lambda _, grp: (None, _eval_channel_chunk(grp)), None, f_grouped
  )
  return out_grouped.reshape((c_pad, l_q_loc, nx_loc, ny_loc))[:num_channels]


def _interpolate_3d_grid_matmul(
    fields_4d: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    sigma: jax.Array,
    sigma_nodes: Array,
    *,
    order: str,
    limiter: str | None,
    vertical_order: Literal['linear', 'cubic'],
    limiter_bound_reference: Array | None,
    max_lat_shift: int,
    max_vertical_shift: int,
) -> jax.Array:
  """Grid-aligned 3-D MXU matmul interpolation (single-device and SPMD)."""
  sigma_nodes = jnp.asarray(sigma_nodes)
  mesh = grid.spmd_mesh
  if mesh is None:
    f_ext = _extend_with_pole_halo(fields_4d, grid)
    return _matmul_3d_local(
        f_ext,
        lon,
        sin_lat,
        sigma,
        grid=grid,
        sigma_nodes=sigma_nodes,
        order=order,
        limiter=limiter,
        vertical_order=vertical_order,
        limiter_bound_reference=limiter_bound_reference,
        k_offset=0,
        buf_z_origin=0,
        j_offset=0,
        buf_lat_origin=0,
        max_lat_shift=max_lat_shift,
        max_vertical_shift=max_vertical_shift,
    )

  z_shards = mesh.shape['z'] if 'z' in mesh.axis_names else 1
  x_shards = mesh.shape['x']
  l_src = fields_4d.shape[1]
  l_q = lon.shape[0]
  l_src_pad = _round_to_multiple(l_src, z_shards)
  l_q_pad = _round_to_multiple(l_q, z_shards)

  f_pad = (
      jnp.pad(
          fields_4d,
          [(0, 0), (0, l_src_pad - l_src), (0, 0), (0, 0)],
          mode='edge',
      )
      if l_src_pad > l_src
      else fields_4d
  )
  if l_q_pad > l_q:
    q_pad_width = [(0, l_q_pad - l_q), (0, 0), (0, 0)]
    lon_pad = jnp.pad(lon, q_pad_width, mode='edge')
    sin_lat_pad = jnp.pad(sin_lat, q_pad_width, mode='edge')
    sigma_pad = jnp.pad(sigma, q_pad_width, mode='edge')
  else:
    lon_pad, sin_lat_pad, sigma_pad = lon, sin_lat, sigma

  z_spec = 'z' if z_shards > 1 else None
  f_spec = P(None, z_spec, 'x', 'y')
  q_spec = P(z_spec, 'x', 'y')
  nodes_spec = P(None)
  ref_spec = P(None) if limiter_bound_reference is not None else None

  def _sharded(f_loc, lon_loc, sin_lat_loc, sigma_loc, nodes_loc, ref_loc):
    if x_shards > 1:
      f_x = lax.all_gather(f_loc, 'x', axis=2, tiled=True)
    else:
      f_x = f_loc
    f_x = f_x[:, :, : grid.longitude_nodes, :]
    f_xy_buf, j_offset, buf_lat_origin = _gather_lat_with_halo(
        f_x, grid, max_lat_shift
    )

    if z_shards == 1:
      f_xyz_buf = f_xy_buf[:, :l_src, :, :]
      k_offset = 0
      buf_z_origin = 0
    else:
      iz = lax.axis_index('z')
      l_src_loc = f_xy_buf.shape[1]
      l_q_loc = lon_loc.shape[0]
      h_z = max_vertical_shift + 2
      if l_src == l_q and l_src == l_src_pad and l_src_loc >= h_z:
        perm_down = [(s, (s + 1) % z_shards) for s in range(z_shards)]
        perm_up = [(s, (s - 1) % z_shards) for s in range(z_shards)]
        top_halo = lax.ppermute(f_xy_buf[:, -h_z:, :, :], 'z', perm=perm_down)
        bot_halo = lax.ppermute(f_xy_buf[:, :h_z, :, :], 'z', perm=perm_up)
        f_xyz_buf = jnp.concatenate([top_halo, f_xy_buf, bot_halo], axis=1)
        k_offset = iz * l_q_loc
        buf_z_origin = iz * l_src_loc - h_z
      else:
        f_xyz_buf = lax.all_gather(f_xy_buf, 'z', axis=1, tiled=True)[
            :, :l_src, :, :
        ]
        k_offset = iz * l_q_loc
        buf_z_origin = 0

    return _matmul_3d_local(
        f_xyz_buf,
        lon_loc,
        sin_lat_loc,
        sigma_loc,
        grid=grid,
        sigma_nodes=nodes_loc,
        order=order,
        limiter=limiter,
        vertical_order=vertical_order,
        limiter_bound_reference=ref_loc,
        k_offset=k_offset,
        buf_z_origin=buf_z_origin,
        j_offset=j_offset,
        buf_lat_origin=buf_lat_origin,
        max_lat_shift=max_lat_shift,
        max_vertical_shift=max_vertical_shift,
    )

  out_pad = jax.shard_map(
      _sharded,
      mesh=mesh,
      in_specs=(f_spec, q_spec, q_spec, q_spec, nodes_spec, ref_spec),
      out_specs=f_spec,
  )(f_pad, lon_pad, sin_lat_pad, sigma_pad, sigma_nodes, limiter_bound_reference)
  return out_pad[:, :l_q, :, :]


@functools.partial(
    jax.jit,
    static_argnames=(
        'grid',
        'order',
        'limiter',
        'vertical_order',
        'method',
        'max_lat_shift',
        'max_vertical_shift',
    ),
)
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
    vertical_order: Literal['linear', 'cubic'] = 'linear',
    limiter_bound_reference: Array | None = None,
    method: InterpolationMethod | None = None,
    max_lat_shift: int | None = 6,
    max_vertical_shift: int | None = 6,
) -> jnp.ndarray:
  """Interpolates a 3-D (or multi-channel 3-D) field at (lon, sin_lat, sigma).

  Horizontal interpolation follows `GridInterpolator` (tensor-product
  Lagrange of the given order); vertical interpolation is Lagrange in σ
  between `sigma_nodes` — linear by default, or 4-point cubic with
  `vertical_order='cubic'`, degraded to linear within the first and last
  cells (the standard operational rule, e.g. in the IFS). Both use constant
  extrapolation beyond the first and last nodes (matching
  `vertical_interpolation.interp` and the zero-gradient boundary
  conditions of `sigma_coordinates.centered_vertical_advection`).

  Args:
    field: nodal values of shape [len(sigma_nodes), *grid.nodal_shape], or
      [*channels, len(sigma_nodes), *grid.nodal_shape] to interpolate multiple
      fields at the same departure points in a single batched pass.
    grid: the horizontal grid holding the source data.
    lon: longitudes in radians, of any shape (typically [layers, *nodal_shape]
      for departure points of full-level arrivals).
    sin_lat: sine of latitudes, same shape as `lon`.
    sigma: σ coordinates of the interpolation points, same shape as `lon`.
    sigma_nodes: increasing σ values at which `field` levels live (layer
      centers for prognostic fields, layer boundaries for vertical velocity).
    order: horizontal interpolation order, 'linear' or 'cubic'.
    limiter: None for unlimited interpolation, or 'quasi_monotone' to clip
      interpolated values to the range of the 2✕2✕2 bracketing cell corners
      (the Bermejo & Staniforth 1992 limiter). The vertical extent of the
      bracketing cell is always the two nodes around the departure σ,
      regardless of `vertical_order`.
    vertical_order: vertical interpolation order, 'linear' or 'cubic'.
    limiter_bound_reference: optional per-level values of shape
      [len(sigma_nodes)] expressing the limiter's bound in a shifted
      variable: corners are shifted by the reference at their levels and
      the interpolated value by the reference interpolated at the
      departure σ before clipping (and unshifted after). Interpolation
      itself is unchanged, and a constant reference is exactly a no-op;
      used to bound `T′ = T − T_ref(σ)` transport in full temperature,
      where a bound on T′ across levels with different `T_ref` would be
      gauge-dependent. Ignored without a limiter.
    method: 'gather', 'matmul', or None (auto-selected per backend/mesh).
    max_lat_shift: maximum meridional shift in latitude cells for windowed
      grid-aligned 'matmul' interpolation (None disables windowing).
    max_vertical_shift: maximum vertical shift in levels for windowed
      grid-aligned 'matmul' interpolation (None disables windowing).

  Returns:
    Interpolated values of shape `(*channels, *lon.shape)`.
  """
  _validate_limiter(limiter)
  resolved = _resolve_interpolation_method(method, grid)
  if vertical_order not in ('linear', 'cubic'):
    raise ValueError(f'unsupported {vertical_order=}')
  field = jnp.asarray(field)
  lon = jnp.asarray(lon)
  sin_lat = jnp.asarray(sin_lat)
  sigma = jnp.asarray(sigma)
  min_levels = 2 if vertical_order == 'linear' else 4
  if len(sigma_nodes) < min_levels:
    raise ValueError(
        f'{vertical_order} vertical interpolation requires at least'
        f' {min_levels} levels'
    )
  if (
      field.ndim < 3
      or field.shape[-3] != len(sigma_nodes)
      or field.shape[-2:] != grid.nodal_shape
  ):
    raise ValueError(
        f'expected field of shape [..., {len(sigma_nodes)}, '
        f'{grid.nodal_shape[0]}, {grid.nodal_shape[1]}]; got {field.shape}'
    )

  channel_shape = field.shape[:-3]
  c_total = math.prod(channel_shape)
  if c_total == 0 or lon.size == 0:
    return jnp.zeros((*channel_shape, *lon.shape), dtype=field.dtype)

  fields_4d = field.reshape((c_total,) + field.shape[-3:])
  if resolved == 'gather':
    out_4d = _interpolate_3d_gather(
        fields_4d,
        grid,
        lon,
        sin_lat,
        sigma,
        sigma_nodes,
        order=order,
        limiter=limiter,
        vertical_order=vertical_order,
        limiter_bound_reference=limiter_bound_reference,
    )
  elif (
      lon.ndim == 3
      and lon.shape[1:] == grid.nodal_shape
      and max_lat_shift is not None
      and max_vertical_shift is not None
  ):
    out_4d = _interpolate_3d_grid_matmul(
        fields_4d,
        grid,
        lon,
        sin_lat,
        sigma,
        jnp.asarray(sigma_nodes),
        order=order,
        limiter=limiter,
        vertical_order=vertical_order,
        limiter_bound_reference=limiter_bound_reference,
        max_lat_shift=max_lat_shift,
        max_vertical_shift=max_vertical_shift,
    )
  else:
    out_4d = _matmul_3d_points(
        fields_4d,
        grid,
        lon,
        sin_lat,
        sigma,
        jnp.asarray(sigma_nodes),
        order=order,
        limiter=limiter,
        vertical_order=vertical_order,
        limiter_bound_reference=limiter_bound_reference,
    )
  return out_4d.reshape((*channel_shape, *lon.shape))


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
    cartesian: unit vectors on the sphere, of shape [3, *points].
    sigma: σ coordinates of departure points, or None for horizontal-only
      trajectories.
    lon: longitudes in radians (derived from `cartesian`).
    sin_lat: sine of latitudes (derived from `cartesian`).
  """

  cartesian: Array
  sigma: Array | None = None

  @property
  def lon(self) -> jnp.ndarray:
    lon, _ = cartesian_to_lon_sin_lat(self.cartesian)
    return lon

  @property
  def sin_lat(self) -> jnp.ndarray:
    _, sin_lat = cartesian_to_lon_sin_lat(self.cartesian)
    return sin_lat


def horizontal_departure_points(
    u: Array,
    v: Array,
    grid: spherical_harmonic.Grid,
    dt: float,
    *,
    iterations: int = 2,
    initial_guess: DeparturePoints | None = None,
    method: InterpolationMethod | None = None,
    max_lat_shift: int | None = 6,
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
    method: 'gather', 'matmul', or None (auto-selected per backend/mesh).
    max_lat_shift: maximum meridional shift in latitude cells for windowed
      grid-aligned 'matmul' interpolation (None disables windowing).

  Returns:
    DeparturePoints with fields of shape [*batch, longitude_nodes,
    latitude_nodes] and `sigma=None`.
  """
  _validate_interpolation_method(method)
  u = jnp.asarray(u)
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  arrival = lon_lat_to_cartesian(
      jnp.broadcast_to(lon_mesh, u.shape),
      jnp.broadcast_to(sin_lat_mesh, u.shape),
  )
  wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
  angular_dt = dt / grid.radius

  departure = arrival if initial_guess is None else initial_guess.cartesian
  for _ in range(iterations):
    midpoint = _normalize((arrival + departure) / 2)
    lon_mid, sin_lat_mid = cartesian_to_lon_sin_lat(midpoint)
    wind_mid = _interpolate_2d_multi(
        wind,
        grid,
        lon_mid,
        sin_lat_mid,
        order='linear',
        limiter=None,
        method=method,
        max_lat_shift=max_lat_shift,
    )
    departure = _normalize(arrival - angular_dt * wind_mid)

  return DeparturePoints(cartesian=departure, sigma=None)


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
    method: InterpolationMethod | None = None,
    max_lat_shift: int | None = 6,
    max_vertical_shift: int | None = 6,
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
    method: 'gather', 'matmul', or None (auto-selected per backend/mesh).
    max_lat_shift: maximum meridional shift in latitude cells for windowed
      grid-aligned 'matmul' interpolation (None disables windowing).
    max_vertical_shift: maximum vertical shift in levels for windowed
      grid-aligned 'matmul' interpolation (None disables windowing).

  Returns:
    DeparturePoints with fields of shape [layers, longitude_nodes,
    latitude_nodes], including departure σ coordinates.
  """
  _validate_interpolation_method(method)
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
      interpolate_3d,
      grid=grid,
      order='linear',
      method=method,
      max_lat_shift=max_lat_shift,
      max_vertical_shift=max_vertical_shift,
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
    wind_mid = interpolate(wind, sigma_nodes=centers, **point)
    sigma_dot_mid = interpolate(sigma_dot, sigma_nodes=boundaries, **point)
    departure = _normalize(arrival - angular_dt * wind_mid)
    sigma_departure = jnp.clip(
        sigma_arrival - dt * sigma_dot_mid, float(centers[0]), float(centers[-1])
    )

  return DeparturePoints(cartesian=departure, sigma=sigma_departure)


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
      [levels, longitude_nodes, latitude_nodes], or batched as
      [channels, longitude_nodes, latitude_nodes] /
      [channels, levels, longitude_nodes, latitude_nodes].
    departure: horizontal departure points with fields shaped like the spatial
      dimensions of `field`.
    interpolator: horizontal interpolation rule.

  Returns:
    The transported field at arrival points, shaped like `field` with trailing
    axes matching `departure.lon`.
  """
  field = jnp.asarray(field)
  lon = departure.lon
  sin_lat = departure.sin_lat
  if field.ndim == 4 or (field.ndim == 3 and lon.ndim == 2):
    out = _interpolate_2d_multi(
        field,
        interpolator.grid,
        lon,
        sin_lat,
        order=interpolator.order,
        limiter=interpolator.limiter,
        method=interpolator.method,
        max_lat_shift=interpolator.max_lat_shift,
    )
  else:
    out = interpolator(field, lon, sin_lat)
  return _zero_nodal_padding(out, interpolator.grid)


def transport_scalar(
    field: Array,
    departure: DeparturePoints,
    vertical: sigma_coordinates.SigmaCoordinates,
    interpolator: GridInterpolator,
    *,
    vertical_order: Literal['linear', 'cubic'] = 'linear',
    limiter_bound_reference: Array | None = None,
    max_vertical_shift: int | None = 6,
) -> jnp.ndarray:
  """Remaps scalar field(s) to arrival points along 3-D trajectories.

  Args:
    field: nodal values at layer centers, of shape
      [*channels, layers, longitude_nodes, latitude_nodes].
    departure: 3-D departure points (with σ coordinates).
    vertical: vertical coordinates on which `field` lives: anything with
      increasing `.centers` node values (`SigmaCoordinates` or
      `VerticalNodes`).
    interpolator: interpolation rule (its grid, order, limiter, method, and
      `max_lat_shift` settings are used).
    vertical_order: vertical interpolation order (see `interpolate_3d`).
    limiter_bound_reference: optional per-level bound shift for the
      limiter (see `interpolate_3d`).
    max_vertical_shift: maximum vertical shift in levels for windowed
      grid-aligned 'matmul' interpolation (None disables windowing).

  Returns:
    The transported field(s) at arrival points, of shape
    `[*channels, *departure.lon.shape]`.
  """
  if departure.sigma is None:
    raise ValueError('transport_scalar requires 3-D departure points')
  out = interpolate_3d(
      field,
      interpolator.grid,
      departure.lon,
      departure.sin_lat,
      departure.sigma,
      sigma_nodes=vertical.centers,
      order=interpolator.order,
      limiter=interpolator.limiter,
      vertical_order=vertical_order,
      limiter_bound_reference=limiter_bound_reference,
      method=interpolator.method,
      max_lat_shift=interpolator.max_lat_shift,
      max_vertical_shift=max_vertical_shift,
  )
  return _zero_nodal_padding(out, interpolator.grid)


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
  u_arr, v_arr = tangent_wind(wind_arrival, lon_mesh, sin_lat_mesh)
  return (
      _zero_nodal_padding(u_arr, grid),
      _zero_nodal_padding(v_arr, grid),
  )


def transport_wind(
    u: Array,
    v: Array,
    departure: DeparturePoints,
    vertical: sigma_coordinates.SigmaCoordinates,
    interpolator: GridInterpolator,
    *,
    rotate: bool = True,
    planetary_rotation_rate: float | None = None,
    vertical_order: Literal['linear', 'cubic'] = 'linear',
    max_vertical_shift: int | None = 6,
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
    interpolator: interpolation rule. Its limiter (if any) clips each
      interpolated Cartesian component of the *relative* wind to its
      bracketing-cell range, before parallel transport and before the
      analytic planetary term is applied — the IFS operationally applies
      quasi-monotone interpolation to momentum too. Adds diffusion on the
      dynamics near sharp features.
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
    vertical_order: vertical interpolation order (see `interpolate_3d`).
    max_vertical_shift: maximum vertical shift in levels for windowed
      grid-aligned 'matmul' interpolation (None disables windowing).

  Returns:
    Tuple (u, v) of transported winds at arrival points.
  """
  if departure.sigma is None:
    raise ValueError('transport_wind requires 3-D departure points')
  grid = interpolator.grid
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
  wind_departure = interpolate_3d(
      wind,
      grid,
      departure.lon,
      departure.sin_lat,
      departure.sigma,
      sigma_nodes=vertical.centers,
      order=interpolator.order,
      limiter=interpolator.limiter,
      vertical_order=vertical_order,
      method=interpolator.method,
      max_lat_shift=interpolator.max_lat_shift,
      max_vertical_shift=max_vertical_shift,
  )
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
    interpolator: interpolation rule; its limiter applies to the wind
      components (see `transport_wind`).
    rotate: whether to parallel-transport interpolated vectors from the
      departure to the arrival tangent plane.
    planetary_rotation_rate: if not None, transport the planetary momentum
      `v + 2Ω ✕ R` (see `transport_wind`).

  Returns:
    Tuple (u, v) of transported winds at arrival points.
  """
  grid = interpolator.grid
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
  wind_departure = _interpolate_2d_multi(
      wind,
      grid,
      departure.lon,
      departure.sin_lat,
      order=interpolator.order,
      limiter=interpolator.limiter,
      method=interpolator.method,
      max_lat_shift=interpolator.max_lat_shift,
  )
  return _finish_wind_transport(
      wind_departure, departure, grid, rotate, planetary_rotation_rate
  )
