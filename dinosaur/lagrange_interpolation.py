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

"""Lagrange interpolation stencils and gather kernels on the Gaussian grid.

The parts of `semi_lagrangian`'s interpolation that do not depend on how the
stencil values are read: the cross-pole halo extension of the latitude axis,
the horizontal and vertical Lagrange stencils (indices and weights) of a set
of query points, the clamp that limits grid-aligned queries to a window
around their arrival point, and the cell bounds of the quasi-monotone
limiter. It also holds the 'gather' kernels, which read the stencil values
with indexed gathers: the fast path on a single CPU or GPU.

`semi_lagrangian_matmul` implements the same interpolation with matrix
products, for TPUs and SPMD meshes. Both are the implementation of the
public API in `semi_lagrangian` and may change without notice.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Literal, NamedTuple

from dinosaur import jax_numpy_utils
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

Array = typing.Array

# All `einsum`s should be done at highest available precision.
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)

# Largest 1-D node/table axis whose lookup is done with a one-hot MXU dot (or a
# broadcast compare) instead of a scalar gather or `searchsorted`. The one-hot
# has one column per node, so this bounds its size.
ONE_HOT_LOOKUP_MAX_NODES = 512


def effective_window_shift(max_shift: int) -> int:
  """Window half-width actually used: a `max_shift` of 0 is treated as 1."""
  return max(1, max_shift)


# Clamping departures to the stencil windows.
#
# The 'matmul' interpolation builds each grid-aligned query's stencil inside a
# window around its arrival point; a stencil point outside that window would
# silently get no weight. Instead, whenever windows are in use (both backends,
# whenever the caller gives the shifts), the interpolation query is clamped
# into the window before its stencil is built: latitude and σ are limited so
# that the bracketing cell is at most `max(1, shift)` cells from the arrival
# cell, the displacement limit a CFL condition would impose. Every window of
# the 'matmul' paths (main, limiter bracketing cell, trilinear, level-blocked)
# contains the cells this allows, so with the clamp 'matmul' and 'gather'
# interpolate the same query and agree to rounding.
#
# The upper bound is one ulp (of the query dtype) below the first node beyond
# the window, which keeps the bracketing cell inside the window under every
# cell search (`searchsorted` and the TPU threshold one-hot alike) while the
# interpolated value is the window-edge value to rounding. The lower bound is
# the lowest node of the lowest allowed cell, whose cell search returns that
# cell. The clamp acts on latitude (after the arctan2 of sin_lat) and on σ
# inside the stencil builders, so every gather, matmul and fused path clamps
# the same query, and the bounds are static per arrival row/level (no table
# lookups or gathers in the per-query work).


def _next_below(x: np.ndarray, dtype: jnp.dtype) -> np.ndarray | jax.Array:
  """The largest `dtype` value strictly below each `x` (rounded to `dtype`)."""
  dtype = np.dtype(dtype)
  if dtype in (np.float16, np.float32, np.float64):
    return np.nextafter(x.astype(dtype), np.asarray(-np.inf, dtype))
  return jnp.nextafter(jnp.asarray(x, dtype), jnp.asarray(-jnp.inf, dtype))


def lat_clamp_bounds(
    grid: spherical_harmonic.Grid,
    max_lat_shift: int,
    order: str,
    dtype: jnp.dtype,
) -> tuple[np.ndarray | jax.Array, np.ndarray | jax.Array]:
  """Latitude (radians) clamp of grid-aligned queries, per arrival row.

  A query in nodal latitude column `j` arrives at Gaussian row
  `min(j, latitude_nodes - 1)` (padding columns map to the last physical row,
  as in `_arrival_lat_rows`), i.e. at row `a = j + HALO_WIDTH` of the
  halo-extended axis `φ_ext`. Its bracketing cell is limited to
  `[a - S, a + S]` (S = `max(1, max_lat_shift)`), intersected with the cells
  a stencil of `order` can use; so φ is clamped to
  `[φ_ext[lo_c], φ_ext[hi_c + 1])`. Physical latitudes never reach the halo
  cells beyond the poles, so near the poles only the equatorward bound acts.

  The limit is deliberately on the bracketing cell, not on the distance: the
  `2S + 1` allowed cells are `a - S ... a + S`, which is what lets every
  stencil window be a fixed set of rows around `a`. Cell `a` starts at the
  arrival node, so when the clamp acts the query may lie up to `S` cells from
  the arrival point towards lower latitudes but up to `S + 1` cells towards
  higher ones.

  Args:
    grid: the horizontal grid.
    max_lat_shift: window half-width in latitude rows (0 is treated as 1).
    order: horizontal interpolation order.
    dtype: dtype of the query latitudes.

  Returns:
    `(lo, hi)`, each of shape [grid.nodal_shape[1]] and dtype `dtype`.
  """
  phi_extended = latitude_halo_axis(grid)
  offsets = _stencil_offsets(order)
  c_min = -int(offsets[0])
  c_max = phi_extended.size - 1 - int(offsets[-1])
  shift = effective_window_shift(max_lat_shift)
  arrival = (
      np.minimum(np.arange(grid.nodal_shape[1]), grid.latitude_nodes - 1)
      + HALO_WIDTH
  )
  lo_cell = np.clip(arrival - shift, c_min, c_max)
  hi_cell = np.clip(arrival + shift, c_min, c_max)
  lo = phi_extended[lo_cell].astype(np.dtype(dtype))
  hi = _next_below(phi_extended[hi_cell + 1], dtype)
  return lo, hi


def _static_take(values: jax.Array, indices: np.ndarray) -> jax.Array:
  """`values[indices]` for static non-decreasing `indices`, without a gather.

  Built from static slices (runs of consecutive indices) and broadcasts (runs
  of a repeated index), so it lowers to slices and a concatenate on every
  backend.
  """
  pieces = []
  start = 0
  n = len(indices)
  while start < n:
    end = start + 1
    if end < n and indices[end] == indices[start]:
      while end < n and indices[end] == indices[start]:
        end += 1
      i0 = int(indices[start])
      pieces.append(jnp.broadcast_to(values[i0 : i0 + 1], (end - start,)))
    else:
      while end < n and indices[end] == indices[end - 1] + 1:
        end += 1
      i0 = int(indices[start])
      pieces.append(values[i0 : i0 + end - start])
    start = end
  return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)


def level_clamp_bounds(
    nodes: Array,
    num_query_levels: int,
    max_vertical_shift: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
  """Vertical clamp of grid-aligned queries, per arrival level.

  Query level `q` arrives at node level `k = min(q, len(nodes) - 1)` (as in
  the 'matmul' windows). Its bracketing cell is limited to `[k - S, k + S]`
  (S = `max(1, max_vertical_shift)`), intersected with the `len(nodes) - 1`
  cells; so σ is clamped to `[nodes[lo_k], nodes[hi_k + 1])`, closed at the
  top when `hi_k + 1` is the last node (where σ is clipped anyway). The node
  values are those of `nodes` cast to `dtype`, as in the stencil builder.

  As in `lat_clamp_bounds` the limit is on the bracketing cell, which keeps
  the vertical windows a fixed set of levels around `k`: cell `k` starts at
  the arrival level, so a clamped query may lie up to `S` cells from it
  towards smaller σ but up to `S + 1` cells towards larger σ.

  Args:
    nodes: increasing vertical nodes, possibly traced.
    num_query_levels: number of query levels.
    max_vertical_shift: window half-width in levels (0 is treated as 1).
    dtype: dtype of the query σ.

  Returns:
    `(lo, hi)`, each of shape [num_query_levels].
  """
  nodes = lax.stop_gradient(jnp.asarray(nodes, dtype=dtype))
  n = nodes.shape[0]
  shift = effective_window_shift(max_vertical_shift)
  k = np.minimum(np.arange(num_query_levels), n - 1)
  lo_index = np.maximum(k - shift, 0)
  hi_index = np.minimum(k + shift + 1, n - 1)
  lo = _static_take(nodes, lo_index)
  hi = _static_take(nodes, hi_index)
  open_top = hi_index < n - 1
  if open_top.any():
    hi = jnp.where(open_top, jnp.nextafter(hi, jnp.full_like(hi, -jnp.inf)), hi)
  return lo, hi


def is_windowed_3d(
    lon: jax.Array,
    grid: spherical_harmonic.Grid,
    max_lat_shift: int | None,
    max_vertical_shift: int | None,
) -> bool:
  """Whether 3-D queries are grid-aligned with both windows given."""
  return (
      lon.ndim == 3
      and lon.shape[1:] == grid.nodal_shape
      and max_lat_shift is not None
      and max_vertical_shift is not None
  )


def window_clamp_bounds_3d(
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    sigma: jax.Array,
    sigma_nodes: Array,
    order: str,
    max_lat_shift: int | None,
    max_vertical_shift: int | None,
) -> tuple[tuple[Array, Array] | None, tuple[Array, Array] | None]:
  """Clamp bounds, broadcastable against `lon`, of windowed 3-D queries.

  Returns `(lat_bounds, sigma_bounds)` for grid-aligned queries with both
  shifts given (`is_windowed_3d`), and `(None, None)` otherwise.
  """
  if not is_windowed_3d(lon, grid, max_lat_shift, max_vertical_shift):
    return None, None
  lat_bounds = lat_clamp_bounds(grid, max_lat_shift, order, sin_lat.dtype)
  lo, hi = level_clamp_bounds(
      sigma_nodes, lon.shape[0], max_vertical_shift, sigma.dtype
  )
  return lat_bounds, (
      lo[:, jnp.newaxis, jnp.newaxis],
      hi[:, jnp.newaxis, jnp.newaxis],
  )


def target_platform(grid: spherical_harmonic.Grid) -> str:
  """Platform ('cpu', 'gpu' or 'tpu') the interpolation is compiled for.

  The method auto-selection and the TPU-specific layouts and memory budgets
  are chosen at trace time. The devices of the grid's mesh decide, when it
  has one: the program may be traced where the default backend is another
  platform (ahead-of-time lowering or export, a CPU host coordinating a TPU
  mesh). Without a mesh only the default backend is known at trace time
  (the arrays are tracers).
  """
  mesh = grid.spmd_mesh
  if mesh is not None:
    platforms = {device.platform for device in mesh.devices.flat}
    if len(platforms) == 1:
      return platforms.pop()
  return jax.default_backend()


def targets_tpu(grid: spherical_harmonic.Grid) -> bool:
  return target_platform(grid) == 'tpu'


#  =============================================================================
#  Stencils.
#  =============================================================================


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


def _uniform_lagrange_weights_tuple(
    alpha: jax.Array, order: str
) -> tuple[jax.Array, ...]:
  """Closed-form Lagrange weights on unit-spaced nodes, `alpha` in [0, 1)."""
  if order == 'linear':
    return (1.0 - alpha, alpha)
  if order == 'cubic':
    am1 = alpha - 1.0
    am2 = alpha - 2.0
    ap1 = alpha + 1.0
    return (
        (-1.0 / 6.0) * alpha * am1 * am2,
        0.5 * ap1 * am1 * am2,
        -0.5 * ap1 * alpha * am2,
        (1.0 / 6.0) * ap1 * alpha * am1,
    )
  raise ValueError(f'unknown interpolation {order=}')


def _contract_stencil(
    values: Array, *weights: Array, platform: str
) -> jnp.ndarray:
  """Contracts gathered stencil values with per-dimension weights.

  Used by the 2-D gather interpolation, where `values` has one trailing axis
  per weight vector (the tensor-product stencil) and each `weights[k]` has
  shape [..., stencil_size_k]. The contraction is an einsum on CPU, whose
  dot_general operands are materialized: that is the fastest form on CPU
  (0.55✕ the fused form) but measurably slower on GPU, where a
  broadcast-multiply + reduce fuses with the producing gather and skips the
  materialization (1.15–1.20✕ isolated on A100 at T170/L32, and
  considerably more in a full model step, where the skipped
  materializations compound: 32.0 → 26.3 ms/step with linear vertical and
  58.4 → 33.3 ms/step with cubic on the 2-day ERA5 benchmark). The platform
  (`target_platform`) is known at trace time, so the form is picked per
  platform. The 3-D
  gather interpolation accumulates the vertical, longitude and latitude
  stencils level by level itself and does not use this function.
  """
  if platform == 'cpu':
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


HALO_WIDTH = 2  # supports stencils up to 4 points wide in latitude.


def latitude_halo_axis(grid: spherical_harmonic.Grid) -> np.ndarray:
  """Returns latitudes (radians) extended with `HALO_WIDTH` rows per pole.

  A point beyond the pole at "latitude" φ > π/2 corresponds to the physical
  location (π - φ, λ + π), so halo rows take coordinates mirrored about the
  pole. The extended axis is strictly increasing, which makes latitude (not
  its sine, which folds at the poles) the coordinate used for interpolation.
  """
  phi = grid.latitudes[: grid.latitude_nodes]
  if not np.all(np.diff(phi) > 0):
    raise ValueError('expected latitudes in increasing order')
  south = -np.pi - phi[HALO_WIDTH - 1 :: -1]
  north = np.pi - phi[: -HALO_WIDTH - 1 : -1]
  extended = np.concatenate([south, phi, north])
  if not np.all(np.diff(extended) > 0):
    # e.g. 'equiangular_with_poles' grids, whose mirrored halo coordinates
    # duplicate the pole nodes and would produce 0/0 Lagrange weights.
    raise ValueError(
        'halo-extended latitudes are not strictly increasing; grids with '
        'nodes at the poles are not supported'
    )
  return extended


def extend_with_pole_halo(field: Array, grid: spherical_harmonic.Grid) -> Array:
  """Appends `HALO_WIDTH` cross-pole halo rows on both latitude ends.

  Halo values are taken from the antipodal longitude (a roll by half the
  longitude nodes), which is exact for even numbers of longitude nodes. Only
  scalar fields (including Cartesian components of tangent vectors) may be
  extended this way; (u, v) wind components would need sign flips. Any nodal
  padding beyond `(longitude_nodes, latitude_nodes)` is stripped first.

  Args:
    field: array of shape [..., *grid.nodal_shape].
    grid: the horizontal grid.

  Returns:
    Array of shape [..., longitude_nodes, latitude_nodes + 2 * HALO_WIDTH].
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
  south = jnp.flip(rolled[..., :HALO_WIDTH], axis=-1)
  north = jnp.flip(rolled[..., -HALO_WIDTH:], axis=-1)
  return jnp.concatenate([south, phys, north], axis=-1)


def _stencil_offsets(order: str) -> np.ndarray:
  """Returns stencil offsets relative to the lower bracketing node."""
  if order == 'linear':
    return np.arange(2)
  elif order == 'cubic':
    return np.arange(-1, 3)
  else:
    raise ValueError(f'unknown interpolation {order=}')


def _searchsorted_right_minus_one(
    nodes_1d: jax.Array, x: jax.Array, is_tpu: bool
) -> jax.Array:
  """`searchsorted(nodes_1d, x, side='right') - 1` without `[..., 1]` tiles."""
  if nodes_1d.shape[0] <= ONE_HOT_LOOKUP_MAX_NODES and (x.ndim <= 2 or is_tpu):
    return (x[..., jnp.newaxis] >= nodes_1d).sum(axis=-1, dtype=jnp.int32) - 1
  return (jnp.searchsorted(nodes_1d, x, side='right') - 1).astype(jnp.int32)


def _tpu_step_diff_one_hot(
    x: jax.Array, thresholds_1d: jax.Array, dtype: jnp.dtype
) -> jax.Array:
  """Builds the exact 1-D cell one-hot from boolean threshold step diffs."""
  num_thresh = thresholds_1d.shape[0]
  if num_thresh == 0:
    return jnp.ones((*x.shape, 1), dtype=dtype)
  ge = x[..., jnp.newaxis] >= thresholds_1d
  if num_thresh == 1:
    oh_bool = jnp.concatenate([~ge, ge], axis=-1)
  else:
    oh_bool = jnp.concatenate(
        [~ge[..., :1], ge[..., :-1] & ~ge[..., 1:], ge[..., -1:]], axis=-1
    )
  return oh_bool.astype(dtype)


class HorizontalStencil(NamedTuple):
  """Horizontal interpolation stencil, with one array per stencil point.

  The per-point tuples have 2 entries for 'linear' and 4 for 'cubic' order;
  every array has the shape of the queries. Keeping the stencil points in
  tuples (rather than a trailing stencil axis) avoids `[..., 4]` arrays.

  Attributes:
    lon_indices: per stencil point, longitude node indices (wrapped).
    lon_weights: per stencil point, interpolation weights along longitude.
    lat_indices: per stencil point, row indices on the halo-extended
      latitude axis.
    lat_weights: per stencil point, interpolation weights along latitude.
    lon_cell: longitude node (wrapped) at which the bracketing cell starts.
    lat_cell: row of the halo-extended latitude axis at which the bracketing
      cell starts.
    extended_latitude_size: size of the halo-extended latitude axis.
    linear_fractions: `(alpha_lon, alpha_lat)`, the fractional position of
      the query in its bracketing cell, or None unless requested.
  """

  lon_indices: tuple[jax.Array, ...]
  lon_weights: tuple[jax.Array, ...]
  lat_indices: tuple[jax.Array, ...]
  lat_weights: tuple[jax.Array, ...]
  lon_cell: jax.Array
  lat_cell: jax.Array
  extended_latitude_size: int
  linear_fractions: tuple[jax.Array, jax.Array] | None


def horizontal_stencil(
    grid: spherical_harmonic.Grid,
    lon: Array,
    sin_lat: Array,
    order: str,
    *,
    include_linear_fractions: bool = False,
    lat_bounds: tuple[Array, Array] | None = None,
) -> HorizontalStencil:
  """Computes the horizontal stencil of the queries `(lon, sin_lat)`.

  `lat_bounds`, if given, is the `(lo, hi)` latitude clamp of the queries
  (from `lat_clamp_bounds`, broadcastable against `lon`): the query
  latitude is clipped to it before the stencil is located, so every result
  (indices, weights, cells, fractions) belongs to the clamped query.
  """
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
  if lat_bounds is not None:
    # Clamped on φ itself (not on sin_lat before the arctan2, whose rounding
    # could move a point on a window edge into the next cell).
    phi = jnp.clip(phi, lat_bounds[0], lat_bounds[1])

  # Longitude nodes are uniform, so the bracketing index is a floor divide.
  # Weights are computed in closed form from the fractional cell coordinate;
  # only gather indices are wrapped.
  n_lon = grid.longitude_nodes
  lon0 = float(grid.nodal_axes[0][0])
  lon_spacing = 2 * np.pi / n_lon
  u_lon = (lon - lon0) / lon_spacing
  lon_cell = jnp.floor(u_lon).astype(jnp.int32)
  alpha_lon = u_lon - lon_cell.astype(u_lon.dtype)
  lon_weights = _uniform_lagrange_weights_tuple(alpha_lon, order)
  lon_indices = tuple((lon_cell + int(off)) % n_lon for off in offsets)

  # Latitude nodes are non-uniform; locate the bracketing ring on the
  # halo-extended axis and compute Lagrange weights on actual node latitudes.
  phi_extended = latitude_halo_axis(grid)
  c_min = -int(offsets[0])
  c_max = phi_extended.size - 1 - int(offsets[-1])
  alpha_lat = None
  is_tpu = targets_tpu(grid)
  if is_tpu and phi_extended.size <= ONE_HOT_LOOKUP_MAX_NODES:
    n_starts = c_max - c_min + 1
    thresholds = jnp.asarray(
        phi_extended[c_min + 1 : c_max + 1], dtype=phi.dtype
    )
    oh_lat = _tpu_step_diff_one_hot(phi, thresholds, phi.dtype)
    dot_prec = jax_numpy_utils.resolve_dot_precision(
        None, oh_lat, oh_lat, lhs_exact_in_bf16=True
    )
    cell_np = np.arange(c_min, c_max + 1, dtype=np.float64)
    if order == 'linear':
      lin_np = np.stack(
          [phi_extended[:-1], 1.0 / np.diff(phi_extended), cell_np], axis=-1
      )
      packed = jnp.dot(
          oh_lat, jnp.asarray(lin_np, dtype=phi.dtype), precision=dot_prec
      )
      frac = (phi - packed[..., 0]) * packed[..., 1]
      lat_cell = jnp.rint(packed[..., 2]).astype(jnp.int32)
      lat_indices = (lat_cell, lat_cell + 1)
      lat_weights = (1.0 - frac, frac)
      if include_linear_fractions:
        alpha_lat = frac
    else:
      nodes_np = [phi_extended[a : a + n_starts] for a in range(4)]
      inv_den_np = [
          1.0 / math.prod(nodes_np[i] - nodes_np[j] for j in range(4) if j != i)
          for i in range(4)
      ]
      cols_np = [*nodes_np, *inv_den_np, cell_np]
      if include_linear_fractions:
        cols_np.append(1.0 / (nodes_np[2] - nodes_np[1]))
      cub_np = np.stack(cols_np, axis=-1)
      packed = jnp.dot(
          oh_lat, jnp.asarray(cub_np, dtype=phi.dtype), precision=dot_prec
      )
      d0 = phi - packed[..., 0]
      d1 = phi - packed[..., 1]
      d2 = phi - packed[..., 2]
      d3 = phi - packed[..., 3]
      lat_weights = (
          d1 * d2 * d3 * packed[..., 4],
          d0 * d2 * d3 * packed[..., 5],
          d0 * d1 * d3 * packed[..., 6],
          d0 * d1 * d2 * packed[..., 7],
      )
      lat_cell = jnp.rint(packed[..., 8]).astype(jnp.int32)
      lat_indices = tuple(lat_cell + int(off) for off in offsets)
      if include_linear_fractions:
        alpha_lat = d1 * packed[..., 9]
  else:
    phi_ext_jax = jnp.asarray(phi_extended, dtype=phi.dtype)
    lat_cell = _searchsorted_right_minus_one(phi_ext_jax, phi, is_tpu)
    # Physical points always satisfy |φ| <= π/2, which keeps cubic stencils
    # within the extended axis; clip for numerical safety only.
    lat_cell = jnp.clip(lat_cell, c_min, c_max)
    lat_indices = tuple(lat_cell + int(off) for off in offsets)
    lat_nodes = tuple(phi_ext_jax[idx] for idx in lat_indices)
    lat_weights = _lagrange_weights_tuple(phi, lat_nodes)
    if include_linear_fractions:
      c0_pos = c_min
      alpha_lat = (phi - lat_nodes[c0_pos]) / (
          lat_nodes[c0_pos + 1] - lat_nodes[c0_pos]
      )

  return HorizontalStencil(
      lon_indices=lon_indices,
      lon_weights=lon_weights,
      lat_indices=lat_indices,
      lat_weights=lat_weights,
      lon_cell=lon_cell % n_lon,
      lat_cell=lat_cell,
      extended_latitude_size=phi_extended.size,
      linear_fractions=(
          (alpha_lon, alpha_lat) if include_linear_fractions else None
      ),
  )


@dataclasses.dataclass
class _StackedHorizontalStencil:
  """Horizontal stencil with the stencil points stacked on a trailing axis.

  The layout of the 2-D gather interpolation, which gathers the whole
  tensor-product stencil at once.

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


def _stacked_horizontal_stencil(
    grid: spherical_harmonic.Grid,
    lon: Array,
    sin_lat: Array,
    order: str,
    lat_bounds: tuple[Array, Array] | None = None,
) -> _StackedHorizontalStencil:
  """Computes gather indices and weights for points (lon, sin_lat)."""
  stencil = horizontal_stencil(grid, lon, sin_lat, order, lat_bounds=lat_bounds)
  return _StackedHorizontalStencil(
      lon_index=jnp.stack(stencil.lon_indices, axis=-1),
      lat_index=jnp.stack(stencil.lat_indices, axis=-1),
      lon_weights=jnp.stack(stencil.lon_weights, axis=-1),
      lat_weights=jnp.stack(stencil.lat_weights, axis=-1),
      extended_latitude_size=stencil.extended_latitude_size,
  )


#  =============================================================================
#  Limiter and gather kernels.
#  =============================================================================

_LIMITERS = ('quasi_monotone',)


def validate_limiter(limiter: str | None) -> None:
  if limiter is not None and limiter not in _LIMITERS:
    raise ValueError(
        f'unknown interpolation {limiter=}; valid values are None (no '
        f'limiting) or one of {_LIMITERS}'
    )


def _linear_cell_slice(order: str) -> slice:
  """Slice selecting the 2-point bracketing cell within a stencil."""
  start = -int(_stencil_offsets(order)[0])
  return slice(start, start + 2)


def cell_bounds_2d(f_buf: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Precomputes 2✕2 bracketing-cell (min, max) on [..., N_lon, N_lat_buf]."""
  f_next_lon = jnp.roll(f_buf, shift=-1, axis=-2)
  lo = jnp.minimum(f_buf, f_next_lon)
  hi = jnp.maximum(f_buf, f_next_lon)
  lo = jnp.minimum(lo[..., :-1], lo[..., 1:])
  hi = jnp.maximum(hi[..., :-1], hi[..., 1:])
  return lo, hi


def cell_bounds_3d(f_buf: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Bracketing-cell (min, max) of 2✕2✕2 on [C, L_buf, N_lon, N_lat_buf]."""
  lo, hi = cell_bounds_2d(f_buf)
  lo = jnp.minimum(lo[:, :-1, :, :], lo[:, 1:, :, :])
  hi = jnp.maximum(hi[:, :-1, :, :], hi[:, 1:, :, :])
  return lo, hi


def _gather_2d_flat_index(stencil: _StackedHorizontalStencil) -> jax.Array:
  return (
      stencil.lon_index[..., :, jnp.newaxis] * stencil.extended_latitude_size
      + stencil.lat_index[..., jnp.newaxis, :]
  )


def _interpolate_horizontal_from_flat_index(
    field_extended: Array,
    flat_index: Array,
    stencil: _StackedHorizontalStencil,
    order: str,
    limiter: str | None,
    platform: str,
) -> jnp.ndarray:
  """Interpolates one [lon, lat_ext] field given precomputed flat indices."""
  values = jnp.take(field_extended.ravel(), flat_index, mode='clip')
  result = _contract_stencil(
      values, stencil.lon_weights, stencil.lat_weights, platform=platform
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


def interpolate_2d_gather(
    fields: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    *,
    order: str,
    limiter: str | None,
    max_lat_shift: int | None,
) -> jnp.ndarray:
  """Gather-based 2-D interpolation of [C, nx, ny] at shared [*points].

  Grid-aligned queries are clamped to the window of `max_lat_shift` (if not
  None), exactly as the 'matmul' method clamps them.
  """
  lat_bounds = (
      lat_clamp_bounds(grid, max_lat_shift, order, sin_lat.dtype)
      if lon.shape == grid.nodal_shape and max_lat_shift is not None
      else None
  )
  stencil = _stacked_horizontal_stencil(
      grid, lon, sin_lat, order, lat_bounds=lat_bounds
  )
  flat_index = _gather_2d_flat_index(stencil)
  extended = extend_with_pole_halo(fields, grid)
  platform = target_platform(grid)
  return jax.vmap(
      lambda ext: _interpolate_horizontal_from_flat_index(
          ext, flat_index, stencil, order, limiter, platform
      )
  )(extended)


class VerticalStencil(NamedTuple):
  """Vertical interpolation stencil, with one array per stencil point.

  The per-point tuples have 2 entries for 'linear' and 4 for 'cubic' order;
  every array has the shape of the queries.

  Attributes:
    indices: per stencil point, level indices.
    weights: per stencil point, interpolation weights.
    bracket_offset: position within the stencil of the level at which the
      bracketing cell starts.
    cell: level at which the bracketing cell starts.
    linear_fraction: fractional position of the query in its bracketing
      cell, or None unless requested.
  """

  indices: tuple[jax.Array, ...]
  weights: tuple[jax.Array, ...]
  bracket_offset: jax.Array
  cell: jax.Array
  linear_fraction: jax.Array | None


def vertical_stencil(
    sigma: Array,
    sigma_nodes: Array,
    vertical_order: Literal['linear', 'cubic'],
    *,
    is_tpu: bool,
    include_linear_fractions: bool = False,
    sigma_bounds: tuple[Array, Array] | None = None,
) -> VerticalStencil:
  """Computes the vertical stencil of the queries `sigma`.

  σ is clipped to the node range (constant extrapolation). `sigma_bounds`, if
  given, is the `(lo, hi)` σ clamp of the queries (from
  `level_clamp_bounds`, broadcastable against `sigma`, in `sigma.dtype`);
  it lies within the node range and is applied in place of that clip.
  `is_tpu` selects the one-hot table lookup used on TPU (`targets_tpu`).
  """
  if vertical_order not in ('linear', 'cubic'):
    raise ValueError(f'unknown {vertical_order=}')
  sigma = jnp.asarray(sigma)
  sigma_nodes = jnp.asarray(sigma_nodes, dtype=sigma.dtype)
  num_levels = sigma_nodes.shape[0]
  if sigma_bounds is None:
    sigma = jnp.clip(sigma, sigma_nodes[0], sigma_nodes[-1])
  else:
    # The window bounds lie within [sigma_nodes[0], sigma_nodes[-1]].
    sigma = jnp.clip(sigma, sigma_bounds[0], sigma_bounds[1])
  if (
      is_tpu
      and (vertical_order == 'linear' or num_levels >= 4)
      and num_levels <= ONE_HOT_LOOKUP_MAX_NODES
  ):
    n_cells = num_levels - 1
    oh_cell = _tpu_step_diff_one_hot(sigma, sigma_nodes[1:-1], sigma.dtype)
    cell_f = jnp.arange(n_cells, dtype=sigma.dtype)
    inv_w = 1.0 / (sigma_nodes[1:] - sigma_nodes[:-1])
    if vertical_order == 'linear':
      lin_tbl = jnp.stack([sigma_nodes[:-1], inv_w, cell_f], axis=-1)
      dot_prec = jax_numpy_utils.resolve_dot_precision(
          None, oh_cell, lin_tbl, lhs_exact_in_bf16=True
      )
      packed = jnp.dot(oh_cell, lin_tbl, precision=dot_prec)
      fraction = (sigma - packed[..., 0]) * packed[..., 1]
      level_cell = jnp.rint(packed[..., 2]).astype(jnp.int32)
      return VerticalStencil(
          indices=(level_cell, level_cell + 1),
          weights=(1.0 - fraction, fraction),
          bracket_offset=jnp.zeros_like(level_cell),
          cell=level_cell,
          linear_fraction=fraction if include_linear_fractions else None,
      )
    nodes_cols = [
        jnp.concatenate(
            [
                sigma_nodes[a : a + 1],
                sigma_nodes[a : num_levels - 3 + a],
                sigma_nodes[num_levels - 4 + a : num_levels - 3 + a],
            ],
            axis=0,
        )
        for a in range(4)
    ]
    inv_denoms = [
        1.0
        / math.prod(nodes_cols[i] - nodes_cols[j] for j in range(4) if j != i)
        for i in range(4)
    ]
    cub_tbl = jnp.stack(
        [*nodes_cols, *inv_denoms, cell_f, sigma_nodes[:-1], inv_w], axis=-1
    )
    dot_prec = jax_numpy_utils.resolve_dot_precision(
        None, oh_cell, cub_tbl, lhs_exact_in_bf16=True
    )
    packed = jnp.dot(oh_cell, cub_tbl, precision=dot_prec)
    d0 = sigma - packed[..., 0]
    d1 = sigma - packed[..., 1]
    d2 = sigma - packed[..., 2]
    d3 = sigma - packed[..., 3]
    cubic_weights = (
        d1 * d2 * d3 * packed[..., 4],
        d0 * d2 * d3 * packed[..., 5],
        d0 * d1 * d3 * packed[..., 6],
        d0 * d1 * d2 * packed[..., 7],
    )
    level_cell = jnp.rint(packed[..., 8]).astype(jnp.int32)
    start = jnp.clip(level_cell - 1, 0, num_levels - 4)
    level_indices = tuple(start + a for a in range(4))
    fraction = (sigma - packed[..., 9]) * packed[..., 10]
    bracket_offset = level_cell - start
    boundary_cell = (level_cell == 0) | (level_cell == num_levels - 2)
    level_weights = tuple(
        jnp.where(
            boundary_cell,
            jnp.where(bracket_offset == a, 1.0 - fraction, 0.0)
            + jnp.where(bracket_offset + 1 == a, fraction, 0.0),
            cubic_weights[a],
        )
        for a in range(4)
    )
    return VerticalStencil(
        indices=level_indices,
        weights=level_weights,
        bracket_offset=bracket_offset,
        cell=level_cell,
        linear_fraction=fraction if include_linear_fractions else None,
    )

  level_cell = _searchsorted_right_minus_one(sigma_nodes, sigma, is_tpu)
  level_cell = jnp.clip(level_cell, 0, num_levels - 2)
  below = sigma_nodes[level_cell]
  above = sigma_nodes[level_cell + 1]
  fraction = (sigma - below) / (above - below)
  if vertical_order == 'linear':
    return VerticalStencil(
        indices=(level_cell, level_cell + 1),
        weights=(1 - fraction, fraction),
        bracket_offset=jnp.zeros_like(level_cell),
        cell=level_cell,
        linear_fraction=fraction if include_linear_fractions else None,
    )
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
  return VerticalStencil(
      indices=level_indices,
      weights=level_weights,
      bracket_offset=bracket_offset,
      cell=level_cell,
      linear_fraction=fraction if include_linear_fractions else None,
  )


def interpolate_3d_gather(
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
    limited_channel_start: int = 0,
    trilinear_channel_start: int | None = None,
    lat_bounds: tuple[Array, Array] | None = None,
    sigma_bounds: tuple[Array, Array] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Gather-based 3-D interpolation on channel-minor layout sharing one stencil.

  Packs channels into the contiguous minor axis (`[N_grid, C]`) so all channels
  at each stencil corner share a single cache line, contracts the 3-D Lagrange
  weights in a cascaded reduction without materializing `[..., 4, 4, 4]` index
  or value tensors, and extracts both quasi-monotone cell bounds and optional
  trilinear values (`trilinear_channel_start`) from the already-gathered
  interior `2x2x2` corners with zero extra gathers. `lat_bounds` and
  `sigma_bounds` optionally clamp the queries (see `window_clamp_bounds_3d`);
  all outputs then belong to the clamped queries.

  Returns `(values, trilinear)`: the interpolated channels, [C, *queries], and
  the trilinear values of the channels from `trilinear_channel_start` on
  (None if `trilinear_channel_start` is None).
  """
  c_total = fields_4d.shape[0]
  eff_lim_start = (
      min(c_total, max(0, limited_channel_start))
      if limiter is not None
      else c_total
  )
  has_lim = limiter is not None and eff_lim_start < c_total
  if has_lim and limiter != 'quasi_monotone':
    raise ValueError(f'unimplemented interpolation {limiter=}')

  need_tri = trilinear_channel_start is not None
  eff_tri_start = (
      min(c_total, max(0, trilinear_channel_start)) if need_tri else c_total
  )
  c_tri = c_total - eff_tri_start
  reuse_raw_for_tri = (
      need_tri and order == 'linear' and vertical_order == 'linear'
  )
  need_tri_box = need_tri and c_tri > 0 and not reuse_raw_for_tri

  h_stencil = horizontal_stencil(
      grid,
      lon,
      sin_lat,
      order,
      include_linear_fractions=need_tri_box,
      lat_bounds=lat_bounds,
  )
  lon_idx, lon_w = h_stencil.lon_indices, h_stencil.lon_weights
  lat_idx, lat_w = h_stencil.lat_indices, h_stencil.lat_weights
  ext_size = h_stencil.extended_latitude_size
  a_lon, a_lat = h_stencil.linear_fractions or (None, None)
  lvl_idx, lvl_w, b_off, _, f_lvl = vertical_stencil(
      sigma,
      jnp.asarray(sigma_nodes),
      vertical_order,
      is_tpu=targets_tpu(grid),
      include_linear_fractions=need_tri_box,
      sigma_bounds=sigma_bounds,
  )

  extended = extend_with_pole_halo(fields_4d, grid)
  dtype = extended.dtype
  ext_c = jnp.moveaxis(extended, 0, -1).reshape(-1, c_total)
  n_lon = grid.longitude_nodes
  q_shape = lon.shape

  ref_levels = None
  ref_at_dep = None
  if has_lim and limiter_bound_reference is not None:
    ref_arr = jnp.asarray(limiter_bound_reference, dtype=dtype)
    ref_centered = ref_arr - jnp.mean(ref_arr)
    ref_levels = [ref_centered[zi][..., jnp.newaxis] for zi in lvl_idx]
    ref_at_dep = sum(
        lw_a * ref_centered[zi] for zi, lw_a in zip(lvl_idx, lvl_w)
    )[..., jnp.newaxis]

  s_lvl = len(lvl_idx)
  s_lon = len(lon_idx)
  s_lat = len(lat_idx)
  c_cell0 = 0 if order == 'linear' else 1
  need_box = has_lim or need_tri_box
  box_start = (
      min(
          eff_lim_start if has_lim else c_total,
          eff_tri_start if need_tri_box else c_total,
      )
      if need_box
      else c_total
  )
  lim_rel = eff_lim_start - box_start
  tri_rel = eff_tri_start - box_start
  c_lim = c_total - eff_lim_start

  stride_z = n_lon * ext_size
  x_base = [xi * ext_size for xi in lon_idx]
  lat_w_exp = [w[..., jnp.newaxis] for w in lat_w]
  lon_w_exp = [w[..., jnp.newaxis] for w in lon_w]
  if need_tri_box:
    w_xl0 = (1.0 - a_lon)[..., jnp.newaxis]
    w_xl1 = a_lon[..., jnp.newaxis]
    w_yl0 = (1.0 - a_lat)[..., jnp.newaxis]
    w_yl1 = a_lat[..., jnp.newaxis]

  def _eval_level(za, ref_a):
    acc_x = 0.0
    cell_rows = []
    for bi in range(s_lon):
      zxb = za + x_base[bi]
      acc_y = 0.0
      row_c = []
      in_b = need_box and bi in (c_cell0, c_cell0 + 1)
      for ci in range(s_lat):
        v_abc = jnp.take(ext_c, zxb + lat_idx[ci], axis=0, mode='clip')
        acc_y = acc_y + v_abc * lat_w_exp[ci]
        if in_b and ci in (c_cell0, c_cell0 + 1):
          row_c.append(v_abc[..., box_start:])
      acc_x = acc_x + acc_y * lon_w_exp[bi]
      if in_b:
        cell_rows.append(row_c)

    lo_a = hi_a = tri_a = None
    if need_box:
      c00, c01 = cell_rows[0]
      c10, c11 = cell_rows[1]
      if has_lim:
        l00, l01 = c00[..., lim_rel:], c01[..., lim_rel:]
        l10, l11 = c10[..., lim_rel:], c11[..., lim_rel:]
        lo_a = jnp.minimum(jnp.minimum(l00, l01), jnp.minimum(l10, l11))
        hi_a = jnp.maximum(jnp.maximum(l00, l01), jnp.maximum(l10, l11))
        if ref_a is not None:
          lo_a = lo_a + ref_a
          hi_a = hi_a + ref_a
      if need_tri_box:
        t00, t01 = c00[..., tri_rel:], c01[..., tri_rel:]
        t10, t11 = c10[..., tri_rel:], c11[..., tri_rel:]
        t0 = t00 * w_yl0 + t01 * w_yl1
        t1 = t10 * w_yl0 + t11 * w_yl1
        tri_a = t0 * w_xl0 + t1 * w_xl1
    return acc_x, lo_a, hi_a, tri_a

  if s_lvl * s_lon * s_lat <= 32:
    acc_z = 0.0
    lo_list = []
    hi_list = []
    tri_list = []
    for a in range(s_lvl):
      ref_a = ref_levels[a] if ref_levels is not None else None
      acc_x, lo_a, hi_a, tri_a = _eval_level(lvl_idx[a] * stride_z, ref_a)
      acc_z = acc_z + acc_x * lvl_w[a][..., jnp.newaxis]
      if has_lim:
        lo_list.append(lo_a)
        hi_list.append(hi_a)
      if need_tri_box:
        tri_list.append(tri_a)
    if s_lvl == 2:
      if has_lim:
        lo = jnp.minimum(lo_list[0], lo_list[1])
        hi = jnp.maximum(hi_list[0], hi_list[1])
      if need_tri_box:
        tri = (
            tri_list[0] * (1.0 - f_lvl)[..., jnp.newaxis]
            + tri_list[1] * f_lvl[..., jnp.newaxis]
        )
    else:
      is0 = (b_off == 0)[..., jnp.newaxis]
      is1 = (b_off == 1)[..., jnp.newaxis]
      if has_lim:
        lo0 = jnp.where(is0, lo_list[0], jnp.where(is1, lo_list[1], lo_list[2]))
        lo1 = jnp.where(is0, lo_list[1], jnp.where(is1, lo_list[2], lo_list[3]))
        hi0 = jnp.where(is0, hi_list[0], jnp.where(is1, hi_list[1], hi_list[2]))
        hi1 = jnp.where(is0, hi_list[1], jnp.where(is1, hi_list[2], hi_list[3]))
        lo = jnp.minimum(lo0, lo1)
        hi = jnp.maximum(hi0, hi1)
      if need_tri_box:
        tr0 = jnp.where(
            is0, tri_list[0], jnp.where(is1, tri_list[1], tri_list[2])
        )
        tr1 = jnp.where(
            is0, tri_list[1], jnp.where(is1, tri_list[2], tri_list[3])
        )
        tri = (
            tr0 * (1.0 - f_lvl)[..., jnp.newaxis]
            + tr1 * f_lvl[..., jnp.newaxis]
        )
  else:
    z_stack = jnp.stack([zi * stride_z for zi in lvl_idx], axis=0)
    lw_stack = jnp.stack(lvl_w, axis=0)[..., jnp.newaxis]
    ref_stack = (
        jnp.stack(ref_levels, axis=0) if ref_levels is not None else None
    )

    def _scan_level(carry, x_a):
      acc_cur, lo_cur, hi_cur, tri_cur = carry
      a, za, lw_a, ref_a = x_a
      acc_x, lo_a, hi_a, tri_a = _eval_level(za, ref_a)
      acc_cur = acc_cur + acc_x * lw_a
      if has_lim:
        is_first = (a == b_off)[..., jnp.newaxis]
        is_second = (a == b_off + 1)[..., jnp.newaxis]
        lo_cur = jnp.where(
            is_first,
            lo_a,
            jnp.where(is_second, jnp.minimum(lo_cur, lo_a), lo_cur),
        )
        hi_cur = jnp.where(
            is_first,
            hi_a,
            jnp.where(is_second, jnp.maximum(hi_cur, hi_a), hi_cur),
        )
      if need_tri_box:
        w_tri_a = jnp.where(
            a == b_off,
            1 - f_lvl,
            jnp.where(a == b_off + 1, f_lvl, jnp.zeros_like(f_lvl)),
        )[..., jnp.newaxis]
        tri_cur = tri_cur + tri_a * w_tri_a
      return (acc_cur, lo_cur, hi_cur, tri_cur), None

    acc_dtype = jnp.result_type(dtype, lon_w[0], lat_w[0], lvl_w[0])
    lo_dtype = (
        jnp.result_type(dtype, ref_levels[0])
        if ref_levels is not None
        else dtype
    )
    tri_dtype = (
        jnp.result_type(dtype, w_xl0, w_yl0, f_lvl) if need_tri_box else dtype
    )
    init_carry = (
        jnp.zeros((*q_shape, c_total), dtype=acc_dtype),
        jnp.zeros((*q_shape, c_lim), dtype=lo_dtype) if has_lim else None,
        jnp.zeros((*q_shape, c_lim), dtype=lo_dtype) if has_lim else None,
        jnp.zeros((*q_shape, c_tri), dtype=tri_dtype) if need_tri_box else None,
    )
    (acc_z, lo, hi, tri), _ = lax.scan(
        _scan_level,
        init_carry,
        (jnp.arange(s_lvl, dtype=jnp.int32), z_stack, lw_stack, ref_stack),
    )

  out_tri = None
  if reuse_raw_for_tri:
    out_tri = jnp.moveaxis(acc_z[..., eff_tri_start:], -1, 0)
  elif need_tri:
    out_tri = (
        jnp.moveaxis(tri, -1, 0)
        if c_tri > 0
        else jnp.zeros((0, *q_shape), dtype=dtype)
    )

  if has_lim:
    lim_vals = acc_z[..., eff_lim_start:]
    if ref_at_dep is not None:
      lim_vals = lim_vals + ref_at_dep
    lim_vals = jnp.clip(lim_vals, lo, hi)
    if ref_at_dep is not None:
      lim_vals = lim_vals - ref_at_dep
    out_c = (
        lim_vals
        if eff_lim_start == 0
        else jnp.concatenate([acc_z[..., :eff_lim_start], lim_vals], axis=-1)
    )
  else:
    out_c = acc_z

  return jnp.moveaxis(out_c, -1, 0), out_tri


def wind_sigma_dot_clamp_bounds(
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    sigma: jax.Array,
    centers: Array,
    boundaries: Array,
    max_lat_shift: int,
    max_vertical_shift: int,
) -> tuple[tuple[Array, Array], tuple[Array, Array], tuple[Array, Array]]:
  """Clamps of the fused (bilinear) `wind` + `sigma_dot` midpoint queries.

  Returns the latitude bounds ([nodal_shape[1]] each) and the σ bounds
  ([layers] each) on the `centers` and on the `boundaries` nodes; the two σ
  windows have the same half-width in cells of their own nodes.
  """
  num_levels = lon.shape[0]
  return (
      lat_clamp_bounds(grid, max_lat_shift, 'linear', sin_lat.dtype),
      level_clamp_bounds(centers, num_levels, max_vertical_shift, sigma.dtype),
      level_clamp_bounds(
          boundaries, num_levels, max_vertical_shift, sigma.dtype
      ),
  )


def interpolate_3d_wind_and_sigma_dot_gather(
    wind: jax.Array,
    sigma_dot: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    sigma: jax.Array,
    centers: Array,
    boundaries: Array,
    *,
    max_lat_shift: int | None,
    max_vertical_shift: int | None,
) -> tuple[jax.Array, jax.Array]:
  """Fused trilinear gather of `wind` and `sigma_dot` sharing one stencil.

  With both shifts given, the queries are clamped to the same windows as in
  `semi_lagrangian_matmul.interpolate_3d_wind_and_sigma_dot`.
  """
  lat_bounds = centers_bounds = boundaries_bounds = None
  if max_lat_shift is not None and max_vertical_shift is not None:
    lat_bounds, centers_bounds, boundaries_bounds = wind_sigma_dot_clamp_bounds(
        grid,
        lon,
        sin_lat,
        sigma,
        centers,
        boundaries,
        max_lat_shift,
        max_vertical_shift,
    )
    centers_bounds, boundaries_bounds = (
        tuple(b[:, jnp.newaxis, jnp.newaxis] for b in bounds)
        for bounds in (centers_bounds, boundaries_bounds)
    )
  h_stencil = horizontal_stencil(
      grid, lon, sin_lat, 'linear', lat_bounds=lat_bounds
  )
  lon_idx, lon_w = h_stencil.lon_indices, h_stencil.lon_weights
  lat_idx, lat_w = h_stencil.lat_indices, h_stencil.lat_weights
  ext_size = h_stencil.extended_latitude_size
  is_tpu = targets_tpu(grid)
  lvl_c_idx, lvl_c_w, _, _, _ = vertical_stencil(
      sigma,
      jnp.asarray(centers),
      'linear',
      is_tpu=is_tpu,
      sigma_bounds=centers_bounds,
  )
  lvl_b_idx, lvl_b_w, _, _, _ = vertical_stencil(
      sigma,
      jnp.asarray(boundaries),
      'linear',
      is_tpu=is_tpu,
      sigma_bounds=boundaries_bounds,
  )
  wind_ext = extend_with_pole_halo(wind, grid)
  wind_c = jnp.moveaxis(wind_ext, 0, -1).reshape(-1, wind.shape[0])
  sd_flat = extend_with_pole_halo(sigma_dot, grid).reshape(-1)
  stride_z = grid.longitude_nodes * ext_size
  x_base = [xi * ext_size for xi in lon_idx]
  lat_w_exp = [w[..., jnp.newaxis] for w in lat_w]
  lon_w_exp = [w[..., jnp.newaxis] for w in lon_w]

  acc_w = 0.0
  acc_sd = 0.0
  for a in range(2):
    zc_a = lvl_c_idx[a] * stride_z
    zb_a = lvl_b_idx[a] * stride_z
    wx = 0.0
    sx = 0.0
    for b in range(2):
      zxcb = zc_a + x_base[b]
      zxbb = zb_a + x_base[b]
      wy = 0.0
      sy = 0.0
      for c in range(2):
        wy = (
            wy
            + jnp.take(wind_c, zxcb + lat_idx[c], axis=0, mode='clip')
            * lat_w_exp[c]
        )
        sy = (
            sy
            + jnp.take(sd_flat, zxbb + lat_idx[c], axis=0, mode='clip')
            * lat_w[c]
        )
      wx = wx + wy * lon_w_exp[b]
      sx = sx + sy * lon_w[b]
    acc_w = acc_w + wx * lvl_c_w[a][..., jnp.newaxis]
    acc_sd = acc_sd + sx * lvl_b_w[a]
  return jnp.moveaxis(acc_w, -1, 0), acc_sd
