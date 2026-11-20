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

"""Matmul implementation of semi-Lagrangian interpolation (TPU and SPMD).

Interpolates nodal fields at departure points without indexed gathers or
scatters: stencil values are selected with one-hot matrix products inside a
fixed window around each query's arrival point. This is the form that runs
efficiently on TPU matrix units and that shards over an SPMD mesh with axes
'z', 'x', 'y' using nearest-neighbour halo exchanges.

Contents, in order: the window arithmetic; the halo exchanges that build each
shard's source buffer; the 2-D kernels; the planners that size channel chunks
and level blocks to the accelerator's memory; the 3-D kernel; the fused wind
+ σ̇ kernel of the departure-point iteration.

This is the implementation of `method='matmul'` of the public API in
`semi_lagrangian` and may change without notice. The stencils and the window
clamp are shared with the gather method and live in `lagrange_interpolation`.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Literal, NamedTuple

from dinosaur import jax_numpy_utils
from dinosaur import lagrange_interpolation
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

Array = typing.Array
P = jax.sharding.PartitionSpec

# Tuning constants of the 'matmul' interpolation. All are empirical settings
# chosen on TPU v5e (128 MiB on-chip VMEM per core) unless stated otherwise;
# none is derived from a model of the hardware, and changing one changes the
# compiled program (not the interpolated values).
_MIB = 1024 * 1024

# Queries evaluated per `lax.scan` step by the arbitrary-point (unwindowed)
# matmul paths; bounds the [queries, nodes] one-hot matrices.
_POINTS_QUERY_CHUNK = 1024

# Channels per `lax.scan` step of the windowed 2-D matmul (the limited variant
# also carries a lower and an upper bound row per channel).
_MATMUL_2D_CHANNEL_CHUNK = 8
_MATMUL_2D_CHANNEL_CHUNK_LIMITED = 4

# Budgets that size the channel chunk of the windowed 3-D matmul. In bytes of
# (8, 128)-tile-padded buffers, per channel chunk: `single_buf` bounds the
# largest single buffer and `ws_buf` the sum of the hoisted buffers alive at
# once (accelerator memory); `vmem` bounds the per-block working set of the
# contraction, relative to TPU v5e's 128 MiB of on-chip VMEM. The non-TPU
# `vmem` is not a hardware limit. `max_channels` caps the chunk.
# TPU values:
_MATMUL_3D_MAX_CHANNELS_TPU = 12
_MATMUL_3D_SINGLE_BUF_LIMIT_TPU = 450 * _MIB
_MATMUL_3D_WS_BUF_LIMIT_TPU = 900 * _MIB
_MATMUL_3D_VMEM_LIMIT_TPU = 112 * _MIB
# CPU/GPU values:
_MATMUL_3D_MAX_CHANNELS_CPU_GPU = 8
_MATMUL_3D_SINGLE_BUF_LIMIT_CPU_GPU = 185 * _MIB
_MATMUL_3D_WS_BUF_LIMIT_CPU_GPU = 440 * _MIB
_MATMUL_3D_VMEM_LIMIT_CPU_GPU = 256 * _MIB

# The planner constants below were chosen from sweeps of the channel chunk and
# the block length on TPU v5e: T63, T85, T106 and T170 at L47, windows of 2
# and 6, limited and unlimited stacks.

# Number of consecutive query levels evaluated per scan block on TPU, when
# blocks are used (see `_plan_blocks_and_channels`); 3 and 4 were slower than
# 2 wherever they were measured.
_TPU_BLOCK_LEVELS = 2

# Cost per level of a block relative to single levels, lane padding aside:
# the block's levels share a vertical window only one level taller than a
# single level's. Measured 0.94 to 0.96 at T85, where the padding is the same
# either way. The padding is the larger effect where it differs: the queries
# of a block's levels lie side by side on the lane axis, so two levels of 192
# (T63) or 320 (T106) longitudes fill 128-lane tiles exactly, where one level
# leaves a third or a fifth of them empty.
_TPU_BLOCK_COST = 0.95

# On-chip memory (VMEM) per core of TPU v5e and v6e. Blocks are only
# considered if their longitude weights fit.
_TPU_VMEM_BYTES = 128 * _MIB

# TPU budget of the part of a block's working set that grows with the channel
# chunk: the vertical window patch and its longitude-contracted result. The
# fastest chunk was the largest one within about this size in every measured
# configuration, including where the whole working set (with the longitude
# weights, or with the limiter's bounds) exceeds VMEM.
_MATMUL_3D_CHUNK_VMEM_TPU = 100 * _MIB

# Fixed cost of one more channel chunk (its stencils and weight matrices are
# built again), relative to the cost of one channel: about a half in the
# per-chunk timings at T63 and T106. Used to trade chunks against padding,
# and blocks (which need smaller chunks) against single levels.
_CHUNK_OVERHEAD_IN_CHANNELS = 0.5

# Largest working set of a block of the fused wind + sigma_dot kernel for
# which it uses blocks of `_TPU_BLOCK_LEVELS` levels.
_WIND_SIGMA_DOT_BLOCK_VMEM_PROBE = 118 * _MIB

# TPU (8, 128) tiling of 32-bit arrays: the minor axis is padded to a multiple
# of 128 lanes and the second-minor axis to a multiple of 8 sublanes. Buffer
# footprint estimates and the K axis of the one-hot products use these.
_TPU_LANES = 128
_TPU_SUBLANES = 8


#  =============================================================================
#  Window arithmetic.
#  =============================================================================


def _round_to_multiple(x: int, multiple: int) -> int:
  return multiple * math.ceil(x / multiple)


def _window_size(
    n_buffer: int, n_source: int, half_width: int, stencil: int
) -> int:
  """Number of rows (levels) in a stencil window.

  The window spans `2 * half_width + stencil` consecutive source nodes: the
  `stencil` nodes of a query whose departure point equals its arrival point,
  plus `half_width` nodes on either side for the displacement. It is
  truncated to the `n_source` source nodes and to the `n_buffer` nodes
  available locally (they differ when the axis is sharded and halo-extended).
  """
  return min(n_buffer, min(n_source, 2 * half_width + stencil))


def _arrival_lat_rows(
    j_offset: int | jax.Array, ny_loc: int, n_lat: int
) -> jax.Array:
  """Global latitude rows of the `ny_loc` local arrival rows, [ny_loc] int32.

  Rows that only exist as padding of the last shard are clipped to the last
  physical row `n_lat - 1`.
  """
  return jnp.minimum(j_offset + jnp.arange(ny_loc, dtype=jnp.int32), n_lat - 1)


def _buffer_window_start(
    win_start: jax.Array,
    buf_origin: int | jax.Array,
    n_buffer: int,
    size: int,
) -> jax.Array:
  """Start of a `size`-node window within the local buffer.

  `win_start` indexes the full source axis; the buffer holds source nodes
  `buf_origin ... buf_origin + n_buffer - 1`. The start is clipped so the
  window lies inside the buffer.
  """
  return jnp.clip(win_start - buf_origin, 0, n_buffer - size)


def _lat_window_starts(
    j_global: jax.Array,
    first_row: jax.Array,
    n_source: int,
    n_buffer: int,
    size: int,
    buf_origin: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Start rows of the per-arrival-row latitude windows, each [ny_loc] int32.

  The window of arrival row `j` covers the `size` consecutive rows
  `win_start[j] ... win_start[j] + size - 1` of an axis of `n_source` rows.
  `first_row[j]` is the unclipped first row the caller wants covered (for
  the stencil windows of `_window_size`: the first stencil row of a query
  with zero displacement, minus the half-width). `win_start` is `first_row`
  clipped to `[0, n_source - size]`, so near the ends of the axis the window
  stops sliding and covers the first or last `size` rows; when the window is
  the whole axis it is 0 and `first_row` is not used. `buf_start` is the same
  start expressed within the local buffer (see `_buffer_window_start`).

  Args:
    j_global: global arrival rows, from `_arrival_lat_rows`.
    first_row: unclipped first row of each window, same shape as `j_global`.
    n_source: number of rows of the (halo-extended) source axis.
    n_buffer: number of rows held in the local buffer.
    size: window size from `_window_size`.
    buf_origin: source row of the first row of the local buffer.

  Returns:
    `(win_start, buf_start)`.
  """
  if size == n_source:
    win_start = jnp.zeros_like(j_global)
  else:
    win_start = jnp.clip(first_row, 0, n_source - size)
  return win_start, _buffer_window_start(win_start, buf_origin, n_buffer, size)


def _z_window_starts(
    first_level: jax.Array,
    n_source: int,
    size: int,
    buf_origin: int | jax.Array = 0,
    n_buffer: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Start level of a vertical window, on the source axis and in the buffer.

  The window covers the `size` consecutive source levels
  `win_start ... win_start + size - 1` out of `n_source`. `first_level` is
  the unclipped first level the caller wants covered (the first stencil level
  of a query with zero displacement, minus the half-width).

  The local buffer holds source levels
  `buf_origin ... buf_origin + n_buffer - 1`. On the first and the last
  z-shard that range extends beyond `[0, n_source)`: those buffer levels are
  the halo the cyclic exchange of `_gather_z_with_halo` filled from the
  opposite end of the column, and no window may include them. The start is
  therefore `first_level` clipped so that the window lies inside
  `[max(0, buf_origin), min(n_source, buf_origin + n_buffer))`, the source
  levels the buffer really holds. Near either end of that range the window
  stops sliding and covers its first or last `size` levels, which is where
  the queries' (clamped) stencils then are. `size` must not exceed the length
  of the range on any shard (`_Matmul3DPlan.l_valid`).

  Args:
    first_level: unclipped first level of the window, int32 scalar.
    n_source: number of levels (or cells) of the source axis.
    size: number of levels (or cells) in the window.
    buf_origin: source level of the first level of the local buffer.
    n_buffer: levels (or cells) in the local buffer; `n_source` if None.

  Returns:
    `(win_start, buf_start)`: the start on the source axis and the same start
    within the buffer, `win_start - buf_origin`.
  """
  if n_buffer is None:
    n_buffer = n_source
  if isinstance(buf_origin, int):
    lo = max(buf_origin, 0)
    hi = min(buf_origin + n_buffer, n_source)
  else:
    lo = jnp.maximum(buf_origin, 0)
    hi = jnp.minimum(buf_origin + n_buffer, n_source)
  if size == n_source:
    win_start = jnp.int32(0)
  else:
    win_start = jnp.clip(first_level, lo, hi - size)
  if isinstance(buf_origin, int) and buf_origin == 0:
    return win_start, win_start
  return win_start, win_start - buf_origin


def _weighted_one_hot(
    indices: tuple[jax.Array, ...],
    weights: tuple[jax.Array, ...],
    num_classes: int,
    dtype: jnp.dtype,
) -> jax.Array:
  """Matrix of stencil weights placed at their node indices.

  Args:
    indices: per stencil point, integer node indices of shape [q].
    weights: per stencil point, interpolation weights of shape [q].
    num_classes: number of nodes.
    dtype: dtype of the result.

  Returns:
    [q, num_classes] matrix whose row `i` holds `weights[k][i]` at column
    `indices[k][i]`, summed over the stencil points `k` in order.
  """
  return sum(
      w[:, jnp.newaxis] * jax.nn.one_hot(idx, num_classes, dtype=dtype)
      for idx, w in zip(indices, weights)
  )


def _weighted_window_one_hot(
    indices: tuple[jax.Array, ...],
    weights: tuple[jax.Array, ...],
    window_arange: jax.Array,
    dtype: jnp.dtype,
    start: jax.Array | None = None,
) -> jax.Array:
  """Stencil weights placed at their node indices within a window.

  Args:
    indices: per stencil point, integer node indices of shape [ny, nx].
    weights: per stencil point, interpolation weights of shape [ny, nx].
    window_arange: [w, 1] window positions `0 ... w - 1`.
    dtype: dtype of the result.
    start: optional window start, subtracted from the indices to give window
      positions: a scalar, or [ny] with one start per row; None when the
      indices already are window positions.

  Returns:
    [ny, w, nx] array whose entry `[j, m, i]` is the sum over stencil points
    `k`, in order, of `weights[k][j, i]` where `indices[k][j, i] - start == m`
    (the comparison is carried out in the index dtype, then cast).
  """
  per_row_start = start is not None and jnp.ndim(start) == 1
  # The per-row start is expanded inside the loop, once per stencil point.
  return sum(
      w[:, jnp.newaxis, :]
      * (
          (
              idx
              if start is None
              else idx - (start[:, jnp.newaxis] if per_row_start else start)
          )[:, jnp.newaxis, :]
          == window_arange
      ).astype(dtype)
      for idx, w in zip(indices, weights)
  )


#  =============================================================================
#  SPMD mesh and halo exchange.
#  =============================================================================


def spatial_mesh(
    grid: spherical_harmonic.Grid,
) -> jax.sharding.Mesh | None:
  """Returns the grid's mesh if it shards a spatial axis, else None.

  A mesh with every spatial axis of size 1 (e.g. one that only has a 'batch'
  or 'ensemble' axis to spread members over devices) does not shard the
  interpolation, which then runs as on a single device.
  """
  mesh = grid.spmd_mesh
  return mesh if spherical_harmonic.has_spatial_sharding(mesh) else None


def _shard_map(f, mesh: jax.sharding.Mesh, in_specs, out_specs):
  """`jax.shard_map` of `f`, manual over the spatial axes of `mesh` only.

  Any other axis of the mesh (e.g. 'batch') is left to automatic sharding,
  so `f` sees it as an ordinary array axis.
  """
  return jax.shard_map(
      f,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=out_specs,
      axis_names=frozenset('zxy') & frozenset(mesh.axis_names),
  )


def _take_1d(values_1d: jax.Array, idx: jax.Array, is_tpu: bool) -> jax.Array:
  """1-D table lookup: a one-hot MXU dot on TPU, an indexed read elsewhere."""
  if (
      is_tpu
      and values_1d.shape[0] <= lagrange_interpolation.ONE_HOT_LOOKUP_MAX_NODES
  ):
    oh = jax.nn.one_hot(idx, values_1d.shape[0], dtype=values_1d.dtype)
    dot_prec = jax_numpy_utils.resolve_dot_precision(
        None, oh, values_1d, lhs_exact_in_bf16=True
    )
    return jnp.dot(oh, values_1d, precision=dot_prec)
  return values_1d[idx]


def _resolve_rhs_binary_dot_precision(
    lhs: jax.Array, rhs: jax.Array, is_tpu: bool
) -> jax.lax.PrecisionLike:
  """Precision of `dot(lhs, rhs)` when `rhs` is an exact 0/1 matrix."""
  if is_tpu and jnp.result_type(lhs, rhs) == jnp.float32:
    return ('highest', 'bfloat16')
  return jax_numpy_utils.resolve_dot_precision(None, lhs, rhs)


def _select_last_axis(
    values: jax.Array, selector: jax.Array, is_tpu: bool
) -> jax.Array:
  """Contracts the last axis of `values` with an exact 0/1 `selector`.

  The selector is the right operand on purpose. On TPU it runs at bfloat16
  operand precision, which is exact for 0/1 entries, while `values` keep full
  float32 precision. JAX transposes a dot product by putting the cotangent in
  the left slot under the same per-operand precision, so with the selector on
  the left (`('bfloat16', 'highest')`) the reverse-mode gradient with respect
  to `values` is rounded to bfloat16.

  Args:
    values: [..., n] array.
    selector: [m, n] matrix of zeros and ones.
    is_tpu: whether the computation targets TPUs.

  Returns:
    [..., m], `values[..., newaxis, :] * selector` summed over the last axis.
  """
  return lax.dot_general(
      values,
      selector,
      (((values.ndim - 1,), (1,)), ((), ())),
      precision=_resolve_rhs_binary_dot_precision(values, selector, is_tpu),
  )


def _gather_lat_with_halo(
    f_x: jax.Array,
    grid: spherical_harmonic.Grid,
    max_lat_shift: int,
) -> tuple[jax.Array, int | jax.Array, int | jax.Array]:
  """Builds local latitude buffer [..., n_lon, n_lat_buf] with pole halos."""
  n_lon = grid.longitude_nodes
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
    return lagrange_interpolation.extend_with_pole_halo(f_x, grid), 0, 0

  iy = lax.axis_index('y')
  ny_loc = f_x.shape[-1]
  pad_y = grid.nodal_padding[1]
  # The windows use `effective_window_shift`, so the halo must too.
  h_lat = (
      lagrange_interpolation.effective_window_shift(max_lat_shift)
      + lagrange_interpolation.HALO_WIDTH
  )
  if pad_y == 0 and ny_loc >= h_lat:
    perm_to_north = [(s, (s + 1) % y_shards) for s in range(y_shards)]
    perm_to_south = [(s, (s - 1) % y_shards) for s in range(y_shards)]
    recv_left = lax.ppermute(f_x[..., -h_lat:], 'y', perm=perm_to_north)
    recv_right = lax.ppermute(f_x[..., :h_lat], 'y', perm=perm_to_south)

    south_pole = jnp.roll(
        jnp.flip(f_x[..., : lagrange_interpolation.HALO_WIDTH], axis=-1),
        shift=half_turn,
        axis=-2,
    )
    north_pole = jnp.roll(
        jnp.flip(f_x[..., -lagrange_interpolation.HALO_WIDTH :], axis=-1),
        shift=half_turn,
        axis=-2,
    )
    zeros_pad = jnp.zeros(
        f_x.shape[:-1] + (h_lat - lagrange_interpolation.HALO_WIDTH,),
        dtype=f_x.dtype,
    )
    pole_left = jnp.concatenate([zeros_pad, south_pole], axis=-1)
    pole_right = jnp.concatenate([north_pole, zeros_pad], axis=-1)

    left_halo = jnp.where(iy == 0, pole_left, recv_left)
    right_halo = jnp.where(iy == y_shards - 1, pole_right, recv_right)
    f_xy_buf = jnp.concatenate([left_halo, f_x, right_halo], axis=-1)
    j_offset = iy * ny_loc
    buf_lat_origin = lagrange_interpolation.HALO_WIDTH + iy * ny_loc - h_lat
    return f_xy_buf, j_offset, buf_lat_origin

  f_all_y = lax.all_gather(f_x, 'y', axis=-1, tiled=True)
  return (
      lagrange_interpolation.extend_with_pole_halo(f_all_y, grid),
      iy * ny_loc,
      0,
  )


def mesh_z_shards(mesh: jax.sharding.Mesh | None) -> int:
  """Number of shards along the mesh's vertical axis 'z' (1 if it has none)."""
  return mesh.shape['z'] if mesh is not None and 'z' in mesh.axis_names else 1


def _edge_pad_levels(x: jax.Array, axis: int, n_padded: int) -> jax.Array:
  """Pads the level axis `axis` of `x` to `n_padded` levels by edge values.

  The level axis is sharded along 'z', so its length must be a multiple of the
  number of z-shards (`_round_to_multiple`). The padded levels repeat the last
  real level, so they stay valid inputs to the interpolation and are sliced
  off the result by the caller. Returns `x` unchanged when no padding is
  needed.
  """
  n = x.shape[axis]
  if n_padded == n:
    return x
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (0, n_padded - n)
  return jnp.pad(x, pad_width, mode='edge')


def _gather_longitude(
    f_loc: jax.Array, grid: spherical_harmonic.Grid, x_shards: int
) -> jax.Array:
  """All-gathers longitude (axis -2) of a field sharded over 'x'.

  Must be called inside a `shard_map` whose mesh has axis 'x'. `f_loc` has
  shape [..., nx_loc, ny_loc]. Every stencil may reach any longitude (the
  displacement limit `max_lat_shift` applies to latitude only), so each shard
  needs all of them. The result is trimmed to the physical
  `grid.longitude_nodes`, dropping the nodal padding that makes the axis
  divisible by `x_shards`. With one x-shard there is nothing to gather.
  """
  if x_shards > 1:
    f_x = lax.all_gather(f_loc, 'x', axis=f_loc.ndim - 2, tiled=True)
  else:
    f_x = f_loc
  return f_x[..., : grid.longitude_nodes, :]


def _gather_longitude_and_lat_halo(
    f_loc: jax.Array,
    grid: spherical_harmonic.Grid,
    max_lat_shift: int,
    x_shards: int,
) -> tuple[jax.Array, int | jax.Array, int | jax.Array]:
  """Shard-local buffer of a field sharded over (x, y), for `_matmul_*_local`.

  Must be called inside a `shard_map` whose mesh has axes 'x' and 'y'. The
  field `f_loc` has shape [..., nx_loc, ny_loc] (any leading axes, e.g.
  [channels, levels]) and is sharded along x (longitude) and y (latitude).

  Longitude is all-gathered over 'x' (`_gather_longitude`). Latitude is then
  extended by `_gather_lat_with_halo`: pole halos
  plus, between y-shards, a `max_lat_shift + HALO_WIDTH` row halo exchanged by
  `ppermute`. That width covers the latitude window of `_window_size` /
  `_lat_window_starts` (half-width `max_lat_shift`, plus the stencil's own
  rows). When the shard has fewer rows than the halo, or the latitude axis is
  padded, the y axis is all-gathered instead; see `_gather_lat_with_halo`.

  Args:
    f_loc: local block [..., nx_loc, ny_loc].
    grid: horizontal grid of the (global) field.
    max_lat_shift: latitude window half-width of the interpolation.
    x_shards: number of shards along mesh axis 'x'.

  Returns:
    `(buf, j_offset, buf_lat_origin)`: the buffer [..., longitude_nodes,
    n_lat_buf], the global latitude row of this shard's first arrival row, and
    the global (pole-extended) latitude row of the buffer's first row.
  """
  f_x = _gather_longitude(f_loc, grid, x_shards)
  return _gather_lat_with_halo(f_x, grid, max_lat_shift)


def _gather_z_with_halo(
    f_xy_buf: jax.Array,
    *,
    z_shards: int,
    n_src: int,
    n_src_padded: int,
    n_query: int,
    n_query_loc: int,
    max_vertical_shift: int,
) -> tuple[jax.Array, int | jax.Array, int | jax.Array, int]:
  """Shard-local vertical buffer of a z-sharded field for `_matmul_3d_local`.

  Must be called inside a `shard_map`. `f_xy_buf` has shape [C, l_src_loc,
  nx, ny_buf] (the output of `_gather_longitude_and_lat_halo`) with the level
  axis 1 sharded along 'z' over `n_src_padded` levels, `n_src` of them real.

  With one z-shard the buffer is the real levels. Otherwise, when the source
  and query level counts agree (each shard's query levels are the arrival
  levels of its own source levels), a halo of `S + 2` levels,
  `S = max(1, max_vertical_shift)`, is exchanged between neighbouring z-shards
  by `ppermute`: it covers the vertical window of `_window_size` /
  `_z_window_starts` (half-width `S`, plus the stencil's own levels). The
  exchange is cyclic, so the halo above the first shard and the halo below
  the last one hold levels from the opposite end of the column. They are not
  source levels of those shards: `_z_window_starts` keeps every window off
  them, and the returned `z_halo` tells the caller how many buffer levels at
  one end may be such. The z axis is all-gathered instead (and trimmed to the
  real levels) when the halo does not fit in a shard (`l_src_loc < S + 2`),
  or the source and query level counts differ, or the source levels were
  padded.

  Args:
    f_xy_buf: local block [C, l_src_loc, nx, ny_buf].
    z_shards: number of shards along mesh axis 'z'.
    n_src: number of real source levels.
    n_src_padded: source levels after `_round_to_multiple` to `z_shards`.
    n_query: number of real query levels.
    n_query_loc: query levels held by this shard.
    max_vertical_shift: vertical window half-width of the interpolation.

  Returns:
    `(buf, k_offset, buf_z_origin, z_halo)`: the buffer, the global level of
    this shard's first query level, the global level of the buffer's first
    level, and the width of the exchanged halo (0 when there is none): every
    shard's buffer holds at least `buf.shape[1] - z_halo` real source levels.
  """
  if z_shards == 1:
    return f_xy_buf[:, :n_src, :, :], 0, 0, 0
  iz = lax.axis_index('z')
  l_src_loc = f_xy_buf.shape[1]
  h_z = lagrange_interpolation.effective_window_shift(max_vertical_shift) + 2
  if n_src == n_query and n_src == n_src_padded and l_src_loc >= h_z:
    perm_down = [(s, (s + 1) % z_shards) for s in range(z_shards)]
    perm_up = [(s, (s - 1) % z_shards) for s in range(z_shards)]
    top_halo = lax.ppermute(f_xy_buf[:, -h_z:, :, :], 'z', perm=perm_down)
    bot_halo = lax.ppermute(f_xy_buf[:, :h_z, :, :], 'z', perm=perm_up)
    f_xyz_buf = jnp.concatenate([top_halo, f_xy_buf, bot_halo], axis=1)
    return f_xyz_buf, iz * n_query_loc, iz * l_src_loc - h_z, h_z
  f_xyz_buf = lax.all_gather(f_xy_buf, 'z', axis=1, tiled=True)[:, :n_src, :, :]
  return f_xyz_buf, iz * n_query_loc, 0, 0


#  =============================================================================
#  2-D interpolation.
#  =============================================================================


def interpolate_2d_points(
    fields: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    *,
    order: str,
    limiter: str | None,
) -> jnp.ndarray:
  """MXU matmul 2-D interpolation of [C, nx, ny] at arbitrary [*points]."""
  f_ext = lagrange_interpolation.extend_with_pole_halo(
      fields, grid
  )  # [C, n_lon, n_lat_ext]
  dtype = f_ext.dtype
  c_total, n_lon, n_lat_ext = f_ext.shape
  point_shape = lon.shape
  q_total = lon.size
  if q_total == 0 or c_total == 0:
    return jnp.zeros((c_total, *point_shape), dtype=dtype)

  lon_flat = jnp.asarray(lon).reshape(-1)
  sin_lat_flat = jnp.asarray(sin_lat).reshape(-1)
  q_chunk = min(q_total, _POINTS_QUERY_CHUNK)
  q_padded = _round_to_multiple(q_total, q_chunk)
  if q_padded > q_total:
    lon_flat = jnp.pad(lon_flat, [(0, q_padded - q_total)], mode='edge')
    sin_lat_flat = jnp.pad(sin_lat_flat, [(0, q_padded - q_total)], mode='edge')

  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, f_ext, f_ext)
  is_tpu = lagrange_interpolation.targets_tpu(grid)
  f_2d = f_ext.reshape(c_total * n_lon, n_lat_ext)
  bounds_2d = None
  if limiter is not None:
    lo_2d, hi_2d = lagrange_interpolation.cell_bounds_2d(f_ext)
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
        _,
    ) = lagrange_interpolation.horizontal_stencil(grid, lon_q, sin_lat_q, order)
    w_lat_mat = _weighted_one_hot(
        lat_indices, lat_weights, n_lat_ext, dtype
    )  # [q, n_lat_ext]
    f_at_lat = jnp.dot(w_lat_mat, f_2d.T, precision=dot_prec_f32).reshape(
        lon_q.shape[0], c_total, n_lon
    )
    w_lon_mat = _weighted_one_hot(
        lon_indices, lon_weights, n_lon, dtype
    )  # [q, n_lon]
    val = (f_at_lat * w_lon_mat[:, jnp.newaxis, :]).sum(axis=-1).T  # [C, q]

    if limiter is not None:
      p_lat_mat = jax.nn.one_hot(lat_cell0, n_lat_ext - 1, dtype=dtype)
      b_at_lat = _select_last_axis(bounds_2d, p_lat_mat, is_tpu).T.reshape(
          lon_q.shape[0], 2 * c_total, n_lon
      )
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
        (
            lon_flat.reshape(n_chunks, q_chunk),
            sin_lat_flat.reshape(n_chunks, q_chunk),
        ),
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
    lat_bounds: tuple[Array, Array],
) -> jax.Array:
  """Windowed MXU matmul 2-D interpolation on [C, n_lon, n_lat_buf].

  `lat_bounds` holds the `lat_clamp_bounds` of the `ny_loc` local arrival
  rows, which keep every query's stencil inside its window.
  """
  dtype = f_buf.dtype
  num_channels, n_lon, n_lat_buf = f_buf.shape
  n_lat = grid.latitude_nodes
  n_lat_ext = n_lat + 2 * lagrange_interpolation.HALO_WIDTH
  nx_loc, ny_loc = lon_q.shape
  eff_lat_shift = lagrange_interpolation.effective_window_shift(max_lat_shift)
  s_lat = 2 if order == 'linear' else 4
  w_lat_size = _window_size(n_lat_buf, n_lat_ext, eff_lat_shift, s_lat)
  lat_start_offset = (2 if order == 'linear' else 1) - eff_lat_shift

  (
      lon_indices,
      lon_weights,
      lat_indices,
      lat_weights,
      lon_cell0,
      lat_cell0,
      _,
      _,
  ) = lagrange_interpolation.horizontal_stencil(
      grid, lon_q, sin_lat_q, order, lat_bounds=lat_bounds
  )

  j_global = _arrival_lat_rows(j_offset, ny_loc, n_lat)
  win_lat_start, buf_lat_start = _lat_window_starts(
      j_global,
      j_global + lat_start_offset,
      n_lat_ext,
      n_lat_buf,
      w_lat_size,
      buf_lat_origin,
  )
  # Place `w_lat_size` before `ny_loc` so `(c_cur, w_lat_size)` is already
  # contiguous after `dot(f_2d, m_lat_2d_t)` — avoiding 4-D transposes and
  # keeping `w_lat_size` in axis=-1 (minor) for TPU tile alignment.
  m_lat_win = jax.nn.one_hot(
      jnp.arange(w_lat_size, dtype=jnp.int32)[:, jnp.newaxis]
      + buf_lat_start[jnp.newaxis, :],
      n_lat_buf,
      dtype=dtype,
  )  # [w_lat_size, ny_loc, n_lat_buf]
  m_lat_2d_t = m_lat_win.reshape(w_lat_size * ny_loc, n_lat_buf).T

  dot_prec_rhs_bf16 = _resolve_rhs_binary_dot_precision(
      f_buf, m_lat_2d_t, lagrange_interpolation.targets_tpu(grid)
  )
  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, f_buf, f_buf)

  w_lon_mat = sum(
      w.T[:, :, jnp.newaxis] * jax.nn.one_hot(idx.T, n_lon, dtype=dtype)
      for idx, w in zip(lon_indices, lon_weights)
  )  # [ny_loc, nx_loc, n_lon]
  w_lat_mat = sum(
      w.T[:, :, jnp.newaxis]
      * jax.nn.one_hot(
          idx.T - win_lat_start[:, jnp.newaxis], w_lat_size, dtype=dtype
      )
      for idx, w in zip(lat_indices, lat_weights)
  )  # [ny_loc, nx_loc, w_lat_size]

  if limiter is not None:
    w_lat_cell_size = _window_size(
        n_lat_buf - 1, n_lat_ext - 1, eff_lat_shift, 2
    )
    win_lat_cell_start, buf_lat_cell_start = _lat_window_starts(
        j_global,
        j_global + 1 - eff_lat_shift,
        n_lat_ext - 1,
        n_lat_buf - 1,
        w_lat_cell_size,
        buf_lat_origin,
    )
    m_lat_cell_win = jax.nn.one_hot(
        jnp.arange(w_lat_cell_size, dtype=jnp.int32)[:, jnp.newaxis]
        + buf_lat_cell_start[jnp.newaxis, :],
        n_lat_buf - 1,
        dtype=dtype,
    )  # [w_lat_cell_size, ny_loc, n_lat_buf - 1]
    m_lat_cell_2d_t = m_lat_cell_win.reshape(
        w_lat_cell_size * ny_loc, n_lat_buf - 1
    ).T
    p_lon_mat = jax.nn.one_hot(lon_cell0.T, n_lon, dtype=dtype)
    p_lat_mat = jax.nn.one_hot(
        lat_cell0.T - win_lat_cell_start[:, jnp.newaxis],
        w_lat_cell_size,
        dtype=dtype,
    )

  def _eval_chunk(f_chunk: jax.Array) -> jax.Array:
    c_cur = f_chunk.shape[0]
    f_2d = f_chunk.transpose(1, 0, 2).reshape(n_lon * c_cur, n_lat_buf)
    f_lat_win = (
        jnp.dot(f_2d, m_lat_2d_t, precision=dot_prec_rhs_bf16)
        .reshape(n_lon, c_cur * w_lat_size, ny_loc)
        .transpose(2, 0, 1)
    )  # [ny_loc, n_lon, c_cur * w_lat_size]
    gathered_lon = jnp.einsum(
        'jim,jmk->jik', w_lon_mat, f_lat_win, precision=dot_prec_f32
    )
    if c_cur == 1:
      val = (gathered_lon * w_lat_mat).sum(axis=-1).T[jnp.newaxis, :, :]
    else:
      val = (
          (
              gathered_lon.reshape(ny_loc, nx_loc, c_cur, w_lat_size)
              * w_lat_mat[:, :, jnp.newaxis, :]
          )
          .sum(axis=-1)
          .transpose(2, 1, 0)
      )

    if limiter is not None:
      lo_buf, hi_buf = lagrange_interpolation.cell_bounds_2d(f_chunk)
      bounds_buf = jnp.concatenate([lo_buf, hi_buf], axis=0)
      b_2d = bounds_buf.transpose(1, 0, 2).reshape(
          n_lon * 2 * c_cur, n_lat_buf - 1
      )
      b_lat_win = (
          jnp.dot(b_2d, m_lat_cell_2d_t, precision=dot_prec_rhs_bf16)
          .reshape(n_lon, 2 * c_cur * w_lat_cell_size, ny_loc)
          .transpose(2, 0, 1)
      )
      # [ny_loc, K, nx_loc] -> [ny_loc, nx_loc, K]; the one-hot is the right
      # operand so that gradients keep full precision (`_select_last_axis`).
      b_lon = lax.dot_general(
          b_lat_win,
          p_lon_mat,
          (((1,), (2,)), ((0,), (0,))),
          precision=dot_prec_rhs_bf16,
      ).transpose(0, 2, 1)
      b_val = (
          (
              b_lon.reshape(ny_loc, nx_loc, 2 * c_cur, w_lat_cell_size)
              * p_lat_mat[:, :, jnp.newaxis, :]
          )
          .sum(axis=-1)
          .transpose(2, 1, 0)
      )
      val = jnp.clip(val, b_val[:c_cur], b_val[c_cur:])
    return val

  chunk_size = (
      _MATMUL_2D_CHANNEL_CHUNK_LIMITED
      if limiter is not None
      else _MATMUL_2D_CHANNEL_CHUNK
  )
  if num_channels <= chunk_size:
    return _eval_chunk(f_buf)

  c_pad = _round_to_multiple(num_channels, chunk_size)
  f_pad = (
      jnp.pad(f_buf, [(0, c_pad - num_channels), (0, 0), (0, 0)])
      if c_pad > num_channels
      else f_buf
  )
  f_grouped = f_pad.reshape((c_pad // chunk_size, chunk_size, n_lon, n_lat_buf))
  _, out_grouped = lax.scan(
      lambda _, grp: (None, _eval_chunk(grp)), None, f_grouped
  )
  return out_grouped.reshape((c_pad, nx_loc, ny_loc))[:num_channels]


def interpolate_2d_grid(
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
  mesh = spatial_mesh(grid)
  lat_lo, lat_hi = lagrange_interpolation.lat_clamp_bounds(
      grid, max_lat_shift, order, sin_lat.dtype
  )
  if mesh is None:
    f_ext = lagrange_interpolation.extend_with_pole_halo(fields, grid)
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
        lat_bounds=(lat_lo, lat_hi),
    )

  x_shards = mesh.shape['x']
  f_spec = P(None, 'x', 'y')
  q_spec = P('x', 'y')

  def _sharded(f_loc, lon_loc, sin_lat_loc, lat_lo_loc, lat_hi_loc):
    f_xy_buf, j_offset, buf_lat_origin = _gather_longitude_and_lat_halo(
        f_loc, grid, max_lat_shift, x_shards
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
        lat_bounds=(lat_lo_loc, lat_hi_loc),
    )

  # The clamp bounds are sharded like the latitude axis, so each shard holds
  # those of its own arrival rows.
  return _shard_map(
      _sharded,
      mesh,
      in_specs=(f_spec, q_spec, q_spec, P('y'), P('y')),
      out_specs=f_spec,
  )(fields, lon, sin_lat, jnp.asarray(lat_lo), jnp.asarray(lat_hi))


def interpolate_2d_grid_levels(
    fields: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    *,
    order: str,
    limiter: str | None,
    max_lat_shift: int,
) -> jax.Array:
  """Grid-aligned 2-D matmul interpolation of [C, levels, nx, ny], per level.

  Level `k` of every channel is interpolated at its own queries
  `(lon[k], sin_lat[k])`; the levels are scanned (on each z-shard, if the
  mesh shards 'z').
  """
  levels = fields.shape[1]
  mesh = spatial_mesh(grid)
  lat_lo, lat_hi = lagrange_interpolation.lat_clamp_bounds(
      grid, max_lat_shift, order, sin_lat.dtype
  )
  if mesh is None:
    f_ext = lagrange_interpolation.extend_with_pole_halo(fields, grid)
    f_by_l = jnp.swapaxes(f_ext, 0, 1)
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
                j_offset=0,
                buf_lat_origin=0,
                max_lat_shift=max_lat_shift,
                lat_bounds=(lat_lo, lat_hi),
            ),
        ),
        None,
        (f_by_l, lon, sin_lat),
    )
    return jnp.swapaxes(out_by_l, 0, 1)

  z_shards = mesh_z_shards(mesh)
  x_shards = mesh.shape['x']
  l_pad = _round_to_multiple(levels, z_shards)
  fields_pad = _edge_pad_levels(fields, 1, l_pad)
  lon_pad = _edge_pad_levels(lon, 0, l_pad)
  sin_lat_pad = _edge_pad_levels(sin_lat, 0, l_pad)
  z_spec = 'z' if z_shards > 1 else None
  f_spec = P(None, z_spec, 'x', 'y')
  q_spec = P(z_spec, 'x', 'y')

  def _sharded_levels(f_loc, lon_loc, sin_lat_loc, lat_lo_loc, lat_hi_loc):
    f_xy_buf, j_offset, buf_lat_origin = _gather_longitude_and_lat_halo(
        f_loc, grid, max_lat_shift, x_shards
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
                lat_bounds=(lat_lo_loc, lat_hi_loc),
            ),
        ),
        None,
        (f_by_l, lon_loc, sin_lat_loc),
    )
    return jnp.swapaxes(out_by_l, 0, 1)

  out_pad = _shard_map(
      _sharded_levels,
      mesh,
      in_specs=(f_spec, q_spec, q_spec, P('y'), P('y')),
      out_specs=f_spec,
  )(
      fields_pad,
      lon_pad,
      sin_lat_pad,
      jnp.asarray(lat_lo),
      jnp.asarray(lat_hi),
  )
  return out_pad[:, :levels, :, :]


#  =============================================================================
#  3-D interpolation.
#  =============================================================================


def interpolate_3d_points(
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
  f_ext = lagrange_interpolation.extend_with_pole_halo(
      fields_4d, grid
  )  # [C, L_src, n_lon, n_lat_ext]
  dtype = f_ext.dtype
  is_tpu = lagrange_interpolation.targets_tpu(grid)
  c_total, l_src, n_lon, n_lat_ext = f_ext.shape
  point_shape = lon.shape
  q_total = lon.size
  if q_total == 0 or c_total == 0:
    return jnp.zeros((c_total, *point_shape), dtype=dtype)

  lon_flat = jnp.asarray(lon).reshape(-1)
  sin_lat_flat = jnp.asarray(sin_lat).reshape(-1)
  sigma_flat = jnp.asarray(sigma).reshape(-1)
  q_chunk = min(q_total, _POINTS_QUERY_CHUNK)
  q_padded = _round_to_multiple(q_total, q_chunk)
  if q_padded > q_total:
    pad_w = [(0, q_padded - q_total)]
    lon_flat = jnp.pad(lon_flat, pad_w, mode='edge')
    sin_lat_flat = jnp.pad(sin_lat_flat, pad_w, mode='edge')
    sigma_flat = jnp.pad(sigma_flat, pad_w, mode='edge')

  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, f_ext, f_ext)

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
    lo_b, hi_b = lagrange_interpolation.cell_bounds_3d(f_for_bounds)
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
        _,
    ) = lagrange_interpolation.horizontal_stencil(grid, lon_q, sin_lat_q, order)
    level_indices, level_weights, _, level_cell0, _ = (
        lagrange_interpolation.vertical_stencil(
            sigma_q, sigma_nodes, vertical_order, is_tpu=is_tpu
        )
    )

    w_lat_mat = _weighted_one_hot(
        lat_indices, lat_weights, n_lat_ext, dtype
    )  # [q, n_lat_ext]
    w_lon_mat = _weighted_one_hot(
        lon_indices, lon_weights, n_lon, dtype
    )  # [q, n_lon]
    w_z_mat = _weighted_one_hot(
        level_indices, level_weights, l_src, dtype
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
        b_at_lat = _select_last_axis(b_l, p_lat_mat, is_tpu).T.reshape(
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
            w * _take_1d(ref_centered, idx, is_tpu)
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


@dataclasses.dataclass(frozen=True)
class _Matmul3DSpec:
  """What a windowed 3-D matmul interpolation computes, on every shard.

  Attributes:
    grid: the horizontal grid holding the source data.
    order: horizontal interpolation order.
    vertical_order: vertical interpolation order.
    limiter: the limiter, applied to the channels from `limited_channel_start`
      on, or None.
    max_lat_shift: latitude displacement limit, in cells.
    max_vertical_shift: vertical displacement limit, in cells.
    limited_channel_start: first limited channel.
    trilinear_channel_start: first channel that also gets a trilinear output,
      or None.
  """

  grid: spherical_harmonic.Grid
  order: str
  vertical_order: str
  limiter: str | None
  max_lat_shift: int
  max_vertical_shift: int
  limited_channel_start: int = 0
  trilinear_channel_start: int | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class _LocalBuffer:
  """The source data one shard holds, and where it sits in the global field.

  Attributes:
    values: [C, L_buf, n_lon, n_lat_buf]: source levels `z_origin ...` and
      halo-extended latitude rows `lat_origin ...`.
    k_offset: global level of the shard's first query level.
    z_origin: source level of the first buffer level.
    z_halo: width of the cyclic vertical halo of `values`, from
      `_gather_z_with_halo` (0 without one).
    j_offset: global latitude row of the shard's first arrival row.
    lat_origin: halo-extended latitude row of the first buffer row.
  """

  values: jax.Array
  k_offset: int | jax.Array = 0
  z_origin: int | jax.Array = 0
  z_halo: int = 0
  j_offset: int | jax.Array = 0
  lat_origin: int | jax.Array = 0


@dataclasses.dataclass(frozen=True, eq=False)
class _LocalQueries:
  """The queries one shard evaluates, with their window clamps.

  Attributes:
    lon: query longitudes, [l_q_loc, nx_loc, ny_loc]; query `[k, i, j]`
      arrives at local row `j` and local level `k`.
    sin_lat: sine of the query latitudes, same shape.
    sigma: query σ, same shape.
    lat_bounds: `lat_clamp_bounds` of the local arrival rows, [ny_loc] each.
    sigma_bounds: `level_clamp_bounds` of the local query levels,
      [l_q_loc] each. Together with `lat_bounds` they keep every stencil
      inside its window.
  """

  lon: jax.Array
  sin_lat: jax.Array
  sigma: jax.Array
  lat_bounds: tuple[Array, Array]
  sigma_bounds: tuple[Array, Array]


@dataclasses.dataclass(frozen=True)
class _Matmul3DPlan:
  """Static geometry and memory budget of one windowed 3-D matmul call.

  Every field is a Python value fixed at trace time. The buffer
  `f_buf` has shape `[num_channels, l_buf, n_lon, n_lat_buf]` and the local
  queries `[l_q_loc, nx_loc, ny_loc]`; `l_src` / `n_lat_ext` are the numbers of
  source levels and halo-extended latitude rows the windows slide over.

  Attributes:
    dtype: dtype of the interpolated field.
    is_tpu: whether the TPU-specific layouts and budgets apply.
    order: horizontal interpolation order.
    vertical_order: vertical interpolation order.
    n_lon: number of longitude nodes.
    n_lat_ext: number of halo-extended latitude rows of the source axis.
    l_src: number of source levels.
    num_channels: number of channels of `f_buf`.
    l_buf: number of levels held in the local buffer.
    l_valid: number of those that are real source levels on every shard:
      `l_buf` less the cyclic halo at one end of the first and last z-shard
      (see `_gather_z_with_halo`). The vertical windows are sized to it.
    n_lat_buf: number of latitude rows held in the local buffer.
    l_q_loc: number of local query levels.
    nx_loc: number of local query longitudes.
    ny_loc: number of local arrival latitude rows.
    limiter: the limiter if any channel is limited, else None.
    eff_lat_shift: latitude window half-width, in rows.
    eff_z_shift: vertical window half-width, in levels.
    w_lat_size: rows of the latitude stencil window.
    w_z_size: levels of the vertical stencil window of a single query level.
    w_z_lin_size: levels of the linear-stencil window of a single query level
      (used by the trilinear output).
    w_lat_cell_size: rows of the limiter's cell-bound latitude window (0 if
      there is no limiter).
    w_z_cell_size: levels of the limiter's cell-bound vertical window (0 if
      there is no limiter).
    lat_start_offset: first stencil row of a query at its arrival row, relative
      to that row.
    z_start_offset: the same in the vertical, relative to the arrival level.
    max_block_levels: largest number of query levels that may share one
      vertical window (see `block_window`): a longer block's window would not
      fit in the real source levels of a z-shard's buffer.
    block_levels: number of query levels evaluated per scan block; 1 as
      built by `_plan_matmul_3d`, set by `_plan_blocks_and_channels`.
    allowed_block_levels: the block length that may be used instead of 1
      (`_level_block_length`; 1 if none).
    need_separate_trilinear: whether the trilinear channels need stencils of
      their own (they do not if the main stencil already is trilinear).
    item_bytes: bytes per element.
    n_lon_pad128: `n_lon` padded to a multiple of the TPU lane count.
    nx_pad128: `nx_loc` padded to a multiple of the TPU lane count.
    w_lon_vmem_b: padded bytes of the longitude weights of a single-level
      block.
    max_channels: cap on the channel chunk.
    single_buf_limit: budget of the largest single hoisted buffer, in bytes.
    ws_buf_limit: budget of the sum of the hoisted buffers alive at once.
    vmem_limit: budget of the per-block working set, in bytes.
  """

  dtype: jnp.dtype
  is_tpu: bool
  order: str
  vertical_order: str
  n_lon: int
  n_lat_ext: int
  l_src: int
  num_channels: int
  l_buf: int
  l_valid: int
  n_lat_buf: int
  l_q_loc: int
  nx_loc: int
  ny_loc: int
  limiter: str | None
  eff_lat_shift: int
  eff_z_shift: int
  w_lat_size: int
  w_z_size: int
  w_z_lin_size: int
  w_lat_cell_size: int
  w_z_cell_size: int
  lat_start_offset: int
  z_start_offset: int
  max_block_levels: int
  block_levels: int
  allowed_block_levels: int
  need_separate_trilinear: bool
  item_bytes: int
  n_lon_pad128: int
  nx_pad128: int
  w_lon_vmem_b: int
  max_channels: int
  single_buf_limit: int
  ws_buf_limit: int
  vmem_limit: int

  def block_window(
      self, single_level_size: int, b_len: int, *, cells: bool = False
  ) -> int:
    """Size of the vertical window shared by a block of `b_len` query levels.

    Args:
      single_level_size: the window of one query level (`w_z_size`,
        `w_z_lin_size` or `w_z_cell_size`).
      b_len: query levels in the block.
      cells: whether the window counts cells (pairs of adjacent levels, for
        the limiter's bounds) rather than levels.

    Returns:
      `single_level_size + b_len - 1`, truncated to the source axis and to the
      real source levels of the local buffer.
    """
    spare = 1 if cells else 0
    return min(
        max(1, self.l_valid - spare),
        min(max(1, self.l_src - spare), single_level_size + b_len - 1),
    )

  def fits_budget(self, num_channels: int, limited_start: int = 0) -> bool:
    """Whether a chunk of `num_channels` channels fits the memory budgets.

    The sizes are those of (8, 128)-tile-padded buffers: the hoisted
    latitude-projected field (and cell bounds), the per-block vertical patch,
    and the per-block longitude-contracted result.

    Args:
      num_channels: channels in the chunk.
      limited_start: index within the chunk of the first limited channel.

    Returns:
      True if the largest single buffer, the sum of the hoisted buffers and
      the per-block working set are all within their limits.
    """
    k_c_pad = _round_to_multiple(num_channels * self.w_lat_size, _TPU_SUBLANES)
    # On TPU: the chunk-dependent part of the working set of a block.
    chunk_vmem_b = (
        self.ny_loc
        * self.block_window(self.w_z_size, self.block_levels)
        * k_c_pad
        * (
            self.n_lon_pad128
            + _round_to_multiple(self.block_levels * self.nx_loc, _TPU_LANES)
        )
        * self.item_bytes
    )
    f_lat_b = (
        self.ny_loc * self.l_buf * k_c_pad * self.n_lon_pad128 * self.item_bytes
    )
    f_patch_b = (
        self.ny_loc
        * self.w_z_size
        * k_c_pad
        * self.n_lon_pad128
        * self.item_bytes
    )
    gath_b = (
        self.ny_loc * self.w_z_size * k_c_pad * self.nx_pad128 * self.item_bytes
    )
    vmem_b = self.w_lon_vmem_b + f_patch_b + gath_b
    max_single_b = f_lat_b
    ws_b = f_lat_b
    c_lim_test = (
        max(0, num_channels - limited_start) if self.limiter is not None else 0
    )
    if c_lim_test > 0:
      k_l_pad = _round_to_multiple(
          2 * c_lim_test * self.w_lat_cell_size, _TPU_SUBLANES
      )
      b_lat_b = (
          self.ny_loc
          * max(1, self.l_buf - 1)
          * k_l_pad
          * self.n_lon_pad128
          * self.item_bytes
      )
      b_patch_b = (
          self.ny_loc
          * self.w_z_cell_size
          * k_l_pad
          * self.n_lon_pad128
          * self.item_bytes
      )
      b_lon_b = (
          self.ny_loc
          * self.w_z_cell_size
          * k_l_pad
          * self.nx_pad128
          * self.item_bytes
      )
      max_single_b = max(max_single_b, b_lat_b)
      ws_b = f_lat_b + b_lat_b
      vmem_b = max(vmem_b, self.w_lon_vmem_b + b_patch_b + b_lon_b)
    if self.is_tpu:
      fits_vmem = chunk_vmem_b <= _MATMUL_3D_CHUNK_VMEM_TPU
    else:
      fits_vmem = vmem_b <= self.vmem_limit
    return (
        max_single_b <= self.single_buf_limit
        and ws_b <= self.ws_buf_limit
        and fits_vmem
    )


def _plan_matmul_3d(
    buffer: _LocalBuffer,
    queries: _LocalQueries,
    spec: _Matmul3DSpec,
    sigma_nodes: np.ndarray,
) -> _Matmul3DPlan:
  """Builds the static plan of a windowed 3-D matmul call.

  Args:
    buffer: the local source data.
    queries: the local queries.
    spec: what to interpolate.
    sigma_nodes: source σ nodes.

  Returns:
    The plan, including the TPU or CPU/GPU memory budgets selected by the
    platform.
  """
  f_buf = buffer.values
  lon_q = queries.lon
  grid = spec.grid
  order = spec.order
  vertical_order = spec.vertical_order
  max_lat_shift = spec.max_lat_shift
  max_vertical_shift = spec.max_vertical_shift
  limited_channel_start = spec.limited_channel_start
  trilinear_channel_start = spec.trilinear_channel_start
  limiter = spec.limiter
  z_halo = buffer.z_halo
  dtype = f_buf.dtype
  is_tpu = lagrange_interpolation.targets_tpu(grid)
  n_lon = grid.longitude_nodes
  n_lat = grid.latitude_nodes
  n_lat_ext = n_lat + 2 * lagrange_interpolation.HALO_WIDTH
  l_src = sigma_nodes.shape[0]
  num_channels, l_buf, _, n_lat_buf = f_buf.shape
  l_valid = l_buf - z_halo
  l_q_loc, nx_loc, ny_loc = lon_q.shape

  eff_limiter = limiter if limited_channel_start < num_channels else None
  eff_lat_shift = lagrange_interpolation.effective_window_shift(max_lat_shift)
  eff_z_shift = lagrange_interpolation.effective_window_shift(
      max_vertical_shift
  )
  s_lat = 2 if order == 'linear' else 4
  s_z = 2 if vertical_order == 'linear' else 4
  w_lat_cell_size = 0
  w_z_cell_size = 0
  if eff_limiter is not None:
    w_lat_cell_size = _window_size(
        n_lat_buf - 1, n_lat_ext - 1, eff_lat_shift, 2
    )
    w_z_cell_size = _window_size(l_valid - 1, l_src - 1, eff_z_shift, 2)
  # A block of `b` query levels shares one window of `2 * shift + s_z + b - 1`
  # levels. On a z-shard between two others that window has to fit in the
  # `l_valid` levels a buffer is guaranteed to hold; when the buffer holds the
  # whole column the window is simply truncated to it.
  if l_valid >= l_src:
    max_block_levels = max(1, l_q_loc)
  else:
    max_block_levels = max(1, l_valid - (2 * eff_z_shift + s_z) + 1)

  item_bytes = 8 if dtype == jnp.float64 else 4
  n_lon_pad128 = _round_to_multiple(n_lon, _TPU_LANES)
  nx_pad128 = _round_to_multiple(nx_loc, _TPU_LANES)
  if is_tpu:
    budgets = (
        _MATMUL_3D_MAX_CHANNELS_TPU,
        _MATMUL_3D_SINGLE_BUF_LIMIT_TPU,
        _MATMUL_3D_WS_BUF_LIMIT_TPU,
        _MATMUL_3D_VMEM_LIMIT_TPU,
    )
  else:
    budgets = (
        _MATMUL_3D_MAX_CHANNELS_CPU_GPU,
        _MATMUL_3D_SINGLE_BUF_LIMIT_CPU_GPU,
        _MATMUL_3D_WS_BUF_LIMIT_CPU_GPU,
        _MATMUL_3D_VMEM_LIMIT_CPU_GPU,
    )
  max_channels, single_buf_limit, ws_buf_limit, vmem_limit = budgets
  return _Matmul3DPlan(
      dtype=dtype,
      is_tpu=is_tpu,
      order=order,
      vertical_order=vertical_order,
      n_lon=n_lon,
      n_lat_ext=n_lat_ext,
      l_src=l_src,
      num_channels=num_channels,
      l_buf=l_buf,
      l_valid=l_valid,
      n_lat_buf=n_lat_buf,
      l_q_loc=l_q_loc,
      nx_loc=nx_loc,
      ny_loc=ny_loc,
      limiter=eff_limiter,
      eff_lat_shift=eff_lat_shift,
      eff_z_shift=eff_z_shift,
      w_lat_size=_window_size(n_lat_buf, n_lat_ext, eff_lat_shift, s_lat),
      w_z_size=_window_size(l_valid, l_src, eff_z_shift, s_z),
      w_z_lin_size=_window_size(l_valid, l_src, eff_z_shift, 2),
      w_lat_cell_size=w_lat_cell_size,
      w_z_cell_size=w_z_cell_size,
      lat_start_offset=(2 if order == 'linear' else 1) - eff_lat_shift,
      z_start_offset=(
          -eff_z_shift if vertical_order == 'linear' else (-eff_z_shift - 1)
      ),
      max_block_levels=max_block_levels,
      block_levels=1,
      allowed_block_levels=_level_block_length(
          l_q_loc,
          is_tpu,
          # The longitude weights [ny_loc, n_lon, block levels * nx_loc].
          lambda levels: (
              ny_loc
              * _round_to_multiple(n_lon, _TPU_SUBLANES)
              * _round_to_multiple(levels * nx_loc, _TPU_LANES)
              * item_bytes
          ),
          _TPU_VMEM_BYTES,
          max_block_levels,
      ),
      need_separate_trilinear=trilinear_channel_start is not None
      and not (order == 'linear' and vertical_order == 'linear'),
      item_bytes=item_bytes,
      n_lon_pad128=n_lon_pad128,
      nx_pad128=nx_pad128,
      w_lon_vmem_b=(
          ny_loc
          * _round_to_multiple(n_lon, _TPU_SUBLANES)
          * nx_pad128
          * item_bytes
      ),
      max_channels=max_channels,
      single_buf_limit=single_buf_limit,
      ws_buf_limit=ws_buf_limit,
      vmem_limit=vmem_limit,
  )


class _ChannelSegment(NamedTuple):
  """A run of channels sharing one limiter / trilinear setting."""

  start: int
  stop: int
  limited: bool
  trilinear: bool


@dataclasses.dataclass(frozen=True)
class _ChannelPlan:
  """How the channels of one windowed 3-D matmul call are processed.

  Attributes:
    chunk_size: channels evaluated together (one `lax.scan` step if the
      channels do not fit in a single chunk).
    cost: relative cost of the plan per query level, in channels: every
      channel of every chunk (padding included) plus a fixed cost per chunk.
    segments: None if the channels are evaluated as one stack. Otherwise the
      homogeneous runs the stack is split into, because chunking a stack
      that mixes limited / unlimited / trilinear channels would make the
      chunks non-uniform; each run is then evaluated by its own call.
  """

  chunk_size: int
  segments: tuple[_ChannelSegment, ...] | None
  cost: float


def _plan_channels(
    plan: _Matmul3DPlan,
    limited_channel_start: int,
    trilinear_channel_start: int | None,
) -> _ChannelPlan:
  """Chooses the channel chunk size and splits mixed channels into runs.

  The chunk is the whole stack if it fits the budgets of `plan` with its
  limited channels. Otherwise it is the cheapest size up to the largest one
  that fits (taking every one of its channels to be limited, the worst
  case), counting the padding of the last chunk and a fixed cost per chunk.

  Args:
    plan: the static plan of the call.
    limited_channel_start: first limited channel.
    trilinear_channel_start: first channel that also gets a trilinear output,
      or None.

  Returns:
    The channel plan.
  """
  num_channels = plan.num_channels

  def chunk_cost(chunk: int) -> float:
    num_chunks = -(-num_channels // chunk)
    return num_chunks * (chunk + _CHUNK_OVERHEAD_IN_CHANNELS)

  if num_channels <= plan.max_channels and plan.fits_budget(
      num_channels, limited_channel_start
  ):
    chunk_size = num_channels
    plan_cost = chunk_cost(num_channels)
  else:
    max_chunk = 1
    for c_cand in range(min(num_channels, plan.max_channels), 0, -1):
      if plan.fits_budget(c_cand, 0):
        max_chunk = c_cand
        break

    # The chunks are equal, the last one padded with zero channels. A padded
    # channel costs as much as a real one and every chunk has a fixed cost,
    # so the largest chunk that fits is not always the cheapest: 8 channels
    # are better as 2 chunks of 4 than as 6 + 2 padded to 6, or as 3 chunks
    # of 3.
    chunk_size = min(range(max_chunk, 0, -1), key=chunk_cost)
    plan_cost = chunk_cost(chunk_size)

  if not (
      num_channels > chunk_size
      and (
          0 < limited_channel_start < num_channels
          or (
              trilinear_channel_start is not None
              and 0 < trilinear_channel_start < num_channels
          )
      )
  ):
    return _ChannelPlan(chunk_size, None, plan_cost)

  split_pts = sorted(
      {0, num_channels}
      | (
          {min(num_channels, max(0, limited_channel_start))}
          if plan.limiter is not None
          else set()
      )
      | (
          {min(num_channels, max(0, trilinear_channel_start))}
          if trilinear_channel_start is not None
          else set()
      )
  )
  segments = tuple(
      _ChannelSegment(
          start=s0,
          stop=s1,
          limited=plan.limiter is not None and s0 >= limited_channel_start,
          trilinear=(
              trilinear_channel_start is not None
              and s0 >= trilinear_channel_start
          ),
      )
      for s0, s1 in zip(split_pts[:-1], split_pts[1:])
      if s0 != s1
  )
  return _ChannelPlan(chunk_size, segments, plan_cost)


def _plan_blocks_and_channels(
    plan: _Matmul3DPlan,
    limited_channel_start: int,
    trilinear_channel_start: int | None,
) -> tuple[_Matmul3DPlan, _ChannelPlan]:
  """Chooses the block length and the channel chunks together.

  A block of `plan.allowed_block_levels` levels is cheaper per level than
  single levels (`_TPU_BLOCK_COST`, and less lane padding when the local
  longitudes are not a multiple of the lane count), but its working set is
  larger, so it may only fit smaller channel chunks, of which there are then
  more. Both are planned and the cheaper one is taken.

  Args:
    plan: the static plan of the call, as built by `_plan_matmul_3d`.
    limited_channel_start: first limited channel.
    trilinear_channel_start: first channel that also gets a trilinear output,
      or None.

  Returns:
    `(plan, channel_plan)`, the plan with its `block_levels` set.
  """
  channels = _plan_channels(
      plan, limited_channel_start, trilinear_channel_start
  )
  levels = plan.allowed_block_levels
  if levels == 1:
    return plan, channels
  block_plan = dataclasses.replace(plan, block_levels=levels)
  block_channels = _plan_channels(
      block_plan, limited_channel_start, trilinear_channel_start
  )
  lanes_per_level = _round_to_multiple(levels * plan.nx_loc, _TPU_LANES) / (
      levels * plan.nx_pad128
  )
  if block_channels.cost * _TPU_BLOCK_COST * lanes_per_level < channels.cost:
    return block_plan, block_channels
  return plan, channels


def _matmul_3d_by_segments(
    buffer: _LocalBuffer,
    queries: _LocalQueries,
    spec: _Matmul3DSpec,
    sigma_nodes: np.ndarray,
    limiter_bound_reference: jax.Array | None,
    segments: tuple[_ChannelSegment, ...],
    plan: _Matmul3DPlan,
) -> tuple[jax.Array, jax.Array | None]:
  """Evaluates homogeneous channel runs separately and concatenates them.

  Args:
    buffer: the local source data of all channels.
    queries: the local queries.
    spec: what to interpolate, for the whole stack.
    sigma_nodes: source σ nodes.
    limiter_bound_reference: per-level limiter bound reference, or None.
    segments: the runs, from `_plan_channels`.
    plan: the static plan of the call.

  Returns:
    `(values, trilinear)`: the interpolated channels, [C, l_q_loc, nx_loc,
    ny_loc], and the trilinear output of the trilinear runs (None if
    `spec.trilinear_channel_start` is None).
  """
  outs = []
  outs_lin = []
  for seg in segments:
    values, trilinear = _matmul_3d_local(
        dataclasses.replace(buffer, values=buffer.values[seg.start : seg.stop]),
        queries,
        dataclasses.replace(
            spec,
            limiter=plan.limiter if seg.limited else None,
            limited_channel_start=0,
            trilinear_channel_start=0 if seg.trilinear else None,
        ),
        sigma_nodes,
        limiter_bound_reference,
    )
    outs.append(values)
    if seg.trilinear:
      outs_lin.append(trilinear)
  out_full = outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=0)
  if spec.trilinear_channel_start is None:
    return out_full, None
  # No trilinear channels at all (`trilinear_channel_start == C`) leaves
  # `outs_lin` empty; the gather method returns an empty trilinear block.
  if not outs_lin:
    out_lin_full = jnp.zeros(
        (0, plan.l_q_loc, plan.nx_loc, plan.ny_loc), dtype=plan.dtype
    )
  elif len(outs_lin) == 1:
    out_lin_full = outs_lin[0]
  else:
    out_lin_full = jnp.concatenate(outs_lin, axis=0)
  return out_full, out_lin_full


def _lat_window_selector(
    j_global: jax.Array,
    first_row: jax.Array,
    n_source: int,
    n_buffer: int,
    size: int,
    buf_origin: int | jax.Array,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
  """One-hot selectors of the per-arrival-row latitude windows.

  Args:
    j_global: global arrival rows, [ny_loc], from `_arrival_lat_rows`.
    first_row: unclipped first row of each window, [ny_loc].
    n_source: number of rows of the (halo-extended) source axis.
    n_buffer: number of rows held in the local buffer.
    size: window size from `_window_size`.
    buf_origin: source row of the first row of the local buffer.
    dtype: dtype of the selector.

  Returns:
    `(win_start, m_lat_2d)`: the window starts within the source axis
    ([ny_loc], see `_lat_window_starts`) and the selector
    [ny_loc * size, n_buffer] whose row `j * size + w` picks buffer row
    `buf_start[j] + w`. It is built through `[ny_loc, size, n_buffer]` so that
    `dot(m_lat_2d, ...)` keeps `n_lon` in the last axis of its right-hand side
    with no cross-lane minor-axis transposes.
  """
  win_start, buf_start = _lat_window_starts(
      j_global, first_row, n_source, n_buffer, size, buf_origin
  )
  m_lat_win = jax.nn.one_hot(
      buf_start[:, jnp.newaxis]
      + jnp.arange(size, dtype=jnp.int32)[jnp.newaxis, :],
      n_buffer,
      dtype=dtype,
  )  # [ny_loc, size, n_buffer]
  return win_start, m_lat_win.reshape(j_global.shape[0] * size, n_buffer)


def _project_lat_window(
    stack: jax.Array,
    m_lat_2d: jax.Array,
    window_size: int,
    is_tpu: bool,
) -> jax.Array:
  """Restricts a stack of fields to the latitude window of each arrival row.

  One matmul with the one-hot selector `m_lat_2d` selects, for every
  arrival row, its `window_size` latitude rows from every (channel, level)
  pair at once. The result is laid out `[ny_loc, level, K, n_lon]` with the
  window rows minor within each channel in `K`, and `K` padded with zeros to a
  multiple of the TPU sublane count, so that the later per-block longitude
  contraction `[ny_loc, level * K, n_lon] x [ny_loc, n_lon, nx]` runs on full
  (8, 128) tiles with `n_lon` (contracted) and `nx` (output) in the lanes.

  Args:
    stack: [C, L, n_lon, n_lat] fields (or cell bounds, `n_lat` rows of
      cells).
    m_lat_2d: selector [ny_loc * window_size, n_lat] from
      `_lat_window_selector`.
    window_size: rows per window.
    is_tpu: whether the computation targets TPUs.

  Returns:
    [ny_loc, L, K_pad, n_lon], `K_pad = round_up(C * window_size, 8)`.
  """
  num_channels, num_levels, n_lon, n_lat = stack.shape
  ny_loc = m_lat_2d.shape[0] // window_size
  k_raw = num_channels * window_size
  k_pad = _round_to_multiple(k_raw, _TPU_SUBLANES)
  stack_3d = stack.swapaxes(0, 1).reshape(
      num_levels * num_channels, n_lon, n_lat
  )
  # The window rows leave the minor axis before they are split off it, so
  # that no intermediate has `window_size` in the lanes.
  projected = (
      _select_last_axis(stack_3d, m_lat_2d, is_tpu)
      .transpose(2, 0, 1)
      .reshape(ny_loc, window_size, num_levels * num_channels, n_lon)
      .transpose(0, 2, 1, 3)
      .reshape(ny_loc, num_levels, k_raw, n_lon)
  )
  if k_pad > k_raw:
    projected = jnp.pad(projected, [(0, 0), (0, 0), (0, k_pad - k_raw), (0, 0)])
  return projected


@dataclasses.dataclass(frozen=True, eq=False)
class _Matmul3DOperands:
  """Everything the channel chunks of one `_matmul_3d_local` call share.

  The arrays are built once, outside every `lax.scan`, so that the scans close
  over them instead of rebuilding them per step.

  Attributes:
    plan: static plan of the call.
    grid: the horizontal grid holding the source data.
    sigma_nodes: source σ nodes.
    k_offset: global index of the first local query level.
    buf_z_origin: source level of the first level of the local buffer.
    win_lat_start: [ny_loc] first source row of each arrival row's window.
    m_lat_2d: [ny_loc * w_lat_size, n_lat_buf] latitude window selector.
    win_lat_cell_start: the same for the limiter's cell windows (None without
      a limiter).
    m_lat_cell_2d: the corresponding selector, [ny_loc * w_lat_cell_size,
      n_lat_buf - 1] (None without a limiter).
    ref_buf: [l_buf] limiter bound reference on the buffer levels (None if
      there is none or no limiter).
    lon_q_t: [l_q_loc, ny_loc, nx_loc] query longitudes. The queries are
      transposed once here so that `nx_loc` stays the last axis throughout the
      level scan, with no per-level transposes.
    sin_lat_q_t: query sin(latitude), laid out like `lon_q_t`.
    sigma_q_t: query σ, laid out like `lon_q_t`.
    k_arr: [l_q_loc] local query level indices.
    lat_bounds_t: latitude clamp bounds, each [ny_loc, 1].
    sig_lo_arr: [l_q_loc] lower σ clamp per query level.
    sig_hi_arr: [l_q_loc] upper σ clamp per query level.
    m_arange: [n_lon, 1] longitude positions.
    lat_arange: [w_lat_size, 1] latitude window positions.
    dot_prec_rhs_binary: dot precision for products whose right operand is an
      exact 0/1 matrix.
    dot_prec_f32: dot precision for the field contractions.
    dot_prec_tri: dot precision for the trilinear contraction.
  """

  plan: _Matmul3DPlan
  grid: spherical_harmonic.Grid
  sigma_nodes: np.ndarray
  k_offset: int | jax.Array
  buf_z_origin: int | jax.Array
  win_lat_start: jax.Array
  m_lat_2d: jax.Array
  win_lat_cell_start: jax.Array | None
  m_lat_cell_2d: jax.Array | None
  ref_buf: jax.Array | None
  lon_q_t: jax.Array
  sin_lat_q_t: jax.Array
  sigma_q_t: jax.Array
  k_arr: jax.Array
  lat_bounds_t: tuple[jax.Array, jax.Array]
  sig_lo_arr: jax.Array
  sig_hi_arr: jax.Array
  m_arange: jax.Array
  lat_arange: jax.Array
  dot_prec_rhs_binary: jax.lax.PrecisionLike
  dot_prec_f32: jax.lax.PrecisionLike
  dot_prec_tri: jax.lax.PrecisionLike


@dataclasses.dataclass(frozen=True)
class _ChannelChunk:
  """Static layout of one chunk of channels.

  Attributes:
    c_cur: channels in the chunk.
    lim_start: first limited channel of the chunk.
    tri_start: first trilinear channel of the chunk, or None.
    c_lim: number of limited channels.
    c_tri: number of trilinear channels.
    k_raw: `c_cur * w_lat_size`, the K extent of the projected field.
    k_pad: `k_raw` padded to the TPU sublane count.
    k_lim_raw: `2 * c_lim * w_lat_cell_size`, the K extent of the projected
      cell bounds (a lower and an upper bound per limited channel).
    k_lim_pad: `k_lim_raw` padded to the TPU sublane count.
  """

  c_cur: int
  lim_start: int
  tri_start: int | None
  c_lim: int
  c_tri: int
  k_raw: int
  k_pad: int
  k_lim_raw: int
  k_lim_pad: int


def _channel_chunk(
    plan: _Matmul3DPlan,
    c_cur: int,
    lim_start: int,
    tri_start: int | None,
) -> _ChannelChunk:
  """Layout of a chunk of `c_cur` channels; see `_ChannelChunk`."""
  c_lim = max(0, c_cur - lim_start) if plan.limiter is not None else 0
  c_tri = max(0, c_cur - tri_start) if tri_start is not None else 0
  k_raw = c_cur * plan.w_lat_size
  k_lim_raw = 0
  k_lim_pad = 0
  if c_lim > 0:
    k_lim_raw = 2 * c_lim * plan.w_lat_cell_size
    k_lim_pad = _round_to_multiple(k_lim_raw, _TPU_SUBLANES)
  return _ChannelChunk(
      c_cur=c_cur,
      lim_start=lim_start,
      tri_start=tri_start,
      c_lim=c_lim,
      c_tri=c_tri,
      k_raw=k_raw,
      k_pad=_round_to_multiple(k_raw, _TPU_SUBLANES),
      k_lim_raw=k_lim_raw,
      k_lim_pad=k_lim_pad,
  )


@dataclasses.dataclass(frozen=True, eq=False)
class _LevelBlock:
  """Static layout and window positions of a block of `b_len` query levels.

  The block's levels share one vertical window, one longitude contraction and
  one latitude weight matrix, with the levels folded into the query axis
  (`nx_blk = b_len * nx_loc`, level-major).

  Attributes:
    b_len: query levels in the block.
    nx_blk: `b_len * nx_loc`.
    w_z_blk_size: levels of the block's vertical stencil window.
    w_z_lin_blk_size: levels of the block's linear-stencil window.
    w_z_cell_blk_size: levels of the block's limiter cell-bound window (0
      without limited channels).
    fuse_gath_blk: whether the longitude contraction of the whole block fits
      in on-chip memory at once (always for `b_len == 1`); otherwise the
      levels are contracted one at a time, ordered by optimization barriers.
    z_blk_arange: [w_z_blk_size, 1] vertical window positions.
    z_lin_blk_arange: [w_z_lin_blk_size, 1] linear-stencil window positions.
    z_cell_blk_arange: [w_z_cell_blk_size, 1] cell-bound window positions
      (None without limited channels).
    lat_cell_arange: [w_lat_cell_size, 1] cell-bound latitude window
      positions (None without limited channels).
  """

  b_len: int
  nx_blk: int
  w_z_blk_size: int
  w_z_lin_blk_size: int
  w_z_cell_blk_size: int
  fuse_gath_blk: bool
  z_blk_arange: jax.Array
  z_lin_blk_arange: jax.Array
  z_cell_blk_arange: jax.Array | None
  lat_cell_arange: jax.Array | None


def _level_block(
    plan: _Matmul3DPlan, chunk: _ChannelChunk, b_len: int
) -> _LevelBlock:
  """Layout of blocks of `b_len` query levels of a channel chunk."""
  w_z_blk_size = plan.block_window(plan.w_z_size, b_len)
  w_z_lin_blk_size = plan.block_window(plan.w_z_lin_size, b_len)
  w_z_cell_blk_size = (
      plan.block_window(plan.w_z_cell_size, b_len, cells=True)
      if chunk.c_lim > 0
      else 0
  )
  nx_blk = b_len * plan.nx_loc
  nx_blk_pad128 = _round_to_multiple(nx_blk, _TPU_LANES)
  f_patch_blk_b = (
      plan.ny_loc
      * w_z_blk_size
      * chunk.k_pad
      * plan.n_lon_pad128
      * plan.item_bytes
  )
  w_lon_blk_b = (
      plan.ny_loc
      * _round_to_multiple(plan.n_lon, _TPU_SUBLANES)
      * nx_blk_pad128
      * plan.item_bytes
  )
  gath_blk_b = (
      plan.ny_loc * w_z_blk_size * chunk.k_pad * nx_blk_pad128 * plan.item_bytes
  )
  fuse_gath_blk = (
      b_len == 1
      or (f_patch_blk_b + w_lon_blk_b + gath_blk_b) <= plan.vmem_limit
  )

  z_blk_arange = jnp.arange(w_z_blk_size, dtype=jnp.int32)[:, jnp.newaxis]
  z_lin_blk_arange = jnp.arange(w_z_lin_blk_size, dtype=jnp.int32)[
      :, jnp.newaxis
  ]
  z_cell_blk_arange = (
      jnp.arange(w_z_cell_blk_size, dtype=jnp.int32)[:, jnp.newaxis]
      if chunk.c_lim > 0
      else None
  )
  lat_cell_arange = (
      jnp.arange(plan.w_lat_cell_size, dtype=jnp.int32)[:, jnp.newaxis]
      if chunk.c_lim > 0
      else None
  )
  return _LevelBlock(
      b_len=b_len,
      nx_blk=nx_blk,
      w_z_blk_size=w_z_blk_size,
      w_z_lin_blk_size=w_z_lin_blk_size,
      w_z_cell_blk_size=w_z_cell_blk_size,
      fuse_gath_blk=fuse_gath_blk,
      z_blk_arange=z_blk_arange,
      z_lin_blk_arange=z_lin_blk_arange,
      z_cell_blk_arange=z_cell_blk_arange,
      lat_cell_arange=lat_cell_arange,
  )


def _level_block_length(
    num_levels: int,
    is_tpu: bool,
    block_bytes,
    limit: int,
    max_block_levels: int | None = None,
) -> int:
  """Longest block of consecutive query levels a kernel may evaluate at once.

  On TPU the levels can be evaluated `_TPU_BLOCK_LEVELS` at a time, sharing
  one vertical window and one (wider) longitude contraction, provided what
  `block_bytes` measures for such a block stays within `limit`. Elsewhere,
  and otherwise, one at a time.

  Args:
    num_levels: number of local query levels.
    is_tpu: whether the TPU layouts apply; otherwise blocks have one level.
    block_bytes: function from a block length to the padded bytes that must
      stay within `limit` for a block of that length to be used.
    limit: the budget for `block_bytes`.
    max_block_levels: optional cap on the block length (the longest block
      whose vertical window fits a z-shard's buffer).

  Returns:
    `_TPU_BLOCK_LEVELS` or 1.
  """
  if (
      is_tpu
      and num_levels >= _TPU_BLOCK_LEVELS
      and (max_block_levels is None or max_block_levels >= _TPU_BLOCK_LEVELS)
      and block_bytes(_TPU_BLOCK_LEVELS) <= limit
  ):
    return _TPU_BLOCK_LEVELS
  return 1


def _scan_level_blocks(
    block_fn,
    k_arr: jax.Array,
    lon_q_t: jax.Array,
    sin_lat_q_t: jax.Array,
    sigma_q_t: jax.Array,
    z_bounds: Sequence[jax.Array],
    b_len: int,
):
  """Scans `block_fn` over the query levels in blocks of `b_len` levels.

  The levels are padded to a multiple of `b_len` by repeating the last one
  (the padded results are dropped) and folded into `[n_blocks, b_len, ...]`.

  Args:
    block_fn: maps `(k_b, lon_b, sin_lat_b, sigma_b, *z_bounds_b)` for one
      block to the block's result (`[b_len, ...]` leaves for `b_len > 1`,
      per-level leaves for `b_len == 1`).
    k_arr: [l] query level indices.
    lon_q_t: [l, ny_loc, nx_loc] query longitudes.
    sin_lat_q_t: query sin(latitude), same layout.
    sigma_q_t: query σ, same layout.
    z_bounds: [l] σ clamp bounds, one array per stencil.
    b_len: levels per block.

  Returns:
    The block results stacked over the query levels: a pytree whose leaves
    have a leading axis of size `l`.
  """
  # TODO(dinosaur): reverse-mode memory at high resolution. Differentiating
  # this scan (and the channel scan of `_map_channel_chunks` around it) with
  # respect to the fields and the departure coordinates stores every step's
  # residuals: the one-hot weight matrices and the windowed patches, i.e.
  # several working sets per block and channel chunk for the whole transport,
  # whatever checkpointing the caller applies to its time steps. The gradient
  # tests run at T21/T31, where that is small. Training at TL127 and above
  # will probably need `jax.checkpoint` around `block_fn` (recomputing a
  # block's stencils in the backward pass) to keep it within device memory.
  num_levels, ny_loc, nx_loc = lon_q_t.shape
  if b_len == 1:
    _, scan_res = lax.scan(
        lambda _, inp: (None, block_fn(inp)),
        None,
        (k_arr, lon_q_t, sin_lat_q_t, sigma_q_t, *z_bounds),
    )
    return scan_res
  n_blocks = (num_levels + b_len - 1) // b_len
  pad_l = n_blocks * b_len - num_levels
  if pad_l > 0:
    k_in = jnp.pad(k_arr, (0, pad_l), mode='edge').reshape(n_blocks, b_len)
    lon_in = jnp.pad(
        lon_q_t, [(0, pad_l), (0, 0), (0, 0)], mode='edge'
    ).reshape(n_blocks, b_len, ny_loc, nx_loc)
    sin_lat_in = jnp.pad(
        sin_lat_q_t, [(0, pad_l), (0, 0), (0, 0)], mode='edge'
    ).reshape(n_blocks, b_len, ny_loc, nx_loc)
    sigma_in = jnp.pad(
        sigma_q_t, [(0, pad_l), (0, 0), (0, 0)], mode='edge'
    ).reshape(n_blocks, b_len, ny_loc, nx_loc)
    z_bounds_in = [
        jnp.pad(b, (0, pad_l), mode='edge').reshape(n_blocks, b_len)
        for b in z_bounds
    ]
  else:
    k_in = k_arr.reshape(n_blocks, b_len)
    lon_in = lon_q_t.reshape(n_blocks, b_len, ny_loc, nx_loc)
    sin_lat_in = sin_lat_q_t.reshape(n_blocks, b_len, ny_loc, nx_loc)
    sigma_in = sigma_q_t.reshape(n_blocks, b_len, ny_loc, nx_loc)
    z_bounds_in = [b.reshape(n_blocks, b_len) for b in z_bounds]
  _, scan_res = lax.scan(
      lambda _, inp: (None, block_fn(inp)),
      None,
      (k_in, lon_in, sin_lat_in, sigma_in, *z_bounds_in),
  )
  return jax.tree_util.tree_map(
      lambda x: x.reshape((n_blocks * b_len,) + x.shape[2:])[:num_levels],
      scan_res,
  )


def _trilinear_block_output(
    f_lat: jax.Array,
    ops: _Matmul3DOperands,
    chunk: _ChannelChunk,
    block: _LevelBlock,
    *,
    w_lon_mat: jax.Array,
    w_lat_mat: jax.Array,
    w_z_mat: jax.Array,
    oh_lon: tuple[jax.Array, ...],
    lat_c0: jax.Array,
    lvl_c0: jax.Array,
    alphas: tuple[jax.Array, jax.Array, jax.Array],
    k_glob: jax.Array,
    buf_z_start: jax.Array,
) -> jax.Array:
  """Trilinear (low-order) interpolation of a block's trilinear channels.

  Used when the main stencil is not already trilinear: the weights are the
  linear weights of the cell that brackets the query (`alphas` are the
  fractional positions in it), applied to the trilinear channels of `f_lat`
  (those from `chunk.tri_start` on) in a linear-stencil vertical window.

  Args:
    f_lat: [ny_loc, l_buf, k_pad, n_lon] latitude-projected channel chunk.
    ops: the shared operands of the call.
    chunk: layout of the channel chunk.
    block: layout of the level block.
    w_lon_mat: [ny_loc, n_lon, nx_blk] longitude weights of the main stencil.
    w_lat_mat: [ny_loc, w_lat_size, nx_blk] latitude window weights of the
      main stencil.
    w_z_mat: [ny_loc, w_z_blk_size, nx_blk] vertical window weights of the
      main stencil.
    oh_lon: per longitude stencil point, [ny_loc, n_lon, nx_blk] one-hots.
    lat_c0: [ny_loc, nx_blk] first latitude row of the bracketing cell.
    lvl_c0: [ny_loc, nx_blk] first level of the bracketing cell.
    alphas: fractional position in the bracketing cell, `(lon, lat, z)`.
    k_glob: global level of the block's first query level.
    buf_z_start: start of the main vertical window within the buffer.

  Returns:
    [ny_loc, c_tri, nx_blk] trilinear values.
  """
  plan = ops.plan
  ny_loc = plan.ny_loc
  n_lon = plan.n_lon
  w_lat_size = plan.w_lat_size
  c_tri = chunk.c_tri
  alpha_lon, alpha_lat, alpha_z = alphas
  c0_lon_pos = 0 if plan.order == 'linear' else 1
  if plan.order == 'linear':
    w_lon_lin_mat = w_lon_mat
    w_lat_lin_mat = w_lat_mat
  else:
    a_lon = alpha_lon[:, jnp.newaxis, :]
    w_lon_lin_mat = (1.0 - a_lon) * oh_lon[c0_lon_pos] + a_lon * oh_lon[
        c0_lon_pos + 1
    ]
    a_lat = alpha_lat[:, jnp.newaxis, :]
    lat_rel = (lat_c0 - ops.win_lat_start[:, jnp.newaxis])[:, jnp.newaxis, :]
    w_lat_lin_mat = (1.0 - a_lat) * (lat_rel == ops.lat_arange).astype(
        plan.dtype
    ) + a_lat * (lat_rel + 1 == ops.lat_arange).astype(plan.dtype)
  if plan.vertical_order == 'linear':
    buf_z_lin_start = buf_z_start
    w_z_lin_mat = w_z_mat
  else:
    win_z_lin_start, buf_z_lin_start = _z_window_starts(
        k_glob - plan.eff_z_shift,
        plan.l_src,
        block.w_z_lin_blk_size,
        ops.buf_z_origin,
        plan.l_buf,
    )
    a_z = alpha_z[:, jnp.newaxis, :]
    z_rel = (lvl_c0 - win_z_lin_start)[:, jnp.newaxis, :]
    w_z_lin_mat = (1.0 - a_z) * (z_rel == block.z_lin_blk_arange).astype(
        plan.dtype
    ) + a_z * (z_rel + 1 == block.z_lin_blk_arange).astype(plan.dtype)

  k_tri_raw = c_tri * w_lat_size
  k_tri_pad = _round_to_multiple(k_tri_raw, _TPU_SUBLANES)
  f_tri_win = lax.dynamic_slice(
      f_lat,
      (
          jnp.int32(0),
          buf_z_lin_start,
          jnp.int32(chunk.tri_start * w_lat_size),
          jnp.int32(0),
      ),
      (ny_loc, block.w_z_lin_blk_size, k_tri_raw, n_lon),
  )
  if k_tri_pad > k_tri_raw:
    f_tri_win = jnp.pad(
        f_tri_win,
        [(0, 0), (0, 0), (0, k_tri_pad - k_tri_raw), (0, 0)],
    )
  f_tri_patch = f_tri_win.reshape(
      ny_loc, block.w_z_lin_blk_size * k_tri_pad, n_lon
  )
  tri_lon = jnp.einsum(
      'jkm,jmi->jki', f_tri_patch, w_lon_lin_mat, precision=ops.dot_prec_tri
  )
  tri_after_z = (
      tri_lon.reshape(ny_loc, block.w_z_lin_blk_size, k_tri_pad, block.nx_blk)
      * w_z_lin_mat[:, :, jnp.newaxis, :]
  ).sum(axis=1)
  return (
      tri_after_z[:, :k_tri_raw, :].reshape(
          ny_loc, c_tri, w_lat_size, block.nx_blk
      )
      * w_lat_lin_mat[:, jnp.newaxis, :, :]
  ).sum(axis=2)


def _limit_block_values(
    val_jci: jax.Array,
    b_lat: jax.Array,
    ops: _Matmul3DOperands,
    chunk: _ChannelChunk,
    block: _LevelBlock,
    *,
    oh_lon: tuple[jax.Array, ...],
    lat_c0: jax.Array,
    lvl_c0: jax.Array,
    w_z_mat: jax.Array,
    k_glob: jax.Array,
    buf_z_start: jax.Array,
) -> jax.Array:
  """Clips the limited channels of a block to their bracketing-cell bounds.

  The cell bounds of every query are selected from the latitude-projected
  bounds `b_lat` with one-hot weights of the bracketing cell (in a cell
  window of the block's levels), then each limited channel is clipped to
  its `[lo, hi]`, in the shifted variable if there is a bound reference.

  Args:
    val_jci: [ny_loc, c_cur, nx_blk] interpolated values of the chunk.
    b_lat: [ny_loc, l_buf - 1, k_lim_pad, n_lon] latitude-projected cell
      bounds.
    ops: the shared operands of the call.
    chunk: layout of the channel chunk.
    block: layout of the level block.
    oh_lon: per longitude stencil point, [ny_loc, n_lon, nx_blk] one-hots.
    lat_c0: [ny_loc, nx_blk] first latitude row of the bracketing cell.
    lvl_c0: [ny_loc, nx_blk] first level of the bracketing cell.
    w_z_mat: [ny_loc, w_z_blk_size, nx_blk] vertical window weights.
    k_glob: global level of the block's first query level.
    buf_z_start: start of the vertical window within the buffer.

  Returns:
    `val_jci` with the limited channels clipped.
  """
  plan = ops.plan
  ny_loc = plan.ny_loc
  n_lon = plan.n_lon
  assert (
      ops.win_lat_cell_start is not None
      and block.z_cell_blk_arange is not None
      and block.lat_cell_arange is not None
  )
  c_lim = chunk.c_lim
  win_z_cell_start, buf_z_cell_start = _z_window_starts(
      k_glob - plan.eff_z_shift,
      plan.l_src - 1,
      block.w_z_cell_blk_size,
      ops.buf_z_origin,
      plan.l_buf - 1,
  )
  p_lon_mat = oh_lon[0 if plan.order == 'linear' else 1]
  p_z_mat = (
      (lvl_c0 - win_z_cell_start)[:, jnp.newaxis, :] == block.z_cell_blk_arange
  ).astype(plan.dtype)
  p_lat_mat = (
      (lat_c0 - ops.win_lat_cell_start[:, jnp.newaxis])[:, jnp.newaxis, :]
      == block.lat_cell_arange
  ).astype(plan.dtype)
  b_win = lax.dynamic_slice_in_dim(
      b_lat, buf_z_cell_start, block.w_z_cell_blk_size, axis=1
  )
  b_patch = b_win.reshape(
      ny_loc, block.w_z_cell_blk_size * chunk.k_lim_pad, n_lon
  )
  # `p_lon_mat` is an exact 0/1 one-hot: the bounds keep full precision.
  b_lon = jnp.einsum(
      'jkm,jmi->jki', b_patch, p_lon_mat, precision=ops.dot_prec_rhs_binary
  )
  b_after_z = (
      b_lon.reshape(
          ny_loc, block.w_z_cell_blk_size, chunk.k_lim_pad, block.nx_blk
      )
      * p_z_mat[:, :, jnp.newaxis, :]
  ).sum(axis=1)
  b_jci = (
      b_after_z[:, : chunk.k_lim_raw, :].reshape(
          ny_loc, 2 * c_lim, plan.w_lat_cell_size, block.nx_blk
      )
      * p_lat_mat[:, jnp.newaxis, :, :]
  ).sum(axis=2)
  lo_jci, hi_jci = b_jci[:, :c_lim, :], b_jci[:, c_lim:, :]
  val_lim = val_jci[:, chunk.lim_start :, :]
  if ops.ref_buf is not None:
    ref_win = lax.dynamic_slice_in_dim(
        ops.ref_buf, buf_z_start, block.w_z_blk_size, axis=0
    )
    ref_ji = (w_z_mat * ref_win[jnp.newaxis, :, jnp.newaxis]).sum(axis=1)[
        :, jnp.newaxis, :
    ]
    val_lim = jnp.clip(val_lim + ref_ji, lo_jci, hi_jci) - ref_ji
  else:
    val_lim = jnp.clip(val_lim, lo_jci, hi_jci)
  return (
      val_lim
      if chunk.lim_start == 0
      else jnp.concatenate([val_jci[:, : chunk.lim_start, :], val_lim], axis=1)
  )


def _eval_level_block(
    inputs,
    ops: _Matmul3DOperands,
    chunk: _ChannelChunk,
    block: _LevelBlock,
    f_lat: jax.Array,
    b_lat: jax.Array | None,
):
  """Interpolates one block of query levels of a channel chunk.

  Builds the block's stencils and one-hot weight matrices, contracts the
  latitude-projected field `f_lat` with them (longitude first, then the
  vertical and latitude windows), applies the limiter, and with trilinear
  channels also evaluates their low-order output.

  Args:
    inputs: `(k_b, lon_b, sin_lat_b, sigma_b, sig_lo_b, sig_hi_b)` of one
      block: for `b_len == 1` a scalar level index, [ny_loc, nx_loc] query
      coordinates and scalar σ clamp bounds; otherwise [b_len] indices,
      [b_len, ny_loc, nx_loc] coordinates and [b_len] bounds.
    ops: the shared operands of the call.
    chunk: layout of the channel chunk.
    block: layout of the level block.
    f_lat: [ny_loc, l_buf, k_pad, n_lon] latitude-projected channel chunk.
    b_lat: [ny_loc, l_buf - 1, k_lim_pad, n_lon] latitude-projected cell
      bounds of the limited channels (None without them).

  Returns:
    `(values, trilinear)`. `values` is [c_cur, ny_loc, nx_loc] for
    `b_len == 1`, else [b_len, c_cur, ny_loc, nx_loc]; `trilinear` is the
    trilinear output of the chunk's trilinear channels ([c_tri, ...], same
    layout), or None if it has none.
  """
  plan = ops.plan
  ny_loc = plan.ny_loc
  nx_loc = plan.nx_loc
  n_lon = plan.n_lon
  b_len = block.b_len
  nx_blk = block.nx_blk
  c_cur = chunk.c_cur
  c_tri = chunk.c_tri
  dtype = plan.dtype
  k_b, lon_b, sin_lat_b, sigma_b, sig_lo_b, sig_hi_b = inputs
  if b_len == 1:
    k0_s = k_b
    lon_s = lon_b
    sin_lat_s = sin_lat_b
    sigma_s = sigma_b
    sigma_bounds_s = (sig_lo_b, sig_hi_b)
  else:
    k0_s = k_b[0]
    lon_s = jnp.swapaxes(lon_b, 0, 1).reshape(ny_loc, nx_blk)
    sin_lat_s = jnp.swapaxes(sin_lat_b, 0, 1).reshape(ny_loc, nx_blk)
    sigma_s = jnp.swapaxes(sigma_b, 0, 1).reshape(ny_loc, nx_blk)
    # One clamp per query level of the block (not per block), laid out
    # like the level-major `nx_blk` axis.
    sigma_bounds_s = tuple(
        jnp.broadcast_to(b[:, jnp.newaxis], (b_len, nx_loc)).reshape(1, nx_blk)
        for b in (sig_lo_b, sig_hi_b)
    )

  separate_trilinear = c_tri > 0 and plan.need_separate_trilinear
  h_stencil = lagrange_interpolation.horizontal_stencil(
      ops.grid,
      lon_s,
      sin_lat_s,
      plan.order,
      include_linear_fractions=separate_trilinear,
      lat_bounds=ops.lat_bounds_t,
  )
  lon_idx, lon_w = h_stencil.lon_indices, h_stencil.lon_weights
  lat_idx, lat_w = h_stencil.lat_indices, h_stencil.lat_weights
  lat_c0 = h_stencil.lat_cell
  alpha_lon, alpha_lat = h_stencil.linear_fractions or (None, None)
  lvl_idx, lvl_w, _, lvl_c0, alpha_z = lagrange_interpolation.vertical_stencil(
      sigma_s,
      ops.sigma_nodes,
      plan.vertical_order,
      is_tpu=plan.is_tpu,
      include_linear_fractions=separate_trilinear,
      sigma_bounds=sigma_bounds_s,
  )

  k_glob = jnp.minimum(ops.k_offset + k0_s, plan.l_src - 1)
  win_z_start, buf_z_start = _z_window_starts(
      k_glob + plan.z_start_offset,
      plan.l_src,
      block.w_z_blk_size,
      ops.buf_z_origin,
      plan.l_buf,
  )

  oh_lon = tuple(
      (idx[:, jnp.newaxis, :] == ops.m_arange).astype(dtype) for idx in lon_idx
  )
  w_lon_mat = sum(w[:, jnp.newaxis, :] * oh for w, oh in zip(lon_w, oh_lon))
  w_z_mat = _weighted_window_one_hot(
      lvl_idx, lvl_w, block.z_blk_arange, dtype, win_z_start
  )
  w_lat_mat = _weighted_window_one_hot(
      lat_idx, lat_w, ops.lat_arange, dtype, ops.win_lat_start
  )

  # Slice the block vertical window once from `f_lat` across all `b_len`
  # query levels so overlapping vertical levels are reused in VMEM.
  f_win = lax.dynamic_slice_in_dim(
      f_lat, buf_z_start, block.w_z_blk_size, axis=1
  )

  val_lin_jci = None
  if c_tri > 0 and plan.need_separate_trilinear:
    assert (
        alpha_lon is not None and alpha_lat is not None and alpha_z is not None
    )
    val_lin_jci = _trilinear_block_output(
        f_lat,
        ops,
        chunk,
        block,
        w_lon_mat=w_lon_mat,
        w_lat_mat=w_lat_mat,
        w_z_mat=w_z_mat,
        oh_lon=oh_lon,
        lat_c0=lat_c0,
        lvl_c0=lvl_c0,
        alphas=(alpha_lon, alpha_lat, alpha_z),
        k_glob=k_glob,
        buf_z_start=buf_z_start,
    )

  f_patch = f_win.reshape(ny_loc, block.w_z_blk_size * chunk.k_pad, n_lon)
  if block.fuse_gath_blk:
    gathered_lon = jnp.einsum(
        'jkm,jmi->jki', f_patch, w_lon_mat, precision=ops.dot_prec_f32
    )
    after_z = (
        gathered_lon.reshape(ny_loc, block.w_z_blk_size, chunk.k_pad, nx_blk)
        * w_z_mat[:, :, jnp.newaxis, :]
    ).sum(axis=1)
    val_jci = (
        after_z[:, : chunk.k_raw, :].reshape(
            ny_loc, c_cur, plan.w_lat_size, nx_blk
        )
        * w_lat_mat[:, jnp.newaxis, :, :]
    ).sum(axis=2)
  else:
    val_parts = []
    f_patch_cur = f_patch
    for b_idx in range(b_len):
      s0 = b_idx * nx_loc
      s1 = s0 + nx_loc
      g_b = jnp.einsum(
          'jkm,jmi->jki',
          f_patch_cur,
          w_lon_mat[:, :, s0:s1],
          precision=ops.dot_prec_f32,
      )
      az_b = (
          g_b.reshape(ny_loc, block.w_z_blk_size, chunk.k_pad, nx_loc)
          * w_z_mat[:, :, jnp.newaxis, s0:s1]
      ).sum(axis=1)
      v_b = (
          az_b[:, : chunk.k_raw, :].reshape(
              ny_loc, c_cur, plan.w_lat_size, nx_loc
          )
          * w_lat_mat[:, jnp.newaxis, :, s0:s1]
      ).sum(axis=2)
      if b_idx + 1 < b_len:
        v_b, f_patch_cur = lax.optimization_barrier((v_b, f_patch_cur))
      val_parts.append(v_b)
    val_jci = jnp.concatenate(val_parts, axis=2)

  if c_tri > 0 and not plan.need_separate_trilinear:
    val_lin_jci = val_jci[:, chunk.tri_start :, :]

  if chunk.c_lim > 0:
    assert b_lat is not None
    val_jci = _limit_block_values(
        val_jci,
        b_lat,
        ops,
        chunk,
        block,
        oh_lon=oh_lon,
        lat_c0=lat_c0,
        lvl_c0=lvl_c0,
        w_z_mat=w_z_mat,
        k_glob=k_glob,
        buf_z_start=buf_z_start,
    )

  if b_len == 1:
    return (
        jnp.swapaxes(val_jci, 0, 1),
        jnp.swapaxes(val_lin_jci, 0, 1) if c_tri > 0 else None,
    )

  out_blk = val_jci.reshape(ny_loc, c_cur, b_len, nx_loc).transpose(2, 1, 0, 3)
  out_lin_blk = None
  if c_tri > 0:
    out_lin_blk = val_lin_jci.reshape(ny_loc, c_tri, b_len, nx_loc).transpose(
        2, 1, 0, 3
    )
  return out_blk, out_lin_blk


def _matmul_3d_channel_chunk(
    f_chunk: jax.Array,
    lim_start: int,
    tri_start: int | None,
    ops: _Matmul3DOperands,
) -> tuple[jax.Array, jax.Array | None]:
  """Interpolates a chunk of channels at every local query.

  Projects the chunk (and the limited channels' cell bounds) onto the
  latitude windows, then evaluates the query levels in blocks.

  Args:
    f_chunk: [c_cur, l_buf, n_lon, n_lat_buf] channels of the chunk.
    lim_start: first limited channel of the chunk.
    tri_start: first trilinear channel of the chunk, or None.
    ops: the shared operands of the call.

  Returns:
    `(values, trilinear)`: [c_cur, l_q_loc, nx_loc, ny_loc], and the trilinear
    output of the chunk's trilinear channels, [c_tri, l_q_loc, nx_loc, ny_loc]
    (with zero channels if the chunk has none; None if `tri_start` is None).
  """
  plan = ops.plan
  dtype = plan.dtype
  l_q_loc = plan.l_q_loc
  nx_loc = plan.nx_loc
  ny_loc = plan.ny_loc
  chunk = _channel_chunk(plan, f_chunk.shape[0], lim_start, tri_start)
  f_lat = _project_lat_window(
      f_chunk, ops.m_lat_2d, plan.w_lat_size, plan.is_tpu
  )

  b_lat = None
  if chunk.c_lim > 0:
    assert ops.m_lat_cell_2d is not None
    f_lim = f_chunk[lim_start:]
    f_for_bounds = (
        f_lim + ops.ref_buf[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
        if ops.ref_buf is not None
        else f_lim
    )
    lo_buf, hi_buf = lagrange_interpolation.cell_bounds_3d(f_for_bounds)
    # The bounds must reach the clip unrounded, or a constant field (or the
    # zero of a non-negative one) would not stay inside its own bounds. Both
    # matmuls that select them have an exact 0/1 right operand and keep the
    # left one at full float32 precision: this one and the longitude
    # selection in `_limit_block_values`. The remaining selections there are
    # elementwise products with 0/1 masks, summed over a single non-zero term.
    b_lat = _project_lat_window(
        jnp.concatenate([lo_buf, hi_buf], axis=0),
        ops.m_lat_cell_2d,
        plan.w_lat_cell_size,
        plan.is_tpu,
    )

  no_trilinear = (
      None
      if tri_start is None
      else jnp.zeros((0, l_q_loc, nx_loc, ny_loc), dtype=dtype)
  )
  if l_q_loc == 1:
    block = _level_block(plan, chunk, 1)
    out_0, out_lin_0 = _eval_level_block(
        (
            jnp.int32(0),
            ops.lon_q_t[0],
            ops.sin_lat_q_t[0],
            ops.sigma_q_t[0],
            ops.sig_lo_arr[0],
            ops.sig_hi_arr[0],
        ),
        ops,
        chunk,
        block,
        f_lat,
        b_lat,
    )
    return (
        jnp.swapaxes(out_0, 1, 2)[:, jnp.newaxis, :, :],
        jnp.swapaxes(out_lin_0, 1, 2)[:, jnp.newaxis, :, :]
        if chunk.c_tri > 0
        else no_trilinear,
    )

  b_len = plan.block_levels
  block = _level_block(plan, chunk, b_len)
  scan_res = _scan_level_blocks(
      functools.partial(
          _eval_level_block,
          ops=ops,
          chunk=chunk,
          block=block,
          f_lat=f_lat,
          b_lat=b_lat,
      ),
      ops.k_arr,
      ops.lon_q_t,
      ops.sin_lat_q_t,
      ops.sigma_q_t,
      (ops.sig_lo_arr, ops.sig_hi_arr),
      b_len,
  )
  # [l_q_loc, c, ny_loc, nx_loc] -> [c, l_q_loc, nx_loc, ny_loc].
  out_all, out_lin_all = jax.tree_util.tree_map(
      lambda x: x.transpose(1, 0, 3, 2), scan_res
  )
  return out_all, out_lin_all if chunk.c_tri > 0 else no_trilinear


def _map_channel_chunks(
    eval_chunk,
    f_buf: jax.Array,
    plan: _Matmul3DPlan,
    chunk_size: int,
    limited_channel_start: int,
    trilinear_channel_start: int | None,
) -> tuple[jax.Array, jax.Array | None]:
  """Evaluates the channels in chunks of `chunk_size` channels.

  A single chunk is evaluated directly. When every channel is limited and
  trilinear alike (or the setting is uniform), the channels are zero-padded
  to a multiple of the chunk and scanned. Otherwise the chunks differ in
  their limited / trilinear channels and are evaluated one after another.

  Args:
    eval_chunk: `_matmul_3d_channel_chunk` with the operands bound.
    f_buf: local buffer, [C, L_buf, n_lon, n_lat_buf].
    plan: the static plan of the call.
    chunk_size: channels per chunk, from `_plan_channels`.
    limited_channel_start: first limited channel.
    trilinear_channel_start: first trilinear channel, or None.

  Returns:
    `(values, trilinear)`: the interpolated channels, [C, l_q_loc, nx_loc,
    ny_loc], and the trilinear output (None if `trilinear_channel_start` is
    None).
  """
  # See the note on reverse-mode memory in `_scan_level_blocks`.
  num_channels = plan.num_channels
  l_q_loc = plan.l_q_loc
  nx_loc = plan.nx_loc
  ny_loc = plan.ny_loc
  if num_channels <= chunk_size:
    return eval_chunk(f_buf, limited_channel_start, trilinear_channel_start)

  if limited_channel_start == 0 and trilinear_channel_start in (None, 0):
    c_pad = _round_to_multiple(num_channels, chunk_size)
    f_pad = (
        jnp.pad(f_buf, [(0, c_pad - num_channels), (0, 0), (0, 0), (0, 0)])
        if c_pad > num_channels
        else f_buf
    )
    f_grouped = f_pad.reshape(
        c_pad // chunk_size, chunk_size, plan.l_buf, plan.n_lon, plan.n_lat_buf
    )
    _, out_grouped = lax.scan(
        lambda _, grp: (None, eval_chunk(grp, 0, trilinear_channel_start)),
        None,
        f_grouped,
    )
    out_g, out_lin_g = jax.tree_util.tree_map(
        lambda x: x.reshape(c_pad, l_q_loc, nx_loc, ny_loc)[:num_channels],
        out_grouped,
    )
    return out_g, out_lin_g

  outs = []
  outs_lin = []
  for c0 in range(0, num_channels, chunk_size):
    c1 = min(c0 + chunk_size, num_channels)
    sub_len = c1 - c0
    sub_lim = max(0, min(sub_len, limited_channel_start - c0))
    if trilinear_channel_start is None or c1 <= trilinear_channel_start:
      sub_tri = None
    else:
      sub_tri = max(0, trilinear_channel_start - c0)
    values, trilinear = eval_chunk(f_buf[c0:c1], sub_lim, sub_tri)
    outs.append(values)
    if sub_tri is not None:
      outs_lin.append(trilinear)
  out_cat = jnp.concatenate(outs, axis=0)
  if trilinear_channel_start is None:
    return out_cat, None
  out_lin_cat = (
      jnp.concatenate(outs_lin, axis=0)
      if outs_lin
      else jnp.zeros((0, l_q_loc, nx_loc, ny_loc), dtype=plan.dtype)
  )
  return out_cat, out_lin_cat


def _matmul_3d_local(
    buffer: _LocalBuffer,
    queries: _LocalQueries,
    spec: _Matmul3DSpec,
    sigma_nodes: np.ndarray,
    limiter_bound_reference: jax.Array | None,
) -> tuple[jax.Array, jax.Array | None]:
  """Windowed MXU matmul 3-D interpolation of one shard's queries.

  The channels are interpolated in chunks sized to the memory budget
  (`_plan_channels`); within a chunk, the field is first restricted to each
  arrival row's latitude window (`_project_lat_window`), then the query
  levels are evaluated in blocks (`_eval_level_block`, scanned by
  `_scan_level_blocks`).

  Args:
    buffer: the local source data.
    queries: the local queries and their clamps.
    spec: what to interpolate.
    sigma_nodes: source σ nodes.
    limiter_bound_reference: per-level limiter bound reference, or None.

  Returns:
    `(values, trilinear)`: [C, l_q_loc, nx_loc, ny_loc], and the trilinear
    output of the channels from `spec.trilinear_channel_start` on,
    [C - trilinear_channel_start, l_q_loc, nx_loc, ny_loc] (None if
    `spec.trilinear_channel_start` is None).
  """
  limited_channel_start = spec.limited_channel_start
  trilinear_channel_start = spec.trilinear_channel_start
  plan, channel_plan = _plan_blocks_and_channels(
      _plan_matmul_3d(buffer, queries, spec, sigma_nodes),
      limited_channel_start,
      trilinear_channel_start,
  )
  if channel_plan.segments is not None:
    return _matmul_3d_by_segments(
        buffer,
        queries,
        spec,
        sigma_nodes,
        limiter_bound_reference,
        channel_plan.segments,
        plan,
    )

  grid = spec.grid
  f_buf = buffer.values
  j_offset = buffer.j_offset
  buf_lat_origin = buffer.lat_origin
  buf_z_origin = buffer.z_origin
  k_offset = buffer.k_offset
  lon_q, sin_lat_q, sigma_q = queries.lon, queries.sin_lat, queries.sigma
  lat_bounds = queries.lat_bounds
  sigma_bounds = queries.sigma_bounds
  dtype = plan.dtype
  j_global = _arrival_lat_rows(j_offset, plan.ny_loc, grid.latitude_nodes)
  win_lat_start, m_lat_2d = _lat_window_selector(
      j_global,
      j_global + plan.lat_start_offset,
      plan.n_lat_ext,
      plan.n_lat_buf,
      plan.w_lat_size,
      buf_lat_origin,
      dtype,
  )
  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, f_buf, f_buf)

  ref_buf = None
  if plan.limiter is not None and limiter_bound_reference is not None:
    ref_arr = jnp.asarray(limiter_bound_reference, dtype=dtype)
    ref_centered = ref_arr - jnp.mean(ref_arr)
    k_buf_levels = jnp.clip(
        buf_z_origin + jnp.arange(plan.l_buf, dtype=jnp.int32),
        0,
        plan.l_src - 1,
    )
    ref_buf = _take_1d(ref_centered, k_buf_levels, plan.is_tpu)

  win_lat_cell_start = None
  m_lat_cell_2d = None
  if plan.limiter is not None:
    win_lat_cell_start, m_lat_cell_2d = _lat_window_selector(
        j_global,
        j_global + 1 - plan.eff_lat_shift,
        plan.n_lat_ext - 1,
        plan.n_lat_buf - 1,
        plan.w_lat_cell_size,
        buf_lat_origin,
        dtype,
    )

  # Queries are pre-transposed once, outside every `lax.scan`, and laid out
  # [ny_loc, nx] inside the scans.
  sig_lo_arr, sig_hi_arr = sigma_bounds
  ops = _Matmul3DOperands(
      plan=plan,
      grid=grid,
      sigma_nodes=sigma_nodes,
      k_offset=k_offset,
      buf_z_origin=buf_z_origin,
      win_lat_start=win_lat_start,
      m_lat_2d=m_lat_2d,
      win_lat_cell_start=win_lat_cell_start,
      m_lat_cell_2d=m_lat_cell_2d,
      ref_buf=ref_buf,
      lon_q_t=jnp.swapaxes(lon_q, 1, 2),
      sin_lat_q_t=jnp.swapaxes(sin_lat_q, 1, 2),
      sigma_q_t=jnp.swapaxes(sigma_q, 1, 2),
      k_arr=jnp.arange(plan.l_q_loc, dtype=jnp.int32),
      lat_bounds_t=tuple(jnp.asarray(b)[:, jnp.newaxis] for b in lat_bounds),
      sig_lo_arr=sig_lo_arr,
      sig_hi_arr=sig_hi_arr,
      m_arange=jnp.arange(plan.n_lon, dtype=jnp.int32)[:, jnp.newaxis],
      lat_arange=jnp.arange(plan.w_lat_size, dtype=jnp.int32)[:, jnp.newaxis],
      dot_prec_rhs_binary=_resolve_rhs_binary_dot_precision(
          f_buf, m_lat_2d, plan.is_tpu
      ),
      dot_prec_f32=dot_prec_f32,
      dot_prec_tri=(
          lax.DotAlgorithmPreset.BF16_BF16_F32_X3
          if (plan.is_tpu and dtype == jnp.float32)
          else dot_prec_f32
      ),
  )
  return _map_channel_chunks(
      functools.partial(_matmul_3d_channel_chunk, ops=ops),
      f_buf,
      plan,
      channel_plan.chunk_size,
      limited_channel_start,
      trilinear_channel_start,
  )


def interpolate_3d_grid(
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
    limited_channel_start: int = 0,
    trilinear_channel_start: int | None = None,
) -> tuple[jax.Array, jax.Array | None]:
  """Grid-aligned 3-D MXU matmul interpolation (single-device and SPMD).

  Returns `(values, trilinear)`; see `_matmul_3d_local`.
  """
  sigma_nodes = jnp.asarray(sigma_nodes)
  mesh = spatial_mesh(grid)
  lat_lo, lat_hi = lagrange_interpolation.lat_clamp_bounds(
      grid, max_lat_shift, order, sin_lat.dtype
  )
  z_shards = mesh_z_shards(mesh)
  # Bounds for every (padded) query level; sharded along 'z' with them.
  sig_lo, sig_hi = lagrange_interpolation.level_clamp_bounds(
      sigma_nodes,
      _round_to_multiple(lon.shape[0], z_shards),
      max_vertical_shift,
      sigma.dtype,
  )
  spec = _Matmul3DSpec(
      grid=grid,
      order=order,
      vertical_order=vertical_order,
      limiter=limiter,
      max_lat_shift=max_lat_shift,
      max_vertical_shift=max_vertical_shift,
      limited_channel_start=limited_channel_start,
      trilinear_channel_start=trilinear_channel_start,
  )
  if mesh is None:
    return _matmul_3d_local(
        _LocalBuffer(
            lagrange_interpolation.extend_with_pole_halo(fields_4d, grid)
        ),
        _LocalQueries(lon, sin_lat, sigma, (lat_lo, lat_hi), (sig_lo, sig_hi)),
        spec,
        sigma_nodes,
        limiter_bound_reference,
    )

  x_shards = mesh.shape['x']
  l_src = fields_4d.shape[1]
  l_q = lon.shape[0]
  l_src_pad = _round_to_multiple(l_src, z_shards)
  l_q_pad = _round_to_multiple(l_q, z_shards)

  f_pad = _edge_pad_levels(fields_4d, 1, l_src_pad)
  lon_pad = _edge_pad_levels(lon, 0, l_q_pad)
  sin_lat_pad = _edge_pad_levels(sin_lat, 0, l_q_pad)
  sigma_pad = _edge_pad_levels(sigma, 0, l_q_pad)

  z_spec = 'z' if z_shards > 1 else None
  f_spec = P(None, z_spec, 'x', 'y')
  q_spec = P(z_spec, 'x', 'y')
  nodes_spec = P(None)
  ref_spec = P(None) if limiter_bound_reference is not None else None
  out_specs = (f_spec, f_spec if trilinear_channel_start is not None else None)

  def _sharded(
      f_loc,
      lon_loc,
      sin_lat_loc,
      sigma_loc,
      nodes_loc,
      ref_loc,
      lat_lo_loc,
      lat_hi_loc,
      sig_lo_loc,
      sig_hi_loc,
  ):
    f_xy_buf, j_offset, buf_lat_origin = _gather_longitude_and_lat_halo(
        f_loc, grid, max_lat_shift, x_shards
    )
    f_xyz_buf, k_offset, buf_z_origin, z_halo = _gather_z_with_halo(
        f_xy_buf,
        z_shards=z_shards,
        n_src=l_src,
        n_src_padded=l_src_pad,
        n_query=l_q,
        n_query_loc=lon_loc.shape[0],
        max_vertical_shift=max_vertical_shift,
    )

    return _matmul_3d_local(
        _LocalBuffer(
            values=f_xyz_buf,
            k_offset=k_offset,
            z_origin=buf_z_origin,
            z_halo=z_halo,
            j_offset=j_offset,
            lat_origin=buf_lat_origin,
        ),
        _LocalQueries(
            lon_loc,
            sin_lat_loc,
            sigma_loc,
            (lat_lo_loc, lat_hi_loc),
            (sig_lo_loc, sig_hi_loc),
        ),
        spec,
        nodes_loc,
        ref_loc,
    )

  out_pad = _shard_map(
      _sharded,
      mesh,
      in_specs=(
          f_spec,
          q_spec,
          q_spec,
          q_spec,
          nodes_spec,
          ref_spec,
          P('y'),
          P('y'),
          P(z_spec),
          P(z_spec),
      ),
      out_specs=out_specs,
  )(
      f_pad,
      lon_pad,
      sin_lat_pad,
      sigma_pad,
      sigma_nodes,
      limiter_bound_reference,
      jnp.asarray(lat_lo),
      jnp.asarray(lat_hi),
      sig_lo,
      sig_hi,
  )
  return jax.tree_util.tree_map(lambda x: x[:, :l_q, :, :], out_pad)


#  =============================================================================
#  Fused wind and vertical-velocity interpolation (departure points).
#  =============================================================================


def _matmul_3d_wind_and_sigma_dot_local(
    wind_buf: jax.Array,
    sd_buf: jax.Array,
    lon_q: jax.Array,
    sin_lat_q: jax.Array,
    sigma_q: jax.Array,
    *,
    grid: spherical_harmonic.Grid,
    centers: jax.Array,
    boundaries: jax.Array,
    j_offset: int | jax.Array,
    buf_lat_origin: int | jax.Array,
    max_lat_shift: int,
    max_vertical_shift: int,
    lat_bounds: tuple[Array, Array],
    centers_bounds: tuple[Array, Array],
    boundaries_bounds: tuple[Array, Array],
) -> tuple[jax.Array, jax.Array]:
  """Fused linear 3-D interpolation of `wind` (centers) + `sigma_dot` (faces).

  `lat_bounds` ([ny_loc] each) clamp the queries' latitude, and
  `centers_bounds` / `boundaries_bounds` ([l_q_loc] each) their σ for the
  `wind` and `sigma_dot` stencils respectively (each node set has its own
  window, see `lagrange_interpolation.level_clamp_bounds`).
  """
  dtype = wind_buf.dtype
  is_tpu = lagrange_interpolation.targets_tpu(grid)
  n_lon = grid.longitude_nodes
  n_lat = grid.latitude_nodes
  n_lat_ext = n_lat + 2 * lagrange_interpolation.HALO_WIDTH
  l_src_c = centers.shape[0]
  l_src_b = boundaries.shape[0]
  _, l_buf_c, _, n_lat_buf = wind_buf.shape
  _, l_buf_b, _, _ = sd_buf.shape
  l_q_loc, nx_loc, ny_loc = lon_q.shape

  eff_lat_shift = lagrange_interpolation.effective_window_shift(max_lat_shift)
  eff_z_shift = lagrange_interpolation.effective_window_shift(
      max_vertical_shift
  )
  w_lat_size = _window_size(n_lat_buf, n_lat_ext, eff_lat_shift, 2)
  w_z_c_size = _window_size(l_buf_c, l_src_c, eff_z_shift, 2)
  w_z_b_size = _window_size(l_buf_b, l_src_b, eff_z_shift, 2)

  j_global = _arrival_lat_rows(j_offset, ny_loc, n_lat)
  win_lat_start, m_lat_2d = _lat_window_selector(
      j_global,
      j_global + 2 - eff_lat_shift,
      n_lat_ext,
      n_lat_buf,
      w_lat_size,
      buf_lat_origin,
      dtype,
  )

  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, wind_buf, wind_buf)

  # [ny_loc, level, K_pad, n_lon], see `_project_lat_window`.
  k_w_raw = 3 * w_lat_size
  k_w_pad = _round_to_multiple(k_w_raw, _TPU_SUBLANES)
  wind_lat = _project_lat_window(wind_buf, m_lat_2d, w_lat_size, is_tpu)
  k_sd_raw = w_lat_size
  k_sd_pad = _round_to_multiple(k_sd_raw, _TPU_SUBLANES)
  sd_lat = _project_lat_window(sd_buf, m_lat_2d, w_lat_size, is_tpu)

  lon_q_t = jnp.swapaxes(lon_q, 1, 2)
  sin_lat_q_t = jnp.swapaxes(sin_lat_q, 1, 2)
  sigma_q_t = jnp.swapaxes(sigma_q, 1, 2)
  k_arr = jnp.arange(l_q_loc, dtype=jnp.int32)
  lat_bounds_t = tuple(jnp.asarray(b)[:, jnp.newaxis] for b in lat_bounds)
  z_bounds = (*centers_bounds, *boundaries_bounds)
  m_arange = jnp.arange(n_lon, dtype=jnp.int32)[:, jnp.newaxis]
  lat_arange = jnp.arange(w_lat_size, dtype=jnp.int32)[:, jnp.newaxis]

  item_bytes = 8 if dtype == jnp.float64 else 4
  n_lon_pad128 = _round_to_multiple(n_lon, _TPU_LANES)

  def _block_bytes(b_cand: int) -> int:
    wz_c = min(l_buf_c, min(l_src_c, w_z_c_size + b_cand - 1))
    wz_b = min(l_buf_b, min(l_src_b, w_z_b_size + b_cand - 1))
    k_tot = wz_c * k_w_pad + wz_b * k_sd_pad
    nx_c_pad128 = _round_to_multiple(b_cand * nx_loc, _TPU_LANES)
    lhs_b = ny_loc * k_tot * n_lon_pad128 * item_bytes
    rhs_b = (
        ny_loc
        * _round_to_multiple(n_lon, _TPU_SUBLANES)
        * nx_c_pad128
        * item_bytes
    )
    out_b = ny_loc * k_tot * nx_c_pad128 * item_bytes
    return lhs_b + rhs_b + out_b

  b_len = _level_block_length(
      l_q_loc, is_tpu, _block_bytes, _WIND_SIGMA_DOT_BLOCK_VMEM_PROBE
  )

  w_z_c_blk = min(l_buf_c, min(l_src_c, w_z_c_size + b_len - 1))
  w_z_b_blk = min(l_buf_b, min(l_src_b, w_z_b_size + b_len - 1))
  z_c_arange = jnp.arange(w_z_c_blk, dtype=jnp.int32)[:, jnp.newaxis]
  z_b_arange = jnp.arange(w_z_b_blk, dtype=jnp.int32)[:, jnp.newaxis]
  k_wind_flat = w_z_c_blk * k_w_pad
  nx_blk = b_len * nx_loc

  def _scan_block(inputs):
    k_b, lon_b, sin_lat_b, sigma_b, *z_bounds_b = inputs
    if b_len == 1:
      k0_s = k_b
      lon_s, sin_lat_s, sigma_s = lon_b, sin_lat_b, sigma_b
      z_bounds_s = z_bounds_b
    else:
      k0_s = k_b[0]
      lon_s = jnp.swapaxes(lon_b, 0, 1).reshape(ny_loc, nx_blk)
      sin_lat_s = jnp.swapaxes(sin_lat_b, 0, 1).reshape(ny_loc, nx_blk)
      sigma_s = jnp.swapaxes(sigma_b, 0, 1).reshape(ny_loc, nx_blk)
      # One clamp per query level of the block, like the `nx_blk` axis.
      z_bounds_s = [
          jnp.broadcast_to(b[:, jnp.newaxis], (b_len, nx_loc)).reshape(
              1, nx_blk
          )
          for b in z_bounds_b
      ]

    h_stencil = lagrange_interpolation.horizontal_stencil(
        grid, lon_s, sin_lat_s, 'linear', lat_bounds=lat_bounds_t
    )
    lon_idx, lon_w = h_stencil.lon_indices, h_stencil.lon_weights
    lat_idx, lat_w = h_stencil.lat_indices, h_stencil.lat_weights
    lvl_c_idx, lvl_c_w, _, _, _ = lagrange_interpolation.vertical_stencil(
        sigma_s,
        centers,
        'linear',
        is_tpu=is_tpu,
        sigma_bounds=tuple(z_bounds_s[:2]),
    )
    lvl_b_idx, lvl_b_w, _, _, _ = lagrange_interpolation.vertical_stencil(
        sigma_s,
        boundaries,
        'linear',
        is_tpu=is_tpu,
        sigma_bounds=tuple(z_bounds_s[2:]),
    )

    k_glob_c = jnp.minimum(k0_s, l_src_c - 1)
    # Neither buffer is z-sharded, so both hold their whole column.
    win_z_c_start, _ = _z_window_starts(
        k_glob_c - eff_z_shift, l_src_c, w_z_c_blk
    )
    win_z_b_start, _ = _z_window_starts(
        k_glob_c - eff_z_shift, l_src_b, w_z_b_blk
    )

    w_lon_mat = _weighted_window_one_hot(lon_idx, lon_w, m_arange, dtype)
    w_lat_mat = _weighted_window_one_hot(
        lat_idx, lat_w, lat_arange, dtype, win_lat_start
    )
    w_z_c_mat = _weighted_window_one_hot(
        lvl_c_idx, lvl_c_w, z_c_arange, dtype, win_z_c_start
    )
    w_z_b_mat = _weighted_window_one_hot(
        lvl_b_idx, lvl_b_w, z_b_arange, dtype, win_z_b_start
    )

    wind_win = lax.dynamic_slice_in_dim(
        wind_lat, win_z_c_start, w_z_c_blk, axis=1
    )
    wind_patch = wind_win.reshape(ny_loc, k_wind_flat, n_lon)
    sd_win = lax.dynamic_slice_in_dim(sd_lat, win_z_b_start, w_z_b_blk, axis=1)
    sd_patch = sd_win.reshape(ny_loc, w_z_b_blk * k_sd_pad, n_lon)

    both_patch = jnp.concatenate([wind_patch, sd_patch], axis=1)
    gathered = jnp.einsum(
        'jkm,jmi->jki', both_patch, w_lon_mat, precision=dot_prec_f32
    )
    wind_after_z = (
        gathered[:, :k_wind_flat, :].reshape(ny_loc, w_z_c_blk, k_w_pad, nx_blk)
        * w_z_c_mat[:, :, jnp.newaxis, :]
    ).sum(axis=1)
    wind_jci = (
        wind_after_z[:, :k_w_raw, :].reshape(ny_loc, 3, w_lat_size, nx_blk)
        * w_lat_mat[:, jnp.newaxis, :, :]
    ).sum(axis=2)
    sd_after_z = (
        gathered[:, k_wind_flat:, :].reshape(
            ny_loc, w_z_b_blk, k_sd_pad, nx_blk
        )
        * w_z_b_mat[:, :, jnp.newaxis, :]
    ).sum(axis=1)
    sd_jci = (sd_after_z[:, :k_sd_raw, :] * w_lat_mat).sum(axis=1)[
        :, jnp.newaxis, :
    ]
    both_jci = jnp.concatenate([wind_jci, sd_jci], axis=1)
    if b_len == 1:
      return jnp.swapaxes(both_jci, 0, 1)
    return both_jci.reshape(ny_loc, 4, b_len, nx_loc).transpose(2, 1, 0, 3)

  if l_q_loc == 1:
    out_0 = _scan_block(
        (
            jnp.int32(0),
            lon_q_t[0],
            sin_lat_q_t[0],
            sigma_q_t[0],
            *(b[0] for b in z_bounds),
        )
    )
    out_4d = jnp.swapaxes(out_0, 1, 2)[:, jnp.newaxis, :, :]
  else:
    out_4d = _scan_level_blocks(
        _scan_block, k_arr, lon_q_t, sin_lat_q_t, sigma_q_t, z_bounds, b_len
    ).transpose(1, 0, 3, 2)
  return out_4d[:3], out_4d[3]


def interpolate_3d_wind_and_sigma_dot(
    wind: jax.Array,
    sigma_dot: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    sigma: jax.Array,
    centers: Array,
    boundaries: Array,
    *,
    max_lat_shift: int,
    max_vertical_shift: int,
) -> tuple[jax.Array, jax.Array]:
  """Fused grid-aligned MXU interpolation of `wind` and `sigma_dot`."""
  centers_jax = jnp.asarray(centers)
  boundaries_jax = jnp.asarray(boundaries)
  sd_4d = sigma_dot[jnp.newaxis, ...]
  mesh = spatial_mesh(grid)
  lat_bounds, centers_bounds, boundaries_bounds = (
      lagrange_interpolation.wind_sigma_dot_clamp_bounds(
          grid,
          lon,
          sin_lat,
          sigma,
          centers_jax,
          boundaries_jax,
          max_lat_shift,
          max_vertical_shift,
      )
  )
  if mesh is None:
    wind_ext = lagrange_interpolation.extend_with_pole_halo(wind, grid)
    sd_ext = lagrange_interpolation.extend_with_pole_halo(sd_4d, grid)
    return _matmul_3d_wind_and_sigma_dot_local(
        wind_ext,
        sd_ext,
        lon,
        sin_lat,
        sigma,
        grid=grid,
        centers=centers_jax,
        boundaries=boundaries_jax,
        j_offset=0,
        buf_lat_origin=0,
        max_lat_shift=max_lat_shift,
        max_vertical_shift=max_vertical_shift,
        lat_bounds=lat_bounds,
        centers_bounds=centers_bounds,
        boundaries_bounds=boundaries_bounds,
    )

  x_shards = mesh.shape['x']
  f_spec = P(None, None, 'x', 'y')
  q_spec = P(None, 'x', 'y')

  def _sharded(
      w_loc, sd_loc, lon_loc, sin_lat_loc, sigma_loc, c_loc, b_loc, *bounds
  ):
    w_x = _gather_longitude(w_loc, grid, x_shards)
    sd_x = _gather_longitude(sd_loc, grid, x_shards)
    w_buf, j_offset, buf_lat_origin = _gather_lat_with_halo(
        w_x, grid, max_lat_shift
    )
    sd_buf, _, _ = _gather_lat_with_halo(sd_x, grid, max_lat_shift)
    w_mid, sd_mid = _matmul_3d_wind_and_sigma_dot_local(
        w_buf,
        sd_buf,
        lon_loc,
        sin_lat_loc,
        sigma_loc,
        grid=grid,
        centers=c_loc,
        boundaries=b_loc,
        j_offset=j_offset,
        buf_lat_origin=buf_lat_origin,
        max_lat_shift=max_lat_shift,
        max_vertical_shift=max_vertical_shift,
        lat_bounds=bounds[0:2],
        centers_bounds=bounds[2:4],
        boundaries_bounds=bounds[4:6],
    )
    return w_mid, sd_mid[jnp.newaxis, ...]

  wind_mid, sd_mid_4d = _shard_map(
      _sharded,
      mesh,
      in_specs=(f_spec, f_spec, q_spec, q_spec, q_spec, P(None), P(None))
      + (P('y'),) * 2
      + (P(None),) * 4,
      out_specs=(f_spec, f_spec),
  )(
      wind,
      sd_4d,
      lon,
      sin_lat,
      sigma,
      centers_jax,
      boundaries_jax,
      *(jnp.asarray(b) for b in lat_bounds),
      *centers_bounds,
      *boundaries_bounds,
  )
  return wind_mid, sd_mid_4d[0]
