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

# Tuning constants of the 'matmul' interpolation. All are empirical settings
# chosen on TPU v5e (128 MiB on-chip VMEM per core) unless stated otherwise;
# none is derived from a model of the hardware, and changing one changes the
# compiled program (not the interpolated values).
_MIB = 1024 * 1024

# Largest 1-D node/table axis whose lookup is done with a one-hot MXU dot (or a
# broadcast compare) instead of a scalar gather or `searchsorted`. The one-hot
# has one column per node, so this bounds its size.
_ONE_HOT_LOOKUP_MAX_NODES = 512

# Queries evaluated per `lax.scan` step by the arbitrary-point (unwindowed)
# matmul paths; bounds the [queries, nodes] one-hot matrices.
_POINTS_QUERY_CHUNK = 1024

# Channels per `lax.scan` step of the windowed 2-D matmul (the limited variant
# also carries a lower and an upper bound row per channel).
_MATMUL_2D_CHANNEL_CHUNK = 8
_MATMUL_2D_CHANNEL_CHUNK_LIMITED = 4

# Budgets that size the channel chunk of the windowed 3-D matmul. In bytes of
# (8, 128)-tile-padded buffers, per channel chunk: `single_buf` bounds the
# largest single buffer, `ws_buf` the sum of the hoisted buffers alive at
# once, and `vmem` the per-block working set of the gather/contraction, which
# on TPU must fit in on-chip VMEM without spilling (hence below 128 MiB). The
# non-TPU `vmem` is not a hardware limit. `max_channels` caps the chunk.
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

# Numbers of consecutive query levels fused into one scan block, tried in this
# order (1 is the fallback), on TPU only.
_BLOCK_LEVEL_CANDIDATES = (4, 2)

# Largest per-block working set (bytes) accepted when probing the block
# length: the 3-D scalar/limiter path and the fused wind + sigma_dot path.
# Both sit just below TPU v5e's 128 MiB VMEM.
_MATMUL_3D_BLOCK_VMEM_PROBE = 116 * _MIB
_WIND_SIGMA_DOT_BLOCK_VMEM_PROBE = 118 * _MIB

# TPU (8, 128) tiling of 32-bit arrays: the minor axis is padded to a multiple
# of 128 lanes and the second-minor axis to a multiple of 8 sublanes. Buffer
# footprint estimates and the K axis of the one-hot products use these.
_TPU_LANES = 128
_TPU_SUBLANES = 8

# Relative widening of the limiter's cell bounds on TPU in float32: the 3-pass
# `bf16_3x` MXU selection of a bound is exact only to the 16-bit split ulp
# (2^-16 ~ 1.5e-5), so the bounds are widened by slightly more than that to
# keep constant fields and non-negative fields exactly inside them.
_BF16X3_BOUND_WIDENING = 1.6e-5


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


def _effective_window_shift(max_shift: int) -> int:
  """Window half-width actually used: a `max_shift` of 0 is treated as 1."""
  return max(1, max_shift)


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


def _z_window_start(
    first_level: jax.Array, n_source: int, size: int
) -> jax.Array:
  """Start level of the vertical window of a query level, int32 scalar.

  The window covers the `size` consecutive source levels
  `start ... start + size - 1` out of `n_source`. `first_level` is the
  unclipped first level the caller wants covered (the first stencil level of
  a query with zero displacement, minus the half-width). The start is
  `first_level` clipped to `[0, n_source - size]`, so near the ends of the
  axis the window stops sliding and covers the first or last `size` levels;
  when the window is the whole axis it is 0 and `first_level` is not used.
  """
  if size == n_source:
    return jnp.int32(0)
  return jnp.clip(first_level, 0, n_source - size)


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


def _uniform_lagrange_weights_tuple(
    alpha: jax.Array, order: str
) -> tuple[jax.Array, ...]:
  """Closed-form Lagrange weights on unit-spaced stencil offsets for `alpha` in [0, 1)."""
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


def _take_1d(values_1d: jax.Array, idx: jax.Array) -> jax.Array:
  """1-D table lookup using MXU `one_hot` dot on TPU to avoid scalar VPU gathers."""
  if (
      jax.default_backend() == 'tpu'
      and values_1d.shape[0] <= _ONE_HOT_LOOKUP_MAX_NODES
  ):
    oh = jax.nn.one_hot(idx, values_1d.shape[0], dtype=values_1d.dtype)
    dot_prec = jax_numpy_utils.resolve_dot_precision(
        None, oh, values_1d, lhs_exact_in_bf16=True
    )
    return jnp.dot(oh, values_1d, precision=dot_prec)
  return values_1d[idx]


def _resolve_rhs_binary_dot_precision(
    lhs: jax.Array, rhs: jax.Array
) -> jax.lax.PrecisionLike:
  """Precision for `dot(lhs, rhs)` when `rhs` is an exact binary (0/1) matrix."""
  if (
      jax.default_backend() == 'tpu'
      and jnp.result_type(lhs, rhs) == jnp.float32
  ):
    return ('highest', 'bfloat16')
  return jax_numpy_utils.resolve_dot_precision(None, lhs, rhs)


def _contract_stencil(values: Array, *weights: Array) -> jnp.ndarray:
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
  58.4 → 33.3 ms/step with cubic on the 2-day ERA5 benchmark). The backend
  is known at trace time, so the form is picked per platform. The 3-D
  gather interpolation accumulates the vertical, longitude and latitude
  stencils level by level itself and does not use this function.
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


def _searchsorted_right_minus_one(
    nodes_1d: jax.Array, x: jax.Array
) -> jax.Array:
  """Computes `searchsorted(nodes_1d, x, side='right') - 1` without `[..., 1]` tiles."""
  if nodes_1d.shape[0] <= _ONE_HOT_LOOKUP_MAX_NODES and (
      x.ndim <= 2 or jax.default_backend() == 'tpu'
  ):
    return (x[..., jnp.newaxis] >= nodes_1d).sum(axis=-1, dtype=jnp.int32) - 1
  return (jnp.searchsorted(nodes_1d, x, side='right') - 1).astype(jnp.int32)


def _tpu_step_diff_one_hot(
    x: jax.Array, thresholds_1d: jax.Array, dtype: jnp.dtype
) -> jax.Array:
  """Builds exact 1-D cell one-hot directly from boolean threshold step diffs."""
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


def _horizontal_stencil_tuple(
    grid: spherical_harmonic.Grid,
    lon: Array,
    sin_lat: Array,
    order: str,
    *,
    include_linear_fractions: bool = False,
):
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
  phi_extended = _latitude_halo_axis(grid)
  c_min = -int(offsets[0])
  c_max = phi_extended.size - 1 - int(offsets[-1])
  alpha_lat = None
  if (
      jax.default_backend() == 'tpu'
      and phi_extended.size <= _ONE_HOT_LOOKUP_MAX_NODES
  ):
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
    lat_cell = _searchsorted_right_minus_one(phi_ext_jax, phi)
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

  base = (
      lon_indices,
      lon_weights,
      lat_indices,
      lat_weights,
      lon_cell % n_lon,
      lat_cell,
      phi_extended.size,
  )
  if include_linear_fractions:
    return (*base, (alpha_lon, alpha_lat))
  return base


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
  q_chunk = min(q_total, _POINTS_QUERY_CHUNK)
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
  num_channels, n_lon, n_lat_buf = f_buf.shape
  n_lat = grid.latitude_nodes
  n_lat_ext = n_lat + 2 * _HALO_WIDTH
  nx_loc, ny_loc = lon_q.shape
  eff_lat_shift = _effective_window_shift(max_lat_shift)
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
  ) = _horizontal_stencil_tuple(grid, lon_q, sin_lat_q, order)

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

  dot_prec_lhs_bf16 = jax_numpy_utils.resolve_dot_precision(
      None, m_lat_win, f_buf, lhs_exact_in_bf16=True
  )
  dot_prec_rhs_bf16 = _resolve_rhs_binary_dot_precision(f_buf, m_lat_2d_t)
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
      lo_buf, hi_buf = _cell_bounds_2d(f_chunk)
      bounds_buf = jnp.concatenate([lo_buf, hi_buf], axis=0)
      b_2d = bounds_buf.transpose(1, 0, 2).reshape(
          n_lon * 2 * c_cur, n_lat_buf - 1
      )
      b_lat_win = (
          jnp.dot(b_2d, m_lat_cell_2d_t, precision=dot_prec_rhs_bf16)
          .reshape(n_lon, 2 * c_cur * w_lat_cell_size, ny_loc)
          .transpose(2, 0, 1)
      )
      b_lon = jnp.einsum(
          'jim,jmk->jik', p_lon_mat, b_lat_win, precision=dot_prec_lhs_bf16
      )
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
    ):
      mesh = grid.spmd_mesh
      if mesh is None:
        f_ext = _extend_with_pole_halo(fields, grid)
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
                ),
            ),
            None,
            (f_by_l, lon, sin_lat),
        )
        return jnp.swapaxes(out_by_l, 0, 1)

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

    if resolved == 'matmul':
      _, out_by_l = lax.scan(
          lambda _, inp: (
              None,
              _interpolate_2d_multi(
                  inp[0],
                  grid,
                  inp[1],
                  inp[2],
                  order=order,
                  limiter=limiter,
                  method=resolved,
                  max_lat_shift=max_lat_shift,
              ),
          ),
          None,
          (jnp.swapaxes(fields, 0, 1), lon, sin_lat),
      )
      return jnp.swapaxes(out_by_l, 0, 1)

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
    method: 'gather' (1-D gather + fused contraction, fastest on single GPU;
      no window), 'matmul' (MXU matrix multiplication without unstructured
      gather/scatter, designed for TPU and SPMD sharding; windowed, see
      `max_lat_shift`), or None (auto-selects 'matmul' on TPU or when
      `grid.spmd_mesh` is set, and 'gather' otherwise).
    max_lat_shift: half-width, in latitude rows, of the window inside which
      the 'matmul' method builds each query's stencil. For queries laid out
      on the grid (`lon.shape == grid.nodal_shape`, or with a leading levels
      axis) the window has `2 * max_lat_shift + stencil` latitude rows
      (stencil is 2 for 'linear' and 4 for 'cubic'), starting `max_lat_shift`
      rows before the first stencil row of a point that equals its arrival
      node, clipped to the ends of the halo-extended latitude axis. A stencil
      point outside the window contributes nothing, so the value is wrong
      there (it is not clamped). A value of 0 is treated as 1. None disables
      windowing, as does any other query layout (the queries are then
      interpolated without a window). Ignored by 'gather'.
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
    *,
    include_linear_fractions: bool = False,
):
  """Returns tuple-unrolled (level_indices, level_weights, bracket_offset, level_cell)."""
  if vertical_order not in ('linear', 'cubic'):
    raise ValueError(f'unknown {vertical_order=}')
  sigma = jnp.asarray(sigma)
  sigma_nodes = jnp.asarray(sigma_nodes, dtype=sigma.dtype)
  num_levels = sigma_nodes.shape[0]
  sigma = jnp.clip(sigma, sigma_nodes[0], sigma_nodes[-1])
  if (
      jax.default_backend() == 'tpu'
      and (vertical_order == 'linear' or num_levels >= 4)
      and num_levels <= _ONE_HOT_LOOKUP_MAX_NODES
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
      base = (
          (level_cell, level_cell + 1),
          (1.0 - fraction, fraction),
          jnp.zeros_like(level_cell),
          level_cell,
      )
      return (*base, fraction) if include_linear_fractions else base
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
    base = (level_indices, level_weights, bracket_offset, level_cell)
    return (*base, fraction) if include_linear_fractions else base

  level_cell = _searchsorted_right_minus_one(sigma_nodes, sigma)
  level_cell = jnp.clip(level_cell, 0, num_levels - 2)
  below = sigma_nodes[level_cell]
  above = sigma_nodes[level_cell + 1]
  fraction = (sigma - below) / (above - below)
  if vertical_order == 'linear':
    base = (
        (level_cell, level_cell + 1),
        (1 - fraction, fraction),
        jnp.zeros_like(level_cell),
        level_cell,
    )
    return (*base, fraction) if include_linear_fractions else base
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
  base = (level_indices, level_weights, bracket_offset, level_cell)
  return (*base, fraction) if include_linear_fractions else base


def _interpolate_3d_gather(
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
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
  """Gather-based 3-D interpolation on channel-minor layout sharing one stencil.

  Packs channels into the contiguous minor axis (`[N_grid, C]`) so all channels
  at each stencil corner share a single cache line, contracts the 3-D Lagrange
  weights in a cascaded reduction without materializing `[..., 4, 4, 4]` index
  or value tensors, and extracts both quasi-monotone cell bounds and optional
  trilinear values (`trilinear_channel_start`) from the already-gathered
  interior `2x2x2` corners with zero extra gathers.
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

  if need_tri_box:
    (
        lon_idx,
        lon_w,
        lat_idx,
        lat_w,
        _,
        _,
        ext_size,
        (a_lon, a_lat),
    ) = _horizontal_stencil_tuple(
        grid, lon, sin_lat, order, include_linear_fractions=True
    )
    lvl_idx, lvl_w, b_off, _, f_lvl = _vertical_stencil_tuple(
        sigma,
        jnp.asarray(sigma_nodes),
        vertical_order,
        include_linear_fractions=True,
    )
  else:
    lon_idx, lon_w, lat_idx, lat_w, _, _, ext_size = _horizontal_stencil_tuple(
        grid, lon, sin_lat, order
    )
    lvl_idx, lvl_w, b_off, _ = _vertical_stencil_tuple(
        sigma, jnp.asarray(sigma_nodes), vertical_order
    )
    a_lon = a_lat = f_lvl = None

  extended = _extend_with_pole_halo(fields_4d, grid)
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
    ref_stack = jnp.stack(ref_levels, axis=0) if ref_levels is not None else None

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
        jnp.result_type(dtype, ref_levels[0]) if ref_levels is not None else dtype
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

  out = jnp.moveaxis(out_c, -1, 0)
  return (out, out_tri) if need_tri else out


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
  q_chunk = min(q_total, _POINTS_QUERY_CHUNK)
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
            w * _take_1d(ref_centered, idx)
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
    limited_channel_start: int = 0,
    trilinear_channel_start: int | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Windowed MXU matmul 3-D interpolation on [C, L_buf, n_lon, n_lat_buf]."""
  dtype = f_buf.dtype
  is_tpu = jax.default_backend() == 'tpu'
  n_lon = grid.longitude_nodes
  n_lat = grid.latitude_nodes
  n_lat_ext = n_lat + 2 * _HALO_WIDTH
  l_src = sigma_nodes.shape[0]
  num_channels, l_buf, _, n_lat_buf = f_buf.shape
  l_q_loc, nx_loc, ny_loc = lon_q.shape

  eff_limiter = limiter if limited_channel_start < num_channels else None
  eff_lat_shift = _effective_window_shift(max_lat_shift)
  eff_z_shift = _effective_window_shift(max_vertical_shift)
  s_lat = 2 if order == 'linear' else 4
  s_z = 2 if vertical_order == 'linear' else 4
  w_lat_size = _window_size(n_lat_buf, n_lat_ext, eff_lat_shift, s_lat)
  w_z_size = _window_size(l_buf, l_src, eff_z_shift, s_z)
  lat_start_offset = (2 if order == 'linear' else 1) - eff_lat_shift
  z_start_offset = (
      -eff_z_shift if vertical_order == 'linear' else (-eff_z_shift - 1)
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
  # Build `m_lat_win` in `[ny_loc, w_lat_size, n_lat_buf]` so `dot(m_lat_2d, ...)`
  # keeps `n_lon` in `axis=-1` without cross-lane minor-axis transposes.
  m_lat_win = jax.nn.one_hot(
      buf_lat_start[:, jnp.newaxis]
      + jnp.arange(w_lat_size, dtype=jnp.int32)[jnp.newaxis, :],
      n_lat_buf,
      dtype=dtype,
  )  # [ny_loc, w_lat_size, n_lat_buf]
  m_lat_2d = m_lat_win.reshape(ny_loc * w_lat_size, n_lat_buf)

  dot_prec_lhs_bf16 = jax_numpy_utils.resolve_dot_precision(
      None, m_lat_2d, f_buf, lhs_exact_in_bf16=True
  )
  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, f_buf, f_buf)

  ref_buf = None
  if eff_limiter is not None and limiter_bound_reference is not None:
    ref_arr = jnp.asarray(limiter_bound_reference, dtype=dtype)
    ref_centered = ref_arr - jnp.mean(ref_arr)
    k_buf_levels = jnp.clip(
        buf_z_origin + jnp.arange(l_buf, dtype=jnp.int32), 0, l_src - 1
    )
    ref_buf = _take_1d(ref_centered, k_buf_levels)

  w_lat_cell_size = 0
  w_z_cell_size = 0
  win_lat_cell_start = None
  m_lat_cell_2d = None
  if eff_limiter is not None:
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
        buf_lat_cell_start[:, jnp.newaxis]
        + jnp.arange(w_lat_cell_size, dtype=jnp.int32)[jnp.newaxis, :],
        n_lat_buf - 1,
        dtype=dtype,
    )  # [ny_loc, w_lat_cell_size, n_lat_buf - 1]
    m_lat_cell_2d = m_lat_cell_win.reshape(
        ny_loc * w_lat_cell_size, n_lat_buf - 1
    )
    w_z_cell_size = _window_size(l_buf - 1, l_src - 1, eff_z_shift, 2)

  need_separate_trilinear = trilinear_channel_start is not None and not (
      order == 'linear' and vertical_order == 'linear'
  )

  # Size channel chunks so hoisted `[ny_loc, l_buf, k_pad, n_lon]` buffers stay
  # compact in HBM AND per-level `b_len = 1` working sets fit inside TPU v5e's
  # 128 MiB on-chip VMEM without spilling.
  item_bytes = 8 if dtype == jnp.float64 else 4
  n_lon_pad128 = _round_to_multiple(n_lon, _TPU_LANES)
  nx_pad128 = _round_to_multiple(nx_loc, _TPU_LANES)
  w_lon_vmem_b = (
      ny_loc * _round_to_multiple(n_lon, _TPU_SUBLANES) * nx_pad128 * item_bytes
  )
  if is_tpu:
    max_c_cap = _MATMUL_3D_MAX_CHANNELS_TPU
    single_buf_limit = _MATMUL_3D_SINGLE_BUF_LIMIT_TPU
    ws_buf_limit = _MATMUL_3D_WS_BUF_LIMIT_TPU
    vmem_limit = _MATMUL_3D_VMEM_LIMIT_TPU
  else:
    max_c_cap = _MATMUL_3D_MAX_CHANNELS_CPU_GPU
    single_buf_limit = _MATMUL_3D_SINGLE_BUF_LIMIT_CPU_GPU
    ws_buf_limit = _MATMUL_3D_WS_BUF_LIMIT_CPU_GPU
    vmem_limit = _MATMUL_3D_VMEM_LIMIT_CPU_GPU

  def _fits_budget(c_test: int, lim_s: int = 0) -> bool:
    k_c_pad = _round_to_multiple(c_test * w_lat_size, _TPU_SUBLANES)
    f_lat_b = ny_loc * l_buf * k_c_pad * n_lon_pad128 * item_bytes
    f_patch_b = ny_loc * w_z_size * k_c_pad * n_lon_pad128 * item_bytes
    gath_b = ny_loc * w_z_size * k_c_pad * nx_pad128 * item_bytes
    vmem_b = w_lon_vmem_b + f_patch_b + gath_b
    max_single_b = f_lat_b
    ws_b = f_lat_b
    c_lim_test = max(0, c_test - lim_s) if eff_limiter is not None else 0
    if c_lim_test > 0:
      k_l_pad = _round_to_multiple(
          2 * c_lim_test * w_lat_cell_size, _TPU_SUBLANES
      )
      b_lat_b = (
          ny_loc * max(1, l_buf - 1) * k_l_pad * n_lon_pad128 * item_bytes
      )
      b_patch_b = ny_loc * w_z_cell_size * k_l_pad * n_lon_pad128 * item_bytes
      b_lon_b = ny_loc * w_z_cell_size * k_l_pad * nx_pad128 * item_bytes
      max_single_b = max(max_single_b, b_lat_b)
      ws_b = f_lat_b + b_lat_b
      vmem_b = max(vmem_b, w_lon_vmem_b + b_patch_b + b_lon_b)
    return (
        max_single_b <= single_buf_limit
        and ws_b <= ws_buf_limit
        and vmem_b <= vmem_limit
    )

  if num_channels <= max_c_cap and _fits_budget(
      num_channels, limited_channel_start
  ):
    chunk_size = num_channels
  else:
    chunk_size = 1
    for c_cand in range(min(num_channels, max_c_cap), 0, -1):
      if _fits_budget(c_cand, 0):
        chunk_size = c_cand
        break

  # If channel chunking is needed with mixed channel options, split into
  # homogeneous channel groups so each `lax.scan` chunk is uniform.
  if num_channels > chunk_size and (
      0 < limited_channel_start < num_channels
      or (
          trilinear_channel_start is not None
          and 0 < trilinear_channel_start < num_channels
      )
  ):
    common_kw = dict(
        lon_q=lon_q,
        sin_lat_q=sin_lat_q,
        sigma_q=sigma_q,
        grid=grid,
        sigma_nodes=sigma_nodes,
        order=order,
        vertical_order=vertical_order,
        limiter_bound_reference=limiter_bound_reference,
        k_offset=k_offset,
        buf_z_origin=buf_z_origin,
        j_offset=j_offset,
        buf_lat_origin=buf_lat_origin,
        max_lat_shift=max_lat_shift,
        max_vertical_shift=max_vertical_shift,
    )
    split_pts = sorted(
        {0, num_channels}
        | (
            {min(num_channels, max(0, limited_channel_start))}
            if eff_limiter is not None
            else set()
        )
        | (
            {min(num_channels, max(0, trilinear_channel_start))}
            if trilinear_channel_start is not None
            else set()
        )
    )
    outs = []
    outs_lin = []
    for s0, s1 in zip(split_pts[:-1], split_pts[1:]):
      if s0 == s1:
        continue
      sub_lim = eff_limiter if s0 >= limited_channel_start else None
      sub_tri = (
          0
          if (trilinear_channel_start is not None and s0 >= trilinear_channel_start)
          else None
      )
      res = _matmul_3d_local(
          f_buf[s0:s1],
          limiter=sub_lim,
          limited_channel_start=0,
          trilinear_channel_start=sub_tri,
          **common_kw,
      )
      if sub_tri is not None:
        outs.append(res[0])
        outs_lin.append(res[1])
      else:
        outs.append(res)
    out_full = outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=0)
    if trilinear_channel_start is not None:
      # No trilinear channels at all (`trilinear_channel_start == C`) leaves
      # `outs_lin` empty; the gather method returns an empty trilinear block.
      if not outs_lin:
        out_lin_full = jnp.zeros((0, l_q_loc, nx_loc, ny_loc), dtype=dtype)
      elif len(outs_lin) == 1:
        out_lin_full = outs_lin[0]
      else:
        out_lin_full = jnp.concatenate(outs_lin, axis=0)
      return out_full, out_lin_full
    return out_full

  # Pre-transpose query coordinates once outside `lax.scan` so `nx_loc` stays
  # in `axis=-1` throughout `_scan_block` with zero per-level transposes.
  lon_q_t = jnp.swapaxes(lon_q, 1, 2)
  sin_lat_q_t = jnp.swapaxes(sin_lat_q, 1, 2)
  sigma_q_t = jnp.swapaxes(sigma_q, 1, 2)
  k_arr = jnp.arange(l_q_loc, dtype=jnp.int32)
  m_arange = jnp.arange(n_lon, dtype=jnp.int32)[:, jnp.newaxis]
  w_z_lin_size = _window_size(l_buf, l_src, eff_z_shift, 2)
  lat_arange = jnp.arange(w_lat_size, dtype=jnp.int32)[:, jnp.newaxis]
  c0_lon_pos = 0 if order == 'linear' else 1
  dot_prec_tri = (
      lax.DotAlgorithmPreset.BF16_BF16_F32_X3
      if (is_tpu and dtype == jnp.float32)
      else dot_prec_f32
  )

  def _eval_channel_chunk(
      f_chunk: jax.Array,
      lim_start: int,
      tri_start: int | None,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    c_cur = f_chunk.shape[0]
    c_lim = max(0, c_cur - lim_start) if eff_limiter is not None else 0
    c_tri = max(0, c_cur - tri_start) if tri_start is not None else 0

    k_raw = c_cur * w_lat_size
    k_pad = _round_to_multiple(k_raw, _TPU_SUBLANES)
    f_2d_t = (
        f_chunk.swapaxes(0, 1)
        .reshape(l_buf * c_cur, n_lon, n_lat_buf)
        .transpose(2, 0, 1)
        .reshape(n_lat_buf, l_buf * c_cur * n_lon)
    )
    f_lat = (
        jnp.dot(m_lat_2d, f_2d_t, precision=dot_prec_lhs_bf16)
        .reshape(ny_loc, w_lat_size, l_buf * c_cur, n_lon)
        .transpose(0, 2, 1, 3)
        .reshape(ny_loc, l_buf, k_raw, n_lon)
    )
    if k_pad > k_raw:
      f_lat = jnp.pad(f_lat, [(0, 0), (0, 0), (0, k_pad - k_raw), (0, 0)])

    b_lat = None
    k_lim_raw = 0
    k_lim_pad = 0
    if c_lim > 0:
      assert m_lat_cell_2d is not None
      k_lim_raw = 2 * c_lim * w_lat_cell_size
      k_lim_pad = _round_to_multiple(k_lim_raw, _TPU_SUBLANES)
      f_lim = f_chunk[lim_start:]
      f_for_bounds = (
          f_lim + ref_buf[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
          if ref_buf is not None
          else f_lim
      )
      lo_buf, hi_buf = _cell_bounds_3d(f_for_bounds)
      if is_tpu and dtype == jnp.float32:
        # Widen cell bounds by 16-bit split ulp (`2^-16`) so 3-pass `bf16_3x`
        # MXU selection preserves constants and non-negativity (`0.0`) exactly.
        lo_buf = lo_buf - _BF16X3_BOUND_WIDENING * jnp.abs(lo_buf)
        hi_buf = hi_buf + _BF16X3_BOUND_WIDENING * jnp.abs(hi_buf)
      b_2d_t = (
          jnp.concatenate([lo_buf, hi_buf], axis=0)
          .swapaxes(0, 1)
          .reshape((l_buf - 1) * 2 * c_lim, n_lon, n_lat_buf - 1)
          .transpose(2, 0, 1)
          .reshape(n_lat_buf - 1, (l_buf - 1) * 2 * c_lim * n_lon)
      )
      b_lat = (
          jnp.dot(m_lat_cell_2d, b_2d_t, precision=dot_prec_tri)
          .reshape(ny_loc, w_lat_cell_size, (l_buf - 1) * 2 * c_lim, n_lon)
          .transpose(0, 2, 1, 3)
          .reshape(ny_loc, l_buf - 1, k_lim_raw, n_lon)
      )
      if k_lim_pad > k_lim_raw:
        b_lat = jnp.pad(
            b_lat, [(0, 0), (0, 0), (0, k_lim_pad - k_lim_raw), (0, 0)]
        )

    def _eval_levels_grouped(b_len: int):
      w_z_blk_size = min(l_buf, min(l_src, w_z_size + b_len - 1))
      w_z_lin_blk_size = min(l_buf, min(l_src, w_z_lin_size + b_len - 1))
      w_z_cell_blk_size = (
          min(
              max(1, l_buf - 1),
              min(max(1, l_src - 1), w_z_cell_size + b_len - 1),
          )
          if c_lim > 0
          else 0
      )
      nx_blk = b_len * nx_loc
      nx_blk_pad128 = _round_to_multiple(nx_blk, _TPU_LANES)
      f_patch_blk_b = ny_loc * w_z_blk_size * k_pad * n_lon_pad128 * item_bytes
      w_lon_blk_b = (
          ny_loc
          * _round_to_multiple(n_lon, _TPU_SUBLANES)
          * nx_blk_pad128
          * item_bytes
      )
      gath_blk_b = ny_loc * w_z_blk_size * k_pad * nx_blk_pad128 * item_bytes
      fuse_gath_blk = (
          b_len == 1 or (f_patch_blk_b + w_lon_blk_b + gath_blk_b) <= vmem_limit
      )

      z_blk_arange = jnp.arange(w_z_blk_size, dtype=jnp.int32)[:, jnp.newaxis]
      z_lin_blk_arange = jnp.arange(w_z_lin_blk_size, dtype=jnp.int32)[
          :, jnp.newaxis
      ]
      z_cell_blk_arange = (
          jnp.arange(w_z_cell_blk_size, dtype=jnp.int32)[:, jnp.newaxis]
          if c_lim > 0
          else None
      )
      lat_cell_arange = (
          jnp.arange(w_lat_cell_size, dtype=jnp.int32)[:, jnp.newaxis]
          if c_lim > 0
          else None
      )

      def _scan_block(inputs):
        k_b, lon_b, sin_lat_b, sigma_b = inputs
        if b_len == 1:
          k0_s = k_b
          lon_s = lon_b
          sin_lat_s = sin_lat_b
          sigma_s = sigma_b
        else:
          k0_s = k_b[0]
          lon_s = jnp.swapaxes(lon_b, 0, 1).reshape(ny_loc, nx_blk)
          sin_lat_s = jnp.swapaxes(sin_lat_b, 0, 1).reshape(ny_loc, nx_blk)
          sigma_s = jnp.swapaxes(sigma_b, 0, 1).reshape(ny_loc, nx_blk)

        if c_tri > 0 and need_separate_trilinear:
          (
              lon_idx,
              lon_w,
              lat_idx,
              lat_w,
              _,
              lat_c0,
              _,
              (alpha_lon, alpha_lat),
          ) = _horizontal_stencil_tuple(
              grid, lon_s, sin_lat_s, order, include_linear_fractions=True
          )
          lvl_idx, lvl_w, _, lvl_c0, alpha_z = _vertical_stencil_tuple(
              sigma_s,
              sigma_nodes,
              vertical_order,
              include_linear_fractions=True,
          )
        else:
          (
              lon_idx,
              lon_w,
              lat_idx,
              lat_w,
              _,
              lat_c0,
              _,
          ) = _horizontal_stencil_tuple(grid, lon_s, sin_lat_s, order)
          lvl_idx, lvl_w, _, lvl_c0 = _vertical_stencil_tuple(
              sigma_s, sigma_nodes, vertical_order
          )
          alpha_lon = alpha_lat = alpha_z = None

        k_glob = jnp.minimum(k_offset + k0_s, l_src - 1)
        win_z_start = _z_window_start(
            k_glob + z_start_offset, l_src, w_z_blk_size
        )
        buf_z_start = _buffer_window_start(
            win_z_start, buf_z_origin, l_buf, w_z_blk_size
        )

        oh_lon = tuple(
            (idx[:, jnp.newaxis, :] == m_arange).astype(dtype)
            for idx in lon_idx
        )
        w_lon_mat = sum(
            w[:, jnp.newaxis, :] * oh for w, oh in zip(lon_w, oh_lon)
        )
        w_z_mat = sum(
            w[:, jnp.newaxis, :]
            * (
                (idx - win_z_start)[:, jnp.newaxis, :] == z_blk_arange
            ).astype(dtype)
            for idx, w in zip(lvl_idx, lvl_w)
        )
        w_lat_mat = sum(
            w[:, jnp.newaxis, :]
            * (
                (idx - win_lat_start[:, jnp.newaxis])[:, jnp.newaxis, :]
                == lat_arange
            ).astype(dtype)
            for idx, w in zip(lat_idx, lat_w)
        )

        # Slice the block vertical window once from `f_lat` across all `b_len`
        # query levels so overlapping vertical levels are reused in VMEM.
        f_win = lax.dynamic_slice_in_dim(
            f_lat, buf_z_start, w_z_blk_size, axis=1
        )

        val_lin_jci = None
        if c_tri > 0 and need_separate_trilinear:
          assert (
              alpha_lon is not None
              and alpha_lat is not None
              and alpha_z is not None
          )
          if order == 'linear':
            w_lon_lin_mat = w_lon_mat
            w_lat_lin_mat = w_lat_mat
          else:
            a_lon = alpha_lon[:, jnp.newaxis, :]
            w_lon_lin_mat = (1.0 - a_lon) * oh_lon[c0_lon_pos] + a_lon * oh_lon[
                c0_lon_pos + 1
            ]
            a_lat = alpha_lat[:, jnp.newaxis, :]
            lat_rel = (lat_c0 - win_lat_start[:, jnp.newaxis])[
                :, jnp.newaxis, :
            ]
            w_lat_lin_mat = (1.0 - a_lat) * (lat_rel == lat_arange).astype(
                dtype
            ) + a_lat * (lat_rel + 1 == lat_arange).astype(dtype)
          if vertical_order == 'linear':
            buf_z_lin_start = buf_z_start
            w_z_lin_mat = w_z_mat
          else:
            win_z_lin_start = _z_window_start(
                k_glob - eff_z_shift, l_src, w_z_lin_blk_size
            )
            buf_z_lin_start = _buffer_window_start(
                win_z_lin_start, buf_z_origin, l_buf, w_z_lin_blk_size
            )
            a_z = alpha_z[:, jnp.newaxis, :]
            z_rel = (lvl_c0 - win_z_lin_start)[:, jnp.newaxis, :]
            w_z_lin_mat = (1.0 - a_z) * (
                z_rel == z_lin_blk_arange
            ).astype(dtype) + a_z * (z_rel + 1 == z_lin_blk_arange).astype(
                dtype
            )

          k_tri_raw = c_tri * w_lat_size
          k_tri_pad = _round_to_multiple(k_tri_raw, _TPU_SUBLANES)
          f_tri_win = lax.dynamic_slice(
              f_lat,
              (
                  jnp.int32(0),
                  buf_z_lin_start,
                  jnp.int32(tri_start * w_lat_size),
                  jnp.int32(0),
              ),
              (ny_loc, w_z_lin_blk_size, k_tri_raw, n_lon),
          )
          if k_tri_pad > k_tri_raw:
            f_tri_win = jnp.pad(
                f_tri_win,
                [(0, 0), (0, 0), (0, k_tri_pad - k_tri_raw), (0, 0)],
            )
          f_tri_patch = f_tri_win.reshape(
              ny_loc, w_z_lin_blk_size * k_tri_pad, n_lon
          )
          tri_lon = jnp.einsum(
              'jkm,jmi->jki', f_tri_patch, w_lon_lin_mat, precision=dot_prec_tri
          )
          tri_after_z = (
              tri_lon.reshape(ny_loc, w_z_lin_blk_size, k_tri_pad, nx_blk)
              * w_z_lin_mat[:, :, jnp.newaxis, :]
          ).sum(axis=1)
          val_lin_jci = (
              tri_after_z[:, :k_tri_raw, :].reshape(
                  ny_loc, c_tri, w_lat_size, nx_blk
              )
              * w_lat_lin_mat[:, jnp.newaxis, :, :]
          ).sum(axis=2)

        f_patch = f_win.reshape(ny_loc, w_z_blk_size * k_pad, n_lon)
        if fuse_gath_blk:
          gathered_lon = jnp.einsum(
              'jkm,jmi->jki', f_patch, w_lon_mat, precision=dot_prec_f32
          )
          after_z = (
              gathered_lon.reshape(ny_loc, w_z_blk_size, k_pad, nx_blk)
              * w_z_mat[:, :, jnp.newaxis, :]
          ).sum(axis=1)
          val_jci = (
              after_z[:, :k_raw, :].reshape(ny_loc, c_cur, w_lat_size, nx_blk)
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
                precision=dot_prec_f32,
            )
            az_b = (
                g_b.reshape(ny_loc, w_z_blk_size, k_pad, nx_loc)
                * w_z_mat[:, :, jnp.newaxis, s0:s1]
            ).sum(axis=1)
            v_b = (
                az_b[:, :k_raw, :].reshape(ny_loc, c_cur, w_lat_size, nx_loc)
                * w_lat_mat[:, jnp.newaxis, :, s0:s1]
            ).sum(axis=2)
            if b_idx + 1 < b_len:
              v_b, f_patch_cur = lax.optimization_barrier((v_b, f_patch_cur))
            val_parts.append(v_b)
          val_jci = jnp.concatenate(val_parts, axis=2)

        if c_tri > 0 and not need_separate_trilinear:
          val_lin_jci = val_jci[:, tri_start:, :]

        if c_lim > 0:
          assert (
              b_lat is not None
              and win_lat_cell_start is not None
              and z_cell_blk_arange is not None
              and lat_cell_arange is not None
          )
          win_z_cell_start = _z_window_start(
              k_glob - eff_z_shift, l_src - 1, w_z_cell_blk_size
          )
          buf_z_cell_start = _buffer_window_start(
              win_z_cell_start, buf_z_origin, l_buf - 1, w_z_cell_blk_size
          )
          p_lon_mat = oh_lon[c0_lon_pos]
          p_z_mat = (
              (lvl_c0 - win_z_cell_start)[:, jnp.newaxis, :]
              == z_cell_blk_arange
          ).astype(dtype)
          p_lat_mat = (
              (lat_c0 - win_lat_cell_start[:, jnp.newaxis])[:, jnp.newaxis, :]
              == lat_cell_arange
          ).astype(dtype)
          b_win = lax.dynamic_slice_in_dim(
              b_lat, buf_z_cell_start, w_z_cell_blk_size, axis=1
          )
          b_patch = b_win.reshape(ny_loc, w_z_cell_blk_size * k_lim_pad, n_lon)
          b_lon = jnp.einsum(
              'jkm,jmi->jki', b_patch, p_lon_mat, precision=dot_prec_tri
          )
          b_after_z = (
              b_lon.reshape(ny_loc, w_z_cell_blk_size, k_lim_pad, nx_blk)
              * p_z_mat[:, :, jnp.newaxis, :]
          ).sum(axis=1)
          b_jci = (
              b_after_z[:, :k_lim_raw, :].reshape(
                  ny_loc, 2 * c_lim, w_lat_cell_size, nx_blk
              )
              * p_lat_mat[:, jnp.newaxis, :, :]
          ).sum(axis=2)
          lo_jci, hi_jci = b_jci[:, :c_lim, :], b_jci[:, c_lim:, :]
          val_lim = val_jci[:, lim_start:, :]
          if ref_buf is not None:
            ref_win = lax.dynamic_slice_in_dim(
                ref_buf, buf_z_start, w_z_blk_size, axis=0
            )
            ref_ji = (w_z_mat * ref_win[jnp.newaxis, :, jnp.newaxis]).sum(
                axis=1
            )[:, jnp.newaxis, :]
            val_lim = jnp.clip(val_lim + ref_ji, lo_jci, hi_jci) - ref_ji
          else:
            val_lim = jnp.clip(val_lim, lo_jci, hi_jci)
          val_jci = (
              val_lim
              if lim_start == 0
              else jnp.concatenate([val_jci[:, :lim_start, :], val_lim], axis=1)
          )

        if b_len == 1:
          if c_tri > 0:
            return jnp.swapaxes(val_jci, 0, 1), jnp.swapaxes(val_lin_jci, 0, 1)
          return jnp.swapaxes(val_jci, 0, 1)

        out_blk = val_jci.reshape(ny_loc, c_cur, b_len, nx_loc).transpose(
            2, 1, 0, 3
        )
        if c_tri > 0:
          out_lin_blk = val_lin_jci.reshape(
              ny_loc, c_tri, b_len, nx_loc
          ).transpose(2, 1, 0, 3)
          return out_blk, out_lin_blk
        return out_blk

      return _scan_block

    if l_q_loc == 1:
      _scan_1 = _eval_levels_grouped(1)
      res_0 = _scan_1(
          (jnp.int32(0), lon_q_t[0], sin_lat_q_t[0], sigma_q_t[0])
      )
      if c_tri > 0:
        out_0, out_lin_0 = res_0
        return (
            jnp.swapaxes(out_0, 1, 2)[:, jnp.newaxis, :, :],
            jnp.swapaxes(out_lin_0, 1, 2)[:, jnp.newaxis, :, :],
        )
      out_single = jnp.swapaxes(res_0, 1, 2)[:, jnp.newaxis, :, :]
      if tri_start is not None:
        return out_single, jnp.zeros((0, 1, nx_loc, ny_loc), dtype=dtype)
      return out_single

    b_len = 1
    if is_tpu:
      for b_cand in _BLOCK_LEVEL_CANDIDATES:
        if l_q_loc < b_cand:
          continue
        w_z_c = min(l_buf, min(l_src, w_z_size + b_cand - 1))
        f_blk_vmem = ny_loc * w_z_c * k_pad * n_lon_pad128 * item_bytes
        g_1_vmem = ny_loc * w_z_c * k_pad * nx_pad128 * item_bytes
        nx_c_pad128 = _round_to_multiple(b_cand * nx_loc, _TPU_LANES)
        w_lon_c_vmem = (
            ny_loc
            * _round_to_multiple(n_lon, _TPU_SUBLANES)
            * nx_c_pad128
            * item_bytes
        )
        b_vmem = 0
        if c_lim > 0:
          w_zc_c = min(
              max(1, l_buf - 1),
              min(max(1, l_src - 1), w_z_cell_size + b_cand - 1),
          )
          b_blk_vmem = ny_loc * w_zc_c * k_lim_pad * n_lon_pad128 * item_bytes
          b_lon_vmem = ny_loc * w_zc_c * k_lim_pad * nx_c_pad128 * item_bytes
          b_vmem = b_blk_vmem + w_lon_c_vmem + b_lon_vmem
        if (
            max(f_blk_vmem + w_lon_vmem_b + g_1_vmem, b_vmem)
            <= _MATMUL_3D_BLOCK_VMEM_PROBE
        ):
          b_len = b_cand
          break

    _scan_fn = _eval_levels_grouped(b_len)
    if b_len == 1:
      _, scan_res = lax.scan(
          lambda _, inp: (None, _scan_fn(inp)),
          None,
          (k_arr, lon_q_t, sin_lat_q_t, sigma_q_t),
      )
      if c_tri > 0:
        out_l, out_lin_l = scan_res
        return out_l.transpose(1, 0, 3, 2), out_lin_l.transpose(1, 0, 3, 2)
      out_all = scan_res.transpose(1, 0, 3, 2)
    else:
      n_blocks = (l_q_loc + b_len - 1) // b_len
      pad_l = n_blocks * b_len - l_q_loc
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
      else:
        k_in = k_arr.reshape(n_blocks, b_len)
        lon_in = lon_q_t.reshape(n_blocks, b_len, ny_loc, nx_loc)
        sin_lat_in = sin_lat_q_t.reshape(n_blocks, b_len, ny_loc, nx_loc)
        sigma_in = sigma_q_t.reshape(n_blocks, b_len, ny_loc, nx_loc)
      _, scan_res = lax.scan(
          lambda _, inp: (None, _scan_fn(inp)),
          None,
          (k_in, lon_in, sin_lat_in, sigma_in),
      )
      if c_tri > 0:
        out_blk, out_lin_blk = scan_res
        out_l = out_blk.reshape(
            n_blocks * b_len, c_cur, ny_loc, nx_loc
        )[:l_q_loc]
        out_lin_l = out_lin_blk.reshape(
            n_blocks * b_len, c_tri, ny_loc, nx_loc
        )[:l_q_loc]
        return out_l.transpose(1, 0, 3, 2), out_lin_l.transpose(1, 0, 3, 2)
      out_all = (
          scan_res.reshape(n_blocks * b_len, c_cur, ny_loc, nx_loc)[:l_q_loc]
          .transpose(1, 0, 3, 2)
      )

    if tri_start is not None:
      return out_all, jnp.zeros((0, l_q_loc, nx_loc, ny_loc), dtype=dtype)
    return out_all

  if num_channels <= chunk_size:
    return _eval_channel_chunk(
        f_buf, limited_channel_start, trilinear_channel_start
    )

  if limited_channel_start == 0 and trilinear_channel_start in (None, 0):
    c_pad = _round_to_multiple(num_channels, chunk_size)
    f_pad = (
        jnp.pad(f_buf, [(0, c_pad - num_channels), (0, 0), (0, 0), (0, 0)])
        if c_pad > num_channels
        else f_buf
    )
    f_grouped = f_pad.reshape(
        c_pad // chunk_size, chunk_size, l_buf, n_lon, n_lat_buf
    )
    _, out_grouped = lax.scan(
        lambda _, grp: (
            None,
            _eval_channel_chunk(grp, 0, trilinear_channel_start),
        ),
        None,
        f_grouped,
    )
    if trilinear_channel_start is not None:
      out_g, out_lin_g = out_grouped
      return (
          out_g.reshape(c_pad, l_q_loc, nx_loc, ny_loc)[:num_channels],
          out_lin_g.reshape(c_pad, l_q_loc, nx_loc, ny_loc)[:num_channels],
      )
    return out_grouped.reshape(c_pad, l_q_loc, nx_loc, ny_loc)[:num_channels]

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
    res = _eval_channel_chunk(f_buf[c0:c1], sub_lim, sub_tri)
    if sub_tri is not None:
      outs.append(res[0])
      outs_lin.append(res[1])
    else:
      outs.append(res)
  out_cat = jnp.concatenate(outs, axis=0)
  if trilinear_channel_start is not None:
    out_lin_cat = (
        jnp.concatenate(outs_lin, axis=0)
        if outs_lin
        else jnp.zeros((0, l_q_loc, nx_loc, ny_loc), dtype=dtype)
    )
    return out_cat, out_lin_cat
  return out_cat


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
    limited_channel_start: int = 0,
    trilinear_channel_start: int | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
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
        limited_channel_start=limited_channel_start,
        trilinear_channel_start=trilinear_channel_start,
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
  out_specs = (f_spec, f_spec) if trilinear_channel_start is not None else f_spec

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
        limited_channel_start=limited_channel_start,
        trilinear_channel_start=trilinear_channel_start,
    )

  out_pad = jax.shard_map(
      _sharded,
      mesh=mesh,
      in_specs=(f_spec, q_spec, q_spec, q_spec, nodes_spec, ref_spec),
      out_specs=out_specs,
  )(f_pad, lon_pad, sin_lat_pad, sigma_pad, sigma_nodes, limiter_bound_reference)
  if trilinear_channel_start is not None:
    return out_pad[0][:, :l_q, :, :], out_pad[1][:, :l_q, :, :]
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
    method: 'gather', 'matmul', or None (auto-selected: 'matmul' on TPU or
      when `grid.spmd_mesh` is set, 'gather' otherwise). 'gather' reads the
      stencil points directly and has no window. 'matmul' selects them with
      one-hot matrix products built inside a fixed window around each
      query's arrival point (see `max_lat_shift` and `max_vertical_shift`).
    max_lat_shift: half-width, in latitude rows, of the 'matmul' window. For
      queries laid out on the grid (`lon.shape == (layers, *grid.nodal_shape)`)
      each query's stencil is built inside a window of
      `2 * max_lat_shift + stencil` latitude rows (stencil is 2 for 'linear'
      and 4 for 'cubic'). The window starts `max_lat_shift` rows before the
      first stencil row of a departure point that equals its arrival point,
      and is clipped to the ends of the halo-extended latitude axis. A
      stencil point outside the window contributes nothing, so the value
      is wrong there (it is not clamped). A value of 0 is treated as 1.
      None disables windowing, as does any other query layout (the queries
      are then interpolated without a window). Ignored by 'gather'.
    max_vertical_shift: half-width, in levels, of the same window in the
      vertical: `2 * max_vertical_shift + stencil` levels (stencil is 2 for
      `vertical_order='linear'` and 4 for 'cubic'), starting
      `max_vertical_shift` levels before the first stencil level of a
      departure point that equals its arrival point, clipped to the ends of
      `sigma_nodes`. A stencil point outside it contributes nothing. A value
      of 0 is treated as 1. Windowing needs both shifts to be set; None for
      either disables it. The limiter's bracketing-cell corners are read from
      a window of the same half-width. Ignored by 'gather'.

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


def _interpolate_3d_mixed(
    fields_4d: Array,
    grid: spherical_harmonic.Grid,
    lon: Array,
    sin_lat: Array,
    sigma: Array,
    sigma_nodes: np.ndarray,
    *,
    order: str = 'cubic',
    limiter: str | None = None,
    limited_channel_start: int = 0,
    trilinear_channel_start: int | None = None,
    vertical_order: Literal['linear', 'cubic'] = 'linear',
    limiter_bound_reference: Array | None = None,
    method: InterpolationMethod | None = None,
    max_lat_shift: int | None = 6,
    max_vertical_shift: int | None = 6,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
  """Interpolates a 4-D stack with mixed channel limiting and optional trilinear slice."""
  _validate_limiter(limiter)
  resolved = _resolve_interpolation_method(method, grid)
  fields_4d = jnp.asarray(fields_4d)
  lon = jnp.asarray(lon)
  sin_lat = jnp.asarray(sin_lat)
  sigma = jnp.asarray(sigma)
  c_total = fields_4d.shape[0]
  eff_lim_start = min(c_total, max(0, limited_channel_start))

  if resolved == 'gather':
    min_levels = 2 if vertical_order == 'linear' else 4
    if len(sigma_nodes) < min_levels:
      raise ValueError(
          f'{vertical_order} vertical interpolation requires at least'
          f' {min_levels} levels'
      )
    if (
        fields_4d.ndim != 4
        or fields_4d.shape[1] != len(sigma_nodes)
        or fields_4d.shape[2:] != grid.nodal_shape
    ):
      raise ValueError(
          f'expected fields_4d of shape [C, {len(sigma_nodes)}, '
          f'{grid.nodal_shape[0]}, {grid.nodal_shape[1]}]; got'
          f' {fields_4d.shape}'
      )
    if c_total == 0 or lon.size == 0:
      out_zero = jnp.zeros((c_total, *lon.shape), dtype=fields_4d.dtype)
      if trilinear_channel_start is not None:
        eff_tri = min(c_total, max(0, trilinear_channel_start))
        return out_zero, jnp.zeros(
            (c_total - eff_tri, *lon.shape), dtype=fields_4d.dtype
        )
      return out_zero
    return _interpolate_3d_gather(
        fields_4d,
        grid,
        lon,
        sin_lat,
        sigma,
        sigma_nodes,
        order=order,
        limiter=limiter if eff_lim_start < c_total else None,
        vertical_order=vertical_order,
        limiter_bound_reference=(
            limiter_bound_reference
            if limiter is not None and eff_lim_start < c_total
            else None
        ),
        limited_channel_start=eff_lim_start,
        trilinear_channel_start=trilinear_channel_start,
    )

  if (
      c_total > 0
      and lon.size > 0
      and lon.ndim == 3
      and lon.shape[1:] == grid.nodal_shape
      and max_lat_shift is not None
      and max_vertical_shift is not None
  ):
    return _interpolate_3d_grid_matmul(
        fields_4d,
        grid,
        lon,
        sin_lat,
        sigma,
        jnp.asarray(sigma_nodes),
        order=order,
        limiter=limiter if eff_lim_start < c_total else None,
        vertical_order=vertical_order,
        limiter_bound_reference=limiter_bound_reference,
        max_lat_shift=max_lat_shift,
        max_vertical_shift=max_vertical_shift,
        limited_channel_start=eff_lim_start,
        trilinear_channel_start=trilinear_channel_start,
    )

  common_kw = dict(
      grid=grid,
      lon=lon,
      sin_lat=sin_lat,
      sigma=sigma,
      sigma_nodes=sigma_nodes,
      order=order,
      vertical_order=vertical_order,
      method=method,
      max_lat_shift=max_lat_shift,
      max_vertical_shift=max_vertical_shift,
  )
  if limiter is None or eff_lim_start == 0 or eff_lim_start == c_total:
    eff_lim = limiter if eff_lim_start < c_total else None
    out = interpolate_3d(
        fields_4d,
        limiter=eff_lim,
        limiter_bound_reference=limiter_bound_reference if eff_lim else None,
        **common_kw,
    )
  else:
    out_unlim = interpolate_3d(fields_4d[:eff_lim_start], limiter=None, **common_kw)
    out_lim = interpolate_3d(
        fields_4d[eff_lim_start:],
        limiter=limiter,
        limiter_bound_reference=limiter_bound_reference,
        **common_kw,
    )
    out = jnp.concatenate([out_unlim, out_lim], axis=0)

  if trilinear_channel_start is not None:
    if (
        order == 'linear'
        and vertical_order == 'linear'
        and (limiter is None or trilinear_channel_start < eff_lim_start)
    ):
      out_lin = out[trilinear_channel_start:]
    else:
      out_lin = interpolate_3d(
          fields_4d[trilinear_channel_start:],
          grid=grid,
          lon=lon,
          sin_lat=sin_lat,
          sigma=sigma,
          sigma_nodes=sigma_nodes,
          order='linear',
          limiter=None,
          vertical_order='linear',
          method=method,
          max_lat_shift=max_lat_shift,
          max_vertical_shift=max_vertical_shift,
      )
    return out, out_lin
  return out


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
) -> tuple[jax.Array, jax.Array]:
  """Fused linear 3-D MXU interpolation of `wind` (centers) + `sigma_dot` (boundaries)."""
  dtype = wind_buf.dtype
  is_tpu = jax.default_backend() == 'tpu'
  n_lon = grid.longitude_nodes
  n_lat = grid.latitude_nodes
  n_lat_ext = n_lat + 2 * _HALO_WIDTH
  l_src_c = centers.shape[0]
  l_src_b = boundaries.shape[0]
  _, l_buf_c, _, n_lat_buf = wind_buf.shape
  _, l_buf_b, _, _ = sd_buf.shape
  l_q_loc, nx_loc, ny_loc = lon_q.shape

  eff_lat_shift = _effective_window_shift(max_lat_shift)
  eff_z_shift = _effective_window_shift(max_vertical_shift)
  w_lat_size = _window_size(n_lat_buf, n_lat_ext, eff_lat_shift, 2)
  w_z_c_size = _window_size(l_buf_c, l_src_c, eff_z_shift, 2)
  w_z_b_size = _window_size(l_buf_b, l_src_b, eff_z_shift, 2)

  j_global = _arrival_lat_rows(j_offset, ny_loc, n_lat)
  win_lat_start, buf_lat_start = _lat_window_starts(
      j_global,
      j_global + 2 - eff_lat_shift,
      n_lat_ext,
      n_lat_buf,
      w_lat_size,
      buf_lat_origin,
  )
  m_lat_win = jax.nn.one_hot(
      buf_lat_start[:, jnp.newaxis]
      + jnp.arange(w_lat_size, dtype=jnp.int32)[jnp.newaxis, :],
      n_lat_buf,
      dtype=dtype,
  )
  m_lat_2d = m_lat_win.reshape(ny_loc * w_lat_size, n_lat_buf)

  dot_prec_lhs_bf16 = jax_numpy_utils.resolve_dot_precision(
      None, m_lat_2d, wind_buf, lhs_exact_in_bf16=True
  )
  dot_prec_f32 = jax_numpy_utils.resolve_dot_precision(None, wind_buf, wind_buf)

  k_w_raw = 3 * w_lat_size
  k_w_pad = _round_to_multiple(k_w_raw, _TPU_SUBLANES)
  wind_2d_t = (
      wind_buf.swapaxes(0, 1)
      .reshape(l_buf_c * 3, n_lon, n_lat_buf)
      .transpose(2, 0, 1)
      .reshape(n_lat_buf, l_buf_c * 3 * n_lon)
  )
  wind_lat = (
      jnp.dot(m_lat_2d, wind_2d_t, precision=dot_prec_lhs_bf16)
      .reshape(ny_loc, w_lat_size, l_buf_c * 3, n_lon)
      .transpose(0, 2, 1, 3)
      .reshape(ny_loc, l_buf_c, k_w_raw, n_lon)
  )
  if k_w_pad > k_w_raw:
    wind_lat = jnp.pad(wind_lat, [(0, 0), (0, 0), (0, k_w_pad - k_w_raw), (0, 0)])

  k_sd_raw = w_lat_size
  k_sd_pad = _round_to_multiple(k_sd_raw, _TPU_SUBLANES)
  sd_2d_t = sd_buf[0].transpose(2, 0, 1).reshape(n_lat_buf, l_buf_b * n_lon)
  sd_lat = (
      jnp.dot(m_lat_2d, sd_2d_t, precision=dot_prec_lhs_bf16)
      .reshape(ny_loc, w_lat_size, l_buf_b, n_lon)
      .transpose(0, 2, 1, 3)
  )
  if k_sd_pad > k_sd_raw:
    sd_lat = jnp.pad(sd_lat, [(0, 0), (0, 0), (0, k_sd_pad - k_sd_raw), (0, 0)])

  lon_q_t = jnp.swapaxes(lon_q, 1, 2)
  sin_lat_q_t = jnp.swapaxes(sin_lat_q, 1, 2)
  sigma_q_t = jnp.swapaxes(sigma_q, 1, 2)
  k_arr = jnp.arange(l_q_loc, dtype=jnp.int32)
  m_arange = jnp.arange(n_lon, dtype=jnp.int32)[:, jnp.newaxis]
  lat_arange = jnp.arange(w_lat_size, dtype=jnp.int32)[:, jnp.newaxis]

  item_bytes = 8 if dtype == jnp.float64 else 4
  n_lon_pad128 = _round_to_multiple(n_lon, _TPU_LANES)
  b_len = 1
  if is_tpu and l_q_loc > 1:
    for b_cand in _BLOCK_LEVEL_CANDIDATES:
      if l_q_loc < b_cand:
        continue
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
      if lhs_b + rhs_b + out_b <= _WIND_SIGMA_DOT_BLOCK_VMEM_PROBE:
        b_len = b_cand
        break

  w_z_c_blk = min(l_buf_c, min(l_src_c, w_z_c_size + b_len - 1))
  w_z_b_blk = min(l_buf_b, min(l_src_b, w_z_b_size + b_len - 1))
  z_c_arange = jnp.arange(w_z_c_blk, dtype=jnp.int32)[:, jnp.newaxis]
  z_b_arange = jnp.arange(w_z_b_blk, dtype=jnp.int32)[:, jnp.newaxis]
  k_wind_flat = w_z_c_blk * k_w_pad
  nx_blk = b_len * nx_loc

  def _scan_block(inputs):
    k_b, lon_b, sin_lat_b, sigma_b = inputs
    if b_len == 1:
      k0_s = k_b
      lon_s, sin_lat_s, sigma_s = lon_b, sin_lat_b, sigma_b
    else:
      k0_s = k_b[0]
      lon_s = jnp.swapaxes(lon_b, 0, 1).reshape(ny_loc, nx_blk)
      sin_lat_s = jnp.swapaxes(sin_lat_b, 0, 1).reshape(ny_loc, nx_blk)
      sigma_s = jnp.swapaxes(sigma_b, 0, 1).reshape(ny_loc, nx_blk)

    lon_idx, lon_w, lat_idx, lat_w, _, _, _ = _horizontal_stencil_tuple(
        grid, lon_s, sin_lat_s, 'linear'
    )
    lvl_c_idx, lvl_c_w, _, _ = _vertical_stencil_tuple(
        sigma_s, centers, 'linear'
    )
    lvl_b_idx, lvl_b_w, _, _ = _vertical_stencil_tuple(
        sigma_s, boundaries, 'linear'
    )

    k_glob_c = jnp.minimum(k0_s, l_src_c - 1)
    win_z_c_start = _z_window_start(
        k_glob_c - eff_z_shift, l_src_c, w_z_c_blk
    )
    win_z_b_start = _z_window_start(
        k_glob_c - eff_z_shift, l_src_b, w_z_b_blk
    )

    w_lon_mat = sum(
        w[:, jnp.newaxis, :]
        * (idx[:, jnp.newaxis, :] == m_arange).astype(dtype)
        for idx, w in zip(lon_idx, lon_w)
    )
    w_lat_mat = sum(
        w[:, jnp.newaxis, :]
        * (
            (idx - win_lat_start[:, jnp.newaxis])[:, jnp.newaxis, :]
            == lat_arange
        ).astype(dtype)
        for idx, w in zip(lat_idx, lat_w)
    )
    w_z_c_mat = sum(
        w[:, jnp.newaxis, :]
        * ((idx - win_z_c_start)[:, jnp.newaxis, :] == z_c_arange).astype(dtype)
        for idx, w in zip(lvl_c_idx, lvl_c_w)
    )
    w_z_b_mat = sum(
        w[:, jnp.newaxis, :]
        * ((idx - win_z_b_start)[:, jnp.newaxis, :] == z_b_arange).astype(dtype)
        for idx, w in zip(lvl_b_idx, lvl_b_w)
    )

    wind_win = lax.dynamic_slice_in_dim(
        wind_lat, win_z_c_start, w_z_c_blk, axis=1
    )
    wind_patch = wind_win.reshape(ny_loc, k_wind_flat, n_lon)
    sd_win = lax.dynamic_slice_in_dim(
        sd_lat, win_z_b_start, w_z_b_blk, axis=1
    )
    sd_patch = sd_win.reshape(ny_loc, w_z_b_blk * k_sd_pad, n_lon)

    both_patch = jnp.concatenate([wind_patch, sd_patch], axis=1)
    gathered = jnp.einsum(
        'jkm,jmi->jki', both_patch, w_lon_mat, precision=dot_prec_f32
    )
    wind_after_z = (
        gathered[:, :k_wind_flat, :].reshape(
            ny_loc, w_z_c_blk, k_w_pad, nx_blk
        )
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
        (jnp.int32(0), lon_q_t[0], sin_lat_q_t[0], sigma_q_t[0])
    )
    out_4d = jnp.swapaxes(out_0, 1, 2)[:, jnp.newaxis, :, :]
  elif b_len == 1:
    _, out_l = lax.scan(
        lambda _, inp: (None, _scan_block(inp)),
        None,
        (k_arr, lon_q_t, sin_lat_q_t, sigma_q_t),
    )
    out_4d = out_l.transpose(1, 0, 3, 2)
  else:
    n_blocks = (l_q_loc + b_len - 1) // b_len
    pad_l = n_blocks * b_len - l_q_loc
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
    else:
      k_in = k_arr.reshape(n_blocks, b_len)
      lon_in = lon_q_t.reshape(n_blocks, b_len, ny_loc, nx_loc)
      sin_lat_in = sin_lat_q_t.reshape(n_blocks, b_len, ny_loc, nx_loc)
      sigma_in = sigma_q_t.reshape(n_blocks, b_len, ny_loc, nx_loc)
    _, out_blk = lax.scan(
        lambda _, inp: (None, _scan_block(inp)),
        None,
        (k_in, lon_in, sin_lat_in, sigma_in),
    )
    out_4d = (
        out_blk.reshape(n_blocks * b_len, 4, ny_loc, nx_loc)[:l_q_loc]
        .transpose(1, 0, 3, 2)
    )
  return out_4d[:3], out_4d[3]


def _interpolate_3d_wind_and_sigma_dot_matmul(
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
  mesh = grid.spmd_mesh
  if mesh is None:
    wind_ext = _extend_with_pole_halo(wind, grid)
    sd_ext = _extend_with_pole_halo(sd_4d, grid)
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
    )

  x_shards = mesh.shape['x']
  f_spec = P(None, None, 'x', 'y')
  q_spec = P(None, 'x', 'y')

  def _sharded(w_loc, sd_loc, lon_loc, sin_lat_loc, sigma_loc, c_loc, b_loc):
    if x_shards > 1:
      w_x = lax.all_gather(w_loc, 'x', axis=2, tiled=True)
      sd_x = lax.all_gather(sd_loc, 'x', axis=2, tiled=True)
    else:
      w_x, sd_x = w_loc, sd_loc
    w_x = w_x[:, :, : grid.longitude_nodes, :]
    sd_x = sd_x[:, :, : grid.longitude_nodes, :]
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
    )
    return w_mid, sd_mid[jnp.newaxis, ...]

  wind_mid, sd_mid_4d = jax.shard_map(
      _sharded,
      mesh=mesh,
      in_specs=(f_spec, f_spec, q_spec, q_spec, q_spec, P(None), P(None)),
      out_specs=(f_spec, f_spec),
  )(wind, sd_4d, lon, sin_lat, sigma, centers_jax, boundaries_jax)
  return wind_mid, sd_mid_4d[0]


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
    method: 'gather', 'matmul', or None (auto-selected per backend/mesh); see
      `interpolate_3d`.
    max_lat_shift: latitude half-width of the 'matmul' window, in rows; the
      wind interpolated at a trajectory midpoint that lies further than this
      from its arrival row is wrong (not clamped). 0 is treated as 1, None
      disables windowing, and 'gather' has no window. See `interpolate_3d`.

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

  can_eval_at_arrival = (
      initial_guess is None
      and grid.nodal_padding == (0, 0)
      and u.shape[-2:] == grid.nodal_shape
  )
  mid_lat_shift = (
      None if max_lat_shift is None else max(1, (max_lat_shift + 1) // 2)
  )
  departure = arrival if initial_guess is None else initial_guess.cartesian
  for iter_idx in range(iterations):
    if iter_idx == 0 and can_eval_at_arrival:
      # On a cold first iteration, `midpoint == arrival` sits directly on the
      # horizontal grid nodes, where bilinear Lagrange interpolation is the
      # identity.
      wind_mid = wind
    else:
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
          max_lat_shift=mid_lat_shift,
      )
    departure = _normalize(arrival - angular_dt * wind_mid)

  return DeparturePoints(cartesian=departure, sigma=None)


def _interpolate_3d_wind_and_sigma_dot_gather(
    wind: jax.Array,
    sigma_dot: jax.Array,
    grid: spherical_harmonic.Grid,
    lon: jax.Array,
    sin_lat: jax.Array,
    sigma: jax.Array,
    centers: Array,
    boundaries: Array,
) -> tuple[jax.Array, jax.Array]:
  """Fused trilinear gather of 3-component `wind` and `sigma_dot` sharing one horizontal stencil."""
  lon_idx, lon_w, lat_idx, lat_w, _, _, ext_size = _horizontal_stencil_tuple(
      grid, lon, sin_lat, 'linear'
  )
  lvl_c_idx, lvl_c_w, _, _ = _vertical_stencil_tuple(
      sigma, jnp.asarray(centers), 'linear'
  )
  lvl_b_idx, lvl_b_w, _, _ = _vertical_stencil_tuple(
      sigma, jnp.asarray(boundaries), 'linear'
  )
  wind_ext = _extend_with_pole_halo(wind, grid)
  wind_c = jnp.moveaxis(wind_ext, 0, -1).reshape(-1, wind.shape[0])
  sd_flat = _extend_with_pole_halo(sigma_dot, grid).reshape(-1)
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
    method: 'gather', 'matmul', or None (auto-selected per backend/mesh); see
      `interpolate_3d`.
    max_lat_shift: latitude half-width of the 'matmul' window, in rows.
    max_vertical_shift: vertical half-width of the 'matmul' window, in levels.
      The winds and σ̇ interpolated at a trajectory midpoint that lies outside
      either window are wrong (not clamped). A shift of 0 is treated as 1,
      None for either disables windowing, and 'gather' has no window. See
      `interpolate_3d`.

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
  centers_np = np.asarray(centers)
  boundaries_np = np.asarray(boundaries)
  u = jnp.asarray(u)

  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  arrival = lon_lat_to_cartesian(
      jnp.broadcast_to(lon_mesh, u.shape),
      jnp.broadcast_to(sin_lat_mesh, u.shape),
  )
  sigma_arrival = jnp.broadcast_to(
      jnp.asarray(centers_np, dtype=u.dtype)[:, jnp.newaxis, jnp.newaxis],
      u.shape,
  )
  wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
  angular_dt = dt / grid.radius
  sigma_dot_arr = jnp.asarray(sigma_dot)
  mid_lat_shift = (
      None if max_lat_shift is None else max(1, (max_lat_shift + 1) // 2)
  )
  mid_vertical_shift = (
      None
      if max_vertical_shift is None
      else max(1, (max_vertical_shift + 1) // 2)
  )
  can_eval_at_arrival = (
      initial_guess is None
      and grid.nodal_padding == (0, 0)
      and u.ndim == 3
      and u.shape == (len(centers_np),) + grid.nodal_shape
      and sigma_dot_arr.shape == (len(boundaries_np),) + grid.nodal_shape
      and len(boundaries_np) == len(centers_np) + 1
      and bool(np.all(np.diff(boundaries_np) > 0))
      and bool(
          np.all(
              (centers_np >= boundaries_np[:-1])
              & (centers_np <= boundaries_np[1:])
          )
      )
  )
  resolved_mid_method = _resolve_interpolation_method(method, grid)
  can_fuse_midpoint = (
      u.ndim == 3
      and u.shape == (len(centers_np),) + grid.nodal_shape
      and sigma_dot_arr.shape == (len(boundaries_np),) + grid.nodal_shape
      and len(centers_np) >= 2
      and len(boundaries_np) >= 2
  )
  use_fused_matmul = (
      resolved_mid_method == 'matmul'
      and can_fuse_midpoint
      and mid_lat_shift is not None
      and mid_vertical_shift is not None
      and (
          grid.spmd_mesh is None
          or 'z' not in grid.spmd_mesh.axis_names
          or grid.spmd_mesh.shape['z'] == 1
      )
  )
  use_fused_gather = resolved_mid_method == 'gather' and can_fuse_midpoint
  interpolate = functools.partial(
      interpolate_3d,
      grid=grid,
      order='linear',
      method=method,
      max_lat_shift=mid_lat_shift,
      max_vertical_shift=mid_vertical_shift,
  )

  if initial_guess is None:
    departure = arrival
    sigma_departure = sigma_arrival
  else:
    departure = initial_guess.cartesian
    sigma_departure = initial_guess.sigma
  for iter_idx in range(iterations):
    if iter_idx == 0 and can_eval_at_arrival:
      # On a cold first iteration, `midpoint == arrival` and `sigma_mid == centers`,
      # so 3-D trilinear interpolation reduces identically to `wind` at layer
      # centers and 1-D vertical linear interpolation of `sigma_dot` from
      # layer boundaries to centers.
      wind_mid = wind
      w_hi = jnp.asarray(
          (centers_np - boundaries_np[:-1])
          / (boundaries_np[1:] - boundaries_np[:-1]),
          dtype=sigma_dot_arr.dtype,
      )[:, jnp.newaxis, jnp.newaxis]
      sigma_dot_mid = (1.0 - w_hi) * sigma_dot_arr[:-1] + w_hi * sigma_dot_arr[1:]
    else:
      midpoint = _normalize((arrival + departure) / 2)
      sigma_mid = (sigma_arrival + sigma_departure) / 2
      lon_mid, sin_lat_mid = cartesian_to_lon_sin_lat(midpoint)
      if use_fused_matmul:
        wind_mid, sigma_dot_mid = _interpolate_3d_wind_and_sigma_dot_matmul(
            wind,
            sigma_dot_arr,
            grid,
            lon_mid,
            sin_lat_mid,
            sigma_mid,
            centers_np,
            boundaries_np,
            max_lat_shift=mid_lat_shift,
            max_vertical_shift=mid_vertical_shift,
        )
      elif use_fused_gather:
        wind_mid, sigma_dot_mid = _interpolate_3d_wind_and_sigma_dot_gather(
            wind,
            sigma_dot_arr,
            grid,
            lon_mid,
            sin_lat_mid,
            sigma_mid,
            centers_np,
            boundaries_np,
        )
      else:
        point = dict(lon=lon_mid, sin_lat=sin_lat_mid, sigma=sigma_mid)
        wind_mid = interpolate(wind, sigma_nodes=centers_np, **point)
        sigma_dot_mid = interpolate(
            sigma_dot_arr, sigma_nodes=boundaries_np, **point
        )
    departure = _normalize(arrival - angular_dt * wind_mid)
    sigma_departure = jnp.clip(
        sigma_arrival - dt * sigma_dot_mid,
        float(centers_np[0]),
        float(centers_np[-1]),
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
      `max_lat_shift` window settings are used).
    vertical_order: vertical interpolation order (see `interpolate_3d`).
    limiter_bound_reference: optional per-level bound shift for the
      limiter (see `interpolate_3d`).
    max_vertical_shift: vertical half-width of the 'matmul' window, in levels;
      the latitude half-width is the interpolator's `max_lat_shift`. A
      departure point outside either window is interpolated wrongly (not
      clamped). 0 is treated as 1, None disables windowing, and the 'gather'
      method has no window. See `interpolate_3d`.

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
    max_vertical_shift: vertical half-width of the 'matmul' window, in levels;
      the latitude half-width is the interpolator's `max_lat_shift`. A
      departure point outside either window is interpolated wrongly (not
      clamped). 0 is treated as 1, None disables windowing, and the 'gather'
      method has no window. See `interpolate_3d`.

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
