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

This module is the public API. The interpolation itself lives in two
implementation modules: `lagrange_interpolation` (stencils, the window clamp
and the gather kernels) and `semi_lagrangian_matmul` (halo exchanges, memory
planners and the matmul kernels for TPUs and SPMD meshes).

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
from dinosaur import lagrange_interpolation
from dinosaur import semi_lagrangian_matmul
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

# Default limit, in grid cells, on the displacement of a departure point from
# its arrival point in latitude and in the vertical (`max_lat_shift` and
# `max_vertical_shift`; see `interpolate_3d`).
DEFAULT_MAX_SHIFT = 6
_INTERPOLATION_METHODS = ('gather', 'matmul')


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
  return jnp.stack(
      [
          -u * sin_lon - v * sin_lat * cos_lon,
          u * cos_lon - v * sin_lat * sin_lon,
          v * cos_lat,
      ]
  )


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


def validate_window_shift(name: str, max_shift: int | None) -> None:
  """Raises unless `max_shift` is None or a non-negative integer.

  Args:
    name: name of the argument, for the error message (`'max_lat_shift'` or
      `'max_vertical_shift'`).
    max_shift: the displacement limit to check.
  """
  if max_shift is None:
    return
  if (
      isinstance(max_shift, bool)
      or not isinstance(max_shift, (int, np.integer))
      or max_shift < 0
  ):
    raise ValueError(
        f'{name} must be None or a non-negative integer; got {max_shift!r}'
    )


def validate_interpolation_method(method: str | None) -> None:
  """Raises unless `method` is None, 'gather' or 'matmul'."""
  if method is not None and method not in _INTERPOLATION_METHODS:
    raise ValueError(
        f'unknown interpolation {method=}; valid values are None (auto) or '
        f'one of {_INTERPOLATION_METHODS}'
    )


def _resolve_interpolation_method(
    method: str | None, grid: spherical_harmonic.Grid
) -> str:
  """Resolves `method=None` lazily at trace time (never at init/import time)."""
  validate_interpolation_method(method)
  if method is not None:
    return method
  if semi_lagrangian_matmul.spatial_mesh(
      grid
  ) is not None or lagrange_interpolation.targets_tpu(grid):
    return 'matmul'
  return 'gather'


def _interpolate_2d_multi(
    fields: Array,
    grid: spherical_harmonic.Grid,
    lon: Array,
    sin_lat: Array,
    *,
    order: str = 'cubic',
    limiter: str | None = None,
    method: InterpolationMethod | None = None,
    max_lat_shift: int | None = DEFAULT_MAX_SHIFT,
) -> jnp.ndarray:
  """Interpolates [C, nx, ny] or [C, levels, nx, ny] at shared points."""
  lagrange_interpolation.validate_limiter(limiter)
  validate_window_shift('max_lat_shift', max_lat_shift)
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
      return lagrange_interpolation.interpolate_2d_gather(
          fields,
          grid,
          lon,
          sin_lat,
          order=order,
          limiter=limiter,
          max_lat_shift=max_lat_shift,
      )
    if lon.shape == grid.nodal_shape and max_lat_shift is not None:
      return semi_lagrangian_matmul.interpolate_2d_grid(
          fields,
          grid,
          lon,
          sin_lat,
          order=order,
          limiter=limiter,
          max_lat_shift=max_lat_shift,
      )
    return semi_lagrangian_matmul.interpolate_2d_points(
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
      return semi_lagrangian_matmul.interpolate_2d_grid_levels(
          fields,
          grid,
          lon,
          sin_lat,
          order=order,
          limiter=limiter,
          max_lat_shift=max_lat_shift,
      )

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
    method: 'gather' (1-D gather + fused contraction, fastest on single GPU),
      'matmul' (MXU matrix multiplication without unstructured
      gather/scatter, designed for TPU and SPMD sharding; windowed, see
      `max_lat_shift`), or None (auto-selects 'matmul' on TPU or when
      `grid.spmd_mesh` shards a spatial axis, and 'gather' otherwise). Both
      give the same result to rounding.
    max_lat_shift: limit, in latitude cells, on the displacement of
      grid-aligned queries (`lon.shape == grid.nodal_shape`, or with a
      leading levels axis), whose point `[..., i, j]` is taken to be the
      departure point of a trajectory arriving at latitude row `j`: its
      latitude is clamped to the window of `max(1, max_lat_shift)` cells
      around the arrival row, by both methods (see `interpolate_3d` for the
      exact bounds). For 'matmul' this is also the half-width of the window
      in which each stencil is built. None disables the clamp and the
      window, as does any other query layout; pass None to interpolate at
      arbitrary points that happen to be laid out like the grid.
  """

  grid: spherical_harmonic.Grid
  order: str = 'cubic'
  limiter: str | None = None
  method: InterpolationMethod | None = dataclasses.field(
      default=None, kw_only=True
  )
  max_lat_shift: int | None = dataclasses.field(
      default=DEFAULT_MAX_SHIFT, kw_only=True
  )

  def __post_init__(self):
    lagrange_interpolation.validate_limiter(self.limiter)
    validate_interpolation_method(self.method)
    validate_window_shift('max_lat_shift', self.max_lat_shift)

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
    max_lat_shift: int | None = DEFAULT_MAX_SHIFT,
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
    max_lat_shift: int | None = DEFAULT_MAX_SHIFT,
    max_vertical_shift: int | None = DEFAULT_MAX_SHIFT,
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
      when `grid.spmd_mesh` shards a spatial axis, 'gather' otherwise).
      'gather' reads the stencil points directly. 'matmul' selects them with
      one-hot matrix products built inside a fixed window around each
      query's arrival point (see `max_lat_shift` and `max_vertical_shift`).
      Both methods evaluate the same (clamped) queries and agree to rounding.
    max_lat_shift: limit, in latitude cells, on the meridional displacement
      of grid-aligned queries: those laid out as
      `lon.shape == (layers, *grid.nodal_shape)`, where query `[k, i, j]` is
      the departure point of a trajectory arriving at latitude row `j` (the
      last physical row for nodal padding) and level
      `min(k, len(sigma_nodes) - 1)`. With `S = max(1, max_lat_shift)`, its
      latitude is clamped to `[φ[j - S], φ[j + S + 1])`, φ being the node
      latitudes by Gaussian row, continued across the poles by mirror
      images (`φ[-1] = -π - φ[0]`, ...): the bracketing cell is at most S
      cells from the cell starting at the arrival row. A departure point
      further away is interpolated on the window edge (the upper edge one
      ulp below its node, which keeps the bracketing cell). The value is
      then a bounded interpolation near the edge, not an extrapolation, and
      the gradient with respect to a clamped coordinate is zero. Bounds that
      would lie beyond a pole are never reached, so near the poles only the
      equatorward limit acts. The clamp is applied by both methods whenever
      the shift is given, so results do not depend on the backend and the
      shift must exceed the largest displacement a time step should
      represent faithfully. 'matmul' builds each stencil in a window of
      `2 * S + stencil` latitude rows (stencil is 2 for 'linear' and 4 for
      'cubic'), clipped to the ends of the halo-extended latitude axis;
      clamping keeps every stencil point inside it. None disables both the
      clamp and the window, as does any other query layout (such queries
      are interpolated where they are). Clamping needs both shifts; None for
      either disables it in both directions.
    max_vertical_shift: the same limit in the vertical, in cells of
      `sigma_nodes` (z): the departure σ of a grid-aligned query arriving at
      level k is clamped to `[z[k - S], z[k + S + 1])` with
      `S = max(1, max_vertical_shift)` (intersected with `[z[0], z[-1]]`).
      For 'matmul' it is the
      half-width of a window of `2 * max(1, max_vertical_shift) + stencil`
      levels (stencil 2 for `vertical_order='linear'`, 4 for 'cubic'). The
      limiter's bracketing cell, the `limiter_bound_reference` shift, and
      any trilinear companion output are those of the clamped query.

  Returns:
    Interpolated values of shape `(*channels, *lon.shape)`.
  """
  lagrange_interpolation.validate_limiter(limiter)
  validate_window_shift('max_lat_shift', max_lat_shift)
  validate_window_shift('max_vertical_shift', max_vertical_shift)
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
    # Windowed queries are clamped exactly as the 'matmul' method clamps them.
    lat_bounds, sigma_bounds = lagrange_interpolation.window_clamp_bounds_3d(
        grid,
        lon,
        sin_lat,
        sigma,
        sigma_nodes,
        order,
        max_lat_shift,
        max_vertical_shift,
    )
    out_4d, _ = lagrange_interpolation.interpolate_3d_gather(
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
        lat_bounds=lat_bounds,
        sigma_bounds=sigma_bounds,
    )
  elif lagrange_interpolation.is_windowed_3d(
      lon, grid, max_lat_shift, max_vertical_shift
  ):
    out_4d, _ = semi_lagrangian_matmul.interpolate_3d_grid(
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
    out_4d = semi_lagrangian_matmul.interpolate_3d_points(
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
    max_lat_shift: int | None = DEFAULT_MAX_SHIFT,
    max_vertical_shift: int | None = DEFAULT_MAX_SHIFT,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Interpolates a stack of fields of which only some channels are limited.

  As `interpolate_3d` for `fields_4d` of shape [C, levels, *nodal_shape], with
  the limiter applied to the channels from `limited_channel_start` on only,
  and optionally a second, trilinear (low-order, unlimited) interpolation of
  the channels from `trilinear_channel_start` on at the same queries, which
  shares the stencil search and the gathered values of the first.

  Returns:
    `(values, trilinear)`: `values` of shape [C, *lon.shape], and `trilinear`
    of shape [C - trilinear_channel_start, *lon.shape], or None if
    `trilinear_channel_start` is None.
  """
  lagrange_interpolation.validate_limiter(limiter)
  validate_window_shift('max_lat_shift', max_lat_shift)
  validate_window_shift('max_vertical_shift', max_vertical_shift)
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
      return out_zero, None
    lat_bounds, sigma_bounds = lagrange_interpolation.window_clamp_bounds_3d(
        grid,
        lon,
        sin_lat,
        sigma,
        sigma_nodes,
        order,
        max_lat_shift,
        max_vertical_shift,
    )
    return lagrange_interpolation.interpolate_3d_gather(
        fields_4d,
        grid,
        lon,
        sin_lat,
        sigma,
        sigma_nodes,
        lat_bounds=lat_bounds,
        sigma_bounds=sigma_bounds,
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
      and lagrange_interpolation.is_windowed_3d(
          lon, grid, max_lat_shift, max_vertical_shift
      )
  ):
    return semi_lagrangian_matmul.interpolate_3d_grid(
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
    out_unlim = interpolate_3d(
        fields_4d[:eff_lim_start], limiter=None, **common_kw
    )
    out_lim = interpolate_3d(
        fields_4d[eff_lim_start:],
        limiter=limiter,
        limiter_bound_reference=limiter_bound_reference,
        **common_kw,
    )
    out = jnp.concatenate([out_unlim, out_lim], axis=0)

  out_lin = None
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
    max_lat_shift: int | None = DEFAULT_MAX_SHIFT,
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
    max_lat_shift: latitude displacement limit, in cells, of the transport
      this solve serves (see `interpolate_3d`). The trajectory midpoint lies
      about half as far from the arrival point as the departure point, so
      the wind is interpolated there with the limit
      `max(1, (max_lat_shift + 1) // 2)`: a midpoint beyond it is clamped to
      that window (on both methods), which also bounds the departure
      displacement. The returned departure points themselves are not
      clamped. None disables the clamp and the 'matmul' window.

  Returns:
    DeparturePoints with fields of shape [*batch, longitude_nodes,
    latitude_nodes] and `sigma=None`.
  """
  validate_interpolation_method(method)
  validate_window_shift('max_lat_shift', max_lat_shift)
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
    max_lat_shift: int | None = DEFAULT_MAX_SHIFT,
    max_vertical_shift: int | None = DEFAULT_MAX_SHIFT,
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
    max_lat_shift: latitude displacement limit, in cells, of the transport
      this solve serves (see `interpolate_3d`).
    max_vertical_shift: vertical displacement limit, in cells, of the same.
      The trajectory midpoint lies about half as far from the arrival point
      as the departure point, so the winds (on layer centers) and σ̇ (on
      layer boundaries) are interpolated there with the limits
      `max(1, (shift + 1) // 2)`, in cells of each one's own nodes: a
      midpoint beyond them is clamped to those windows (on both methods),
      which also bounds the departure displacement. The returned departure
      points themselves are not clamped. None for either shift disables the
      clamp and the 'matmul' window.

  Returns:
    DeparturePoints with fields of shape [layers, longitude_nodes,
    latitude_nodes], including departure σ coordinates.
  """
  validate_interpolation_method(method)
  validate_window_shift('max_lat_shift', max_lat_shift)
  validate_window_shift('max_vertical_shift', max_vertical_shift)
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
      and semi_lagrangian_matmul.mesh_z_shards(grid.spmd_mesh) == 1
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
      # On a cold first iteration, `midpoint == arrival` and
      # `sigma_mid == centers`, so 3-D trilinear interpolation reduces
      # identically to `wind` at layer centers and 1-D vertical linear
      # interpolation of `sigma_dot` from layer boundaries to centers.
      wind_mid = wind
      w_hi = jnp.asarray(
          (centers_np - boundaries_np[:-1])
          / (boundaries_np[1:] - boundaries_np[:-1]),
          dtype=sigma_dot_arr.dtype,
      )[:, jnp.newaxis, jnp.newaxis]
      sigma_dot_mid = (1.0 - w_hi) * sigma_dot_arr[:-1] + w_hi * sigma_dot_arr[
          1:
      ]
    else:
      midpoint = _normalize((arrival + departure) / 2)
      sigma_mid = (sigma_arrival + sigma_departure) / 2
      lon_mid, sin_lat_mid = cartesian_to_lon_sin_lat(midpoint)
      if use_fused_matmul:
        wind_mid, sigma_dot_mid = (
            semi_lagrangian_matmul.interpolate_3d_wind_and_sigma_dot(
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
        )
      elif use_fused_gather:
        wind_mid, sigma_dot_mid = (
            lagrange_interpolation.interpolate_3d_wind_and_sigma_dot_gather(
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
  return interpolator.grid.zero_nodal_padding(out)


def transport_scalar(
    field: Array,
    departure: DeparturePoints,
    vertical: sigma_coordinates.SigmaCoordinates,
    interpolator: GridInterpolator,
    *,
    vertical_order: Literal['linear', 'cubic'] = 'linear',
    limiter_bound_reference: Array | None = None,
    max_vertical_shift: int | None = DEFAULT_MAX_SHIFT,
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
    max_vertical_shift: vertical displacement limit, in levels; the latitude
      limit is the interpolator's `max_lat_shift`. A departure point beyond
      either limit is interpolated at the edge of the window (on both
      methods); None for either disables the clamp and the 'matmul' window.
      See `interpolate_3d`.

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
  return interpolator.grid.zero_nodal_padding(out)


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
      jnp.broadcast_to(lon_mesh, departure.cartesian.shape[1:]),
      jnp.broadcast_to(sin_lat_mesh, departure.cartesian.shape[1:]),
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
  return grid.zero_nodal_padding(u_arr), grid.zero_nodal_padding(v_arr)


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
    max_vertical_shift: int | None = DEFAULT_MAX_SHIFT,
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
    max_vertical_shift: vertical displacement limit, in levels; the latitude
      limit is the interpolator's `max_lat_shift`. A departure point beyond
      either limit is interpolated at the edge of the window (on both
      methods); None for either disables the clamp and the 'matmul' window.
      See `interpolate_3d`.

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


class TransportedWindAndScalars(NamedTuple):
  """Result of `transport_wind_and_scalars`.

  Attributes:
    u: transported zonal wind at arrival points.
    v: transported meridional wind at arrival points.
    scalars: transported scalar fields, [scalars, *departure.lon.shape].
    low_order_scalars: trilinear (low-order, unlimited) interpolation of the
      scalars from `low_order_scalar_start` on, at the same departure points;
      it has zero scalars if none was requested.
  """

  u: jnp.ndarray
  v: jnp.ndarray
  scalars: jnp.ndarray
  low_order_scalars: jnp.ndarray


def transport_wind_and_scalars(
    u: Array,
    v: Array,
    scalars: Array,
    departure: DeparturePoints,
    vertical: sigma_coordinates.SigmaCoordinates,
    interpolator: GridInterpolator,
    *,
    limited_scalar_start: int = 0,
    low_order_scalar_start: int | None = None,
    rotate: bool = True,
    planetary_rotation_rate: float | None = None,
    vertical_order: Literal['linear', 'cubic'] = 'linear',
    max_vertical_shift: int | None = DEFAULT_MAX_SHIFT,
) -> TransportedWindAndScalars:
  """Remaps winds and scalars along the same 3-D trajectories in one pass.

  The result is that of `transport_wind` (with an unlimited interpolator)
  and `transport_scalar`, but the three Cartesian wind components and the
  scalars are interpolated as one stack of fields: the stencils of the
  departure points are located once and every field is read at them
  together.

  Optionally the same pass also returns a second, trilinear interpolation of
  some of the scalars (`low_order_scalar_start`). Mass fixers of the
  Bermejo-Conde type compare a tracer's high-order interpolant with its
  low-order one at the same departure points; obtaining it here reuses the
  stencil search and the values already read.

  Args:
    u: nodal zonal wind (true winds, not cosθ-scaled) at layer centers, of
      shape [layers, longitude_nodes, latitude_nodes].
    v: nodal meridional wind, same shape as `u`.
    scalars: nodal scalar fields at layer centers, of shape
      [scalars, layers, longitude_nodes, latitude_nodes] and the dtype of the
      winds.
    departure: 3-D departure points (with σ coordinates).
    vertical: vertical coordinates on which the fields live (anything with
      increasing `.centers` node values).
    interpolator: interpolation rule (its grid, order, method and
      `max_lat_shift`). Its limiter, if any, is applied to the scalars from
      `limited_scalar_start` on, and never to the wind: use `transport_wind`
      for limited winds.
    limited_scalar_start: index of the first scalar the limiter applies to.
    low_order_scalar_start: if not None, the scalars from this index on are
      also interpolated trilinearly, without limiter.
    rotate: whether to parallel-transport the interpolated wind vectors from
      the departure to the arrival tangent plane (see `transport_wind`).
    planetary_rotation_rate: if not None, transport the planetary momentum
      `v + 2Ω ✕ R` (see `transport_wind`).
    vertical_order: vertical interpolation order (see `interpolate_3d`).
    max_vertical_shift: vertical displacement limit, in levels (see
      `transport_scalar`).

  Returns:
    The transported winds and scalars (`TransportedWindAndScalars`).
  """
  if departure.sigma is None:
    raise ValueError('transport_wind_and_scalars requires 3-D departure points')
  grid = interpolator.grid
  scalars = jnp.asarray(scalars)
  if scalars.ndim != 4:
    raise ValueError(
        'expected scalars of shape [scalars, layers, longitude_nodes,'
        f' latitude_nodes]; got {scalars.shape}'
    )
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
  num_wind = wind.shape[0]
  values, low_order = _interpolate_3d_mixed(
      jnp.concatenate([wind, scalars], axis=0),
      grid,
      departure.lon,
      departure.sin_lat,
      departure.sigma,
      sigma_nodes=vertical.centers,
      order=interpolator.order,
      limiter=interpolator.limiter,
      limited_channel_start=num_wind + limited_scalar_start,
      trilinear_channel_start=(
          None
          if low_order_scalar_start is None
          else num_wind + low_order_scalar_start
      ),
      vertical_order=vertical_order,
      method=interpolator.method,
      max_lat_shift=interpolator.max_lat_shift,
      max_vertical_shift=max_vertical_shift,
  )
  u_arrival, v_arrival = _finish_wind_transport(
      values[:num_wind], departure, grid, rotate, planetary_rotation_rate
  )
  if low_order is None:
    low_order = jnp.zeros((0, *values.shape[1:]), dtype=values.dtype)
  return TransportedWindAndScalars(
      u=u_arrival,
      v=v_arrival,
      scalars=grid.zero_nodal_padding(values[num_wind:]),
      low_order_scalars=grid.zero_nodal_padding(low_order),
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
