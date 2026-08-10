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

"""Routines for horizontal regridding.

Conservative regridding schemes are adapted from:
https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276
"""
import dataclasses
import functools

from dinosaur import jax_numpy_utils
from dinosaur import spherical_harmonic
from dinosaur import typing

import jax
import jax.numpy as jnp
import numpy as np
from sklearn import neighbors


def _assert_increasing(x: typing.Array):
  if isinstance(x, np.ndarray) and not (np.diff(x) > 0).all():
    raise ValueError(f'array is not increasing: {x}')


def _latitude_cell_bounds(x):
  pi_over_2 = jnp.array([np.pi / 2])
  return jnp.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _latitude_overlap(
    source_points: typing.Array,
    target_points: typing.Array,
) -> jnp.ndarray:
  """Calculate the area overlap as a function of latitude."""
  source_bounds = _latitude_cell_bounds(source_points)
  target_bounds = _latitude_cell_bounds(target_points)
  upper = jnp.minimum(
      target_bounds[1:, jnp.newaxis], source_bounds[jnp.newaxis, 1:]
  )
  lower = jnp.maximum(
      target_bounds[:-1, jnp.newaxis], source_bounds[jnp.newaxis, :-1]
  )
  # normalized cell area: integral from lower to upper of cos(latitude)
  return (upper > lower) * (jnp.sin(upper) - jnp.sin(lower))


def conservative_latitude_weights(
    source_points: typing.Array, target_points: typing.Array
) -> jnp.ndarray:
  """Create a weight matrix for conservative regridding along latitude.

  Args:
    source_points: 1D latitude coordinates in units of radians for centers of
      source cells.
    target_points: 1D latitude coordinates in units of radians for centers of
      target cells.

  Returns:
    NumPy array with shape (target, source). Rows sum to 1.
  """
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  weights = _latitude_overlap(source_points, target_points)
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (target_points.size, source_points.size)
  return weights


def _align_phase_with(x, target, period):
  """Align the phase of a periodic number to match another.

  The returned number is equivalent to the original (modulo the period) with
  the smallest distance from the target, among the values
  `{x - period, x, x + period}`.

  Args:
    x: number to adjust.
    target: number with phase to match.
    period: periodicity.

  Returns:
    x possibly shifted up or down by `period`.
  """
  shift_down = x > target + period / 2
  shift_up = x < target - period / 2
  return x + period * shift_up - period * shift_down


def _periodic_upper_bounds(x, period):
  x_plus = _align_phase_with(jnp.roll(x, -1), x, period)
  return (x + x_plus) / 2


def _periodic_lower_bounds(x, period):
  x_minus = _align_phase_with(jnp.roll(x, +1), x, period)
  return (x_minus + x) / 2


def _periodic_overlap(x0, x1, y0, y1, period):
  # valid as long as no intervals are larger than period/2
  y0 = _align_phase_with(y0, x0, period)
  y1 = _align_phase_with(y1, x0, period)
  upper = jnp.minimum(x1, y1)
  lower = jnp.maximum(x0, y0)
  return jnp.maximum(upper - lower, 0)


def _longitude_overlap(
    first_points: typing.Array,
    second_points: typing.Array,
    period: float = 2 * np.pi,
) -> jnp.ndarray:
  """Calculate the area overlap as a function of latitude."""
  first_points = first_points % period
  first_upper = _periodic_upper_bounds(first_points, period)
  first_lower = _periodic_lower_bounds(first_points, period)

  second_points = second_points % period
  second_upper = _periodic_upper_bounds(second_points, period)
  second_lower = _periodic_lower_bounds(second_points, period)

  return jnp.vectorize(functools.partial(_periodic_overlap, period=period))(
      first_lower[:, jnp.newaxis],
      first_upper[:, jnp.newaxis],
      second_lower[jnp.newaxis, :],
      second_upper[jnp.newaxis, :],
  )


def conservative_longitude_weights(
    source_points: typing.Array, target_points: typing.Array
) -> jnp.ndarray:
  """Create a weight matrix for conservative regridding along longitude.

  Args:
    source_points: 1D longitude coordinates in units of radians for centers of
      source cells.
    target_points: 1D longitude coordinates in units of radians for centers of
      target cells.

  Returns:
    NumPy array with shape (new_size, old_size). Rows sum to 1.
  """
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  weights = _longitude_overlap(target_points, source_points)
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (target_points.size, source_points.size)
  return weights


def nearest_neighbor_indices(source_grid, target_grid):
  """Returns Haversine nearest neighbor indices from source_grid to target_grid."""
  lon_source, sin_lat_source = source_grid.nodal_mesh
  lat_source = np.arcsin(sin_lat_source)

  lon_target, sin_lat_target = target_grid.nodal_mesh
  lat_target = np.arcsin(sin_lat_target)

  # construct a BallTree to find nearest neighbor on the surface of a sphere
  index_coords = np.stack([lat_source.ravel(), lon_source.ravel()], axis=-1)
  query_coords = np.stack([lat_target.ravel(), lon_target.ravel()], axis=-1)
  tree = neighbors.BallTree(index_coords, metric='haversine')
  indices = tree.query(query_coords, return_distance=False).squeeze(axis=-1)
  return indices


@dataclasses.dataclass(frozen=True)
class Regridder:
  source_grid: spherical_harmonic.Grid
  target_grid: spherical_harmonic.Grid

  def __call__(self, field: typing.Array) -> jnp.ndarray:
    raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class BilinearRegridder(Regridder):
  """Regrid with bilinear interpolation."""

  @functools.partial(jax.jit, static_argnums=0)
  def __call__(self, field: typing.Array) -> jnp.ndarray:
    batch_interp = jax.vmap(jnp.interp, in_axes=(0, None, None))

    # interpolate latitude
    lat_source = self.source_grid.latitudes
    lat_target = self.target_grid.latitudes
    lat_interp = jnp.vectorize(batch_interp, signature='(a),(b),(b)->(a)')
    field = lat_interp(lat_target, lat_source, field)

    # interpolation longitude
    lon_source = self.source_grid.longitudes
    lon_target = self.target_grid.longitudes
    lon_interp = jnp.vectorize(
        jax.vmap(batch_interp, in_axes=(None, None, -1), out_axes=-1),
        signature='(a),(b),(b,y)->(a,y)',
    )
    field = lon_interp(lon_target, lon_source, field)

    return field


@dataclasses.dataclass(frozen=True)
class NearestRegridder(Regridder):
  """Regrid with nearest neighbor interpolation."""

  @functools.cached_property
  def indices(self):
    """The interpolation indices associated with source_grid."""
    return nearest_neighbor_indices(self.source_grid, self.target_grid)

  @functools.partial(jax.jit, static_argnums=0)
  def nearest_neighbor_2d(self, array: typing.Array) -> jnp.ndarray:
    """2d nearest neighbor interpolation using BallTree."""
    if array.shape != self.source_grid.nodal_shape:
      raise ValueError(
          f'expected {array.shape=} to match {self.source_grid.nodal_shape=}'
      )
    array = array.ravel().take(self.indices)
    return array.reshape(self.target_grid.nodal_shape)

  @functools.partial(jax.jit, static_argnums=0)
  def __call__(self, field: typing.Array) -> jnp.ndarray:
    interp = jnp.vectorize(self.nearest_neighbor_2d, signature='(a,b)->(c,d)')
    return interp(field)


@dataclasses.dataclass(frozen=True)
class ConservativeRegridder(Regridder):
  """Regrid with linear conservative regridding.

  Parameters:
    source_grid: Grid used for inputs.
    target_grid: Grid used for outputs.
    skipna: Whether to ignore nan values when interpolating. If True, acts
      like numpy nanmean and ignores NaN values when computing the mean of
      neighboring points. Returns NaN wherever all neighboring points are Nan.
      If False, cells will be NaN wherever any neighboring point is NaN.
  """
  skipna: bool = False

  @functools.cached_property
  def lat_weights(self):
    return conservative_latitude_weights(
        self.source_grid.latitudes, self.target_grid.latitudes
    )

  @functools.cached_property
  def lon_weights(self):
    return conservative_longitude_weights(
        self.source_grid.longitudes, self.target_grid.longitudes
    )

  @functools.partial(jax.jit, static_argnums=0)
  def _mean(self, field: typing.Array) -> jnp.ndarray:
    """Computes cell-averages of field on the target grid."""
    lon_weights = conservative_longitude_weights(
        self.source_grid.longitudes, self.target_grid.longitudes
    )
    lat_weights = conservative_latitude_weights(
        self.source_grid.latitudes, self.target_grid.latitudes
    )
    # Note, any NaN in input produces all NaN in output
    return jax_numpy_utils.precise_einsum(
        'ab,cd,...bd->...ac', lon_weights, lat_weights, field
    )

  def __call__(self, field: typing.Array) -> jnp.ndarray:
    not_nulls = jnp.logical_not(jnp.isnan(field))
    mean = self._mean(jnp.where(not_nulls, field, 0))
    not_null_fraction = self._mean(not_nulls)
    if self.skipna:
      return mean / not_null_fraction  # intended NaN if not_null_fraction == 0
    else:
      return jnp.where(
          jnp.isclose(not_null_fraction, 1, rtol=1e-3),
          mean / not_null_fraction,
          jnp.nan,
      )
