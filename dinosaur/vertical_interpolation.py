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

"""Routines for regridding between sigma and pressure levels."""
from __future__ import annotations

import dataclasses
import functools
import importlib
from typing import Any, Callable, Dict, Sequence, TypeVar, Union

import dinosaur
from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np


Array = typing.Array
InterpolateFn = Callable[[Array, Array, Array], Array]


def vectorize_vertical_interpolation(
    interpolate_fn: InterpolateFn,
) -> InterpolateFn:
  """Vectorizes vertical `interpolate_fn` function to work on 3+d fields."""
  # inerpolate_fn operates on `x, xp, fp` (target, source loc, source vals).
  # vmap over spatial axes for targets and source values.
  interpolate_fn = jax.vmap(interpolate_fn, (-1, None, -1), out_axes=-1)
  interpolate_fn = jax.vmap(interpolate_fn, (-1, None, -1), out_axes=-1)
  # vmap over multiple vertical targets.
  interpolate_fn = jax.vmap(interpolate_fn, (0, None, None), out_axes=0)
  # vectorize an leading dimensions.
  return jnp.vectorize(interpolate_fn, signature='(a,x,y),(b),(b,x,y)->(a,x,y)')


def _dot_interp(x, xp, fp):
  # pylint: disable=g-doc-return-or-yield,g-doc-args
  """Interpolate with a dot product instead of indexing.

  This is often much faster than the default jnp.interp on TPUs.
  """
  # TODO(shoyer): upstream this into jnp.interp
  n = len(xp)
  i = jnp.arange(n)
  dx = xp[1:] - xp[:-1]
  delta = x - xp[:-1]
  w = delta / dx
  w_left = jnp.pad(1 - w, [(0, 1)])
  w_right = jnp.pad(w, [(1, 0)])
  u = jnp.searchsorted(xp, x, side='right', method='compare_all')
  u = jnp.clip(u, 1, n - 1)
  weights = w_left * (i == (u - 1)) + w_right * (i == u)
  weights = jnp.where(x < xp[0], i == 0, weights)
  weights = jnp.where(x > xp[-1], i == (n - 1), weights)
  return jnp.dot(weights, fp, precision='highest')


@jax.jit
def interp(
    x: typing.Numeric, xp: typing.Array, fp: typing.Array
) -> jnp.ndarray:
  """Optimized version of jnp.interp."""
  return jax.lax.platform_dependent(
      x, xp, fp, tpu=_dot_interp, default=jnp.interp
  )


def _extrapolate_left(y):
  delta = y[1] - y[0]
  return jnp.concatenate([jnp.array([y[0] - delta]), y])


def _extrapolate_right(y):
  delta = y[-1] - y[-2]
  return jnp.concatenate([y, jnp.array([y[-1] + delta])])


def _extrapolate_both(y):
  return _extrapolate_left(_extrapolate_right(y))


def _linear_interp_with_safe_extrap(x, xp, fp, n=1):
  """Linear interpolation with extrapolation for n grid cells at each end."""
  for _ in range(n):
    xp = _extrapolate_both(xp)
    fp = _extrapolate_both(fp)
  return jnp.interp(x, xp, fp, left=np.nan, right=np.nan)


@jax.jit
def linear_interp_with_linear_extrap(
    x: typing.Numeric, xp: typing.Array, fp: typing.Array
) -> jnp.ndarray:
  """Linear interpolation with unlimited linear extrapolation at each end."""
  n = len(xp)
  i = jnp.arange(n)
  dx = xp[1:] - xp[:-1]
  delta = x - xp[:-1]
  w = delta / dx
  w_left = jnp.pad(1 - w, [(0, 1)])
  w_right = jnp.pad(w, [(1, 0)])
  u = jnp.searchsorted(xp, x, side='right', method='compare_all')
  u = jnp.clip(u, 1, n - 1)
  weights = w_left * (i == (u - 1)) + w_right * (i == u)
  return jnp.dot(weights, fp, precision='highest')


# TODO(shoyer): add higher order interpolation schemes, e.g., with splines.
# These aren't supported in JAX yet, but can be found in JAX-Cosmo:
# https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues/29
# https://github.com/DifferentiableUniverseInitiative/jax_cosmo/blob/master/jax_cosmo/scipy/interpolate.py


@dataclasses.dataclass(frozen=True)
class PressureCoordinates:
  """Specifies the vertical coordinate with pressure levels.

  Attributes:
    centers: center of each pressure level, starting at the level closest to the
      top of the atmosphere. Must be monotonic increasing.
    layers: number of vertical layers.
  """

  centers: np.ndarray

  def __init__(self, centers: Union[Sequence[float], np.ndarray]):
    object.__setattr__(self, 'centers', np.asarray(centers))
    if not all(np.diff(self.centers) > 0):
      raise ValueError(
          'Expected `centers` to be monotonically increasing, '
          f'got centers = {self.centers}'
      )

  @property
  def layers(self) -> int:
    return len(self.centers)

  def asdict(self) -> Dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def __hash__(self):
    return hash(tuple(self.centers.tolist()))

  def __eq__(self, other):
    return isinstance(other, PressureCoordinates) and np.array_equal(
        self.centers, other.centers
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
    if len(self.a_boundaries) != len(self.b_boundaries):
      raise ValueError(
          'Expected `a_boundaries` and `b_boundaries` to have the same length, '
          f'got {len(self.a_boundaries)} and {len(self.b_boundaries)}.'
      )

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


@functools.partial(jax.jit, static_argnums=0)
def get_surface_pressure(
    pressure_levels: PressureCoordinates,
    geopotential: typing.Array,
    orography: typing.Array,
    gravity_acceleration: float,
) -> typing.Array:
  """Calculate surface pressure from geopotential on pressure levels.

  Args:
    pressure_levels: 1D array with pressure levels.
    geopotential: array with dimensions [..., level, x, y]
    orography: array with dimensions [1, x, y]
    gravity_acceleration: acceleration due to gravity.

  Returns:
    Array with dimensions [..., x, y].
  """
  # note: relative height must be an increasing function along the level axis,
  # which is why we subtract geopotential (which decreases as you get closer to
  # the surface of the Earth)
  relative_height = orography * gravity_acceleration - geopotential

  @functools.partial(jnp.vectorize, signature='(z,x,y),(z)->(1,x,y)')
  @functools.partial(jax.vmap, in_axes=(-1, None), out_axes=-1)
  @functools.partial(jax.vmap, in_axes=(-1, None), out_axes=-1)
  def find_intercept(rh, levels):
    return linear_interp_with_linear_extrap(0.0, rh, levels)[np.newaxis]

  return find_intercept(relative_height, pressure_levels.centers)


def vertical_interpolation(
    x: typing.Array,
    xp: typing.Array,
    fp: typing.Array,
) -> typing.Array:
  """Computes linear interpolation `f(x)` with constant extrapolation.

  Args:
    x: array of vertical coordinates for which interpolation is computed.
    xp: array of vertical coordinates on which values are known.
    fp: array of values of on the grid `xp`.

  Returns:
    Interpolated values `f(x)`.
  """
  return interp(x, jnp.asarray(xp), fp)


@functools.partial(jax.jit, static_argnums=(1, 2, 4))
def interp_pressure_to_sigma(
    fields: typing.Pytree,
    pressure_coords: PressureCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
    interpolate_fn: InterpolateFn = (
        vectorize_vertical_interpolation(_linear_interp_with_safe_extrap)
    ),
) -> typing.Pytree:
  """Interpolate 3D fields from pressure to sigma levels."""
  desired = sigma_coords.centers[:, np.newaxis, np.newaxis] * surface_pressure
  regrid = lambda x: interpolate_fn(desired, pressure_coords.centers, x)

  def cond_fn(x) -> bool:
    shape = jnp.shape(x)
    return len(shape) >= 3 and shape[-3] == pressure_coords.centers.shape[0]

  return pytree_utils.tree_map_where(
      condition_fn=cond_fn,
      f=regrid,
      g=lambda x: x,
      x=fields,
  )


@functools.partial(jax.jit, static_argnums=(1, 2, 4))
def interp_sigma_to_pressure(
    fields: typing.Pytree,
    pressure_coords: PressureCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
    interpolate_fn: InterpolateFn = (
        vectorize_vertical_interpolation(_linear_interp_with_safe_extrap)
    ),
) -> typing.Pytree:
  """Interpolate 3D fields from sigma to pressure levels."""
  desired = (
      pressure_coords.centers[:, np.newaxis, np.newaxis] / surface_pressure
  )
  regrid = lambda x: interpolate_fn(desired, sigma_coords.centers, x)
  return pytree_utils.tree_map_over_nonscalars(regrid, fields)


SigmaOrPressure = TypeVar(
    'SigmaOrPressure',
    sigma_coordinates.SigmaCoordinates,
    PressureCoordinates,
)


def _interp_centers_to_centers(
    fields: typing.Pytree,
    source_sigma: SigmaOrPressure,
    target_sigma: SigmaOrPressure,
    interpolate_fn: InterpolateFn = vertical_interpolation,
) -> typing.Pytree:
  """Interpolate 3D fields on either sigma or pressure coordinates."""
  # interpolate_fn operates on `x, xp, fp` (target, source loc, source vals).
  # vmap over spatial axes for targets and source values.
  interpolate_fn = jax.vmap(interpolate_fn, (None, None, -1), out_axes=-1)
  interpolate_fn = jax.vmap(interpolate_fn, (None, None, -1), out_axes=-1)
  # vmap over multiple vertical targets.
  interpolate_fn = jax.vmap(interpolate_fn, (0, None, None), out_axes=0)
  # vectorize leading dimensions.
  interpolate_fn = jnp.vectorize(
      interpolate_fn, signature='(a),(b),(b,x,y)->(a,x,y)'
  )

  regrid = lambda x: interpolate_fn(
      target_sigma.centers, source_sigma.centers, x  # currently I have an error
  )

  def cond_fn(x) -> bool:
    x = jnp.asarray(x)
    return x.ndim > 2

  return pytree_utils.tree_map_where(
      condition_fn=cond_fn,
      f=regrid,
      g=lambda x: x,
      x=fields,
  )


interp_sigma_to_sigma = _interp_centers_to_centers
interp_pressure_to_pressure = _interp_centers_to_centers


@functools.partial(jax.jit, static_argnums=(1, 2))
def interp_hybrid_to_sigma(
    fields: typing.Pytree,
    hybrid_coords: HybridCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
) -> typing.Pytree:
  """Interpolate 3D fields from hybrid to sigma levels."""
  # Linear interpolation from values at cell centers on hybrid levels to cell
  # centers on sigma levels. Here we interpolate on sigma coordinates.

  @jax.jit
  @functools.partial(jnp.vectorize, signature='(x,y),(a),(b,x,y)->(a,x,y)')
  @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
  @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
  def regrid(surface_pressure, target_sigmas, field):
    source_sigmas = hybrid_coords.get_sigma_centers(surface_pressure)
    return jax.vmap(_linear_interp_with_safe_extrap, in_axes=(0, None, None))(
        target_sigmas, source_sigmas, field
    )

  return pytree_utils.tree_map_over_nonscalars(
      lambda x: regrid(surface_pressure, sigma_coords.centers, x), fields
  )


def _interval_overlap(
    source_bounds: typing.Array, target_bounds: typing.Array
) -> jnp.ndarray:
  """Calculate the interval overlap between grid cells."""
  # based on https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276
  # (see also horizontal_interpolation.py)
  upper = jnp.minimum(
      target_bounds[1:, jnp.newaxis], source_bounds[jnp.newaxis, 1:]
  )
  lower = jnp.maximum(
      target_bounds[:-1, jnp.newaxis], source_bounds[jnp.newaxis, :-1]
  )
  return jnp.maximum(upper - lower, 0)


def conservative_regrid_weights(
    source_bounds: typing.Array, target_bounds: typing.Array
) -> jnp.ndarray:
  """Create a weight matrix for conservative regridding on pressure levels.

  Args:
    source_bounds: boundaries between increasing pressure levels for the source
      grid. Values must be strictly increasing.
    target_bounds: 1D strictly increasing pressure levels for the target grid.
      Values must be strictly increasing.

  Returns:
    NumPy array with shape (target, source). Rows sum to 1.
  """
  weights = _interval_overlap(source_bounds, target_bounds)
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (target_bounds.size - 1, source_bounds.size - 1)
  return weights


@functools.partial(jax.jit, static_argnums=(1, 2))
def regrid_hybrid_to_sigma(
    fields: typing.Pytree,
    hybrid_coords: HybridCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
) -> typing.Pytree:
  """Conservatively regrid 3D fields from hybrid to sigma levels."""
  # Conservative regridding is a simple area-weighted average of source cells.
  # Here we are regridding in one dimension, so it only depends on the overlap
  # between cell boundaries. Here we calculate both bounds in terms of sigma
  # coordinates (from 0 to 1).

  # Note: c=a-1, but the signature argument does not support arithemtic in
  # dimension specifications. This is verified via assert statements below.
  @jax.jit
  @functools.partial(jnp.vectorize, signature='(x,y),(a),(b,x,y)->(c,x,y)')
  @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
  @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
  def regrid(surface_pressure, sigma_bounds, field):
    assert sigma_bounds.shape == (sigma_coords.layers + 1,)
    if field.shape[0] != hybrid_coords.layers:
      raise ValueError(
          f'Source has {hybrid_coords.layers} layers, but field has'
          f' {fields.shape[0]}'
      )
    hybrid_bounds = hybrid_coords.get_sigma_boundaries(surface_pressure)
    weights = conservative_regrid_weights(hybrid_bounds, sigma_bounds)
    result = jnp.einsum('ab,b->a', weights, field, precision='float32')
    assert result.shape[0] == sigma_coords.layers
    return result

  return pytree_utils.tree_map_over_nonscalars(
      lambda x: regrid(surface_pressure, sigma_coords.boundaries, x), fields
  )


@dataclasses.dataclass(frozen=True)
class Regridder:
  """Regrid vertically, from hybrid to sigma coordinates."""

  # TODO(shoyer): support more generic vertical coordinate systems.
  source_grid: HybridCoordinates | PressureCoordinates
  target_grid: sigma_coordinates.SigmaCoordinates | PressureCoordinates

  def __call__(
      self, field: typing.Array, surface_pressure: typing.Array | None
  ) -> jnp.ndarray:
    raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ConservativeRegridder(Regridder):
  """Regrid with conservative interpolation."""

  def __call__(
      self, field: typing.Array, surface_pressure: typing.Array | None
  ) -> jnp.ndarray:
    if surface_pressure is None:
      raise ValueError(
          'surface_pressure is required for hybrid to sigma regridding'
      )
    return regrid_hybrid_to_sigma(
        field, self.source_grid, self.target_grid, surface_pressure
    )


@dataclasses.dataclass(frozen=True)
class BilinearRegridder(Regridder):
  """Regrid with bilinear interpolation."""

  def __call__(
      self, field: typing.Array, surface_pressure: typing.Array | None
  ) -> jnp.ndarray:
    if isinstance(self.source_grid, HybridCoordinates) and isinstance(
        self.target_grid, sigma_coordinates.SigmaCoordinates
    ):
      if surface_pressure is None:
        raise ValueError(
            'surface_pressure is required for hybrid to sigma regridding'
        )
      return interp_hybrid_to_sigma(
          field, self.source_grid, self.target_grid, surface_pressure
      )
    elif isinstance(self.source_grid, PressureCoordinates) and isinstance(
        self.target_grid, PressureCoordinates
    ):
      return interp_pressure_to_pressure(
          field, self.source_grid, self.target_grid
      )
    else:
      raise ValueError(
          f'Unsupported source grid type: {type(self.source_grid)} '
          f'and target grid type: {type(self.target_grid)}'
      )
