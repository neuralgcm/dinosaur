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

"""A vertical coordinate system based on normalized pressure.

See https://en.wikipedia.org/wiki/Sigma_coordinate_system
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Callable

from dinosaur import jax_numpy_utils
from dinosaur import typing
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


Array = typing.Array

# All `einsum`s should be done at highest available precision.
einsum = functools.partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def _slice_shape_along_axis(
    x: np.ndarray,
    axis: int,
    slice_width: int = 1,
) -> tuple[int, ...]:
  """Returns a shape of `x` sliced along `axis` with width `slice_width`."""
  x_shape = list(x.shape)
  x_shape[axis] = slice_width
  return tuple(x_shape)


def _with_f64_math(
    f: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
  """Returns a function that uses float64 internally."""
  return lambda x: f(x.astype(np.float64)).astype(x.dtype)


# TODO(dkochkov) Consider renaming `SigmaCoordinates` to `Grid` or `SigmaGrid`.
@dataclasses.dataclass(frozen=True)
class SigmaCoordinates:
  """A description of a discrete vertical coordinate system.

  Layers are indexed from the "top" of the atmosphere (𝜎 = 0) to the surface of
  the earth (𝜎 = 1).

  Attributes:
    boundaries: sigma values of the boundaries of horizontal layers. Must be an
      increasing array of values beginning with zero and ending with one. For
      `n` layers, `boundaries` has length `n + 1`.
    internal_boundaries: sigma values of the boundaries _between_ layers. For
      `n` layers, `internal_boundaries` has length `n - 1`.
    centers: sigma values of the centers of horizontal layers. For `n` layers,
      `centers` has length `n`
    layer_thickness: the thickness of each layer in sigma coordinates. For `n`
      layers, `layer_thickness` has length `n`.
    center_to_center: the distances between the centers of each layer in sigma
      coordinates. For `n` layers, `center_to_center` has length `n - 1`.
    layers: the number of layers.
  """

  boundaries: np.ndarray

  def __init__(self, boundaries: np.typing.ArrayLike):
    boundaries = np.asarray(boundaries)
    if not (np.isclose(boundaries[0], 0) and np.isclose(boundaries[-1], 1)):
      raise ValueError(
          'Expected boundaries[0] = 0, boundaries[-1] = 1, '
          f'got boundaries = {boundaries}'
      )
    if not all(np.diff(boundaries) > 0):
      raise ValueError(
          'Expected `boundaries` to be monotonically increasing, '
          f'got boundaries = {boundaries}'
      )
    object.__setattr__(self, 'boundaries', boundaries)

  @property
  def internal_boundaries(self) -> np.ndarray:
    return self.boundaries[1:-1]

  @property
  def centers(self) -> np.ndarray:
    # Use float64 internally so we can convert float32 boundaries to centers and
    # back without any loss of precision.
    return _with_f64_math(lambda x: (x[1:] + x[:-1]) / 2)(self.boundaries)

  @property
  def layer_thickness(self) -> np.ndarray:
    return _with_f64_math(np.diff)(self.boundaries)

  @property
  def center_to_center(self) -> np.ndarray:
    return _with_f64_math(np.diff)(self.centers)

  @property
  def layers(self) -> int:
    return len(self.boundaries) - 1

  @classmethod
  def equidistant(
      cls, layers: int, dtype: np.typing.DTypeLike = np.float32
  ) -> SigmaCoordinates:
    boundaries = np.linspace(0, 1, layers + 1, dtype=dtype)
    return cls(boundaries)

  @classmethod
  def from_centers(cls, centers: np.typing.ArrayLike):
    """Create sigma coordinates from the centers of each layer."""
    # The relationship between cell centers and boundaries is given by:
    #   centers[i] = 0.5 * (boundaries[i] + boundaries[i + 1])
    # Writing this as a matrix and dropping the column corresponding to
    # boundaries[0] (fixed at zero), we have a linear system of N equations and
    # N unknowns that we can solve to obtain cell boundaries.

    def centers_to_boundaries(centers):
      layers = len(centers)
      bounds_to_centers = 0.5 * (np.eye(layers) + np.eye(layers, k=-1))
      unpadded_bounds = np.linalg.solve(bounds_to_centers, centers)
      return np.pad(unpadded_bounds, [(1, 0)])

    boundaries = _with_f64_math(centers_to_boundaries)(centers)
    return cls(boundaries)

  def asdict(self):
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def __hash__(self):
    return hash(tuple(self.centers.tolist()))

  def __eq__(self, other):
    return isinstance(other, SigmaCoordinates) and np.array_equal(
        self.centers, other.centers
    )


# For consistency with commonly accepted notation, we use Greek letters within
# some of the functions below.
# pylint: disable=invalid-name


@jax.named_call
def centered_difference(
    x: np.ndarray, coordinates: SigmaCoordinates, axis: int = -3
) -> np.ndarray:
  """Derivative of `x` with respect to `sigma` along specified `axis`.

  The derivative is approximated as

  (∂x / ∂𝜎)[n + ½] ≈ (x[n + 1] - x[n]) / (𝜎[n + 1] - 𝜎[n])

  So, the derivatives will be located on the 'boundaries' between layers and
  will consist of one fewer values than `x`.

  Args:
    x: an array of values with `x.shape[axis] == coordinates.layers`. These
      values are "located" at `coordinates` layer centers.
    coordinates: a `SigmaCoordinates` object describing the vertical coordinates
      to differentiate against. Must satisfy `coordinates.layers ==
      x.shape[axis]`.
    axis: the axis along which to approximate the derivative. Defaults to -3
      since this is the axis we typically use to index layers.

  Returns:
    An array `x_sigma` with `x_sigma.shape[axis] == x.shape[axis] - 1`,
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
  inv_d𝜎 = 1 / coordinates.center_to_center
  inv_d𝜎_axes = [dx_axes[axis]]
  return einsum(dx, dx_axes, inv_d𝜎, inv_d𝜎_axes, dx_axes, precision='float32')  # pytype: disable=bad-return-type


@jax.named_call
def cumulative_sigma_integral(
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    downward: bool = True,
    cumsum_method: str = 'dot',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Approximates the integral of a quantity `x` with respect to 𝜎.

  Uses a midpoint rule (https://en.wikipedia.org/wiki/Riemann_sum#Midpoint_rule)
  to approximate the integral

    ∫x d𝜎   𝜎= 0...b

  where 𝜎= 0...b if `downward` is True, and 𝜎= 1...b otherwise, and `b` ranges
  over `coordinates.boundaries`. Note that this integral is approximated
  differently to `log_sigma_integral`. This reflects the approach described in

    Durran, Dale. "Numerical Methods for Fluid Dynamics", 2010. § 8.6.3

  Args:
    x: an array of values with `x.shape[axis] == coordinates.layers`. These
      values are "located" at `coordinates` layer centers.
    coordinates: a `SigmaCoordinates` object describing the vertical coordinates
      to differentiate against. Must satisfy `coordinates.layers ==
      x.shape[axis]`.
    axis: the axis along which to approximate the derivative. Defaults to -3
      since this is the axis we typically use to index layers.
    downward: the direction of the integral. If True, the integral is taken
      "down" from the top of the atmosphere (𝜎 = 1). If False, it is taken "up"
      from the ground (𝜎 = 1).
    cumsum_method: method to use for calculating cumsum.
    sharding: optional parallel sharding for x.

  Returns:
    An array with the same shape as `x` containing approximate values of the
    integral of `x` from the 𝜎 = 0 to each layer's {lower, upper} boundary when
    `downward` is set to {True, False}.
  """
  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )
  x_axes = range(x.ndim)
  d𝜎 = coordinates.layer_thickness
  d𝜎_axes = [x_axes[axis]]
  xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
  if downward:
    return jax_numpy_utils.cumsum(
        xd𝜎, axis, method=cumsum_method, sharding=sharding
    )
  else:
    return jax_numpy_utils.reverse_cumsum(
        xd𝜎, axis, method=cumsum_method, sharding=sharding
    )


@jax.named_call
def sigma_integral(
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    keepdims: bool = True,
) -> jax.Array:
  """Calculate a full integral of a quantity `x` with respect to 𝜎."""
  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )
  x_axes = range(x.ndim)
  d𝜎 = coordinates.layer_thickness
  d𝜎_axes = [x_axes[axis]]
  xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
  return xd𝜎.sum(axis=axis, keepdims=keepdims)


@jax.named_call
def cumulative_log_sigma_integral(
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    downward: bool = True,
    cumsum_method: str = 'dot',
) -> jax.Array:
  """Approximates the integral of a quantity `x` with respect to log(𝜎).

  Uses the trapezoid rule (https://en.wikipedia.org/wiki/Trapezoidal_rule) to
  approximate the integral

    ∫x d(log(𝜎))

  where 𝜎= 0...c if `downward` is True, and 𝜎= 1...c otherwise, where `c` ranges
  over `coordinates.centers`. Note that this integral is approximated
  differently to `sigma_integral`. This reflects the approach described in

    Durran, Dale. "Numerical Methods for Fluid Dynamics", 2010. § 8.6.3

  Args:
    x: an array of values with `x.shape[axis] == coordinates.layers`. These
      values are "located" at `coordinates` layer centers.
    coordinates: a `SigmaCoordinates` object describing the vertical coordinates
      to differentiate against. Must satisfy `coordinates.layers ==
      x.shape[axis]`.
    axis: the axis along which to approximate the derivative. Defaults to -3
      since this is the axis we typically use to index layers.
    downward: the direction of the integral. If True, the integral is taken
      "down" from the top of the atmosphere (𝜎 = 1). If False, it is taken "up"
      from the ground (𝜎 = 1).
    cumsum_method: method to use for calculating cumsum.

  Returns:
    An array with the same shape as `x` containing approximate values of the
    integral of `x` from the "surface" (𝜎 = 1) up to each layer center.
  """
  if coordinates.layers != x.shape[axis]:
    raise ValueError(
        '`x.shape[axis]` must be equal to `coordinates.layers`;'
        f'got {x.shape[axis]} and {coordinates.layers}.'
    )
  # To integrate using the trapezoid rule, we linearly interpolate values of
  # `x`. The exception is between the surface (𝜎 = 1) and the center of the
  # first layer. In this interval, we assume a constant value of `x[-1]`.
  x_last = lax.slice_in_dim(x, -1, None, axis=axis)
  x_interpolated = (
      lax.slice_in_dim(x, 1, None, axis=axis)
      + lax.slice_in_dim(x, 0, -1, axis=axis)
  ) / 2
  integrand = jnp.concatenate([x_interpolated, x_last], axis=axis)
  integrand_axes = range(integrand.ndim)
  log𝜎 = jnp.log(coordinates.centers)
  dlog𝜎 = jnp.diff(log𝜎, append=0)
  dlog𝜎_axes = [integrand_axes[axis]]
  xd𝜎 = einsum(integrand, integrand_axes, dlog𝜎, dlog𝜎_axes, integrand_axes)
  if downward:
    return jax_numpy_utils.cumsum(xd𝜎, axis, method=cumsum_method)
  else:
    return jax_numpy_utils.reverse_cumsum(xd𝜎, axis, method=cumsum_method)


@jax.named_call
def centered_vertical_advection(
    w: Array,
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
    w_boundary_values: tuple[Array, Array] | None = None,
    dx_dsigma_boundary_values: tuple[Array, Array] | None = None,
) -> jnp.ndarray:
  """Compute vertical advection using 2nd order finite differences.

  Computes `-(w * ∂x/∂𝜎)[n]` at `coordinates.centers` with averaging.

  This procedure computes the expression using the averaging approximation:

  -(w * ∂x/∂𝜎)[n] ≈ -0.5 * (w[n+0.5] * ∂x/∂𝜎[n+0.5] + w[n-0.5] * ∂x/∂𝜎[n-0.5])

  To obtain values at all levels `w` and `∂x/∂𝜎` is padded with boundary values.
  By default boundary values for both terms are set to zero. Inputs are expected
  to be provided.

  Args:
    w: array of vertical velocities at `coordinates.internal_boundaries`.
    x: array of values at `coordinates.centers` that will be differentiated.
      Must be shape-compatible with `w` along axes other than `axis`.
    coordinates: object describing the vertical levels.
    axis: the axis corresponding to vertical (𝜎) values in `x` and `w`.
    w_boundary_values: optional top, bottom boundary values for `w` with shapes
      broacastable to `w` slices along `axis`.
    dx_dsigma_boundary_values: optional top, bottom boundary values for `∂x/∂𝜎`
      with shapes broacastable to `x` slices along `axis`.

  Returns:
    Values of `-(w * ∂x/∂𝜎)[n]` at level centers.
  """
  if w_boundary_values is None:
    w_slc_shape = _slice_shape_along_axis(w, axis)
    w_boundary_values = (
        jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
        jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
    )
  if dx_dsigma_boundary_values is None:
    x_slc_shape = _slice_shape_along_axis(x, axis)
    dx_dsigma_boundary_values = (
        jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
        jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
    )

  w_boundary_top, w_boundary_bot = w_boundary_values
  w = jnp.concatenate([w_boundary_top, w, w_boundary_bot], axis=axis)

  x_diff = centered_difference(x, coordinates, axis)
  x_diff_boundary_top, x_diff_boundary_bot = dx_dsigma_boundary_values
  x_diff = jnp.concatenate(
      [x_diff_boundary_top, x_diff, x_diff_boundary_bot], axis=axis
  )

  w_times_x_diff = w * x_diff
  return -0.5 * (
      lax.slice_in_dim(w_times_x_diff, 1, None, axis=axis)
      + lax.slice_in_dim(w_times_x_diff, 0, -1, axis=axis)
  )


@jax.named_call
def upwind_vertical_advection(
    w: Array,
    x: Array,
    coordinates: SigmaCoordinates,
    axis: int = -3,
) -> jnp.ndarray:
  """Compute vertical advection using 1st order upwinding."""
  w_slc_shape = _slice_shape_along_axis(w, axis)
  w_boundary_values = (
      jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
      jnp.zeros(w_slc_shape, dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
  )

  x_slc_shape = _slice_shape_along_axis(x, axis)
  dx_dsigma_boundary_values = (
      jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
      jnp.zeros(x_slc_shape, dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
  )

  # https://en.wikipedia.org/wiki/Upwind_scheme#Compact_form
  x_diff = centered_difference(x, coordinates, axis)

  w_boundary_top, w_boundary_bot = w_boundary_values
  w_up = jnp.concatenate([w_boundary_top, w], axis=axis)
  w_down = jnp.concatenate([w, w_boundary_bot], axis=axis)

  x_diff_boundary_top, x_diff_boundary_bot = dx_dsigma_boundary_values
  x_diff_up = jnp.concatenate([x_diff_boundary_top, x_diff], axis=axis)
  x_diff_down = jnp.concatenate([x_diff, x_diff_boundary_bot], axis=axis)
  # tendency (i.e. r.h.s. has a negative sign).
  return -(
      jnp.maximum(w_up, 0) * x_diff_up + jnp.minimum(w_down, 0) * x_diff_down
  )


# pylint: enable=invalid-name
