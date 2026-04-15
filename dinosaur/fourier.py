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

"""Real Fourier basis evaluation, and derivative coefficients."""
from dinosaur import jax_numpy_utils

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def real_basis(wavenumbers: int, nodes: int) -> np.ndarray:
  """Returns the real-valued Fourier basis.

  Args:
    wavenumbers: number of wavenumbers.
    nodes: number of equally spaced nodes in the range [0, 2π). Must satisfy
      wavenumbers >= nodes.

  Returns:
    The nodes ✕ (2 * wavenumbers - 1) matrix F, such that

      F[i, 0] = 1 / √2π
      F[i, 2j - 1] = cos(j xᵢ) / √π    1 ≤ j < wavenumbers
      F[i, 2j]     = sin(j xᵢ) / √π    1 ≤ j < wavenumbers

    where x is a vector of m equally-spaced nodes so xᵢ = i * 2π / nodes.
    i.e., the columns of F are the real Fourier basis functions sampled at x.

    The normalization of the basis functions is chosen such that they have unit
    L²([0, 2π]) norm.
  """
  if nodes < wavenumbers:
    raise ValueError(
        '`real_basis` requires nodes >= wavenumbers; '
        f'got m = {nodes} and n = {wavenumbers}.'
    )

  dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)

  cos = np.real(dft[:, 1:])
  sin = -np.imag(dft[:, 1:])

  f = np.empty(shape=[nodes, 2 * wavenumbers - 1], dtype=np.float64)
  f[:, 0] = 1 / np.sqrt(2 * np.pi)
  f[:, 1::2] = cos
  f[:, 2::2] = sin
  return f


def real_basis_derivative(
    u: jnp.ndarray | jax.Array, /, axis: int = -1
) -> jax.Array:
  """Calculate the derivative of a signal using a real basis.

  Args:
    u: signal to differentiate, in the real Fourier basis.
    axis: the axis along which the transform will be applied.

  Returns:
    The derivative of `u` along `axis`. In particular, if
    `u_x = real_basis_derivative(u)`:

      u_x[..., 0]      = 0
      u_x[..., 2j - 1] =  j * u[..., 2j]      1 ≤ j < wavenumbers
      u_x[..., 2j]     = -j * u[..., 2j - 1]  1 ≤ j < wavenumbers
  """
  if u.shape[axis] % 2 != 1:
    raise ValueError(f'{u.shape=} along {axis=} is not odd')
  if axis >= 0:
    raise ValueError('axis must be negative')
  i = jnp.arange(u.shape[axis]).reshape((-1,) + (1,) * (-1 - axis))
  j = (i + 1) // 2
  u_down = jax_numpy_utils.shift(u, -1, axis)
  u_up = jax_numpy_utils.shift(u, +1, axis)
  return j * jnp.where(i % 2, u_down, -u_up)


def real_basis_with_zero_imag(wavenumbers: int, nodes: int) -> np.ndarray:
  """Real basis with a zero imaginary part."""
  if nodes < wavenumbers:
    raise ValueError(
        '`real_basis` requires nodes >= wavenumbers; '
        f'got m = {nodes} and n = {wavenumbers}.'
    )

  dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)

  cos = np.real(dft[:, 1:])
  sin = -np.imag(dft[:, 1:])

  f = np.empty(shape=[nodes, 2 * wavenumbers], dtype=np.float64)
  f[:, 0] = 1 / np.sqrt(2 * np.pi)
  f[:, 1] = 0
  f[:, 2::2] = cos
  f[:, 3::2] = sin
  return f


def real_basis_derivative_with_zero_imag(
    u: jnp.ndarray | jax.Array,
    axis: int = -1,
    frequency_offset: int | jax.Array = 0,
) -> jax.Array:
  """Calculate the derivative along a real basis with zero imaginary part."""
  if u.shape[axis] % 2:
    raise ValueError(f'{u.shape=} along {axis=} is not even')
  if axis >= 0:
    raise ValueError('axis must be negative')
  i = jnp.arange(u.shape[axis]).reshape((-1,) + (1,) * (-1 - axis))
  j = frequency_offset + i // 2
  u_down = jax_numpy_utils.shift(u, -1, axis)
  u_up = jax_numpy_utils.shift(u, +1, axis)
  return j * jnp.where((i + 1) % 2, u_down, -u_up)


def quadrature_nodes(nodes: int) -> tuple[np.ndarray, np.ndarray]:
  """Returns nodes and weights for the trapezoidal rule.

  Args:
    nodes: the number of nodes and weights to return.

  Returns:
    A pair (nodes, weights); values and weights to use for quadrature, with
    dtype `np.float64`.
  """
  xs = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
  weights = 2 * np.pi / nodes
  return xs, weights
