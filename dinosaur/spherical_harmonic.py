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

"""Spherical harmonics basis evaluation, and differential operators."""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, Callable

from dinosaur import associated_legendre
from dinosaur import fourier
from dinosaur import jax_numpy_utils
from dinosaur import pytree_utils
from dinosaur import typing
import jax
from jax import lax
from jax.experimental import shard_map
import jax.numpy as jnp
import numpy as np


Array = typing.Array
ArrayOrArrayTuple = typing.ArrayOrArrayTuple


# All `einsum`s should be done at highest available precision.
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


LATITUDE_SPACINGS = dict(
    gauss=associated_legendre.gauss_legendre_nodes,
    equiangular=associated_legendre.equiangular_nodes,
    equiangular_with_poles=associated_legendre.equiangular_nodes_with_poles,
)


def get_latitude_nodes(n: int, spacing: str) -> tuple[np.ndarray, np.ndarray]:
  """Computes latitude nodes using the given spacing."""
  get_nodes = LATITUDE_SPACINGS.get(spacing)
  if get_nodes is None:
    raise ValueError(
        f'Unknown spacing: {spacing}'
        f'available spacings are {list(LATITUDE_SPACINGS.keys())}'
    )
  return get_nodes(n)


@dataclasses.dataclass
class _SphericalHarmonicBasis:
  """Data structure representing a basis for spherical harmonics.

  Attributes:
    f: Fourier matrix.
    p: Legendre transform coefficients.
    w: nodal quadrature weights.
  """

  f: np.ndarray
  p: np.ndarray
  w: np.ndarray


@dataclasses.dataclass(frozen=True)
class SphericalHarmonics:
  """Base class for spherical harmonics implementations.

  Attributes:
    longitude_wavenumbers: the maximum (exclusive) wavenumber in the
      longitudinal direction. Indexes along longitudinal wavenumber are
      typically denoted by `m`.
    total_wavenumbers: the maximum (exclusive) sum of the latitudinal and
      longitudinal wavenumbers. Indices along total wavenumber are typically
      denoted by `l`.
    longitude_nodes: the number of nodes in the longitudinal direction. The
      selected nodes will be the equally spaced points in [0, 2π).
    latitude_nodes: the number of nodes in the latitudinal direction. The
      selected nodes will be the Gauss-Legendre quadrature points.
    latitude_spacing: a string indicating the spacing of latitude nodes. If
      'gauss' is passed, then Gauss-Legendre nodes are used. If 'equiangular' or
      'equiangular_with_poles' is passed, then the nodes are equally spaced in
      latitude (without or with points at the poles, respectively).
  """

  longitude_wavenumbers: int = 0
  total_wavenumbers: int = 0
  longitude_nodes: int = 0
  latitude_nodes: int = 0
  latitude_spacing: str = 'gauss'

  @property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and sin(latitude) coordinates of the nodal basis."""
    raise NotImplementedError

  @property
  def nodal_shape(self) -> tuple[int, int]:
    """Shape in the nodal basis."""
    raise NotImplementedError

  @property
  def nodal_padding(self) -> tuple[int, int]:
    """Padding in the nodal basis."""
    raise NotImplementedError

  @property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    """Longitudinal and total wavenumbers (m, l) of the modal basis."""
    raise NotImplementedError

  @property
  def modal_shape(self) -> tuple[int, int]:
    """Shape in the modal basis."""
    raise NotImplementedError

  @property
  def modal_padding(self) -> tuple[int, int]:
    """Padding in the modal basis."""
    raise NotImplementedError

  @property
  def modal_dtype(self) -> np.dtype:
    """Dtype in the modal state."""
    raise NotImplementedError

  @property
  def mask(self) -> np.ndarray:
    """Mask of valid values in modal representation."""
    raise NotImplementedError

  @property
  def basis(self) -> _SphericalHarmonicBasis:
    """Basis functions for these spherical harmonics."""
    raise NotImplementedError

  def inverse_transform(self, x: jax.Array) -> jax.Array:
    """Maps `x` from a modal to nodal representation."""
    raise NotImplementedError

  def transform(self, x: jax.Array) -> jax.Array:
    """Maps `x` from a nodal to modal representation."""
    raise NotImplementedError

  def longitudinal_derivative(self, x: Array) -> Array:
    """Computes `∂x/∂λ` in the modal basis, where λ denotes longitude."""
    raise NotImplementedError


class RealSphericalHarmonics(SphericalHarmonics):
  """Pedagogical implementation of spherical harmonics transforms.

  This transform represents spherical harmonic (modal) coefficients as a two
  dimensional grid of longtitudinal wavenumber (m) and total wavenumber (l)
  values:
      m = [0, +1, -1, +2, -2, ..., +M, -M]
      l = [0, 1, 2, ..., L]
  where `M = longitude_wavenumbers - 1` and `L = total_wavenumbers`.

  Entries with `abs(m) > l` are structural zeros,

  For better performance when using computing forward and inverse transforms,
  but no guaranteed stable representation, use FastSphericalHarmonics, which
  also supports parallelism.
  """

  @functools.cached_property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
    sin_latitude, _ = get_latitude_nodes(
        self.latitude_nodes, self.latitude_spacing
    )
    return longitude, sin_latitude

  @functools.cached_property
  def nodal_shape(self) -> tuple[int, int]:
    return (self.longitude_nodes, self.latitude_nodes)

  @functools.cached_property
  def nodal_padding(self) -> tuple[int, int]:
    return (0, 0)

  @functools.cached_property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    m_pos = np.arange(1, self.longitude_wavenumbers)
    m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
    lon_wavenumbers = np.concatenate([[0], m_pos_neg])  # [0, 1, -1, 2, -2, ...]
    tot_wavenumbers = np.arange(self.total_wavenumbers)
    return lon_wavenumbers, tot_wavenumbers

  @functools.cached_property
  def modal_shape(self) -> tuple[int, int]:
    return (2 * self.longitude_wavenumbers - 1, self.total_wavenumbers)

  @functools.cached_property
  def modal_padding(self) -> tuple[int, int]:
    return (0, 0)

  @functools.cached_property
  def modal_dtype(self) -> np.dtype:
    return np.dtype(np.float32)

  @functools.cached_property
  def mask(self) -> np.ndarray:
    m, l = np.meshgrid(*self.modal_axes, indexing='ij')
    return abs(m) <= l

  @functools.cached_property
  def basis(self) -> _SphericalHarmonicBasis:
    # The product of the arrays `f` and `p` gives the real normalized spherical
    # harmonic basis evaluated on a grid of longitudes λ and latitudes θ:
    #
    #   f[i, 0]      p[0     , j, l] = cₗ₀ P⁰ₗ(sin θⱼ)
    #   f[i, 2m - 1] p[2m - 1, j, l] = cₗₘ cos(m λᵢ) Pᵐₗ(sin θⱼ)
    #   f[i, 2m]     p[2m,     j, l] = cₗₘ sin(m λᵢ) Pᵐₗ(sin θⱼ)
    #
    # where the constants cₗₘ are chosen such that each function has unit L²
    # norm on the unit sphere. The longitudes λᵢ are `longitude_nodes` equally
    # spaced points in [0, 2π). The latitude nodes θⱼ are chosen such that
    # (sin θⱼ) are the Gauss-Legendre quadrature points if
    # `latitude_spacing = 'gauss'`, or θⱼ are `latitude_nodes` equally spaced
    # points if `latitude_spacing = 'equiangular'` (or
    # `'equiangular_with_poles'` for equally spaced points including points at
    # the poles).
    #
    # The shapes of the returned arrays are
    #
    #   f.shape == [longitude_nodes, (2 * longitude_wavenumbers - 1)]
    #   p.shape == [2 * longitude_wavenumbers - 1,
    #               latitude_nodes,
    #               total_wavenumbers]
    f = fourier.real_basis(
        wavenumbers=self.longitude_wavenumbers,
        nodes=self.longitude_nodes,
    )
    _, wf = fourier.quadrature_nodes(self.longitude_nodes)
    x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
    w = wf * wp
    p = associated_legendre.evaluate(
        n_m=self.longitude_wavenumbers, n_l=self.total_wavenumbers, x=x
    )
    # Each associated Legendre polynomial Pᵐₗ with m > 0 is paired with both
    # the sin and cos components of the Fourier basis. As a result, we have to
    # duplicate the rows of the associated Legendre matrix.
    p = np.repeat(p, 2, axis=0)
    # When m = 0, the associated Legendre polynomial is paired only with the
    # constant component of the Fourier matrix, so we only need one copy.
    p = p[1:]
    return _SphericalHarmonicBasis(f=f, p=p, w=w)

  def inverse_transform(self, x):
    p = self.basis.p
    f = self.basis.f
    px = jax.named_call(einsum, name='inv_legendre')('mjl,...ml->...mj', p, x)
    # note: explicitly matrix multiplication seems to be faster than using an
    # explicit FFT at the resolutions we use.
    fpx = jax.named_call(einsum, name='inv_fourier')('im,...mj->...ij', f, px)
    return fpx

  def transform(self, x):
    w = self.basis.w
    f = self.basis.f
    p = self.basis.p
    wx = w * x
    fwx = jax.named_call(einsum, name='fwd_fourier')('im,...ij->...mj', f, wx)
    pfwx = jax.named_call(einsum, name='fwd_legendre')(
        'mjl,...mj->...ml', p, fwx
    )
    return pfwx

  def longitudinal_derivative(self, x: Array) -> Array:
    return fourier.real_basis_derivative(x, axis=-2)


def _round_to_multiple(x: int, multiple: int) -> int:
  return multiple * math.ceil(x / multiple)


P = jax.sharding.PartitionSpec
shmap = shard_map.shard_map


@jax.named_call
def _unstack_m(x: jax.Array, /, mesh: jax.sharding.Mesh | None) -> jax.Array:
  """Unstack positive and negative values of `m` along a separate dimension."""

  def unstack(x):
    shape = x.shape[:-2] + (2, x.shape[-2] // 2) + x.shape[-1:]
    return jnp.reshape(x, shape, order='F')

  if mesh is None:
    return unstack(x)

  assert x.ndim in {2, 3}, x.shape
  z = None if x.shape[0] == 1 else 'z'
  in_spec = P(z, 'x', 'y') if x.ndim == 3 else P('x', 'y')
  out_spec = P(z, None, 'x', 'y') if x.ndim == 3 else P(None, 'x', 'y')
  return shmap(unstack, mesh, (in_spec,), out_spec)(x)


@jax.named_call
def _stack_m(x: jax.Array, /, mesh: jax.sharding.Mesh | None) -> jax.Array:
  """Stack a separate "sign" dimension into single dimension for `m`."""

  def stack(x):
    shape = x.shape[:-3] + (-1,) + x.shape[-1:]
    return jnp.reshape(x, shape, order='F')

  if mesh is None:
    return stack(x)

  assert x.ndim in {3, 4}, x.shape
  z = None if x.shape[0] == 1 else 'z'
  in_spec = P(z, None, 'x', 'y') if x.ndim == 4 else P(None, 'x', 'y')
  out_spec = P(z, 'x', 'y') if x.ndim == 4 else P('x', 'y')
  return shmap(stack, mesh, (in_spec,), out_spec)(x)


@jax.named_call
def _fourier_derivative_for_real_basis_with_zero_imag(
    x: jax.Array, /, mesh: jax.sharding.Mesh | None
) -> jax.Array:
  """Calculate a Fourier basis derivative."""

  if mesh is None:
    return fourier.real_basis_derivative_with_zero_imag(x, axis=-2)

  # FastHarmonicsWithZeroImage always pads longitudinal frequencies by a
  # multiple of two times the number of X shards, so we can safely differentiate
  # without any distributed communication.

  def differentiate(u):
    axis = -2
    frequency_offset = u.shape[axis] // 2 * lax.axis_index('x')
    return fourier.real_basis_derivative_with_zero_imag(
        u, axis, frequency_offset
    )

  assert x.ndim in {2, 3}, x.shape
  z = None if x.shape[0] == 1 else 'z'
  spec = P(z, 'x', 'y') if x.ndim == 3 else P('x', 'y')
  # TODO(shoyer): understand why this bogus check_rep=False is necessary to
  # avoid crashing
  return shmap(differentiate, mesh, (spec,), spec, check_rep=False)(x)


def _transform_einsum(
    subscripts: str,
    lhs: np.ndarray | jax.Array,
    rhs: jax.Array,
    mesh: jax.sharding.Mesh | None,
    reverse_einsum_arg_order: bool | None,
    precision: str,
) -> jax.Array:
  """einsum for calculating Fourier and Legendre transforms."""
  if mesh is None:
    return jnp.einsum(subscripts, lhs, rhs, precision=precision)

  out_ndim = len(
      jax.eval_shape(functools.partial(jnp.einsum, subscripts), lhs, rhs).shape
  )
  in_subscripts, _ = subscripts.split('->')
  _, rhs_subscripts = in_subscripts.split(',')

  if rhs.ndim == len(rhs_subscripts.replace('...', '')):
    # no ellipsis dimensions
    subscripts = subscripts.replace('...', '')
    in_spec = P(None, 'x', 'y') if rhs.ndim == 3 else P('x', 'y')
    out_spec = P(None, 'x', 'y') if out_ndim == 3 else P('x', 'y')
  elif rhs.ndim == len(rhs_subscripts.replace('...', 'z')):
    # one ellipsis dimension, for 'z'
    subscripts = subscripts.replace('...', 'z')
    z = None if rhs.shape[0] == 1 else 'z'
    in_spec = P(z, None, 'x', 'y') if rhs.ndim == 4 else P(z, 'x', 'y')
    out_spec = P(z, None, 'x', 'y') if out_ndim == 4 else P(z, 'x', 'y')
  else:
    raise ValueError(
        'only 0 or 1 dimensions supported for ... when using a mesh:'
        f' {subscripts}'
    )

  return jax_numpy_utils.sharded_einsum(
      subscripts,
      lhs,
      rhs,
      mesh=mesh,
      rhs_spec=in_spec,
      out_spec=out_spec,
      reverse_arg_order=bool(reverse_einsum_arg_order),
      precision=precision,
  )


@dataclasses.dataclass(frozen=True)
class FastSphericalHarmonics(SphericalHarmonics):
  """Fast implementation of spherical harmonic transformation.

  No stability guarantees are made about the shapes of arrays in the modal
  representation.

  Currently uses an extra imaginary term for m=-0. This can be more efficient
  because the array of Legendre transform coefficients is the same for positive
  and negative coefficients, so this halves the size of the `p` array on the
  MXU.

  This version of spherical harmonics also supports model parallelism, if
  `spmd_mesh` is provided. The additional optional arguments allow for low-level
  control of how the transforms are implemented.
  """

  spmd_mesh: jax.sharding.Mesh | None = None
  base_shape_multiple: int | None = None
  reverse_einsum_arg_order: bool | None = None
  stacked_fourier_transforms: bool | None = None
  transform_precision: str = 'tensorfloat32'

  def __post_init__(self):
    model_parallelism = self.spmd_mesh is not None and any(
        self.spmd_mesh.shape[dim] > 1 for dim in 'zxy'
    )

    if self.base_shape_multiple is None:
      shape_multiple = 8 if model_parallelism else 1
      object.__setattr__(self, 'base_shape_multiple', shape_multiple)

    if self.reverse_einsum_arg_order is None:
      object.__setattr__(self, 'reverse_einsum_arg_order', model_parallelism)

    if self.stacked_fourier_transforms is None:
      # it's faster to avoid explicitly stacking outputs from Fourier
      # transforms, but only if we don't have to do additional multiplications
      # on the MXU.
      unstacked_matmuls = math.ceil(self.longitude_wavenumbers / 128)
      stacked_matmuls = 2 * math.ceil(self.longitude_wavenumbers / 256)
      stack = stacked_matmuls <= unstacked_matmuls
      object.__setattr__(self, 'stacked_fourier_transforms', stack)

  @functools.cached_property
  def nodal_limits(self) -> tuple[int, int]:
    return (self.longitude_nodes, self.latitude_nodes)

  @functools.cached_property
  def modal_limits(self) -> tuple[int, int]:
    return (2 * self.longitude_wavenumbers, self.total_wavenumbers)

  def _mesh_shape(self) -> tuple[int, int]:
    if self.spmd_mesh is not None:
      return (self.spmd_mesh.shape['x'], self.spmd_mesh.shape['y'])
    else:
      return (1, 1)

  @functools.cached_property
  def nodal_shape(self) -> tuple[int, int]:
    base = self.base_shape_multiple or 1
    x_shards, y_shards = self._mesh_shape()
    shape_multiples = (base * x_shards, base * y_shards)
    return tuple(map(_round_to_multiple, self.nodal_limits, shape_multiples))

  @functools.cached_property
  def modal_shape(self) -> tuple[int, int]:
    base = self.base_shape_multiple or 1
    x_shards, y_shards = self._mesh_shape()
    # twice the padding for x to handle positive and negative m when unstacked
    shape_multiples = (2 * base * x_shards, base * y_shards)
    return tuple(map(_round_to_multiple, self.modal_limits, shape_multiples))

  @functools.cached_property
  def nodal_padding(self) -> tuple[int, int]:
    return tuple(x - y for x, y in zip(self.nodal_shape, self.nodal_limits))

  @functools.cached_property
  def modal_padding(self) -> tuple[int, int]:
    return tuple(x - y for x, y in zip(self.modal_shape, self.modal_limits))

  @functools.cached_property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    nodal_pad_x, nodal_pad_y = self.nodal_padding
    longitude, _ = fourier.quadrature_nodes(self.longitude_nodes)
    longitude = np.pad(longitude, [(0, nodal_pad_x)])
    sin_latitude, _ = get_latitude_nodes(
        self.latitude_nodes, self.latitude_spacing
    )
    sin_latitude = np.pad(sin_latitude, [(0, nodal_pad_y)])
    return longitude, sin_latitude

  @functools.cached_property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    modal_pad_x, modal_pad_y = self.modal_padding
    m_pos = np.arange(1, self.longitude_wavenumbers)
    m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
    lon_wavenumbers = np.pad(
        np.concatenate([[0, 0], m_pos_neg]), [(0, modal_pad_x)]
    )
    tot_wavenumbers = np.pad(
        np.arange(self.total_wavenumbers), [(0, modal_pad_y)]
    )
    return lon_wavenumbers, tot_wavenumbers

  @functools.cached_property
  def modal_dtype(self) -> np.dtype:
    return np.dtype(np.float32)

  @functools.cached_property
  def mask(self) -> np.ndarray:
    m, l = np.meshgrid(*self.modal_axes, indexing='ij')
    i, j = np.meshgrid(*(np.arange(s) for s in self.modal_shape), indexing='ij')
    i_lim, j_lim = self.modal_limits
    return (abs(m) <= l) & (i != 1) & (i < i_lim) & (j < j_lim)

  @functools.cached_property
  def basis(self) -> _SphericalHarmonicBasis:
    # The product of the arrays `f` and `p` gives the real normalized spherical
    # harmonic basis evaluated on a grid of longitudes λ and latitudes θ:
    #
    #   f[i, 2m    ]  p[2m,     j, l] = cₗₘ cos(m λᵢ) Pᵐₗ(sin θⱼ)
    #   f[i, 2m + 1]  p[2m + 1, j, l] = cₗₘ sin(m λᵢ) Pᵐₗ(sin θⱼ)
    #
    # where the constants cₗₘ are chosen such that each function has unit L²
    # norm on the unit sphere. The longitudes λᵢ are `longitude_nodes` equally
    # spaced points in [0, 2π). The latitude nodes θⱼ are chosen such that
    # (sin θⱼ) are the Gauss-Legendre quadrature points if
    # `latitude_spacing = 'gauss'`, or θⱼ are `latitude_nodes` equally spaced
    # points if `latitude_spacing = 'equiangular'` (or
    # `'equiangular_with_poles'` for equally spaced points including points at
    # the poles).
    #
    # The shapes of the returned arrays are
    #
    #   f.shape == (longitude_nodes, 2*longitude_wavenumbers)
    #   p.shape == (2*longitude_wavenumbers, latitude_nodes, total_wavenumbers)
    nodal_pad_x, nodal_pad_y = self.nodal_padding
    modal_pad_x, modal_pad_y = self.modal_padding

    f = fourier.real_basis_with_zero_imag(
        wavenumbers=self.longitude_wavenumbers,
        nodes=self.longitude_nodes,
    )
    f = np.pad(f, [(0, nodal_pad_x), (0, modal_pad_x)])
    if self.stacked_fourier_transforms:
      f = np.reshape(f, (-1, 2, f.shape[-1] // 2), order='F')

    _, wf = fourier.quadrature_nodes(self.longitude_nodes)
    x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
    w = wf * wp
    w = np.pad(w, [(0, nodal_pad_y)])

    p = associated_legendre.evaluate(
        n_m=self.longitude_wavenumbers, n_l=self.total_wavenumbers, x=x
    )
    p = np.pad(p, [(0, modal_pad_x // 2), (0, nodal_pad_y), (0, modal_pad_y)])

    return _SphericalHarmonicBasis(f=f, p=p, w=w)

  def inverse_transform(self, x):
    p = self.basis.p
    f = self.basis.f
    mesh = self.spmd_mesh
    einsum_args = (self.reverse_einsum_arg_order, self.transform_precision)

    # TODO(shoyer): consider supporting a "stacked" modal representation with
    # positive & negative values of `m` separated. This would allow for omitting
    # this call to _unstack_m().
    x = _unstack_m(x, mesh)
    x = jax.named_call(_transform_einsum, name='inv_legendre')(
        'mjl,...sml->...smj', p, x, mesh, *einsum_args
    )
    if self.stacked_fourier_transforms:
      # note: explicit matrix multiplication seems to be faster than using an
      # explicit FFT at the resolutions we use.
      x = jax.named_call(_transform_einsum, name='inv_fourier')(
          'ism,...smj->...ij', f, x, mesh, *einsum_args
      )
    else:
      x = _stack_m(x, mesh)
      x = jax.named_call(_transform_einsum, name='inv_fourier')(
          'im,...mj->...ij', f, x, mesh, *einsum_args
      )
    return x

  def transform(self, x):
    w = self.basis.w
    f = self.basis.f
    p = self.basis.p
    mesh = self.spmd_mesh
    einsum_args = (self.reverse_einsum_arg_order, self.transform_precision)

    x = w * x
    if self.stacked_fourier_transforms:
      x = jax.named_call(_transform_einsum, name='fwd_fourier')(
          'ism,...ij->...smj', f, x, mesh, *einsum_args
      )
    else:
      x = jax.named_call(_transform_einsum, name='fwd_fourier')(
          'im,...ij->...mj', f, x, mesh, *einsum_args
      )
      x = _unstack_m(x, mesh)
    x = jax.named_call(_transform_einsum, name='fwd_legendre')(
        'mjl,...smj->...sml', p, x, mesh, *einsum_args
    )
    x = _stack_m(x, mesh)
    return x

  def longitudinal_derivative(self, x: Array) -> Array:
    return _fourier_derivative_for_real_basis_with_zero_imag(x, self.spmd_mesh)


@dataclasses.dataclass(frozen=True)
class RealSphericalHarmonicsWithZeroImag(FastSphericalHarmonics):
  """Deprecated alias for `FastSphericalHarmonics`."""


def _vertical_pad(
    field: jax.Array, mesh: jax.sharding.Mesh | None
) -> tuple[jax.Array, int | None]:
  if field.ndim < 3 or field.shape[0] == 1 or mesh is None:
    return field, None
  assert field.ndim == 3, field.shape
  z_multiple = mesh.shape['z']
  z_padding = _round_to_multiple(field.shape[0], z_multiple) - field.shape[0]
  return jnp.pad(field, [(0, z_padding), (0, 0), (0, 0)]), z_padding


def _vertical_crop(field: jax.Array, padding: int | None) -> jax.Array:
  if not padding:
    return field
  assert field.ndim == 3, field.shape
  return jax.lax.slice_in_dim(field, 0, -padding, axis=0)


def _with_vertical_padding(
    f: Callable[[jax.Array], jax.Array], mesh: jax.sharding.Mesh | None
) -> Callable[[jax.Array], jax.Array]:
  """Apply a function with vertical padding on a mesh.

  This is useful for implementing sharded Grid operations even the case where
  the z dimension has some irregular size (e.g., the 37 pressure levels for
  ERA5).

  Args:
    f: function to apply on padded data.
    mesh: SPMD mesh.

  Returns:
    Function that can be applied to non-padded arrays.
  """

  def g(x):
    x, padding = _vertical_pad(x, mesh)
    return _vertical_crop(f(x), padding)

  return g


SPHERICAL_HARMONICS_IMPL_KEY = 'spherical_harmonics_impl'
SPMD_MESH_KEY = 'spmd_mesh'


SphericalHarmonicsImpl = Callable[..., SphericalHarmonics]


@dataclasses.dataclass(frozen=True)
class Grid:
  """A class that represents real-space and spectral grids over the sphere.

  The number of wavenumbers and nodes is entirely flexible, although in practice
  one should use one of the established conventions used by the constructors
  below. Typically both wavenumbers and nodes should be specified, unless you
  only need operations in real or spectral space.

  Attributes:
    longitude_wavenumbers: the maximum (exclusive) wavenumber in the
      longitudinal direction. Indexes along longitudinal wavenumber are
      typically denoted by `m`. Must satisfy `longitude_wavenumbers <=
      total_wavenumbers`.
    total_wavenumbers: the maximum (exclusive) sum of the latitudinal and
      longitudinal wavenumbers. Indices along total wavenumber are typically
      denoted by `l`. Must satisfy `longitude_wavenumbers <= total_wavenumbers`.
    longitude_nodes: the number of nodes in the longitudinal direction. The
      selected nodes will be the equally spaced points in [0, 2π) incremented by
      longitude_offset.
    latitude_nodes: the number of nodes in the latitudinal direction. The
      selected nodes will be the Gauss-Legendre quadrature points.
    latitude_spacing: a string indicating the spacing of latitude nodes. If
      'gauss' is passed, then Gauss-Legendre nodes are used. If 'equiangular' or
      'equiangular_with_poles' is passed, then the nodes are equally spaced in
      latitude (without or with points at the poles, respectively).
    longitude_offset: the value of the first longitude node, in radians.
    radius: radius of the sphere. If `None` a default value of `1` is used.
    spherical_harmonics_impl: class providing an implementation of spherical
      harmonics.
    spmd_mesh: mesh to use for parallelism in the single program multiple device
      (SPMD) paradigm with distributed JAX arrays, if any. Required if using
      model parallelism.
  """

  longitude_wavenumbers: int = 0
  total_wavenumbers: int = 0
  longitude_nodes: int = 0
  latitude_nodes: int = 0
  latitude_spacing: str = 'gauss'
  longitude_offset: float = 0.0
  radius: float | None = None
  spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics
  spmd_mesh: jax.sharding.Mesh | None = None

  def __post_init__(self):
    if self.radius is None:
      # TODO(shoyer): consider updating this to scales.RADIUS instead
      object.__setattr__(self, 'radius', 1.0)

    if self.latitude_spacing not in LATITUDE_SPACINGS:
      raise ValueError(
          f'Unsupported `latitude_spacing` "{self.latitude_spacing}". '
          f'Supported values are: {list(LATITUDE_SPACINGS)}.'
      )

    if self.spmd_mesh is not None:
      if not {'x', 'y'} <= set(self.spmd_mesh.axis_names):
        raise ValueError(
            "mesh is missing one or more of the required axis names 'x' and "
            f"'y': {self.spmd_mesh}"
        )
      assert isinstance(self.spherical_harmonics, FastSphericalHarmonics)

  @classmethod
  def with_wavenumbers(
      cls,
      longitude_wavenumbers: int,
      dealiasing: str = 'quadratic',
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics,
      radius: float | None = None,
  ) -> Grid:
    """Construct a `Grid` by specifying only wavenumbers."""

    # The number of nodes is chosen for de-aliasing.
    order = {'linear': 2, 'quadratic': 3, 'cubic': 4}[dealiasing]
    longitude_nodes = order * longitude_wavenumbers + 1
    latitude_nodes = math.ceil(longitude_nodes / 2)

    return cls(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=longitude_wavenumbers + 1,
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
    )

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      gaussian_nodes: int,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      radius: float | None = None,
      spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics,
  ) -> Grid:
    """Construct a `Grid` by specifying max wavenumber & the number of nodes.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
      gaussian_nodes: number of nodes on the Gaussian grid between the equator
        and a pole.
      latitude_spacing: either 'gauss' or 'equiangular'. This determines the
        spacing of nodal grid points in the latitudinal (north-south) direction.
      longitude_offset: the value of the first longitude node, in radians.
      radius: radius of the sphere. If `None` a default values of `1` is used.
      spherical_harmonics_impl: class providing an implementation of spherical
        harmonics.

    Returns:
      Constructed Grid object.
    """

    return cls(
        longitude_wavenumbers=max_wavenumber + 1,
        total_wavenumbers=max_wavenumber + 2,
        longitude_nodes=4 * gaussian_nodes,
        latitude_nodes=2 * gaussian_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
    )

  # The factory methods below return "standard" grids that appear in the
  # literature. See, e.g. https://doi.org/10.5194/tc-12-1499-2018 and
  # https://www.ecmwf.int/en/forecasts/documentation-and-support/data-spatial-coordinate-systems

  # The number in these names correspond to the maximum resolved wavenumber,
  # which is one less than the number of wavenumbers used in the Grid
  # constructor. An additional total wavenumber is added because the top
  # wavenumber is clipped from the initial state and each calculation of
  # explicit tendencies.

  # The names for these factory methods (including capilatization) are
  # standard in the literature.
  # pylint:disable=invalid-name

  # T* grids can model quadratic terms without aliasing, because the maximum
  # total wavenumber is <= 2/3 of the number of latitudinal nodes. ECMWF
  # sometimes calls these "TQ" (truncated quadratic) grids.

  @classmethod
  def T21(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=21, gaussian_nodes=16, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=24, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=42, gaussian_nodes=32, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=85, gaussian_nodes=64, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=106, gaussian_nodes=80, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=119, gaussian_nodes=90, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=170, gaussian_nodes=128, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=213, gaussian_nodes=160, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=340, gaussian_nodes=256, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=425, gaussian_nodes=320, **kwargs)

  # TL* grids do not truncate any frequencies, and hence can only model linear
  # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
  # advection (which eliminates quadratic terms) up to 2016, which it switched
  # to "cubic" grids for resolutions above TL1279:
  # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

  @classmethod
  def TL31(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=16, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=47, gaussian_nodes=24, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=63, gaussian_nodes=32, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=95, gaussian_nodes=48, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=127, gaussian_nodes=64, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=159, gaussian_nodes=80, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=179, gaussian_nodes=90, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=255, gaussian_nodes=128, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=639, gaussian_nodes=320, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> Grid:
    return cls.construct(max_wavenumber=1279, gaussian_nodes=640, **kwargs)

  # pylint:enable=invalid-name

  def asdict(self) -> dict[str, Any]:
    """Returns grid attributes as a dictionary."""
    items = dataclasses.asdict(self)
    items[SPHERICAL_HARMONICS_IMPL_KEY] = self.spherical_harmonics_impl.__name__  # pylint:disable=attribute-error
    if self.spmd_mesh is not None:
      items[SPMD_MESH_KEY] = ','.join(
          f'{k}={v}' for k, v in self.spmd_mesh.shape.items()
      )
    else:
      items[SPMD_MESH_KEY] = ''
    return items

  # pylint:disable=g-missing-from-attributes

  @functools.cached_property
  def spherical_harmonics(self) -> SphericalHarmonics:
    """Implementation of spherical harmonic transformations."""
    if self.spmd_mesh is not None:
      kwargs = dict(spmd_mesh=self.spmd_mesh)
    else:
      kwargs = dict()
    return self.spherical_harmonics_impl(
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
        **kwargs,
    )

  @property
  def longitudes(self) -> np.ndarray:
    return self.nodal_axes[0]

  @property
  def latitudes(self) -> np.ndarray:
    return np.arcsin(self.nodal_axes[1])

  @functools.cached_property
  def nodal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and sin(latitude) coordinates of the nodal basis."""
    lon, sin_lat = self.spherical_harmonics.nodal_axes
    return lon + self.longitude_offset, sin_lat

  @functools.cached_property
  def nodal_shape(self) -> tuple[int, int]:
    return self.spherical_harmonics.nodal_shape

  @functools.cached_property
  def nodal_padding(self) -> tuple[int, int]:
    return self.spherical_harmonics.nodal_padding

  @functools.cached_property
  def nodal_mesh(self) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(*self.nodal_axes, indexing='ij')

  @functools.cached_property
  def modal_axes(self) -> tuple[np.ndarray, np.ndarray]:
    """Longitudinal and total wavenumbers (m, l) of the modal basis."""
    return self.spherical_harmonics.modal_axes

  @functools.cached_property
  def modal_shape(self) -> tuple[int, int]:
    return self.spherical_harmonics.modal_shape

  @functools.cached_property
  def modal_padding(self) -> tuple[int, int]:
    return self.spherical_harmonics.modal_padding

  @functools.cached_property
  def mask(self) -> np.ndarray:
    """Modal mask."""
    return self.spherical_harmonics.mask

  @functools.cached_property
  def modal_mesh(self) -> tuple[np.ndarray, np.ndarray]:
    """Mesh of longitudinal and total wavenumbers (m, l) for the modal basis."""
    return np.meshgrid(*self.spherical_harmonics.modal_axes, indexing='ij')

  @functools.cached_property
  def cos_lat(self) -> jnp.ndarray:
    _, sin_lat = self.nodal_axes
    return np.sqrt(1 - sin_lat**2)

  @functools.cached_property
  def sec2_lat(self) -> jnp.ndarray:
    _, sin_lat = self.nodal_axes
    return 1 / (1 - sin_lat**2)  # pytype: disable=bad-return-type  # jnp-array

  @functools.cached_property
  def laplacian_eigenvalues(self) -> np.ndarray:
    _, l = self.modal_axes
    return -l * (l + 1) / (self.radius**2)

  # pylint:enable=g-missing-from-attributes

  @jax.named_call
  def to_nodal(self, x: typing.Pytree) -> typing.Pytree:
    """Maps `x` from a modal to nodal representation."""
    f = _with_vertical_padding(
        self.spherical_harmonics.inverse_transform, self.spmd_mesh
    )
    return pytree_utils.tree_map_over_nonscalars(f, x)

  @jax.named_call
  def to_modal(self, z: typing.Pytree) -> typing.Pytree:
    """Maps `x` from a nodal to modal representation."""
    f = _with_vertical_padding(
        self.spherical_harmonics.transform, self.spmd_mesh
    )
    return pytree_utils.tree_map_over_nonscalars(f, z)

  @jax.named_call
  def laplacian(self, x: Array) -> jnp.ndarray:
    """Computes `∇²(x)` in the spectral basis."""
    return x * self.laplacian_eigenvalues

  @jax.named_call
  def inverse_laplacian(self, x: Array) -> jnp.ndarray:
    """Computes `(∇²)⁻¹(x)` in the spectral basis."""
    with np.errstate(divide='ignore', invalid='ignore'):
      inverse_eigenvalues = 1 / self.laplacian_eigenvalues
    inverse_eigenvalues[0] = 0
    inverse_eigenvalues[self.total_wavenumbers :] = 0
    assert not np.isnan(inverse_eigenvalues).any()
    return x * inverse_eigenvalues

  @jax.named_call
  def clip_wavenumbers(self, x: typing.Pytree, n: int = 1) -> typing.Pytree:
    """Zeros out the highest `n` total wavenumbers."""
    if n <= 0:
      raise ValueError(f'`n` must be >= 0; got {n}.')

    def clip(x):
      # Multiplication by the mask is significantly faster than directly using
      # `x.at[..., -n:].set(0)`
      num_zeros = n + self.modal_padding[-1]
      mask = jnp.ones(self.modal_shape[-1], x.dtype).at[-num_zeros:].set(0)
      return x * mask

    return pytree_utils.tree_map_over_nonscalars(clip, x)

  @functools.cached_property
  def _derivative_recurrence_weights(self) -> tuple[np.ndarray, np.ndarray]:
    m, l = self.modal_mesh
    a = np.sqrt(self.mask * (l**2 - m**2) / (4 * l**2 - 1))
    a[:, 0] = 0
    b = np.sqrt(self.mask * ((l + 1) ** 2 - m**2) / (4 * (l + 1) ** 2 - 1))
    b[:, -1] = 0
    return a, b

  @jax.named_call
  def d_dlon(self, x: Array) -> Array:
    """Computes `∂x/∂λ` where λ denotes longitude."""
    return _with_vertical_padding(
        self.spherical_harmonics.longitudinal_derivative, self.spmd_mesh
    )(x)

  @jax.named_call
  def cos_lat_d_dlat(self, x: Array) -> Array:
    """Computes `cosθ ∂x/∂θ`, where θ denotes latitude.

    Args:
      x: input field in the spectral basis.

    Returns:
      `cosθ ∂x/∂θ` with potential numerical artifact in the highest wavenumber.
    """
    _, l = self.modal_mesh
    a, b = self._derivative_recurrence_weights
    x_lm1 = jax_numpy_utils.shift(((l + 1) * a) * x, -1, axis=-1)
    x_lp1 = jax_numpy_utils.shift((-l * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1

  @jax.named_call
  def sec_lat_d_dlat_cos2(self, x: Array) -> Array:
    """Computes `secθ ∂/∂θ(cos²θ x)`, where θ denotes latitude."""
    _, l = self.modal_mesh
    a, b = self._derivative_recurrence_weights
    x_lm1 = jax_numpy_utils.shift(((l - 1) * a) * x, -1, axis=-1)
    x_lp1 = jax_numpy_utils.shift((-(l + 2) * b) * x, +1, axis=-1)
    return x_lm1 + x_lp1

  @jax.named_call
  def cos_lat_grad(self, x: Array, clip: bool = True) -> Array:
    """Computes `cosθ ∇(x)` where θ denotes latitude."""
    # we clip the last wavenumber to remove numerical artifacts in d_dlat.
    raw = self.d_dlon(x) / self.radius, self.cos_lat_d_dlat(x) / self.radius
    if clip:
      return self.clip_wavenumbers(raw)
    return raw  # pytype: disable=bad-return-type  # jnp-array

  @jax.named_call
  def k_cross(self, v: ArrayOrArrayTuple) -> Array:
    """Computes `k ✕ v`,  where k is the normal unit vector."""
    return -v[1], v[0]  # pytype: disable=bad-return-type  # jnp-array

  @jax.named_call
  def div_cos_lat(
      self,
      v: ArrayOrArrayTuple,
      clip: bool = True,
  ) -> Array:
    """Computes `∇ · (v cosθ)` where θ denotes latitude."""
    raw = (self.d_dlon(v[0]) + self.sec_lat_d_dlat_cos2(v[1])) / self.radius
    if clip:
      return self.clip_wavenumbers(raw)
    return raw

  @jax.named_call
  def curl_cos_lat(
      self,
      v: ArrayOrArrayTuple,
      clip: bool = True,
  ) -> Array:
    """Computes `k · ∇ ✕ (v cosθ)` where θ denotes latitude."""
    raw = (self.d_dlon(v[1]) - self.sec_lat_d_dlat_cos2(v[0])) / self.radius
    if clip:
      return self.clip_wavenumbers(raw)
    return raw

  @property
  def quadrature_weights(self) -> np.ndarray:
    """Calculates quadrature weights in nodal space."""
    return np.broadcast_to(self.spherical_harmonics.basis.w, self.nodal_shape)

  @jax.named_call
  def integrate(self, z: Array) -> Array:
    """Approximates the integral of nodal values `z` over the sphere."""
    w = self.spherical_harmonics.basis.w * self.radius**2
    return einsum('y,...xy->...', w, z)


# In the spectral basis, a constant field of ones has this value in entry
# [0, 0]. This is a consequence of the way we normalize Legendre polynomials.
_CONSTANT_NORMALIZATION_FACTOR = 3.5449077


def add_constant(x: jnp.ndarray, c: float | Array) -> jnp.ndarray:
  """Adds the constant `c` to the array `x` in the spectral basis."""
  return x.at[..., 0, 0].add(_CONSTANT_NORMALIZATION_FACTOR * c)


@jax.named_call
def get_cos_lat_vector(
    vorticity: Array,
    divergence: Array,
    grid: Grid,
    clip: bool = True,
) -> Array:
  """Computes `v cosθ`, where θ denotes latitude."""
  stream_function = grid.inverse_laplacian(vorticity)
  velocity_potential = grid.inverse_laplacian(divergence)
  return jax.tree_util.tree_map(
      lambda x, y: x + y,
      grid.cos_lat_grad(velocity_potential, clip=clip),
      grid.k_cross(grid.cos_lat_grad(stream_function, clip=clip)),
  )


@functools.partial(jax.jit, static_argnames=('grid', 'clip'))
def uv_nodal_to_vor_div_modal(
    grid: Grid,
    u_nodal: Array,
    v_nodal: Array,
    clip: bool = True,
) -> tuple[Array, Array]:
  """Converts nodal `u, v` velocities to a modal `vort, div` representation."""
  u_over_cos_lat = grid.to_modal(u_nodal / grid.cos_lat)
  v_over_cos_lat = grid.to_modal(v_nodal / grid.cos_lat)

  vorticity = grid.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
  divergence = grid.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
  return vorticity, divergence


@functools.partial(jax.jit, static_argnames=('grid', 'clip'))
def vor_div_to_uv_nodal(
    grid: Grid,
    vorticity: Array,
    divergence: Array,
    clip: bool = True,
) -> tuple[Array, Array]:
  """Converts modal `vorticity, divergence` to a nodal `u, v` representation."""
  u_cos_lat, v_cos_lat = get_cos_lat_vector(
      vorticity, divergence, grid, clip=clip
  )
  u_nodal = grid.to_nodal(u_cos_lat) / grid.cos_lat
  v_nodal = grid.to_nodal(v_cos_lat) / grid.cos_lat
  return u_nodal, v_nodal
