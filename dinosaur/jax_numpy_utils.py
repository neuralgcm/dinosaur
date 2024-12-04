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

"""jax.numpy functions that could belong upstream."""
from __future__ import annotations

import functools
import math
import re

import jax
from jax import lax
from jax.experimental import shard_map
import jax.numpy as jnp
import numpy as np


# pylint: disable=logging-fstring-interpolation


@jax.named_call
def _single_device_dot_cumsum(
    x: jax.Array, axis: int, reverse: bool = False
) -> jax.Array:
  """cumsum() implemented via matrix-multiplciation."""
  # This is much faster on TPUs than the default cumsum.
  if not -x.ndim <= axis < x.ndim:
    raise ValueError(f'invalid {axis=}')
  if axis < 0:
    axis = axis + x.ndim
  size = x.shape[axis]
  i = jnp.arange(size)[:, jnp.newaxis]
  j = jnp.arange(size)[jnp.newaxis, :]
  op = jnp.greater_equal if reverse else jnp.less_equal
  w = op(i, j).astype(np.float32)
  out_axes = list(range(x.ndim))
  out_axes[axis] = x.ndim
  return jnp.einsum(
      w,
      [axis, x.ndim],
      x,
      list(range(x.ndim)),
      out_axes,
      precision=('bfloat16', 'highest'),
  )


def _parallel_dot_cumsum(
    x: jax.Array, axis: int, reverse: bool, axis_name: str
) -> jax.Array:
  """Parallel implementation of dot cumsum using lax primitives."""
  partials = _single_device_dot_cumsum(x, axis=axis, reverse=reverse)
  last_partial = lax.index_in_dim(partials, 0 if reverse else -1, axis)
  sums = lax.all_gather(last_partial, axis_name, tiled=True)
  axis_index = lax.axis_index(axis_name)
  op = jnp.greater if reverse else jnp.less
  total = partials
  terms = sums[1:] if reverse else sums[:-1]
  start = 1 if reverse else 0
  for i, term in enumerate(terms, start=start):
    total += op(i, axis_index) * term
  return total


def _dot_cumsum(
    x: jax.Array,
    axis: int,
    sharding: jax.sharding.NamedSharding | None,
    reverse: bool = False,
) -> jax.Array:
  """Dot product based cumsum() with custom parallel implementation."""
  if sharding is None or sharding.spec[axis] is None:
    # axis being summed along is not sharded
    return _single_device_dot_cumsum(x, axis, reverse=reverse)

  mesh = sharding.mesh
  spec = sharding.spec

  @jax.jit
  @functools.partial(
      shard_map.shard_map,
      mesh=mesh,
      in_specs=(spec,),
      out_specs=spec,
      check_rep=False,
  )
  def dot_cumsum(x):
    return _parallel_dot_cumsum(
        x, axis=axis, reverse=reverse, axis_name=sharding.spec[axis]
    )

  return dot_cumsum(x)


def cumsum(
    x: np.ndarray | jax.Array,
    axis: int,
    method: str = 'dot',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Optimized version of cumsum for short axes on TPUs."""
  if method == 'dot':
    return _dot_cumsum(x, axis, sharding=sharding)
  elif method == 'jax':
    return jnp.cumsum(x, axis)
  else:
    raise ValueError(f'invalid {method=}')


def reverse_cumsum(
    x: np.ndarray | jax.Array,
    axis: int,
    method: str = 'dot',
    sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Performs cumsum in reverse order along `axis`."""
  if method == 'dot':
    return _dot_cumsum(x, axis, reverse=True, sharding=sharding)
  elif method == 'jax':
    return jnp.flip(jnp.cumsum(jnp.flip(x, axis), axis), axis)
  else:
    raise ValueError(f'invalid {method=}')


# XLA can generate rather inefficient code for Gather and DynamicSlice,
# especially on TPUs and when using automatic parallelization (because Gather
# can in principle be between arbitrary machines), batching (because the
# batching rule for DynamicSlice turns it into Gather) or reverse mode automatic
# differentiation (because the transpose of Gather is Scatter, which TPUs
# implement even less efficiently than Gather).
#
# To work around this issue, we need to reimplement jax.numpy operations
# (indexing regular array indexing) in terms of lax.slice and lax.pad, which are
# much easier for XLA to optimize.


def pad_in_dim(
    x: np.ndarray | jax.Array, pad_width: tuple[int, int], axis: int
) -> jax.Array:
  """Pad an array along an axis."""
  padding_value = jnp.array(0, dtype=x.dtype)
  padding_config = [(0, 0, 0)] * x.ndim
  padding_config[axis] = pad_width + (0,)  # add "interior" padding for lax.pad
  return lax.pad(x, padding_value, padding_config)


def shift(x: np.ndarray | jax.Array, offset: int, axis: int) -> jax.Array:
  """Like jnp.roll, but zero pads and implemented with different primitives."""
  if abs(offset) >= x.shape[axis]:
    return jnp.zeros_like(x)
  if offset > 0:
    sliced = lax.slice_in_dim(x, 0, x.shape[axis] - offset, axis=axis)
    return pad_in_dim(sliced, (offset, 0), axis=axis)
  else:
    sliced = lax.slice_in_dim(x, -offset, x.shape[axis], axis=axis)
    return pad_in_dim(sliced, (0, -offset), axis=axis)


def diff(x, axis=-1):
  """Like jnp.diff, but implemented in terms of simpler primitives."""
  upper = lax.slice_in_dim(x, 1, None, axis=axis)
  lower = lax.slice_in_dim(x, 0, -1, axis=axis)
  return upper - lower


def _reversed_arg_order_einsum(
    subscripts: str, x: jax.Array, y: jax.Array, **kwargs: ...
) -> jax.Array:
  # Surprisingly, XLA can generate code with up to ~10% speed differences
  # depending on the order of einsum arguments. There is no consistent pattern
  # about which is better.
  in_subscripts, out_subscripts = subscripts.split('->')
  lhs_subscripts, rhs_subscripts = in_subscripts.split(',')
  new_subscripts = f'{rhs_subscripts},{lhs_subscripts}->{out_subscripts}'
  return jnp.einsum(new_subscripts, y, x, **kwargs)


def _allgather_matmul_twoway(
    einsum_spec: str,
    lhs: jax.Array,
    rhs: jax.Array,
    split_axis: int,
    axis_name: str | tuple[str, str],
    reverse_arg_order: bool = False,
    precision: str = 'float32',
) -> jax.Array:
  """All-gather matmul using two way communication.

  This function is designed to be called from within `shmap`. It transfers
  inputs chunks using two way communication along a JAX mesh dimension, and then
  computes matrix-multiplication on each chunk.

  Args:
    einsum_spec: matmul specification.
    lhs: constant coefficients.
    rhs: sharded inputs.
    split_axis: the axis of `lhs` to split.
    axis_name: name of the axis to reduce along.
    reverse_arg_order: whether to reverse the order of arguments when calling
      einsum on chunks.
    precision: floating point precision to use for einsum.

  Returns:
    Result of einsum operation.
  """
  # adapted/simplified from
  # http://google3/third_party/google_research/google_research/scaling_transformer_inference_efficiency/collectives.py;l=327;rcl=521587778
  axis_size = lax.psum(1, axis_name)

  matmul = functools.partial(
      _reversed_arg_order_einsum if reverse_arg_order else jnp.einsum,
      einsum_spec,
      precision=precision,
  )

  if axis_size == 1:
    return matmul(lhs, rhs)

  if axis_size % 2:
    raise ValueError(f'axis_size must be 1 or even: {axis_size}')

  axis_index = lax.axis_index(axis_name)
  chunk_size = lhs.shape[split_axis] // axis_size

  def get_lhs_chunk(i):
    chunk_index = (axis_index + i) % axis_size
    lhs_chunk = lax.dynamic_slice_in_dim(
        lhs, chunk_index * chunk_size, chunk_size, axis=split_axis
    )
    return lhs_chunk

  def indexed_computation(i, rhs_fwd, rhs_bwd):
    lhs_fwd = get_lhs_chunk(-i)
    lhs_bwd = get_lhs_chunk(i + 1)
    return matmul(lhs_fwd, rhs_fwd) + matmul(lhs_bwd, rhs_bwd)

  perm_fwd = [(j, (j + 1) % axis_size) for j in range(axis_size)]
  perm_bwd = [(j, (j - 1) % axis_size) for j in range(axis_size)]

  def collective_matmul(i, carrys):
    accum, rhs_fwd, rhs_bwd = carrys
    rhs_fwd = lax.ppermute(rhs_fwd, axis_name, perm=perm_fwd)
    rhs_bwd = lax.ppermute(rhs_bwd, axis_name, perm=perm_bwd)
    accum += indexed_computation(i, rhs_fwd, rhs_bwd)
    return accum, rhs_fwd, rhs_bwd

  rhs_fwd = rhs
  rhs_bwd = lax.ppermute(rhs, axis_name, perm=perm_bwd)
  accum = indexed_computation(0, rhs_fwd, rhs_bwd)

  accum, rhs_fwd, rhs_bwd = lax.fori_loop(
      1, axis_size // 2, collective_matmul, (accum, rhs_fwd, rhs_bwd)
  )
  return accum


def _matmul_reducescatter_twoway(
    einsum_spec: str,
    lhs: jax.Array,
    rhs: jax.Array,
    scatter_axis: int,
    axis_name: str | tuple[str, str],
    reverse_arg_order: bool = False,
    precision: str = 'float32',
) -> jax.Array:
  """All-gather matmul using two way communication.

  This function is designed to be called from within `shmap`. It computes
  matrix-multiplication on each chunk, and then transfers the result using two
  way communication along a JAX mesh dimension.

  Args:
    einsum_spec: matmul specification.
    lhs: constant coefficients.
    rhs: sharded inputs.
    scatter_axis: the axis of `lhs` to scatter along.
    axis_name: name of the axis to reduce along.
    reverse_arg_order: whether to reverse the order of arguments when calling
      einsum on chunks.
    precision: floating point precision to use for einsum.

  Returns:
    Result of einsum operation.
  """
  # adapted/simplified from
  # http://google3/third_party/google_research/google_research/scaling_transformer_inference_efficiency/collectives.py;l=671;rcl=521587778
  axis_size = lax.psum(1, axis_name)
  matmul = functools.partial(
      _reversed_arg_order_einsum if reverse_arg_order else jnp.einsum,
      einsum_spec,
      precision=precision,
  )

  if axis_size == 1:
    return matmul(lhs, rhs)

  if axis_size % 2:
    raise ValueError(f'axis_size must be 1 or even: {axis_size}')

  axis_index = lax.axis_index(axis_name)
  chunk_size = lhs.shape[scatter_axis] // axis_size

  def indexed_computation(i):
    chunk_index = (axis_index + axis_size // 2 + i) % axis_size
    lhs_chunk = lax.dynamic_slice_in_dim(
        lhs, chunk_index * chunk_size, chunk_size, axis=scatter_axis
    )
    return matmul(lhs_chunk, rhs)

  perm_fwd = [(j, (j + 1) % axis_size) for j in range(axis_size)]
  perm_bwd = [(j, (j - 1) % axis_size) for j in range(axis_size)]

  def collective_matmul(i, carrays):
    accum_fwd, accum_bwd = carrays
    accum_fwd = lax.ppermute(accum_fwd, axis_name, perm=perm_fwd)
    accum_bwd = lax.ppermute(accum_bwd, axis_name, perm=perm_bwd)
    accum_fwd += indexed_computation(-i)
    accum_bwd += indexed_computation(i + 1)
    return accum_fwd, accum_bwd

  accum_fwd = indexed_computation(0)
  accum_bwd = indexed_computation(1)

  (accum_fwd, accum_bwd) = lax.fori_loop(
      1, axis_size // 2, collective_matmul, (accum_fwd, accum_bwd)
  )
  accum_fwd = lax.ppermute(accum_fwd, axis_name, perm=perm_fwd)
  accum = accum_fwd + accum_bwd
  return accum


def _parse_einsum_subscripts(subscripts: str) -> tuple[str, str, str]:
  if '...' in subscripts:
    raise ValueError(f'ellipsis not supported by sharded_einsum: {subscripts}')
  matches = re.fullmatch(r'(\w+)\,(\w+)\-\>(\w+)', subscripts)
  if matches is None:
    raise ValueError(f'{subscripts=} does not match the expected pattern')
  lhs_subscripts, rhs_subscripts, out_subscripts = matches.groups()
  return lhs_subscripts, rhs_subscripts, out_subscripts


def _determine_reduce_subscript(
    lhs_subscripts: str,
    rhs_subscripts: str,
    out_subscripts: str,
    rhs_spec: jax.sharding.PartitionSpec,
) -> str:
  """Determine the reduce axis subscript from einsum subscripts."""
  subscripts = []
  for subscript in lhs_subscripts:
    if (
        subscript not in out_subscripts
        and subscript in rhs_subscripts
        and rhs_spec[rhs_subscripts.index(subscript)] is not None
    ):
      subscripts.append(subscript)
  if len(subscripts) != 1:
    raise ValueError(f'multiple sharded axes are reduced over: {subscripts}')
  (subscript,) = subscripts  # pylint: disable=unbalanced-tuple-unpacking
  return subscript


def _determine_transfer_subscript(
    lhs_subscripts: str,
    rhs_subscripts: str,
    out_subscripts: str,
    out_spec: jax.sharding.PartitionSpec,
) -> str:
  """Determine the transfer axis subscript from einsum subscripts."""
  subscripts = []
  for subscript in lhs_subscripts:
    if (
        subscript not in rhs_subscripts
        and subscript in out_subscripts
        and out_spec[out_subscripts.index(subscript)] is not None
    ):
      subscripts.append(subscript)
  if len(subscripts) != 1:
    raise ValueError(
        f'multiple sharded axes are transferred over: {subscripts}'
    )
  (subscript,) = subscripts  # pylint: disable=unbalanced-tuple-unpacking
  return subscript


@jax.named_call
def sharded_einsum(
    subscripts: str,
    lhs: np.ndarray | jax.Array,
    rhs: jax.Array,
    /,
    *,
    gather_inputs: bool | None = None,
    reverse_arg_order: bool = False,
    precision: str = 'float32',
    mesh: jax.sharding.Mesh | None,
    rhs_spec: jax.sharding.PartitionSpec,
    out_spec: jax.sharding.PartitionSpec,
) -> jax.Array:
  """Calculate a two argument einsum with a single sharded inputs.

  Designed for supporting `einsum` patterns that arise in the Neural GCM dycore,
  i.e., forward/inverse Fourier and Legendre transforms. Only reductions along
  a single sharded axis are supported.

  Args:
    subscripts: einsum specification string.
    lhs: non-sharded left hand side argument, in the form of a static NumPy or
      JAX array.
    rhs: right hand side argument, potentially sharded along any axes.
    gather_inputs: if `True`, transfer input data between cores to implement the
      parallel einsum; if `False`, transfer output data from the result of
      calculating each chunk. The default behavior is to use whichever strategy
      results in less data movement, i.e., `gather_inputs=False` iff the outputs
      are smaller than the `rhs``.
    reverse_arg_order: if True, call einsum on each chunk like `einsum(..., rhs,
      lhs)` instead of `einsum(..., lhs, rhs)`. This results in different XLA
      optimizations and can occasionally be somewhat faster.
    precision: floating point precision to use for einsum.
    mesh: parallel mesh to use for implementing this operation, or `None`, which
      indicates no sharding.
    rhs_spec: sharding spec for `rhs`.
    out_spec: sharding spec for outputs. Dimensions in `rhs` that are preserved
      in the outputs should be chunked in the way on `rhs_spec`.

  Returns:
    Result of `einsum(subscripts, lhs, rhs)`, sharded according to `out_spec`.
  """
  all_subscripts = _parse_einsum_subscripts(subscripts)
  lhs_subscripts, rhs_subscripts, out_subscripts = all_subscripts

  if mesh is None:
    # validation complete; calculate non-sharded einsum if no mesh supplied
    return jnp.einsum(subscripts, lhs, rhs, precision=precision)

  reduce_subscript = _determine_reduce_subscript(*all_subscripts, rhs_spec)
  transfer_subscript = _determine_transfer_subscript(*all_subscripts, out_spec)
  reduce_axis_name = rhs_spec[rhs_subscripts.index(reduce_subscript)]

  if gather_inputs is None:
    out_shape = jax.eval_shape(
        functools.partial(jnp.einsum, subscripts), lhs, rhs
    ).shape
    gather_inputs = math.prod(out_shape) > math.prod(rhs.shape)

  if gather_inputs:
    lhs_partitions = [
        out_spec[out_subscripts.index(i)] if i in out_subscripts else None
        for i in lhs_subscripts
    ]
    split_axis = lhs_subscripts.index(reduce_subscript)
    matmul_impl = functools.partial(
        _allgather_matmul_twoway, split_axis=split_axis
    )
  else:
    lhs_partitions = [
        rhs_spec[rhs_subscripts.index(i)] if i in rhs_subscripts else None
        for i in lhs_subscripts
    ]
    scatter_axis = lhs_subscripts.index(transfer_subscript)
    matmul_impl = functools.partial(
        _matmul_reducescatter_twoway, scatter_axis=scatter_axis
    )

  lhs_spec = jax.sharding.PartitionSpec(*lhs_partitions)
  in_specs = (lhs_spec, rhs_spec)

  @functools.partial(
      shard_map.shard_map,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=out_spec,
      check_rep=False,  # https://github.com/jax-ml/jax/issues/15894
  )
  @jax.jit  # https://github.com/jax-ml/jax/issues/15723
  def distributed_matmul(lhs, rhs):
    return matmul_impl(
        subscripts,
        lhs,
        rhs,
        axis_name=reduce_axis_name,
        precision=precision,
        reverse_arg_order=reverse_arg_order,
    )

  return distributed_matmul(lhs, rhs)
