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

"""Tests for jax_numpy_utils."""
import math

from absl.testing import absltest
from absl.testing import parameterized
import chex

from dinosaur import jax_numpy_utils

import jax
import numpy as np


chex.set_n_cpu_devices(8)  # set at top level so it works in pytest


class ResolveDotPrecisionTest(parameterized.TestCase):

  @parameterized.parameters(
      ('tensorfloat32', jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3),
      ('bfloat16_3x', jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3),
      ('high', jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3),
      ('bfloat16', jax.lax.DotAlgorithmPreset.BF16_BF16_F32),
      ('highest', 'highest'),
      ('float32', 'float32'),
      (('bfloat16', 'highest'), ('bfloat16', 'highest')),
  )
  def test_legacy_alias_normalization(self, precision, expected):
    x = np.zeros((2, 2), np.float32)
    resolved = jax_numpy_utils.resolve_dot_precision(precision, x, x)
    self.assertEqual(resolved, expected)

  def test_float64_resolves_to_highest(self):
    x = np.zeros((2, 2), np.float64)
    resolved = jax_numpy_utils.resolve_dot_precision(None, x, x)
    self.assertEqual(resolved, jax.lax.Precision.HIGHEST)


class CumsumTest(parameterized.TestCase):

  @parameterized.parameters({'axis': 0}, {'axis': 1}, {'axis': -1})
  def test_cumsum(self, axis):
    x = np.random.RandomState(0).randn(3, 4)
    expected = np.cumsum(x, axis=axis)
    actual = jax_numpy_utils.cumsum(x, axis=axis, method='dot')
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  @parameterized.parameters({'axis': 0}, {'axis': 1}, {'axis': -1})
  def test_reverse_cumsum_consistency(self, axis):
    x = np.random.RandomState(0).randn(3, 4)
    expected = jax_numpy_utils.reverse_cumsum(x, axis=axis, method='jax')
    actual = jax_numpy_utils.reverse_cumsum(x, axis=axis, method='dot')
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  def test_shmap(self):
    x = np.random.RandomState(0).randn(16)
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), axis_names=['z']),
        jax.sharding.PartitionSpec('z'),
    )
    y = jax.device_put(x, sharding)

    expected = jax_numpy_utils.cumsum(x, axis=0)
    actual = jax_numpy_utils.cumsum(y, axis=0, sharding=sharding)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

    expected = jax_numpy_utils.reverse_cumsum(x, axis=0)
    actual = jax_numpy_utils.reverse_cumsum(y, axis=0, sharding=sharding)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  @parameterized.parameters({'axis': 0}, {'axis': 1}, {'axis': -1})
  def test_shmap_2d(self, axis):
    x = np.random.RandomState(0).randn(16, 3)
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), axis_names=['z']),
        jax.sharding.PartitionSpec('z', None),
    )
    y = jax.device_put(x, sharding)

    expected = jax_numpy_utils.cumsum(x, axis=axis)
    actual = jax_numpy_utils.cumsum(y, axis=axis, sharding=sharding)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

    expected = jax_numpy_utils.reverse_cumsum(x, axis=axis)
    actual = jax_numpy_utils.reverse_cumsum(y, axis=axis, sharding=sharding)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


class ShiftTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'inputs': np.array([1, 2, 3]),
          'offset': 0,
          'expected': np.array([1, 2, 3]),
      },
      {
          'inputs': np.array([1, 2, 3]),
          'offset': +1,
          'expected': np.array([0, 1, 2]),
      },
      {
          'inputs': np.array([1, 2, 3]),
          'offset': -1,
          'expected': np.array([2, 3, 0]),
      },
      {
          'inputs': np.array([1, 2, 3]),
          'offset': +5,
          'expected': np.array([0, 0, 0]),
      },
      {
          'inputs': np.array([1, 2, 3]),
          'offset': -5,
          'expected': np.array([0, 0, 0]),
      },
      {
          'inputs': np.ones((2, 3)),
          'offset': +1,
          'axis': 0,
          'expected': np.stack([np.zeros(3), np.ones(3)]),
      },
      {
          'inputs': np.ones((2, 3)),
          'offset': -1,
          'axis': -2,
          'expected': np.stack([np.ones(3), np.zeros(3)]),
      },
      {
          'inputs': np.ones((2, 3), int),
          'offset': +1,
          'axis': 1,
          'expected': np.array([[0, 1, 1], [0, 1, 1]]),
      },
  )
  def test_shift(self, inputs, offset, expected, axis=0):
    actual = jax_numpy_utils.shift(inputs, offset, axis)
    np.testing.assert_array_equal(expected, actual)


P = jax.sharding.PartitionSpec


_EINSUM_CASES = [
    {
        'subscripts': 'ij,jk->ik',
        'lhs_shape': (2, 2),
        'rhs_shape': (2, 2),
        'axis_names': ['x', 'y'],
        'mesh_shape': (2, 2),
        'rhs_spec': P('x', 'y'),
        'out_spec': P('x', 'y'),
    },
    {
        'subscripts': 'ij,jk->ik',
        'lhs_shape': (4, 4),
        'rhs_shape': (4, 3),
        'axis_names': ['x', 'y'],
        'mesh_shape': (4, 1),
        'rhs_spec': P('x', 'y'),
        'out_spec': P('x', 'y'),
    },
    {
        'subscripts': 'ij,jk->ik',
        'lhs_shape': (3, 3),
        'rhs_shape': (3, 4),
        'axis_names': ['x', 'y'],
        'mesh_shape': (1, 4),
        'rhs_spec': P('x', 'y'),
        'out_spec': P('x', 'y'),
    },
    {
        # inverse Legendre
        'subscripts': 'mjl,zsml->zsmj',
        'lhs_shape': (2, 2, 2),
        'rhs_shape': (3, 2, 2, 2),
        'axis_names': ['z', 'x', 'y'],
        'mesh_shape': (1, 2, 2),
        'rhs_spec': P('z', None, 'x', 'y'),
        'out_spec': P('z', None, 'x', 'y'),
    },
    {
        # inverse Fourier
        'subscripts': 'ism,zsmj->zij',
        'lhs_shape': (2, 2, 2),
        'rhs_shape': (3, 2, 2, 2),
        'axis_names': ['z', 'x', 'y'],
        'mesh_shape': (1, 2, 2),
        'rhs_spec': P('z', None, 'x', 'y'),
        'out_spec': P('z', 'x', 'y'),
    },
    {
        # forward Fourier
        'subscripts': 'ism,zij->zsmj',
        'lhs_shape': (4, 2, 4),
        'rhs_shape': (3, 4, 2),
        'axis_names': ['z', 'x', 'y'],
        'mesh_shape': (1, 4, 2),
        'rhs_spec': P('z', 'x', 'y'),
        'out_spec': P('z', None, 'x', 'y'),
    },
    {
        # forward Legendre
        'subscripts': 'mjl,zsmj->zsml',
        'lhs_shape': (2, 2, 2),
        'rhs_shape': (4, 2, 2, 2),
        'axis_names': ['z', 'x', 'y'],
        'mesh_shape': (2, 2, 2),
        'rhs_spec': P('z', None, 'x', 'y'),
        'out_spec': P('z', None, 'x', 'y'),
    },
    {
        # vertical matvec
        'subscripts': 'gh,hml->gml',
        'lhs_shape': (8, 8),
        'rhs_shape': (8, 1, 1),
        'axis_names': ['z', 'x', 'y'],
        'mesh_shape': (8, 1, 1),
        'rhs_spec': P('z', 'x', 'y'),
        'out_spec': P('z', 'x', 'y'),
    },
    {
        # vertical matvec per wavenumber
        'subscripts': 'lgh,hml->gml',
        'lhs_shape': (2, 4, 4),
        'rhs_shape': (4, 2, 2),
        'axis_names': ['z', 'x', 'y'],
        'mesh_shape': (4, 1, 2),
        'rhs_spec': P('z', 'x', 'y'),
        'out_spec': P('z', 'x', 'y'),
    },
]
_GATHER_CASES = [dict(case, gather_inputs=True) for case in _EINSUM_CASES]
_SCATTER_CASES = [dict(case, gather_inputs=False) for case in _EINSUM_CASES]
_FLEX_CASES = [dict(case, gather_inputs=None) for case in _EINSUM_CASES]


class ShardedEinsum(parameterized.TestCase):

  @parameterized.parameters(*_GATHER_CASES, *_SCATTER_CASES, *_FLEX_CASES)
  def test_sharded_einsum(
      self, subscripts, lhs_shape, rhs_shape, axis_names, mesh_shape, **kwargs
  ):
    mesh_size = math.prod(mesh_shape)
    devices = np.array(jax.devices()[:mesh_size]).reshape(mesh_shape)
    mesh = jax.sharding.Mesh(devices, axis_names)
    lhs = 1.0 * np.arange(math.prod(lhs_shape)).reshape(lhs_shape)
    rhs = 2.0 * np.arange(math.prod(rhs_shape)).reshape(rhs_shape)
    expected = np.einsum(subscripts, lhs, rhs)
    actual = jax_numpy_utils.sharded_einsum(
        subscripts, lhs, rhs, mesh=mesh, **kwargs
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
