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

"""Tests for semi_lagrangian_matmul."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import lagrange_interpolation
from dinosaur import semi_lagrangian_matmul
from dinosaur import spherical_harmonic
import jax
import jax.numpy as jnp
import numpy as np


class WindowArithmeticTest(parameterized.TestCase):

  def test_window_size(self):
    window_size = semi_lagrangian_matmul._window_size
    # 2 * half_width + stencil, truncated to the source and to the buffer.
    self.assertEqual(window_size(100, 100, 6, 4), 16)
    self.assertEqual(window_size(100, 10, 6, 4), 10)
    self.assertEqual(window_size(12, 100, 6, 4), 12)

  def test_z_window_on_an_unsharded_column(self):
    starts = semi_lagrangian_matmul._z_window_starts
    for first, expected in ((-3, 0), (5, 5), (30, 16)):
      win_start, buf_start = starts(jnp.int32(first), 32, 16)
      self.assertEqual(int(win_start), expected)
      self.assertEqual(int(buf_start), expected)
    # A window that is the whole column starts at its first level.
    win_start, _ = starts(jnp.int32(7), 32, 32)
    self.assertEqual(int(win_start), 0)

  @parameterized.parameters(
      dict(levels=32, z_shards=4, halo=8, size=16),  # one level per window
      dict(levels=32, z_shards=4, halo=8, size=16 + 3),  # blocks of 4 levels
      dict(levels=20, z_shards=2, halo=8, size=16 + 1),
      dict(levels=40, z_shards=4, halo=8, size=16 + 1),
  )
  def test_z_window_stays_off_the_cyclic_halo(
      self, levels, z_shards, halo, size
  ):
    """Windows of a z-shard cover only source levels its buffer really holds.

    The buffer of shard `iz` is [halo, local levels, halo] and starts at
    source level `iz * local - halo`; on the first and last shard that
    reaches outside the column, where the cyclic exchange put levels from
    its other end.
    """
    local = levels // z_shards
    n_buffer = local + 2 * halo
    size = min(size, local + halo, levels)
    for iz in range(z_shards):
      origin = jnp.int32(iz * local - halo)
      for k in range(iz * local, (iz + 1) * local):
        first = jnp.int32(k - 7)
        win_start, buf_start = semi_lagrangian_matmul._z_window_starts(
            first, levels, size, origin, n_buffer
        )
        win_start, buf_start = int(win_start), int(buf_start)
        with self.subTest(iz=iz, k=k):
          # The buffer position and the source position are the same level.
          self.assertEqual(buf_start + int(origin), win_start)
          # The window is inside the column and inside the buffer.
          self.assertGreaterEqual(win_start, 0)
          self.assertLessEqual(win_start + size, levels)
          self.assertGreaterEqual(buf_start, 0)
          self.assertLessEqual(buf_start + size, n_buffer)
          # It contains the query level.
          self.assertLessEqual(win_start, k)
          self.assertGreater(win_start + size, k)

  def test_lat_window_starts(self):
    j_global = jnp.arange(6, dtype=jnp.int32)
    win_start, buf_start = semi_lagrangian_matmul._lat_window_starts(
        j_global, j_global - 1, n_source=10, n_buffer=10, size=4, buf_origin=0
    )
    np.testing.assert_array_equal(win_start, [0, 0, 1, 2, 3, 4])
    np.testing.assert_array_equal(buf_start, win_start)
    # A buffer that starts at source row 2.
    _, buf_start = semi_lagrangian_matmul._lat_window_starts(
        j_global, j_global + 2, n_source=10, n_buffer=8, size=4, buf_origin=2
    )
    np.testing.assert_array_equal(buf_start, [0, 1, 2, 3, 4, 4])

  def test_weighted_one_hot(self):
    indices = (jnp.asarray([0, 2]), jnp.asarray([1, 3]))
    weights = (jnp.asarray([0.25, 0.5]), jnp.asarray([0.75, 0.5]))
    actual = semi_lagrangian_matmul._weighted_one_hot(
        indices, weights, 4, jnp.float32
    )
    np.testing.assert_array_equal(
        actual, [[0.25, 0.75, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]
    )


class SelectorProductTest(parameterized.TestCase):

  def test_select_last_axis(self):
    rng = np.random.RandomState(0)
    values = jnp.asarray(rng.normal(size=(3, 4, 7)), jnp.float32)
    rows = np.array([6, 0, 0, 3, 5])
    selector = jnp.asarray(np.eye(7)[rows], jnp.float32)
    for is_tpu in (False, True):
      np.testing.assert_array_equal(
          semi_lagrangian_matmul._select_last_axis(values, selector, is_tpu),
          values[..., rows],
      )

  def test_select_last_axis_keeps_cotangents_at_full_precision(self):
    # JAX transposes a dot product with the cotangent as the left operand and
    # the per-operand precision unchanged, so the operand that may be rounded
    # to bfloat16 on TPU has to be the selector, on the right, in the forward
    # product and in its transpose.
    values = jnp.ones((3, 4, 7), jnp.float32)
    selector = jnp.asarray(np.eye(5, 7), jnp.float32)

    def loss(x):
      out = semi_lagrangian_matmul._select_last_axis(x, selector, True)
      return jnp.sum(out**2)

    jaxpr = jax.make_jaxpr(jax.grad(loss))(values).jaxpr
    dots = [e for e in jaxpr.eqns if e.primitive.name == 'dot_general']
    self.assertLen(dots, 2)
    for eqn in dots:
      self.assertEqual(eqn.invars[1].aval.shape, selector.shape)
      self.assertEqual(eqn.params['precision'][0], jax.lax.Precision.HIGHEST)


class PlannerTest(parameterized.TestCase):

  def _plan(
      self, platform, num_channels, limited_start, shift=2, wavenumber=63
  ):
    grid = spherical_harmonic.Grid.construct(
        max_wavenumber=wavenumber,
        gaussian_nodes={42: 32, 63: 48, 85: 64, 106: 80, 170: 128}[wavenumber],
        spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
    )
    layers = 47
    nodes = np.linspace(0.01, 0.99, layers)
    halo = lagrange_interpolation.HALO_WIDTH
    fields = jnp.zeros(
        (
            num_channels,
            layers,
            grid.longitude_nodes,
            grid.latitude_nodes + 2 * halo,
        ),
        jnp.float32,
    )
    queries = jnp.zeros((layers,) + grid.nodal_shape, jnp.float32)
    limiter = 'quasi_monotone' if limited_start < num_channels else None
    spec = semi_lagrangian_matmul._Matmul3DSpec(
        grid, 'cubic', 'cubic', limiter, shift, shift, limited_start
    )
    with mock.patch.object(
        lagrange_interpolation, 'target_platform', return_value=platform
    ):
      plan = semi_lagrangian_matmul._plan_matmul_3d(
          semi_lagrangian_matmul._LocalBuffer(fields),
          semi_lagrangian_matmul._LocalQueries(
              queries, queries, queries, (None, None), (None, None)
          ),
          spec,
          nodes,
      )
    return semi_lagrangian_matmul._plan_blocks_and_channels(
        plan, limited_start, None
    )

  @parameterized.parameters(
      # (channels, largest chunk that fits) -> chunk
      dict(num_channels=8, max_chunk=6, expected=4),
      dict(num_channels=8, max_chunk=3, expected=2),
      dict(num_channels=16, max_chunk=5, expected=4),
      dict(num_channels=7, max_chunk=4, expected=4),
      dict(num_channels=9, max_chunk=4, expected=3),
      dict(num_channels=5, max_chunk=1, expected=1),
  )
  def test_chunk_trades_padding_against_chunks(
      self, num_channels, max_chunk, expected
  ):
    plan, _ = self._plan('gpu', num_channels, 0)
    with mock.patch.object(
        type(plan),
        'fits_budget',
        lambda self, chunk, limited_start=0: chunk <= max_chunk,
    ):
      channels = semi_lagrangian_matmul._plan_channels(plan, 0, None)
    self.assertEqual(channels.chunk_size, expected)
    num_chunks = -(-num_channels // expected)
    self.assertEqual(channels.cost, num_chunks * (expected + 0.5))

  @parameterized.parameters(
      # The configurations the planner constants were measured on (TPU v5e,
      # L47): (wavenumber, window, channels, first limited) -> (chunk, block),
      # the fastest of the measured combinations.
      dict(wavenumber=42, shift=2, channels=8, limited=0, expected=(8, 2)),
      dict(wavenumber=63, shift=2, channels=8, limited=0, expected=(4, 2)),
      dict(wavenumber=63, shift=2, channels=4, limited=4, expected=(4, 2)),
      dict(wavenumber=63, shift=6, channels=8, limited=0, expected=(1, 2)),
      dict(wavenumber=85, shift=2, channels=8, limited=0, expected=(4, 1)),
      dict(wavenumber=85, shift=2, channels=4, limited=4, expected=(4, 1)),
      dict(wavenumber=106, shift=2, channels=8, limited=0, expected=(2, 2)),
      dict(wavenumber=170, shift=2, channels=8, limited=0, expected=(1, 1)),
  )
  def test_tpu_plan(self, wavenumber, shift, channels, limited, expected):
    plan, channel_plan = self._plan(
        'tpu', channels, limited, shift=shift, wavenumber=wavenumber
    )
    self.assertEqual((channel_plan.chunk_size, plan.block_levels), expected)

  def test_blocks_are_a_tpu_layout(self):
    plan, _ = self._plan('gpu', 8, 0)
    self.assertEqual(plan.allowed_block_levels, 1)
    self.assertEqual(plan.block_levels, 1)

  def test_block_length(self):
    block_length = semi_lagrangian_matmul._level_block_length
    fits = dict(block_bytes=lambda levels: 1, limit=1)
    self.assertEqual(block_length(47, True, **fits), 2)
    self.assertEqual(block_length(47, False, **fits), 1)
    self.assertEqual(block_length(1, True, **fits), 1)
    self.assertEqual(block_length(47, True, max_block_levels=1, **fits), 1)
    self.assertEqual(
        block_length(47, True, block_bytes=lambda levels: 2, limit=1), 1
    )


if __name__ == '__main__':
  absltest.main()
