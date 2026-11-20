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

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import semi_lagrangian_matmul
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


if __name__ == '__main__':
  absltest.main()
