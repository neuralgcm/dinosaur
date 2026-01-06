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

"""Tests for filtering."""

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import filtering
from dinosaur import spherical_harmonic

import jax
import numpy as np


class FilteringTest(parameterized.TestCase):

  def test_make_filter_fn(self):
    # make a filter than should only effect 3D variables
    f = filtering._make_filter_fn(np.zeros((2, 1, 3)))
    x = {'a': 1.0, 'b': np.ones((1, 2, 3)), 'c': np.ones((2, 2, 3))}
    y = f(x)
    np.testing.assert_array_equal(y['a'], x['a'])  # not filtered
    np.testing.assert_array_equal(y['b'], x['b'])  # not filtered
    np.testing.assert_array_equal(y['c'], 0.0 * x['c'])  # filtered

  def test_exponential_filter(self):
    grid = spherical_harmonic.Grid.TL127()
    inputs = np.ones(grid.modal_shape)
    scaling = filtering.exponential_filter(grid, attenuation=16)(inputs)
    self.assertAlmostEqual(scaling[0, 0], 1.0)
    self.assertAlmostEqual(scaling[0, -1], 1.125e-7, delta=1e-9)

  @parameterized.parameters(
      dict(order=1),
      dict(order=2),
      dict(order=3),
  )
  def test_horizontal_diffusion_filter(self, order):
    grid = spherical_harmonic.Grid.TL127()
    inputs = np.ones((3,) + grid.modal_shape)
    # verify that we can supply different timescales for different layers
    timescale = np.array([1, 2, 3])[:, np.newaxis, np.newaxis]
    eigenvalues = grid.laplacian_eigenvalues
    scale = 0.1 / (timescale * abs(eigenvalues[-1]) ** order)
    scaling = filtering.horizontal_diffusion_filter(grid, scale, order)(inputs)
    self.assertAlmostEqual(scaling[0, 0, 0], 1.0)
    self.assertAlmostEqual(scaling[1, 0, 0], 1.0)
    self.assertAlmostEqual(scaling[2, 0, 0], 1.0)
    self.assertAlmostEqual(scaling[0, 0, -1], np.exp(-0.1))
    self.assertAlmostEqual(scaling[1, 0, -1], np.exp(-0.1/2))
    self.assertAlmostEqual(scaling[2, 0, -1], np.exp(-0.1/3))

  @parameterized.parameters(
      dict(filter_fn=filtering.exponential_filter,
           filter_kwargs={
               'attenuation': np.expand_dims(np.array([1.0, 2.0]), (1, 2, 3))}
           ),
      dict(filter_fn=filtering.horizontal_diffusion_filter,
           filter_kwargs={
               'scale': np.expand_dims(np.array([1.0, 2.0]), (1, 2, 3))}
           ),
  )
  def test_time_filter_variation(self, filter_fn, filter_kwargs):
    """Tests that filter arguments can be specified as arrays."""
    (time_slices,) = set(x.shape[0] for x in filter_kwargs.values())
    grid = spherical_harmonic.Grid.TL63()
    inputs = np.ones((time_slices, 3) + grid.modal_shape)  # [time, level, ...].
    out = filter_fn(grid, **filter_kwargs)(inputs)
    for i in range(time_slices):
      kwargs = jax.tree.map (lambda x: x[i].item(), filter_kwargs)  # pylint: disable=cell-var-from-loop
      # we expect `out[i]` to be equal to independent application filter_fn.
      expected_out_i = filter_fn(grid, **kwargs)(inputs[i, ...])
      np.testing.assert_allclose(expected_out_i, out[i, ...])


if __name__ == '__main__':
  absltest.main()
