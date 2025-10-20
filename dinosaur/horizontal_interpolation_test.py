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

"""Tests for horizontal_interpolation."""
import functools

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic

import numpy as np


class HorizontalInterpolationTest(parameterized.TestCase):

  def test_conservative_latitude_weights(self):
    source_lat = np.pi / 180 * np.array([-75, -45, -15, 15, 45, 75])
    target_lat = np.pi / 180 * np.array([-45, 45])
    # from Wolfram alpha:
    # integral of cos(x) from 0*pi/6 to 1*pi/6 -> 0.5
    # integral of cos(x) from 1*pi/6 to 2*pi/6 -> (sqrt(3) - 1) / 2
    # integral of cos(x) from 2*pi/6 to 3*pi/6 -> 1 - sqrt(3) / 2
    expected = np.array(
        [
            [1 - np.sqrt(3) / 2, (np.sqrt(3) - 1) / 2, 1 / 2, 0, 0, 0],
            [0, 0, 0, 1 / 2, (np.sqrt(3) - 1) / 2, 1 - np.sqrt(3) / 2],
        ]
    )
    actual = horizontal_interpolation.conservative_latitude_weights(
        source_lat, target_lat
    )
    np.testing.assert_almost_equal(expected, actual)

  @parameterized.parameters(
      (1, 0, 1),
      (-1, 0, -1),
      (5, 0, 5),
      (6, 0, -4),
      (1, 9, 11),
      (5, 9, 5),
  )
  def test_align_phase_with(self, x, y, expected):
    actual = horizontal_interpolation._align_phase_with(x, y, period=10)
    self.assertEqual(actual, expected)

  def test_conservative_longitude_weights(self):
    source_lon = np.pi / 180 * np.array([0, 60, 120, 180, 240, 300])
    target_lon = np.pi / 180 * np.array([0, 90, 180, 270])
    expected = (
        np.array(
            [
                [4, 1, 0, 0, 0, 1],
                [0, 3, 3, 0, 0, 0],
                [0, 0, 1, 4, 1, 0],
                [0, 0, 0, 0, 3, 3],
            ]
        )
        / 6
    )
    actual = horizontal_interpolation.conservative_longitude_weights(
        source_lon, target_lon
    )
    np.testing.assert_allclose(expected, actual, atol=1e-5)

  @parameterized.named_parameters(
      {
          'testcase_name': 'bilinear',
          'regridder_cls': horizontal_interpolation.BilinearRegridder,
      },
      {
          'testcase_name': 'conservative',
          'regridder_cls': horizontal_interpolation.ConservativeRegridder,
      },
      {
          'testcase_name': 'nearest',
          'regridder_cls': horizontal_interpolation.NearestRegridder,
      }
  )
  def test_regridding_shape(self, regridder_cls):
    source_grid = spherical_harmonic.Grid.T85()
    target_grid = spherical_harmonic.Grid.T21()
    regridder = regridder_cls(source_grid, target_grid)

    inputs = np.zeros(source_grid.nodal_shape)
    outputs = regridder(inputs)
    self.assertEqual(outputs.shape, target_grid.nodal_shape)

    batch_inputs = np.zeros((2,) + source_grid.nodal_shape)
    batch_outputs = regridder(batch_inputs)
    self.assertEqual(batch_outputs.shape, (2,) + target_grid.nodal_shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'bilinear',
          'regridder_cls': horizontal_interpolation.BilinearRegridder,
      },
      {
          'testcase_name': 'nearest',
          'regridder_cls': horizontal_interpolation.NearestRegridder,
      },
      {
          'testcase_name': 'conservative_skipna_true',
          'regridder_cls': functools.partial(
              horizontal_interpolation.ConservativeRegridder, skipna=True
          ),
      },
      {
          'testcase_name': 'conservative_skipna_false',
          'regridder_cls': functools.partial(
              horizontal_interpolation.ConservativeRegridder, skipna=False
          ),
      },
  )
  def test_regridding_nans(self, regridder_cls):
    source_grid = spherical_harmonic.Grid.TL255(latitude_spacing='equiangular')
    target_grid = spherical_harmonic.Grid.TL127()
    regridder = regridder_cls(source_grid, target_grid)

    inputs = np.ones(source_grid.nodal_shape)
    in_valid = (
        source_grid.latitudes[np.newaxis, :] ** 2
        + (source_grid.longitudes[:, np.newaxis] - np.pi) ** 2
        < (np.pi / 2) ** 2
    )
    inputs = np.where(in_valid, inputs, np.nan)
    outputs = regridder(inputs)

    out_valid = ~np.isnan(outputs)
    # Non-nan fraction is similar but not identical
    np.testing.assert_allclose(out_valid.mean(), in_valid.mean(), atol=0.1)
    np.testing.assert_allclose(outputs[out_valid], 1.0)

  @parameterized.named_parameters(
      {
          'testcase_name': 'bilinear',
          'regridder_cls': horizontal_interpolation.BilinearRegridder,
      },
      {
          'testcase_name': 'nearest',
          'regridder_cls': horizontal_interpolation.NearestRegridder,
      },
  )
  def test_regridding_exact(self, regridder_cls):
    source_grid = spherical_harmonic.Grid(
        longitude_nodes=4, latitude_nodes=3
    )
    target_grid = spherical_harmonic.Grid(
        longitude_nodes=2, latitude_nodes=3
    )
    regridder = regridder_cls(source_grid, target_grid)
    inputs = np.array([[0, 1, 2], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    expected = np.array([[0, 1, 2], [7, 8, 9]])
    actual = regridder(inputs)
    np.testing.assert_allclose(expected, actual, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
