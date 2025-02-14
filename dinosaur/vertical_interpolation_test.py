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

"""Tests for vertical_interpolation."""

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import sigma_coordinates
from dinosaur import vertical_interpolation
import jax.numpy as jnp
import numpy as np


class InterpTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(x=0.5, expected=10.0),
      dict(x=1.1, expected=11.0),
      dict(x=2.5, expected=22.5),
      dict(x=10.0, expected=30.0),
  )
  def test_interp(self, x, expected):
    xp = jnp.array([1, 2, 4])
    fp = jnp.array([10, 20, 30])
    actual = vertical_interpolation.interp(x, xp, fp)
    self.assertAlmostEqual(actual, expected)

  @parameterized.parameters(
      dict(x=0.5, expected=5.0),
      dict(x=1.1, expected=11.0),
      dict(x=2.5, expected=22.5),
      dict(x=6.0, expected=40.0),
  )
  def test_linear_interp_with_linear_extrap(self, x, expected):
    xp = jnp.array([1, 2, 4])
    fp = jnp.array([10, 20, 30])
    actual = vertical_interpolation.linear_interp_with_linear_extrap(x, xp, fp)
    self.assertAlmostEqual(actual, expected)


class PressureLevelsTest(parameterized.TestCase):

  def test_get_surface_pressure(self):
    levels = np.array([100, 200, 300, 400, 500])
    orography = np.array([[0, 5, 10, 15]])
    geopotential = np.moveaxis(
        [[
            [400, 250, 150, 50, -50],
            [1000, 900, 140, 40, 20],
            [500, 400, 300, 200, 100],
            [600, 500, 400, 300, 200],
        ]],
        -1,
        0,
    )  # reorder from [x, y, level] to [level, x, y]
    expected = np.array([[[450, 390, 500, 550]]])
    actual = vertical_interpolation.get_surface_pressure(
        vertical_interpolation.PressureCoordinates(levels),
        geopotential,
        orography,
        gravity_acceleration=10,
    )
    np.testing.assert_allclose(actual, expected)

  def test_sigma_to_pressure_roundtrip(self):
    sigma_coords = sigma_coordinates.SigmaCoordinates.equidistant(10)
    pressure_coords = vertical_interpolation.PressureCoordinates(
        np.array([100, 200, 300, 400])
    )
    surface_pressure = np.array([[[250, 350, 450]]])
    original = np.moveaxis([[[1, 2, 3, 4]] * 3], -1, 0)
    on_sigma_levels = vertical_interpolation.interp_pressure_to_sigma(
        original,
        pressure_coords,
        sigma_coords,
        surface_pressure,
    )
    self.assertEqual(on_sigma_levels.shape, (10, 1, 3))
    roundtripped = vertical_interpolation.interp_sigma_to_pressure(
        on_sigma_levels, pressure_coords, sigma_coords, surface_pressure
    )
    # mask points below the surface
    expected = original.astype(float)
    expected[2:, :, 0] = np.nan
    expected[3:, :, 1] = np.nan
    np.testing.assert_allclose(roundtripped, expected, atol=1e-6)

  def test_interp_sigma_to_sigma(self):
    source_sigma_coords = sigma_coordinates.SigmaCoordinates(
        np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    )
    target_sigma_coords = sigma_coordinates.SigmaCoordinates(
        np.array([0.0, 0.2, 0.3, 0.5, 0.7, 1.0])
    )
    field_value = np.array([1.0, 2.0, 4.0, 6.0, 8.0])
    field_value = field_value[:, np.newaxis, np.newaxis]
    fields = {'temperature': field_value}
    expected_value = jnp.array([1, 1.75, 3, 5, 7.5])
    expected_value = expected_value[:, np.newaxis, np.newaxis]
    expected = {'temperature': expected_value}
    actual = vertical_interpolation.interp_sigma_to_sigma(
        fields,
        source_sigma_coords,
        target_sigma_coords,
    )
    np.testing.assert_allclose(actual['temperature'], expected['temperature'], atol=1e-6)


class HybridCoordinatesTest(absltest.TestCase):

  def test_ecmwf_137(self):
    coords = vertical_interpolation.HybridCoordinates.ECMWF137()
    self.assertEqual(coords.layers, 137)

    # spot check ph (half-level, on the boundary) from row 84 in the table
    expected = 316.3607
    actual = coords.a_boundaries[84] + coords.b_boundaries[84] * 1013.250
    self.assertAlmostEqual(expected, actual, places=3)

  def test_ufs_127(self):
    coords = vertical_interpolation.HybridCoordinates.UFS127()
    self.assertEqual(coords.layers, 127)

    # spot check pf (full-level, centered) from row 80 in the table
    expected = 572.829345703125
    p_below = coords.a_boundaries[79] + coords.b_boundaries[79] * 1000
    p_above = coords.a_boundaries[80] + coords.b_boundaries[80] * 1000
    actual = (p_below + p_above) / 2
    self.assertAlmostEqual(expected, actual, places=1)

  def test_to_approx_sigma_coords(self):
    hybrid_coords = vertical_interpolation.HybridCoordinates(
        # pressure bounds at [0, 500, 800, 1000]
        a_boundaries=np.array([0, 500, 300, 0]),
        b_boundaries=np.array([0, 0, 0.5, 1.0]),
    )

    expected = np.array([0.0, 0.65, 1.0])
    sigma_coords = hybrid_coords.to_approx_sigma_coords(
        surface_pressure=1000, layers=2
    )
    np.testing.assert_allclose(sigma_coords.boundaries, expected)

    expected = np.array([0.0, 0.5, 0.8, 1.0])
    sigma_coords = hybrid_coords.to_approx_sigma_coords(
        surface_pressure=1000, layers=3
    )
    np.testing.assert_allclose(sigma_coords.boundaries, expected)

    expected = np.array([0.0, 0.25, 0.5, 0.65, 0.8, 0.9, 1.0])
    sigma_coords = hybrid_coords.to_approx_sigma_coords(
        surface_pressure=1000, layers=6
    )
    np.testing.assert_allclose(sigma_coords.boundaries, expected)

  def test_interp_hybrid_to_sigma(self):
    sigma_coords = sigma_coordinates.SigmaCoordinates.equidistant(5)
    hybrid_coords = vertical_interpolation.HybridCoordinates(
        # pressure bounds at [0, 400, 800, 1000] -> centers at [200, 600, 900]
        a_boundaries=np.array([0, 400, 300, 0]),
        b_boundaries=np.array([0, 0, 0.5, 1.0]),
    )
    surface_pressure = np.array([[1000]])
    original = np.array([1.0, 2.0, 3.0])[:, np.newaxis, np.newaxis]
    # linear interpolation (with extrapolation) from y=[1, 2, 3] defined
    # at x=[200, 600, 900] to x=[100, 300, 500, 700, 900].
    expected = np.array([0.75, 1.25, 1.75, 2 + 1 / 3, 3.0])
    actual = vertical_interpolation.interp_hybrid_to_sigma(
        original, hybrid_coords, sigma_coords, surface_pressure
    ).ravel()
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  def test_regrid_hybrid_to_sigma(self):
    sigma_coords = sigma_coordinates.SigmaCoordinates.equidistant(5)
    hybrid_coords = vertical_interpolation.HybridCoordinates(
        # at pressures boundaries [0, 30, 75, 200, 450, 700, 850, 1000]
        a_boundaries=np.array([0, 30, 75, 200, 300, 300, 150, 0]),
        b_boundaries=np.array([0, 0, 0, 0, 0.15, 0.4, 0.7, 1]),
    )
    surface_pressure = np.array([[1000]])
    original = np.arange(1.0, 8.0)[:, np.newaxis, np.newaxis]
    # area weighted averages for cells centered at [100, 300, 500, 700, 900]
    expected = np.array([
        1 * (30 / 200) + 2.0 * (45 / 200) + 3.0 * (125 / 200),
        4.0,
        4.0 * (50 / 200) + 5.0 * (150 / 200),
        5.0 * (100 / 200) + 6.0 * (100 / 200),
        6.0 * (50 / 200) + 7.0 * (150 / 200),
    ])
    actual = vertical_interpolation.regrid_hybrid_to_sigma(
        original, hybrid_coords, sigma_coords, surface_pressure
    ).ravel()
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  def test_interval_overlap(self):
    actual = vertical_interpolation._interval_overlap(
        np.array([1, 4, 6, 9, 10]), np.array([0, 3, 7, 10])
    )
    expected = np.array([[2, 0, 0, 0], [1, 2, 1, 0], [0, 0, 2, 1]])
    np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
  absltest.main()
