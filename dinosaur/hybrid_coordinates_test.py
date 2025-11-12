# Copyright 2025 Google LLC
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
"""Tests for hybrid coordinates and their helper methods."""

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import hybrid_coordinates
from dinosaur import sigma_coordinates
import numpy as np


class HybridCoordinatesTest(parameterized.TestCase):

  def test_initialization_raises_on_unequal_lengths(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Expected `a_boundaries` and `b_boundaries` to have the same length, '
        'got 2 and 3.',
    ):
      hybrid_coordinates.HybridCoordinates(
          a_boundaries=np.array([1, 2]),
          b_boundaries=np.array([1, 2, 3]),
      )

  def test_from_dimensionless_coefficients(self):
    a_coeffs = np.array([0.0, 0.1, 0.2])
    b_coeffs = np.array([0.5, 0.6, 0.7])
    p0 = 1000.0
    coords = hybrid_coordinates.HybridCoordinates.from_coefficients(
        a_coeffs, b_coeffs, p0
    )
    np.testing.assert_allclose(coords.a_boundaries, a_coeffs * p0)
    np.testing.assert_allclose(coords.b_boundaries, b_coeffs)

  def test_analytic_levels(self):
    n_levels = 10
    p_top = 10.0
    p0 = 1000.0
    coords = hybrid_coordinates.HybridCoordinates.analytic_levels(
        n_levels=n_levels, p_top=p_top, p0=p0
    )
    self.assertLen(coords.a_boundaries, n_levels + 1)
    self.assertLen(coords.b_boundaries, n_levels + 1)
    # Check boundary conditions
    self.assertAlmostEqual(coords.a_boundaries[0], p_top)
    self.assertAlmostEqual(coords.b_boundaries[0], 0.0)
    self.assertAlmostEqual(coords.a_boundaries[-1], 0.0)
    self.assertAlmostEqual(coords.b_boundaries[-1], 1.0)

  def test_from_sigma_levels(self):
    sigma_coords = sigma_coordinates.SigmaCoordinates.equidistant(10)
    hybrid_coords = hybrid_coordinates.HybridCoordinates.from_sigma_levels(
        sigma_coords
    )
    np.testing.assert_allclose(hybrid_coords.a_boundaries, 0.0)
    np.testing.assert_allclose(
        hybrid_coords.b_boundaries, sigma_coords.boundaries
    )


if __name__ == '__main__':
  absltest.main()
