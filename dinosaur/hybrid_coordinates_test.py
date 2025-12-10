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
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import units
import jax.numpy as jnp
import numpy as np


def _broadcast(*args):
  """Reshapes `args` so that they will broadcast over `len(args)` dimensions."""
  broadcasted = []
  for j, arg in enumerate(args):
    shape = [1] * len(args)
    shape[j] = -1
    broadcasted.append(np.asarray(arg).reshape(shape))
  return broadcasted


# pylint: disable=unbalanced-tuple-unpacking


def quadratic_function(p, lon, lat):
  """A test function for vertical differentiation and integration."""
  return p**2 * (1 + np.cos(lon) * np.cos(lat))


def quadratic_integral(p, lon, lat):
  """The indefinite integral of `quadratic_function` with respect to `p`."""
  return p**3 / 3 * (1 + np.cos(lon) * np.cos(lat))


def exponential_function(p, lon, lat):
  """A test function for vertical differentiation and integration."""
  return np.exp(p / 1000) * np.cos(lon) * np.sin(lat)


def exponential_integral(p, lon, lat):
  """The indefinite integral of `exponential_function` with respect to `p`."""
  return 1000 * np.exp(p / 1000) * np.cos(lon) * np.sin(lat)


def quadratic_sigma_function(sigma, lon, lat):
  """A test function for vertical differentiation w.r.t. sigma."""
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return sigma**2 * (1 + np.cos(lon) * np.cos(lat))


def quadratic_sigma_derivative(sigma, lon, lat):
  """The derivative of `quadratic_sigma_function` with respect to `sigma`."""
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return 2 * sigma * (1 + np.cos(lon) * np.cos(lat))


def quadratic_sigma_integral(sigma, lon, lat):
  """The indefinite integral of `quadratic_sigma_function` wrt `sigma`."""
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return sigma**3 / 3 * (1 + np.cos(lon) * np.cos(lat))


def exponential_sigma_function(sigma, lon, lat):
  """A test function for vertical differentiation w.r.t. sigma."""
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return np.exp(sigma) * np.cos(lon) * np.sin(lat)


def exponential_sigma_integral(sigma, lon, lat):
  """The indefinite integral of `exponential_sigma_function` wrt `sigma`."""
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return np.exp(sigma) * np.cos(lon) * np.sin(lat)

exponential_sigma_derivative = exponential_sigma_function


# pylint: enable=unbalanced-tuple-unpacking


def _sigma_test_cases():
  return (
      dict(
          testcase_name='QuadraticSigma',
          test_function=quadratic_sigma_function,
          derivative_function=quadratic_sigma_derivative,
          integral_function=quadratic_sigma_integral,
          layers=np.array([10, 20, 40, 80, 160, 320]),
          grid_resolution=8,
      ),
      dict(
          testcase_name='ExponentialSigma',
          test_function=exponential_sigma_function,
          derivative_function=exponential_sigma_derivative,
          integral_function=exponential_sigma_integral,
          layers=np.array([10, 20, 40, 80, 160]),
          grid_resolution=16,
      ),
  )


def _pressure_test_cases():
  return (
      dict(
          testcase_name='Quadratic',
          test_function=quadratic_function,
          integral_function=quadratic_integral,
          layers=np.array([10, 20, 40, 80, 160, 320]),
          grid_resolution=8,
      ),
      dict(
          testcase_name='Exponential',
          test_function=exponential_function,
          integral_function=exponential_integral,
          layers=np.array([10, 20, 40, 80, 160]),
          grid_resolution=16,
      ),
  )


def _test_error_scaling(layers, errors, error_scaling):
  """Checks that `errors` scales with `layers` according to `error_scaling`."""
  log_error_ratios = np.diff(np.log(errors))
  log_expected_ratios = np.diff(np.log(error_scaling(layers)))
  np.testing.assert_allclose(log_error_ratios, log_expected_ratios, atol=0.1)


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
    levels = hybrid_coordinates.HybridCoordinates.from_coefficients(
        a_coeffs, b_coeffs, p0
    )
    np.testing.assert_allclose(levels.a_boundaries, a_coeffs * p0)
    np.testing.assert_allclose(levels.b_boundaries, b_coeffs)

  def test_analytic_levels(self):
    n_levels = 10
    p_top = 10.0
    p0 = 1000.0
    levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        n_levels=n_levels, p_top=p_top, p0=p0
    )
    self.assertLen(levels.a_boundaries, n_levels + 1)
    self.assertLen(levels.b_boundaries, n_levels + 1)
    # Check boundary conditions
    self.assertAlmostEqual(levels.a_boundaries[0], p_top)
    self.assertAlmostEqual(levels.b_boundaries[0], 0.0)
    self.assertAlmostEqual(levels.a_boundaries[-1], 0.0)
    self.assertAlmostEqual(levels.b_boundaries[-1], 1.0)

  def test_from_sigma_levels(self):
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(10)
    hybrid_levels = hybrid_coordinates.HybridCoordinates.from_sigma_levels(
        sigma_levels
    )
    np.testing.assert_allclose(hybrid_levels.a_boundaries, 0.0)
    np.testing.assert_allclose(
        hybrid_levels.b_boundaries, sigma_levels.boundaries
    )

  @parameterized.named_parameters(*_pressure_test_cases())
  def test_cumulative_integral_over_pressure_downward(
      self, test_function, integral_function, layers, grid_resolution, **_
  ):
    """Tests `cumulative_integral_over_pressure` in the downward direction."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = np.meshgrid(grid.nodal_axes[0], grid.nodal_axes[1])
    surface_pressure = 1000.0 + 100 * np.cos(lon) * np.cos(lat)
    total_errors = []
    expected_integral, computed_integral = None, None  # make pytype happy.
    for nlayers in layers:
      levels = hybrid_coordinates.HybridCoordinates.analytic_levels(nlayers)
      centers = levels.pressure_centers(surface_pressure)
      boundaries = levels.pressure_boundaries(surface_pressure)
      x = test_function(centers, lon, lat)
      indefinite_integral = integral_function(boundaries, lon, lat)
      expected_integral = indefinite_integral[1:] - indefinite_integral[0]
      computed_integral = hybrid_coordinates.cumulative_integral_over_pressure(
          x, jnp.asarray(surface_pressure), levels
      )
      total_errors.append(
          np.abs(expected_integral[-1] - computed_integral[-1]).max()
      )
    with self.subTest('Convergence'):
      error_scaling = lambda l: 1 / l**2
      _test_error_scaling(layers, total_errors, error_scaling)
    with self.subTest('Accuracy'):
      np.testing.assert_allclose(
          expected_integral[-1], computed_integral[-1], rtol=1e-2
      )

  @parameterized.named_parameters(*_pressure_test_cases())
  def test_cumulative_integral_over_pressure_upward(
      self, test_function, integral_function, layers, grid_resolution, **_
  ):
    """Tests `cumulative_integral_over_pressure` in the upward direction."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = np.meshgrid(grid.nodal_axes[0], grid.nodal_axes[1])
    surface_pressure = 1000.0 + 100 * np.cos(lon) * np.cos(lat)
    total_errors = []
    expected_integral, computed_integral = None, None  # make pytype happy.
    for nlayers in layers:
      levels = hybrid_coordinates.HybridCoordinates.analytic_levels(nlayers)
      centers = levels.pressure_centers(surface_pressure)
      boundaries = levels.pressure_boundaries(surface_pressure)
      x = test_function(centers, lon, lat)
      indefinite_integral = integral_function(boundaries, lon, lat)
      expected_integral = -(indefinite_integral[:-1] - indefinite_integral[-1])
      computed_integral = hybrid_coordinates.cumulative_integral_over_pressure(
          x, jnp.asarray(surface_pressure), levels, downward=False
      )
      total_errors.append(
          np.abs(expected_integral[0] - computed_integral[0]).max()
      )
    with self.subTest('Convergence'):
      error_scaling = lambda l: 1 / l**2
      _test_error_scaling(layers, total_errors, error_scaling)
    with self.subTest('Accuracy'):
      np.testing.assert_allclose(
          expected_integral[0], computed_integral[0], rtol=1e-2
      )

  @parameterized.named_parameters(*_pressure_test_cases())
  def test_integral_over_pressure(
      self, test_function, integral_function, layers, grid_resolution, **_
  ):
    """Tests `integral_over_pressure` converges to correct values."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = np.meshgrid(grid.nodal_axes[0], grid.nodal_axes[1])
    surface_pressure = 1000.0 + 100 * np.cos(lon) * np.cos(lat)
    total_errors = []
    expected_integral, computed_integral = None, None  # make pytype happy.
    for nlayers in layers:
      levels = hybrid_coordinates.HybridCoordinates.analytic_levels(nlayers)
      centers = levels.pressure_centers(surface_pressure)
      boundaries = levels.pressure_boundaries(surface_pressure)
      x = test_function(centers, lon, lat)
      indefinite_integral = integral_function(boundaries, lon, lat)
      expected_integral = indefinite_integral[-1] - indefinite_integral[0]
      computed_integral = hybrid_coordinates.integral_over_pressure(
          x, jnp.asarray(surface_pressure), levels, keepdims=False
      )
      total_errors.append(
          np.abs(expected_integral - computed_integral).max()
      )
    with self.subTest('Convergence'):
      error_scaling = lambda l: 1 / l**2
      _test_error_scaling(layers, total_errors, error_scaling)
    with self.subTest('Accuracy'):
      np.testing.assert_allclose(
          expected_integral, computed_integral, rtol=1e-2
      )

  @parameterized.named_parameters(*_sigma_test_cases())
  def test_cumulative_integral_over_sigma(
      self, test_function, integral_function, layers, grid_resolution, **_
  ):
    """Tests `cumulative_integral_over_sigma` converges to correct values."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = grid.nodal_axes
    total_errors = []
    expected_integral, computed_integral = None, None  # make pytype happy.
    for nlayers in layers:
      levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
          nlayers, stretch_exponent=1.0, sigma_exponent=1.0)
      centers = levels.b_centers
      boundaries = levels.b_boundaries
      x = test_function(centers, lon, lat)
      indefinite_integral = integral_function(boundaries, lon, lat)
      expected_integral = indefinite_integral[1:] - indefinite_integral[0]
      computed_integral = hybrid_coordinates.cumulative_integral_over_sigma(
          x, levels
      )
      total_errors.append(
          np.abs(expected_integral[-1] - computed_integral[-1]).max()
      )
    with self.subTest('Convergence'):
      error_scaling = lambda l: 1 / l**2
      _test_error_scaling(layers, total_errors, error_scaling)
    with self.subTest('Accuracy'):
      np.testing.assert_allclose(
          expected_integral[-1], computed_integral[-1], rtol=1e-2
      )

  @parameterized.named_parameters(*_sigma_test_cases())
  def test_integral_over_sigma(
      self, test_function, integral_function, layers, grid_resolution, **_
  ):
    """Tests `integral_over_sigma` converges to correct values."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = grid.nodal_axes
    total_errors = []
    expected_integral, computed_integral = None, None  # make pytype happy.
    for nlayers in layers:
      levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
          nlayers, stretch_exponent=1.0, sigma_exponent=1.0)
      centers = levels.b_centers
      boundaries = levels.b_boundaries
      x = test_function(centers, lon, lat)
      indefinite_integral = integral_function(boundaries, lon, lat)
      expected_integral = indefinite_integral[-1] - indefinite_integral[0]
      computed_integral = hybrid_coordinates.integral_over_sigma(
          x, levels, keepdims=False
      )
      total_errors.append(
          np.abs(expected_integral - computed_integral).max()
      )
    with self.subTest('Convergence'):
      error_scaling = lambda l: 1 / l**2
      _test_error_scaling(layers, total_errors, error_scaling)
    with self.subTest('Accuracy'):
      np.testing.assert_allclose(
          expected_integral, computed_integral, rtol=1e-2
      )

  @parameterized.named_parameters(*_sigma_test_cases())
  def test_centered_difference(
      self, test_function, derivative_function, layers, grid_resolution, **_
  ):
    """Tests `centered_difference` against the closed form derivative."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = grid.nodal_axes
    # Note that we only test accuracy for derivatives at the finest resolution.
    levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        layers[-1], sigma_exponent=1.0, stretch_exponent=1.0
    )
    centers = levels.b_centers
    boundaries = levels.b_boundaries[1:-1]
    x = test_function(centers, lon, lat)
    expected_derivative = derivative_function(boundaries, lon, lat)
    computed_derivative = hybrid_coordinates.centered_difference(x, levels)
    np.testing.assert_allclose(
        expected_derivative, computed_derivative, atol=1e-3
    )


if __name__ == '__main__':
  absltest.main()
