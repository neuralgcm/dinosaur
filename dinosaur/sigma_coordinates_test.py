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
from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
import numpy as np


def _broadcast(*args):
  """Reshapes `args` so that they will broadcast over `len(args)` dimensions."""
  broadcasted = []
  for j, arg in enumerate(args):
    shape = [1] * len(args)
    shape[j] = -1
    broadcasted.append(arg.reshape(shape))
  return broadcasted


# pylint: disable=unbalanced-tuple-unpacking


def quadratic_function(sigma, lon, lat):
  """A test function for vertical differentiation and integration."""
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return sigma**2 * (1 + np.cos(lon) * np.cos(lat))


def quadratic_derivative(sigma, lon, lat):
  """The derivative of `quadratic_function` with respect to `sigma`."""
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return 2 * sigma * (1 + np.cos(lon) * np.cos(lat))


def quadratic_integral(sigma, lon, lat):
  """The indefinite integral of `quadratic_function` with respect to `sigma`."""
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return sigma**3 / 3 * (1 + np.cos(lon) * np.cos(lat))


def exponential_function(sigma, lon, lat):
  """A test function for vertical differentiation and integration."""
  sigma, lon, lat = _broadcast(sigma, lon, lat)
  return np.exp(sigma) * np.cos(lon) * np.sin(lat)


exponential_derivative = exponential_integral = exponential_function

# pylint: enable=unbalanced-tuple-unpacking


def test_cases():
  # Note that the test cases use far more layers that we will use in practice.
  # We do this to test numerical accuracy against closed form derivatives and
  # integrals.
  return (
      dict(
          testcase_name='Quadratic',
          test_function=quadratic_function,
          derivative_function=quadratic_derivative,
          integral_function=quadratic_integral,
          layers=np.array([10, 20, 40, 80, 160, 320]),
          grid_resolution=8,
      ),
      dict(
          testcase_name='Exponential',
          test_function=exponential_function,
          derivative_function=exponential_derivative,
          integral_function=exponential_integral,
          layers=np.array([10, 20, 40, 80, 160]),
          grid_resolution=16,
      ),
  )


def _test_error_scaling(layers, errors, error_scaling):
  """Checks that `errors` scales with `layers` according to `error_scaling`."""
  log_error_ratios = np.diff(np.log(errors))
  log_expected_ratios = np.diff(np.log(error_scaling(layers)))
  np.testing.assert_allclose(log_error_ratios, log_expected_ratios, atol=0.075)


class SigmaCoordinatesTest(parameterized.TestCase):

  @parameterized.parameters(
      ([0.0, 0.5, 1.0],),
      ((0.0, 0.5, 1.0),),
      (np.array([0.0, 0.5, 1.0]),),
  )
  def test_initialization_casting(self, boundaries):
    coordinates = sigma_coordinates.SigmaCoordinates(boundaries)
    self.assertIsInstance(coordinates.boundaries, np.ndarray)

    with self.subTest('asdict'):
      self.assertIsInstance(coordinates.asdict()['boundaries'], list)
      np.testing.assert_array_equal(
          coordinates.asdict()['boundaries'], boundaries
      )

  def test_initialization_raises(self):
    with self.subTest('end values'):
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          'Expected boundaries[0] = 0, boundaries[-1] = 1, '
          'got boundaries = [0.2 0.5 1. ]',
      ):
        sigma_coordinates.SigmaCoordinates([0.2, 0.5, 1])
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          'Expected boundaries[0] = 0, boundaries[-1] = 1, '
          'got boundaries = [0.  0.5 0.9]',
      ):
        sigma_coordinates.SigmaCoordinates([0.0, 0.5, 0.9])
    with self.subTest('increasing'):
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          'Expected `boundaries` to be monotonically increasing, '
          'got boundaries = [0.  0.5 0.5 1. ]',
      ):
        sigma_coordinates.SigmaCoordinates([0.0, 0.5, 0.5, 1])

  def test_from_centers(self):
    centers = np.array([0.1, 0.35, 0.75], dtype=np.float32)
    coordinates = sigma_coordinates.SigmaCoordinates.from_centers(centers)
    expected_boundaries = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(coordinates.boundaries, expected_boundaries)
    np.testing.assert_array_equal(coordinates.centers, centers)

  @parameterized.named_parameters(*test_cases())
  def test_centered_difference(
      self, test_function, derivative_function, layers, grid_resolution, **_
  ):
    """Tests `centered_difference` against the closed form derivative."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = grid.nodal_axes
    # Note that we only test accuracy for derivatives at the finest resolution.
    coordinates = sigma_coordinates.SigmaCoordinates.equidistant(layers[-1])
    centers = coordinates.centers
    boundaries = coordinates.internal_boundaries
    x = test_function(centers, lon, lat)
    expected_derivative = derivative_function(boundaries, lon, lat)
    computed_derivative = sigma_coordinates.centered_difference(x, coordinates)
    np.testing.assert_allclose(
        expected_derivative, computed_derivative, atol=1e-3
    )

  @parameterized.named_parameters(*test_cases())
  def test_cumulative_sigma_integral_downward(
      self, test_function, integral_function, layers, grid_resolution, **_
  ):
    """Tests `sigma_integral` in the downward direction."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = grid.nodal_axes
    total_errors = []
    for nlayers in layers:
      coordinates = sigma_coordinates.SigmaCoordinates.equidistant(nlayers)
      centers = coordinates.centers
      boundaries = coordinates.boundaries
      x = test_function(centers, lon, lat)
      indefinite_integral = integral_function(boundaries, lon, lat)
      expected_integral = indefinite_integral[1:] - indefinite_integral[0]
      computed_integral = sigma_coordinates.cumulative_sigma_integral(
          x, coordinates, cumsum_method='jax'
      )
      computed_integral_all = sigma_coordinates.sigma_integral(
          x, coordinates, keepdims=False
      )
      np.testing.assert_allclose(
          computed_integral[-1], computed_integral_all, atol=1e-6
      )
      # To test convergence, we compute the error in the integral at the
      # "bottom" layer.
      total_errors.append(
          np.abs(expected_integral[-1] - computed_integral[-1]).max()
      )
    with self.subTest('Convergence'):
      # Since we use a midpoint method, we expect errors to scale with
      # 1 / layers².
      error_scaling = lambda l: 1 / l**2
      _test_error_scaling(layers, total_errors, error_scaling)
    with self.subTest('Accuracy'):
      # Tests that the integral is accurate at the highest resolution.
      np.testing.assert_allclose(
          expected_integral[-1], computed_integral[-1], atol=1e-2
      )

  @parameterized.named_parameters(*test_cases())
  def test_cumulative_sigma_integral_upward(
      self, test_function, integral_function, layers, grid_resolution, **_
  ):
    """Tests `sigma_integral` in the upward direction."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = grid.nodal_axes
    total_errors = []
    for nlayers in layers:
      coordinates = sigma_coordinates.SigmaCoordinates.equidistant(nlayers)
      centers = coordinates.centers
      boundaries = coordinates.boundaries
      x = test_function(centers, lon, lat)
      indefinite_integral = integral_function(boundaries, lon, lat)
      expected_integral = -(indefinite_integral[:-1] - indefinite_integral[-1])
      computed_integral = sigma_coordinates.cumulative_sigma_integral(
          x, coordinates, downward=False, cumsum_method='jax'
      )
      computed_integral_all = sigma_coordinates.sigma_integral(
          x, coordinates, keepdims=False
      )
      np.testing.assert_allclose(
          computed_integral[0], computed_integral_all, atol=1e-6
      )
      # To test convergence, we compute the error in the integral at the
      # "top" layer.
      total_errors.append(
          np.abs(expected_integral[0] - computed_integral[0]).max()
      )
    with self.subTest('Convergence'):
      # Since we use a midpoint method in  we expect errors to scale with
      # 1 / layers².
      error_scaling = lambda l: 1 / l**2
      _test_error_scaling(layers, total_errors, error_scaling)
    with self.subTest('Accuracy'):
      # Tests that the integral is accurate at the highest resolution.
      np.testing.assert_allclose(
          expected_integral[0], computed_integral[0], atol=1e-2
      )

  @parameterized.named_parameters(*test_cases())
  def test_log_sigma_integral_downward(
      self, test_function, integral_function, layers, grid_resolution, **_
  ):
    """Tests `cumulative_log_sigma_integral` in the downward direction."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = grid.nodal_axes
    total_errors = []
    for nlayers in layers:
      coordinates = sigma_coordinates.SigmaCoordinates.equidistant(nlayers)
      centers = coordinates.centers
      broadcasted_centers = centers[:, np.newaxis, np.newaxis]
      # We integrate ∫f(x) 𝜎 d(log𝜎) =  ∫f(x) d𝜎
      x = test_function(centers, lon, lat) * broadcasted_centers
      indefinite_integral = integral_function(centers, lon, lat)
      integral_boundary = integral_function(np.zeros(1), lon, lat)
      expected_integral = indefinite_integral - integral_boundary
      computed_integral = sigma_coordinates.cumulative_log_sigma_integral(
          x, coordinates
      )
      # To test convergence, we compute the error in the integral at the
      # "bottom" layer.
      total_errors.append(
          np.abs(expected_integral[-1] - computed_integral[-1]).max()
      )
    with self.subTest('Convergence'):
      # Since we use a trapezoidal method in log space, we expect errors to
      # scale with the inverse square of the spacing in log space. Note that if
      # the boundaries were equidistant in log space, this would be equivalent
      # to the usual 1 / layers² scaling.
      def error_scaling(layers):
        expected_scaling = []
        for l in layers:
          centers = sigma_coordinates.SigmaCoordinates.equidistant(l).centers
          log_space_widths = np.diff(np.log(centers))
          expected_scaling.append(np.square(log_space_widths).mean())
        return np.array(expected_scaling)

      _test_error_scaling(layers, total_errors, error_scaling)
    with self.subTest('Accuracy'):
      # Tests that the integral is accurate at the highest resolution.
      np.testing.assert_allclose(
          expected_integral[-1], computed_integral[-1], atol=1e-2
      )

  @parameterized.named_parameters(*test_cases())
  def test_log_sigma_integral_upward(
      self, test_function, integral_function, layers, grid_resolution, **_
  ):
    """Tests `cumulative_log_sigma_integral` in the upward direction."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = grid.nodal_axes
    total_errors = []
    for nlayers in layers:
      coordinates = sigma_coordinates.SigmaCoordinates.equidistant(nlayers)
      centers = coordinates.centers
      broadcasted_centers = centers[:, np.newaxis, np.newaxis]
      # We integrate ∫f(x) 𝜎 d(log𝜎) =  ∫f(x) d𝜎
      x = test_function(centers, lon, lat) * broadcasted_centers
      indefinite_integral = integral_function(centers, lon, lat)
      integral_boundary = integral_function(
          np.array([coordinates.centers[-1]]), lon, lat
      )
      expected_integral = -(indefinite_integral - integral_boundary)
      computed_integral = sigma_coordinates.cumulative_log_sigma_integral(
          x, coordinates, downward=False
      )
      # To test convergence, we compute the error in the integral at the
      # "bottom" layer.
      total_errors.append(
          np.abs(expected_integral[0] - computed_integral[0]).max()
      )
    print(f'TOTAL ERROR: {total_errors}')
    with self.subTest('Convergence'):
      # Since we use a trapezoidal method in log space, we expect errors to
      # scale with the inverse square of the spacing in log space. Note that if
      # the boundaries were equidistant in log space, this would be equivalent
      # to the usual 1 / layers² scaling.
      def error_scaling(layers):
        expected_scaling = []
        for l in layers:
          centers = sigma_coordinates.SigmaCoordinates.equidistant(l).centers
          log_space_widths = np.diff(np.log(centers))
          expected_scaling.append(np.square(log_space_widths).mean())
        return np.array(expected_scaling)

      _test_error_scaling(layers, total_errors, error_scaling)
    with self.subTest('Accuracy'):
      # Tests that the integral is accurate at the highest resolution.
      np.testing.assert_allclose(
          expected_integral[0], computed_integral[0], atol=1e-2
      )

  @parameterized.named_parameters(*test_cases())
  def test_vertical_advection_helper(
      self, test_function, derivative_function, layers, grid_resolution, **_
  ):
    """Tests `sigma_coordinates.centered_vertical_advection` helper function."""
    grid = spherical_harmonic.Grid.with_wavenumbers(grid_resolution)
    lon, lat = grid.nodal_axes
    # Note that we only test accuracy for derivatives at the finest resolution.
    coordinates = sigma_coordinates.SigmaCoordinates.equidistant(layers[-1])
    centers = coordinates.centers
    boundaries = coordinates.internal_boundaries
    velocity_fn = test_function
    derivative_fn = derivative_function

    x = test_function(centers, lon, lat)
    dx_dsigma = derivative_fn(centers, lon, lat)
    w = velocity_fn(boundaries, lon, lat)

    with self.subTest('Default boundary conditions, centered'):
      # By default boundary conditions are set to zeros.
      w_boundary_values = np.zeros_like(w[:2, ...])
      dx_dsigma_boundary_values = np.zeros_like(dx_dsigma[:2, ...])
      # This does not affect the bulk, but modifies boundary to:
      # -(w[inner] * ∂x/∂𝜎[inner] + w[ghost] * ∂x/∂𝜎[ghost]) / 2
      boundaries = -0.5 * (
          w[[0, -1], ...] * derivative_fn(boundaries, lon, lat)[[0, -1], ...]
          + w_boundary_values * dx_dsigma_boundary_values
      )
      expected = -dx_dsigma * velocity_fn(centers, lon, lat)
      expected[[0, -1], ...] = boundaries
      actual = sigma_coordinates.centered_vertical_advection(w, x, coordinates)
      np.testing.assert_allclose(actual, expected, atol=1e-3)

    with self.subTest('Default boundary conditions, upwind'):
      actual = sigma_coordinates.upwind_vertical_advection(w, x, coordinates)
      # Upwinding is only 1st order accurate (vs 2nd order centered advection)
      np.testing.assert_allclose(actual[1:-1], expected[1:-1], atol=5e-2)

    with self.subTest('Custom boundary conditions'):
      # Tests that boundary values can be provided.
      w_boundary_values = velocity_fn(coordinates.boundaries[[0, -1]], lon, lat)
      w_boundary_values = (
          w_boundary_values[[0], ...],
          w_boundary_values[[1], ...],
      )
      dx_dsigma_boundary_values = derivative_fn(
          coordinates.boundaries[[0, -1]], lon, lat
      )
      dx_dsigma_boundary_values = (
          dx_dsigma_boundary_values[[0], ...],
          dx_dsigma_boundary_values[[1], ...],
      )
      # with these values boundaries should remain unmodified.
      expected = -dx_dsigma * velocity_fn(centers, lon, lat)
      actual = sigma_coordinates.centered_vertical_advection(
          w,
          x,
          coordinates,
          w_boundary_values=w_boundary_values,
          dx_dsigma_boundary_values=dx_dsigma_boundary_values,
      )
      np.testing.assert_allclose(actual, expected, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
