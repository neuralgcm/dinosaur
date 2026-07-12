# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for dinosaur.semi_lagrangian."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import semi_lagrangian
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
import jax
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
  """Rotation matrix by `angle` about `axis` (Rodrigues formula)."""
  axis = np.asarray(axis) / np.linalg.norm(axis)
  k = np.array([
      [0, -axis[2], axis[1]],
      [axis[2], 0, -axis[0]],
      [-axis[1], axis[0], 0],
  ])
  return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def solid_body_winds(
    grid: spherical_harmonic.Grid, axis: np.ndarray, omega: float
) -> tuple[np.ndarray, np.ndarray]:
  """True (u, v) winds for solid-body rotation about `axis`."""
  lon, sin_lat = grid.nodal_mesh
  r = np.stack(semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat))
  velocity = omega * np.cross(np.asarray(axis), r, axisa=0, axisb=0, axisc=0)
  u, v = semi_lagrangian.tangent_wind(velocity, lon, sin_lat)
  return np.asarray(u), np.asarray(v)


class X64TestCase(parameterized.TestCase):
  """Base class running tests in float64 for convergence measurements."""

  def setUp(self):
    super().setUp()
    self._x64_was_enabled = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)

  def tearDown(self):
    jax.config.update('jax_enable_x64', self._x64_was_enabled)
    super().tearDown()


class GeometryTest(parameterized.TestCase):

  def test_cartesian_round_trip(self):
    rng = np.random.RandomState(0)
    lon = rng.uniform(0, 2 * np.pi, size=100)
    sin_lat = np.sin(rng.uniform(-np.pi / 2, np.pi / 2, size=100))
    r = semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat)
    np.testing.assert_allclose(
        np.linalg.norm(r, axis=0), np.ones(100), atol=1e-6
    )
    lon2, sin_lat2 = semi_lagrangian.cartesian_to_lon_sin_lat(r)
    np.testing.assert_allclose(sin_lat2, sin_lat, atol=1e-6)
    lon_error = np.abs(np.asarray(lon2) - lon)
    lon_error = np.minimum(lon_error, 2 * np.pi - lon_error)
    np.testing.assert_allclose(lon_error, np.zeros(100), atol=1e-5)

  def test_wind_round_trip(self):
    rng = np.random.RandomState(1)
    lon = rng.uniform(0, 2 * np.pi, size=50)
    sin_lat = np.sin(rng.uniform(-1.5, 1.5, size=50))
    u = rng.normal(size=50)
    v = rng.normal(size=50)
    w = semi_lagrangian.cartesian_wind(u, v, lon, sin_lat)
    # the Cartesian wind is tangent: w · r = 0.
    r = semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat)
    np.testing.assert_allclose((w * r).sum(0), np.zeros(50), atol=1e-6)
    u2, v2 = semi_lagrangian.tangent_wind(w, lon, sin_lat)
    np.testing.assert_allclose(u2, u, atol=1e-6)
    np.testing.assert_allclose(v2, v, atol=1e-6)

  def test_parallel_transport_is_identity_for_same_point(self):
    rng = np.random.RandomState(2)
    lon = rng.uniform(0, 2 * np.pi, size=20)
    sin_lat = np.sin(rng.uniform(-1.5, 1.5, size=20))
    r = semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat)
    w = semi_lagrangian.cartesian_wind(
        rng.normal(size=20), rng.normal(size=20), lon, sin_lat
    )
    np.testing.assert_allclose(
        semi_lagrangian.parallel_transport(w, r, r), w, atol=1e-6
    )

  def test_parallel_transport_properties(self):
    rng = np.random.RandomState(3)
    n = 50
    lon_a = rng.uniform(0, 2 * np.pi, size=n)
    sin_lat_a = np.sin(rng.uniform(-1.5, 1.5, size=n))
    lon_b = lon_a + rng.uniform(-0.3, 0.3, size=n)
    sin_lat_b = np.sin(np.arcsin(sin_lat_a) + rng.uniform(-0.2, 0.2, size=n))
    r_a = semi_lagrangian.lon_lat_to_cartesian(lon_a, sin_lat_a)
    r_b = semi_lagrangian.lon_lat_to_cartesian(lon_b, sin_lat_b)
    w = semi_lagrangian.cartesian_wind(
        rng.normal(size=n), rng.normal(size=n), lon_a, sin_lat_a
    )
    w_b = semi_lagrangian.parallel_transport(w, r_a, r_b)
    with self.subTest('preserves norm'):
      np.testing.assert_allclose(
          np.linalg.norm(w_b, axis=0), np.linalg.norm(w, axis=0), atol=1e-5
      )
    with self.subTest('tangent at target'):
      np.testing.assert_allclose((w_b * r_b).sum(0), np.zeros(n), atol=1e-5)
    with self.subTest('maps source to target'):
      np.testing.assert_allclose(
          semi_lagrangian.parallel_transport(np.asarray(r_a), r_a, r_b),
          r_b,
          atol=1e-5,
      )

  def test_parallel_transport_along_equator(self):
    # Transport from (lon=0) to (lon=π/2) along the equator: the north vector
    # is unchanged and the east vector remains east.
    r_a = np.array([1.0, 0.0, 0.0])[:, np.newaxis]
    r_b = np.array([0.0, 1.0, 0.0])[:, np.newaxis]
    north = np.array([0.0, 0.0, 1.0])[:, np.newaxis]
    east_a = np.array([0.0, 1.0, 0.0])[:, np.newaxis]
    east_b = np.array([-1.0, 0.0, 0.0])[:, np.newaxis]
    np.testing.assert_allclose(
        semi_lagrangian.parallel_transport(north, r_a, r_b), north, atol=1e-6
    )
    np.testing.assert_allclose(
        semi_lagrangian.parallel_transport(east_a, r_a, r_b), east_b, atol=1e-6
    )


class InterpolatorTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(order='linear', monotone=False),
      dict(order='linear', monotone=True),
      dict(order='cubic', monotone=False),
      dict(order='cubic', monotone=True),
  )
  def test_exact_for_constants(self, order, monotone):
    grid = spherical_harmonic.Grid.T21()
    interpolator = semi_lagrangian.GridInterpolator(grid, order, monotone)
    field = 2.5 * jnp.ones(grid.nodal_shape)
    rng = np.random.RandomState(0)
    lon = rng.uniform(0, 2 * np.pi, size=200)
    sin_lat = np.sin(rng.uniform(-np.pi / 2, np.pi / 2, size=200))
    values = interpolator(field, jnp.asarray(lon), jnp.asarray(sin_lat))
    np.testing.assert_allclose(values, 2.5 * np.ones(200), rtol=1e-6)

  @parameterized.parameters(
      dict(order='linear', degree=1),
      dict(order='cubic', degree=3),
  )
  def test_exact_for_polynomials(self, order, degree):
    """Interpolation reproduces polynomials in (lon, lat) up to stencil order."""
    grid = spherical_harmonic.Grid.T21()
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    phi_mesh = np.arcsin(sin_lat_mesh)

    def poly(lon, phi):
      return sum(
          (0.3 + 0.1 * k) * lon**k + (0.5 - 0.2 * k) * phi**k
          for k in range(degree + 1)
      ) + lon**degree * phi**degree * 0.05

    field = jnp.asarray(poly(lon_mesh, phi_mesh))
    interpolator = semi_lagrangian.GridInterpolator(grid, order)
    # Interior points, away from the longitude wrap (where the unwrapped
    # coordinate makes lon**k non-periodic) and away from the pole halo.
    rng = np.random.RandomState(1)
    lon = rng.uniform(1.0, 5.0, size=300)
    phi = rng.uniform(-1.2, 1.2, size=300)
    values = interpolator(field, jnp.asarray(lon), jnp.asarray(np.sin(phi)))
    np.testing.assert_allclose(values, poly(lon, phi), rtol=2e-5, atol=2e-5)

  def test_longitude_wrap_periodicity(self):
    grid = spherical_harmonic.Grid.T21()
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    field = jnp.asarray(np.sin(2 * lon_mesh) * (1 - sin_lat_mesh**2))
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')
    rng = np.random.RandomState(2)
    # points straddling the wrap at lon = 0 == 2π.
    lon = rng.uniform(-0.2, 0.2, size=100)
    sin_lat = np.sin(rng.uniform(-1.0, 1.0, size=100))
    values = interpolator(field, jnp.asarray(lon), jnp.asarray(sin_lat))
    values_shifted = interpolator(
        field, jnp.asarray(lon + 2 * np.pi), jnp.asarray(sin_lat)
    )
    with self.subTest('periodic'):
      # not bit-identical: weights are computed with unwrapped coordinates,
      # so adding 2π perturbs them at float32 rounding level.
      np.testing.assert_allclose(values, values_shifted, atol=1e-5)
    with self.subTest('accurate across the wrap'):
      expected = np.sin(2 * lon) * (1 - sin_lat**2)
      np.testing.assert_allclose(values, expected, atol=1e-4)

  def test_cross_pole_interpolation(self):
    """Interpolation beyond the last ring uses correct cross-pole halos."""
    grid = spherical_harmonic.Grid.T42()
    lon_mesh, sin_lat_mesh = grid.nodal_mesh

    def smooth(r):
      # low-degree polynomial in Cartesian coordinates = band-limited field.
      return 1.0 + r[0] + 2.0 * r[1] * r[2] + r[2] ** 2 - 0.5 * r[0] * r[1]

    field = jnp.asarray(
        smooth(np.stack(semi_lagrangian.lon_lat_to_cartesian(
            lon_mesh, sin_lat_mesh)))
    )
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')
    rng = np.random.RandomState(3)
    # Points at and around both poles, poleward of the last grid ring.
    max_phi = np.arcsin(grid.nodal_axes[1]).max()
    phi = np.concatenate([
        rng.uniform(max_phi, np.pi / 2, size=100),
        rng.uniform(-np.pi / 2, -max_phi, size=100),
        [np.pi / 2, -np.pi / 2],
    ])
    lon = rng.uniform(0, 2 * np.pi, size=phi.size)
    expected = smooth(
        np.stack(semi_lagrangian.lon_lat_to_cartesian(lon, np.sin(phi)))
    )
    values = interpolator(field, jnp.asarray(lon), jnp.asarray(np.sin(phi)))
    np.testing.assert_allclose(values, expected, atol=1e-4)

  def test_batched_interpolation(self):
    grid = spherical_harmonic.Grid.T21()
    rng = np.random.RandomState(4)
    fields = jnp.asarray(rng.normal(size=(3,) + grid.nodal_shape))
    lon = rng.uniform(0, 2 * np.pi, size=(3, 10))
    sin_lat = np.sin(rng.uniform(-1.4, 1.4, size=(3, 10)))
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')
    batched = interpolator(fields, jnp.asarray(lon), jnp.asarray(sin_lat))
    for k in range(3):
      single = interpolator(
          fields[k], jnp.asarray(lon[k]), jnp.asarray(sin_lat[k])
      )
      np.testing.assert_allclose(batched[k], single, atol=1e-6)

  def test_batched_interpolation_shape_mismatch_raises(self):
    grid = spherical_harmonic.Grid.T21()
    fields = jnp.zeros((3,) + grid.nodal_shape)
    points = jnp.zeros((2, 10))
    interpolator = semi_lagrangian.GridInterpolator(grid)
    with self.assertRaisesRegex(ValueError, 'batched interpolation'):
      interpolator(fields, points, points)

  def test_monotone_limiter(self):
    """Cubic interpolation overshoots on sharp fields; the limiter does not."""
    grid = spherical_harmonic.Grid.T21()
    _, sin_lat_mesh = grid.nodal_mesh
    field = jnp.asarray((sin_lat_mesh > 0).astype(np.float64))  # 0/1 step
    rng = np.random.RandomState(5)
    lon = jnp.asarray(rng.uniform(0, 2 * np.pi, size=1000))
    sin_lat = jnp.asarray(np.sin(rng.uniform(-0.3, 0.3, size=1000)))
    unlimited = semi_lagrangian.GridInterpolator(grid, 'cubic', False)(
        field, lon, sin_lat
    )
    limited = semi_lagrangian.GridInterpolator(grid, 'cubic', True)(
        field, lon, sin_lat
    )
    with self.subTest('cubic overshoots without limiter'):
      self.assertLess(np.min(np.asarray(unlimited)), -1e-3)
      self.assertGreater(np.max(np.asarray(unlimited)), 1 + 1e-3)
    with self.subTest('limiter prevents new extrema'):
      self.assertGreaterEqual(np.min(np.asarray(limited)), 0.0)
      self.assertLessEqual(np.max(np.asarray(limited)), 1.0)

  def test_interpolate_levels_linear_in_sigma(self):
    grid = spherical_harmonic.Grid.T21()
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(8)
    centers = vertical.centers
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    horizontal = 1 + 0.3 * np.asarray(sin_lat_mesh)

    field = jnp.asarray(
        (2.0 + 3.0 * centers)[:, np.newaxis, np.newaxis] * horizontal
    )
    rng = np.random.RandomState(6)
    shape = (8, 12)
    lon = jnp.asarray(rng.uniform(0, 2 * np.pi, size=shape))
    sin_lat = jnp.asarray(np.sin(rng.uniform(-1.4, 1.4, size=shape)))
    # includes out-of-range sigma to exercise constant extrapolation.
    sigma = jnp.asarray(rng.uniform(-0.2, 1.2, size=shape))
    values = semi_lagrangian.interpolate_levels(
        field, grid, lon, sin_lat, sigma, sigma_nodes=centers, order='linear'
    )
    sigma_clipped = np.clip(sigma, centers[0], centers[-1])
    horizontal_at_points = 1 + 0.3 * np.asarray(sin_lat)
    expected = (2.0 + 3.0 * sigma_clipped) * horizontal_at_points
    # bilinear horizontal interpolation of a linear-in-sin_lat field is not
    # exact in phi, so allow a modest tolerance.
    np.testing.assert_allclose(values, expected, rtol=1e-3)

  def test_interpolate_levels_monotone(self):
    grid = spherical_harmonic.Grid.T21()
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(6)
    rng = np.random.RandomState(7)
    field = jnp.asarray(
        np.maximum(rng.normal(size=(6,) + grid.nodal_shape), 0.0)
    )
    shape = (6, 20)
    lon = jnp.asarray(rng.uniform(0, 2 * np.pi, size=shape))
    sin_lat = jnp.asarray(np.sin(rng.uniform(-1.5, 1.5, size=shape)))
    sigma = jnp.asarray(rng.uniform(0, 1, size=shape))
    values = semi_lagrangian.interpolate_levels(
        field,
        grid,
        lon,
        sin_lat,
        sigma,
        sigma_nodes=vertical.centers,
        order='cubic',
        monotone=True,
    )
    self.assertGreaterEqual(np.min(np.asarray(values)), 0.0)


class DeparturePointsTest(X64TestCase):

  def test_zero_wind_departure_is_arrival(self):
    grid = spherical_harmonic.Grid.T21()
    zeros = jnp.zeros(grid.nodal_shape)
    departure = semi_lagrangian.horizontal_departure_points(
        zeros, zeros, grid, dt=0.5
    )
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    np.testing.assert_allclose(departure.lon, lon_mesh, atol=1e-12)
    np.testing.assert_allclose(departure.sin_lat, sin_lat_mesh, atol=1e-12)
    self.assertIsNone(departure.sigma)

  def test_zero_wind_departure_is_arrival_3d(self):
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(4),
    )
    grid = coords.horizontal
    zeros = jnp.zeros((4,) + grid.nodal_shape)
    sigma_dot = jnp.zeros((5,) + grid.nodal_shape)
    departure = semi_lagrangian.departure_points_3d(
        zeros, zeros, sigma_dot, coords, dt=0.5
    )
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    np.testing.assert_allclose(
        departure.lon, np.broadcast_to(lon_mesh, departure.lon.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        departure.sigma,
        np.broadcast_to(
            coords.vertical.centers[:, np.newaxis, np.newaxis],
            departure.sigma.shape,
        ),
        atol=1e-12,
    )

  def test_solid_body_departure_points(self):
    """Departure points match the analytic solid-body solution."""
    grid = spherical_harmonic.Grid.T85()
    omega = 1.0
    u, v = solid_body_winds(grid, axis=[0, 0, 1], omega=omega)
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    dt = 0.2
    departure = semi_lagrangian.horizontal_departure_points(
        jnp.asarray(u), jnp.asarray(v), grid, dt=dt
    )
    r_exact = rotation_matrix([0, 0, 1], -omega * dt) @ np.stack(
        semi_lagrangian.lon_lat_to_cartesian(lon_mesh, sin_lat_mesh)
    ).reshape(3, -1)
    error = np.linalg.norm(
        np.asarray(departure.cartesian).reshape(3, -1) - r_exact, axis=0
    )
    self.assertLess(error.max(), 1e-3)

  def test_solid_body_departure_convergence(self):
    """Departure-point error is third order in dt (2nd-order trajectories)."""
    grid = spherical_harmonic.Grid.T85()
    omega = 1.0
    u, v = solid_body_winds(grid, axis=[0, 0, 1], omega=omega)
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    r_arrival = np.stack(
        semi_lagrangian.lon_lat_to_cartesian(lon_mesh, sin_lat_mesh)
    ).reshape(3, -1)

    def max_error(dt):
      departure = semi_lagrangian.horizontal_departure_points(
          jnp.asarray(u), jnp.asarray(v), grid, dt=dt
      )
      r_exact = rotation_matrix([0, 0, 1], -omega * dt) @ r_arrival
      return np.linalg.norm(
          np.asarray(departure.cartesian).reshape(3, -1) - r_exact, axis=0
      ).max()

    errors = [max_error(dt) for dt in [0.4, 0.2, 0.1]]
    ratios = [errors[i] / errors[i + 1] for i in range(2)]
    # local error of a single step is O(dt³): halving dt reduces it ~8x.
    for ratio in ratios:
      self.assertGreater(ratio, 5.0)
      self.assertLess(ratio, 12.0)

  def test_solid_body_departure_over_pole(self):
    """Trajectories crossing the poles are handled by Cartesian iteration."""
    grid = spherical_harmonic.Grid.T85()
    omega = 1.0
    u, v = solid_body_winds(grid, axis=[0, 1, 0], omega=omega)
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    dt = 0.2
    departure = semi_lagrangian.horizontal_departure_points(
        jnp.asarray(u), jnp.asarray(v), grid, dt=dt
    )
    r_exact = rotation_matrix([0, 1, 0], -omega * dt) @ np.stack(
        semi_lagrangian.lon_lat_to_cartesian(lon_mesh, sin_lat_mesh)
    ).reshape(3, -1)
    error = np.linalg.norm(
        np.asarray(departure.cartesian).reshape(3, -1) - r_exact, axis=0
    )
    self.assertLess(error.max(), 1e-3)

  def test_vertical_departure_points(self):
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(8),
    )
    grid = coords.horizontal
    layers = coords.vertical.layers
    zeros = jnp.zeros((layers,) + grid.nodal_shape)
    rate = 0.04
    sigma_dot = rate * jnp.ones((layers + 1,) + grid.nodal_shape)
    dt = 2.0
    departure = semi_lagrangian.departure_points_3d(
        zeros, zeros, sigma_dot, coords, dt=dt
    )
    centers = coords.vertical.centers
    expected = np.clip(
        centers[:, np.newaxis, np.newaxis] - dt * rate,
        centers[0],
        centers[-1],
    )
    np.testing.assert_allclose(
        departure.sigma,
        np.broadcast_to(expected, departure.sigma.shape),
        atol=1e-12,
    )

  def test_3d_departure_matches_2d_for_horizontal_flow(self):
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T42(),
        sigma_coordinates.SigmaCoordinates.equidistant(4),
    )
    grid = coords.horizontal
    layers = coords.vertical.layers
    u2, v2 = solid_body_winds(grid, axis=[1, 1, 1], omega=0.8)
    u = jnp.broadcast_to(u2, (layers,) + grid.nodal_shape)
    v = jnp.broadcast_to(v2, (layers,) + grid.nodal_shape)
    sigma_dot = jnp.zeros((layers + 1,) + grid.nodal_shape)
    dt = 0.1
    departure_3d = semi_lagrangian.departure_points_3d(
        u, v, sigma_dot, coords, dt=dt
    )
    departure_2d = semi_lagrangian.horizontal_departure_points(
        u, v, grid, dt=dt
    )
    np.testing.assert_allclose(
        departure_3d.cartesian, departure_2d.cartesian, atol=1e-10
    )
    np.testing.assert_allclose(
        departure_3d.sigma,
        np.broadcast_to(
            coords.vertical.centers[:, np.newaxis, np.newaxis],
            departure_3d.sigma.shape,
        ),
        atol=1e-12,
    )


class TransportTest(X64TestCase):

  def _coords(self, grid, layers):
    return coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(layers)
    )

  def test_transport_scalar_2d_over_pole(self):
    """A blob advected over the pole matches the analytic rotated field."""
    grid = spherical_harmonic.Grid.T85()
    omega = 1.0
    dt = 0.2
    axis = [0, 1, 0]
    u, v = solid_body_winds(grid, axis=axis, omega=omega)
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    r = np.stack(semi_lagrangian.lon_lat_to_cartesian(lon_mesh, sin_lat_mesh))

    center = np.array([0.0, 0.0, 1.0])  # blob at the north pole

    def blob(r):
      return np.exp(-((1 - np.einsum('c...,c->...', r, center)) / 0.05))

    departure = semi_lagrangian.horizontal_departure_points(
        jnp.asarray(u), jnp.asarray(v), grid, dt=dt
    )
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')
    transported = semi_lagrangian.transport_scalar_2d(
        jnp.asarray(blob(r)), departure, interpolator
    )
    exact = blob(
        np.einsum(
            'ab,b...->a...', rotation_matrix(axis, -omega * dt), r
        )
    )
    np.testing.assert_allclose(transported, exact, atol=2e-3)

  def test_transport_wind_solid_body(self):
    """Wind transport matches independently computed parallel transport."""
    grid = spherical_harmonic.Grid.T85()
    coords = self._coords(grid, layers=4)
    omega = 1.0
    axis = [0, 1, 0]  # over the poles, the hard case
    u2, v2 = solid_body_winds(grid, axis=axis, omega=omega)
    layers = coords.vertical.layers
    u = jnp.broadcast_to(u2, (layers,) + grid.nodal_shape)
    v = jnp.broadcast_to(v2, (layers,) + grid.nodal_shape)
    sigma_dot = jnp.zeros((layers + 1,) + grid.nodal_shape)
    dt = 0.1
    departure = semi_lagrangian.departure_points_3d(
        u, v, sigma_dot, coords, dt=dt
    )
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')
    u_rot, v_rot = semi_lagrangian.transport_wind(
        u, v, departure, coords.vertical, interpolator, rotate=True
    )

    # Independent reference: evaluate the analytic wind at the exact departure
    # points and parallel-transport it to arrival with explicit per-point
    # Rodrigues rotation matrices.
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    r_a = np.stack(
        semi_lagrangian.lon_lat_to_cartesian(lon_mesh, sin_lat_mesh)
    ).reshape(3, -1)
    r_d = rotation_matrix(axis, -omega * dt) @ r_a
    w_d = omega * np.cross(np.asarray(axis), r_d, axisa=0, axisb=0, axisc=0)
    w_a = np.empty_like(w_d)
    for i in range(r_a.shape[1]):
      cross = np.cross(r_d[:, i], r_a[:, i])
      norm = np.linalg.norm(cross)
      angle = np.arctan2(norm, r_d[:, i] @ r_a[:, i])
      w_a[:, i] = rotation_matrix(cross, angle) @ w_d[:, i]
    u_expected, v_expected = semi_lagrangian.tangent_wind(
        w_a.reshape((3,) + grid.nodal_shape),
        lon_mesh,
        sin_lat_mesh,
    )
    # errors here are only interpolation and departure-point errors.
    np.testing.assert_allclose(u_rot[0], u_expected, atol=2e-3)
    np.testing.assert_allclose(v_rot[0], v_expected, atol=2e-3)

    with self.subTest('flow returned up to Coriolis-scale turning'):
      # Transporting a solid-body wind along its own flow does not return the
      # flow exactly: parallel transport differs from the flow map by a
      # rotation of ~(omega * dt * cos θ) about the vertical (θ = angle from
      # the rotation axis) — precisely the turning that Coriolis/metric terms
      # supply in the momentum equations. Pin the O(omega² dt) bound.
      error_rot = max(
          np.abs(np.asarray(u_rot) - u).max(),
          np.abs(np.asarray(v_rot) - v).max(),
      )
      self.assertLess(error_rot, 0.7 * omega**2 * dt)

  def test_transport_scalar_3d(self):
    """3-D transport with uniform vertical motion matches the analytic field."""
    grid = spherical_harmonic.Grid.T42()
    coords = self._coords(grid, layers=16)
    layers = coords.vertical.layers
    omega = 0.5
    axis = [0, 0, 1]
    u2, v2 = solid_body_winds(grid, axis=axis, omega=omega)
    u = jnp.broadcast_to(u2, (layers,) + grid.nodal_shape)
    v = jnp.broadcast_to(v2, (layers,) + grid.nodal_shape)
    rate = 0.05
    sigma_dot = rate * jnp.ones((layers + 1,) + grid.nodal_shape)
    dt = 0.2

    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    centers = coords.vertical.centers

    def analytic(lon, sin_lat, sigma):
      return (
          np.cos(2 * lon) * (1 - np.asarray(sin_lat) ** 2)
          + 0.5 * np.asarray(sigma)
      )

    field = jnp.asarray(
        analytic(lon_mesh, sin_lat_mesh, centers[:, np.newaxis, np.newaxis])
    )
    departure = semi_lagrangian.departure_points_3d(
        u, v, sigma_dot, coords, dt=dt
    )
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')
    transported = semi_lagrangian.transport_scalar(
        field, departure, coords.vertical, interpolator
    )
    sigma_departure = np.clip(
        centers[:, np.newaxis, np.newaxis] - dt * rate,
        centers[0],
        centers[-1],
    )
    exact = analytic(lon_mesh - omega * dt, sin_lat_mesh, sigma_departure)
    np.testing.assert_allclose(transported, exact, atol=1e-3)


class DifferentiabilityTest(parameterized.TestCase):

  def test_gradients_through_transport(self):
    grid = spherical_harmonic.Grid.T21()
    coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(4)
    )
    layers = coords.vertical.layers
    rng = np.random.RandomState(0)
    u = jnp.asarray(0.1 * rng.normal(size=(layers,) + grid.nodal_shape))
    v = jnp.asarray(0.1 * rng.normal(size=(layers,) + grid.nodal_shape))
    sigma_dot = jnp.asarray(
        0.01 * rng.normal(size=(layers + 1,) + grid.nodal_shape)
    )
    field = jnp.asarray(rng.normal(size=(layers,) + grid.nodal_shape))
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')

    @jax.jit
    def loss(u, v, sigma_dot, field):
      departure = semi_lagrangian.departure_points_3d(
          u, v, sigma_dot, coords, dt=0.3
      )
      out = semi_lagrangian.transport_scalar(
          field, departure, coords.vertical, interpolator
      )
      return jnp.sum(out**2)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(u, v, sigma_dot, field)
    for grad in grads:
      self.assertTrue(np.all(np.isfinite(np.asarray(grad))))
    # gradients w.r.t. winds flow through the departure points.
    self.assertGreater(np.abs(np.asarray(grads[0])).max(), 0.0)
    self.assertGreater(np.abs(np.asarray(grads[3])).max(), 0.0)


if __name__ == '__main__':
  absltest.main()
