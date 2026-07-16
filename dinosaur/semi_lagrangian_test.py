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

"""Tests for dinosaur.semi_lagrangian."""


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
      dict(order='linear', limiter=None),
      dict(order='linear', limiter='quasi_monotone'),
      dict(order='cubic', limiter=None),
      dict(order='cubic', limiter='quasi_monotone'),
  )
  def test_exact_for_constants(self, order, limiter):
    grid = spherical_harmonic.Grid.T21()
    interpolator = semi_lagrangian.GridInterpolator(grid, order, limiter)
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

  def test_unknown_limiter_is_rejected(self):
    grid = spherical_harmonic.Grid.T21()
    with self.assertRaisesRegex(ValueError, 'unknown interpolation limiter'):
      semi_lagrangian.GridInterpolator(grid, 'cubic', limiter='bogus')
    with self.assertRaisesRegex(ValueError, 'unknown interpolation limiter'):
      semi_lagrangian.interpolate_3d(
          jnp.zeros((2,) + grid.nodal_shape),
          grid,
          jnp.zeros((3,)),
          jnp.zeros((3,)),
          jnp.zeros((3,)),
          sigma_nodes=np.array([0.25, 0.75]),
          limiter='bogus',
      )

  def test_grids_with_pole_nodes_are_rejected(self):
    grid = spherical_harmonic.Grid(
        longitude_wavenumbers=22,
        total_wavenumbers=23,
        longitude_nodes=64,
        latitude_nodes=33,
        latitude_spacing='equiangular_with_poles',
    )
    interpolator = semi_lagrangian.GridInterpolator(grid)
    field = jnp.zeros(grid.nodal_shape)
    points = jnp.zeros((4,))
    with self.assertRaisesRegex(ValueError, 'nodes at the poles'):
      interpolator(field, points, points)

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
    unlimited = semi_lagrangian.GridInterpolator(grid, 'cubic')(
        field, lon, sin_lat
    )
    limited = semi_lagrangian.GridInterpolator(grid, 'cubic', 'quasi_monotone')(
        field, lon, sin_lat
    )
    with self.subTest('cubic overshoots without limiter'):
      self.assertLess(np.min(np.asarray(unlimited)), -1e-3)
      self.assertGreater(np.max(np.asarray(unlimited)), 1 + 1e-3)
    with self.subTest('limiter prevents new extrema'):
      self.assertGreaterEqual(np.min(np.asarray(limited)), 0.0)
      self.assertLessEqual(np.max(np.asarray(limited)), 1.0)

  def test_interpolate_3d_linear_in_sigma(self):
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
    values = semi_lagrangian.interpolate_3d(
        field, grid, lon, sin_lat, sigma, sigma_nodes=centers, order='linear'
    )
    sigma_clipped = np.clip(sigma, centers[0], centers[-1])
    horizontal_at_points = 1 + 0.3 * np.asarray(sin_lat)
    expected = (2.0 + 3.0 * sigma_clipped) * horizontal_at_points
    # bilinear horizontal interpolation of a linear-in-sin_lat field is not
    # exact in phi, so allow a modest tolerance.
    np.testing.assert_allclose(values, expected, rtol=1e-3)

  def test_interpolate_3d_cubic_in_sigma(self):
    """Cubic vertical interpolation is exact for cubics away from boundaries.

    The horizontal profile is constant so all error is vertical; sigma
    points are restricted to interior cells (the first and last cells
    degrade to linear by design, tested separately).
    """
    grid = spherical_harmonic.Grid.T21()
    centers = sigma_coordinates.SigmaCoordinates.equidistant(10).centers

    def profile(sigma):
      return 1.0 + sigma - 2.0 * sigma**2 + 3.0 * sigma**3

    field = jnp.asarray(
        np.broadcast_to(
            profile(centers)[:, np.newaxis, np.newaxis],
            (10,) + grid.nodal_shape,
        )
    )
    rng = np.random.RandomState(8)
    shape = (6, 20)
    lon = jnp.asarray(rng.uniform(0, 2 * np.pi, size=shape))
    sin_lat = jnp.asarray(np.sin(rng.uniform(-1.4, 1.4, size=shape)))
    # interior cells only: [centers[1], centers[-2]]
    sigma = jnp.asarray(rng.uniform(centers[1], centers[-2], size=shape))
    kwargs = dict(sigma_nodes=centers, order='cubic')
    cubic = semi_lagrangian.interpolate_3d(
        field, grid, lon, sin_lat, sigma, vertical_order='cubic', **kwargs
    )
    np.testing.assert_allclose(cubic, profile(np.asarray(sigma)), rtol=1e-5)
    linear = semi_lagrangian.interpolate_3d(
        field, grid, lon, sin_lat, sigma, vertical_order='linear', **kwargs
    )
    linear_error = np.abs(np.asarray(linear) - profile(np.asarray(sigma)))
    cubic_error = np.abs(np.asarray(cubic) - profile(np.asarray(sigma)))
    self.assertLess(cubic_error.max(), 0.05 * linear_error.max())

  def test_interpolate_3d_cubic_degrades_to_linear_at_boundaries(self):
    """Within the first and last cells, cubic vertical equals linear."""
    grid = spherical_harmonic.Grid.T21()
    centers = sigma_coordinates.SigmaCoordinates.equidistant(6).centers
    rng = np.random.RandomState(9)
    field = jnp.asarray(rng.normal(size=(6,) + grid.nodal_shape))
    shape = (40,)
    lon = jnp.asarray(rng.uniform(0, 2 * np.pi, size=shape))
    sin_lat = jnp.asarray(np.sin(rng.uniform(-1.4, 1.4, size=shape)))
    # points in the first and last cells, plus out-of-range extrapolation.
    sigma = jnp.asarray(
        np.concatenate([
            rng.uniform(-0.1, centers[1], size=20),
            rng.uniform(centers[-2], 1.1, size=20),
        ])
    )
    results = {}
    for vertical_order in ('linear', 'cubic'):
      results[vertical_order] = semi_lagrangian.interpolate_3d(
          field, grid, lon, sin_lat, sigma,
          sigma_nodes=centers, order='cubic', vertical_order=vertical_order,
      )
    np.testing.assert_allclose(
        results['cubic'], results['linear'], rtol=1e-6, atol=1e-6
    )

  def test_interpolate_3d_cubic_requires_four_levels(self):
    grid = spherical_harmonic.Grid.T21()
    with self.assertRaisesRegex(ValueError, 'at least 4 levels'):
      semi_lagrangian.interpolate_3d(
          jnp.zeros((3,) + grid.nodal_shape),
          grid,
          jnp.zeros(()),
          jnp.zeros(()),
          jnp.zeros(()),
          sigma_nodes=np.array([0.2, 0.5, 0.8]),
          vertical_order='cubic',
      )

  @parameterized.parameters(
      dict(vertical_order='linear'), dict(vertical_order='cubic')
  )
  def test_interpolate_3d_monotone(self, vertical_order):
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
    values = semi_lagrangian.interpolate_3d(
        field,
        grid,
        lon,
        sin_lat,
        sigma,
        sigma_nodes=vertical.centers,
        order='cubic',
        limiter='quasi_monotone',
        vertical_order=vertical_order,
    )
    self.assertGreaterEqual(np.min(np.asarray(values)), 0.0)


class ContractStencilTest(parameterized.TestCase):

  @parameterized.parameters(dict(n=2), dict(n=3))
  def test_fused_form_matches_einsum(self, n):
    """The accelerator (fused) contraction equals the einsum form.

    The dispatch in `_contract_stencil` picks by backend, so CPU test runs
    exercise only the einsum branch; this pins the fused branch directly.
    """
    rng = np.random.RandomState(0)
    batch = (3, 5, 7)
    sizes = (2, 4, 4)[-n:]
    values = jnp.asarray(rng.standard_normal(batch + sizes))
    weights = [
        jnp.asarray(rng.standard_normal(batch + (s,))) for s in sizes
    ]
    letters = 'ijkl'[:n]
    expected = jnp.einsum(
        ','.join(f'...{c}' for c in letters) + f',...{letters}->...',
        *weights,
        values,
    )
    actual = semi_lagrangian._contract_stencil_fused(values, *weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


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

  def test_gradients_finite_for_interpolation_at_the_pole(self):
    """Interpolating exactly at a pole must not produce NaN gradients."""
    grid = spherical_harmonic.Grid.T21()
    rng = np.random.RandomState(1)
    field = jnp.asarray(rng.normal(size=grid.nodal_shape))
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')

    def loss(sin_lat):
      lon = jnp.asarray([0.3, 1.0, 2.0])
      return jnp.sum(interpolator(field, lon, sin_lat) ** 2)

    # points at and within float32 rounding of both poles.
    sin_lat = jnp.asarray([1.0, -1.0, 1.0 - 1e-8])
    grad = jax.grad(loss)(sin_lat)
    self.assertTrue(np.all(np.isfinite(np.asarray(grad))))


if __name__ == '__main__':
  absltest.main()
