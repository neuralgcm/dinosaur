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


import functools
import math
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
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


class MonotoneWindTransportTest(parameterized.TestCase):

  def _solid_body(self, grid):
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    cos_lat = np.sqrt(1 - sin_lat_mesh**2)
    u = jnp.asarray(np.broadcast_to(cos_lat, grid.nodal_shape))
    v = jnp.zeros(grid.nodal_shape)
    return u, v

  def test_limiter_is_inert_for_smooth_winds(self):
    """Solid-body winds are within every bracketing cell's range."""
    grid = spherical_harmonic.Grid.T21()
    u, v = self._solid_body(grid)
    departure = semi_lagrangian.horizontal_departure_points(
        u, v, grid, dt=0.1
    )
    results = {}
    for limiter in (None, 'quasi_monotone'):
      interpolator = semi_lagrangian.GridInterpolator(grid, 'cubic', limiter)
      results[limiter] = semi_lagrangian.transport_wind_2d(
          u, v, departure, interpolator
      )
    for a, b in zip(results[None], results['quasi_monotone']):
      np.testing.assert_allclose(a, b, atol=1e-6)

  def test_limiter_acts_on_rough_winds(self):
    """A grid-scale wind field engages the clip (results differ)."""
    grid = spherical_harmonic.Grid.T21()
    u_smooth, v_smooth = self._solid_body(grid)
    rng = np.random.RandomState(0)
    u = u_smooth + 0.3 * jnp.asarray(
        rng.standard_normal(grid.nodal_shape)
    )
    v = v_smooth + 0.3 * jnp.asarray(
        rng.standard_normal(grid.nodal_shape)
    )
    departure = semi_lagrangian.horizontal_departure_points(
        u_smooth, v_smooth, grid, dt=0.3
    )
    unlimited = semi_lagrangian.transport_wind_2d(
        u, v, departure, semi_lagrangian.GridInterpolator(grid, 'cubic')
    )
    limited = semi_lagrangian.transport_wind_2d(
        u,
        v,
        departure,
        semi_lagrangian.GridInterpolator(grid, 'cubic', 'quasi_monotone'),
    )
    difference = max(
        float(np.abs(np.asarray(a) - np.asarray(b)).max())
        for a, b in zip(unlimited, limited)
    )
    self.assertGreater(difference, 1e-4)


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


class MatmulInterpolationTest(parameterized.TestCase):

  @parameterized.product(
      order=('linear', 'cubic'),
      limiter=(None, 'quasi_monotone'),
      query_kind=(
          'grid_aligned',
          'arbitrary_points',
          'batched_levels',
          'single_level_3d',
      ),
  )
  def test_2d_matmul_matches_gather(self, order, limiter, query_kind):
    grid = spherical_harmonic.Grid.T21()
    rng = np.random.RandomState(42)
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    phi_mesh = np.arcsin(sin_lat_mesh)

    if query_kind == 'grid_aligned':
      field = jnp.asarray(rng.normal(size=grid.nodal_shape), dtype=jnp.float32)
      dphi = rng.uniform(-0.15, 0.15, size=grid.nodal_shape)
      phi_q = np.clip(phi_mesh + dphi, -np.pi / 2 + 1e-4, np.pi / 2 - 1e-4)
      lon_q = (lon_mesh + rng.uniform(-0.3, 0.3, size=grid.nodal_shape)) % (
          2 * np.pi
      )
      lon = jnp.asarray(lon_q, dtype=jnp.float32)
      sin_lat = jnp.asarray(np.sin(phi_q), dtype=jnp.float32)
    elif query_kind == 'arbitrary_points':
      field = jnp.asarray(rng.normal(size=grid.nodal_shape), dtype=jnp.float32)
      lon = jnp.asarray(
          rng.uniform(0, 2 * np.pi, size=(7, 13)), dtype=jnp.float32
      )
      sin_lat = jnp.asarray(
          np.sin(rng.uniform(-1.55, 1.55, size=(7, 13))), dtype=jnp.float32
      )
    elif query_kind == 'batched_levels':
      field = jnp.asarray(
          rng.normal(size=(4,) + grid.nodal_shape), dtype=jnp.float32
      )
      lon = jnp.asarray(
          (lon_mesh + rng.uniform(-0.2, 0.2, size=(4,) + grid.nodal_shape))
          % (2 * np.pi),
          dtype=jnp.float32,
      )
      phi_q = np.clip(
          phi_mesh + rng.uniform(-0.1, 0.1, size=(4,) + grid.nodal_shape),
          -1.55,
          1.55,
      )
      sin_lat = jnp.asarray(np.sin(phi_q), dtype=jnp.float32)
    else:
      # Surface fields like u_mean and log_surface_pressure have shape [1, nx, ny].
      field = jnp.asarray(
          rng.normal(size=(1,) + grid.nodal_shape), dtype=jnp.float32
      )
      lon = jnp.asarray(
          (lon_mesh + rng.uniform(-0.2, 0.2, size=(1,) + grid.nodal_shape))
          % (2 * np.pi),
          dtype=jnp.float32,
      )
      sin_lat = jnp.asarray(
          np.sin(
              np.clip(
                  phi_mesh
                  + rng.uniform(-0.1, 0.1, size=(1,) + grid.nodal_shape),
                  -1.55,
                  1.55,
              )
          ),
          dtype=jnp.float32,
      )

    interp_gather = semi_lagrangian.GridInterpolator(
        grid, order, limiter, method='gather'
    )
    interp_matmul = semi_lagrangian.GridInterpolator(
        grid, order, limiter, method='matmul'
    )
    out_gather = interp_gather(field, lon, sin_lat)
    out_matmul = interp_matmul(field, lon, sin_lat)
    np.testing.assert_allclose(out_matmul, out_gather, rtol=1e-5, atol=1e-5)

    if limiter is None and query_kind == 'grid_aligned':
      grad_gather = jax.grad(
          lambda f, x, y: jnp.sum(interp_gather(f, x, y) ** 2),
          argnums=(0, 1, 2),
      )(field, lon, sin_lat)
      grad_matmul = jax.grad(
          lambda f, x, y: jnp.sum(interp_matmul(f, x, y) ** 2),
          argnums=(0, 1, 2),
      )(field, lon, sin_lat)
      for g_m, g_g in zip(grad_matmul, grad_gather):
        np.testing.assert_allclose(g_m, g_g, rtol=1e-4, atol=2e-4)

  @parameterized.product(
      order=('linear', 'cubic'),
      vertical_order=('linear', 'cubic'),
      limiter=(None, 'quasi_monotone'),
      use_ref=(False, True),
      query_kind=('grid_aligned', 'arbitrary_points'),
  )
  def test_3d_matmul_matches_gather(
      self, order, vertical_order, limiter, use_ref, query_kind
  ):
    if use_ref and limiter is None:
      return
    grid = spherical_harmonic.Grid.T21()
    layers = 6
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    rng = np.random.RandomState(17)
    field = jnp.asarray(
        rng.normal(size=(layers,) + grid.nodal_shape), dtype=jnp.float32
    )
    ref = (
        jnp.asarray(np.linspace(220.0, 295.0, layers), dtype=jnp.float32)
        if use_ref
        else None
    )

    if query_kind == 'grid_aligned':
      lon_mesh, sin_lat_mesh = grid.nodal_mesh
      phi_mesh = np.arcsin(sin_lat_mesh)
      shape = (layers,) + grid.nodal_shape
      lon = jnp.asarray(
          (lon_mesh + rng.uniform(-0.25, 0.25, size=shape)) % (2 * np.pi),
          dtype=jnp.float32,
      )
      phi_q = np.clip(phi_mesh + rng.uniform(-0.12, 0.12, size=shape), -1.56, 1.56)
      sin_lat = jnp.asarray(np.sin(phi_q), dtype=jnp.float32)
      sigma_base = centers[:, np.newaxis, np.newaxis]
      sigma = jnp.asarray(
          np.clip(sigma_base + rng.uniform(-0.15, 0.15, size=shape), 0.0, 1.0),
          dtype=jnp.float32,
      )
    else:
      shape = (5, 9)
      lon = jnp.asarray(rng.uniform(0, 2 * np.pi, size=shape), dtype=jnp.float32)
      sin_lat = jnp.asarray(
          np.sin(rng.uniform(-1.5, 1.5, size=shape)), dtype=jnp.float32
      )
      sigma = jnp.asarray(rng.uniform(-0.05, 1.05, size=shape), dtype=jnp.float32)

    common = dict(
        grid=grid,
        lon=lon,
        sin_lat=sin_lat,
        sigma=sigma,
        sigma_nodes=centers,
        order=order,
        limiter=limiter,
        vertical_order=vertical_order,
        limiter_bound_reference=ref,
    )
    out_gather = semi_lagrangian.interpolate_3d(
        field, method='gather', **common
    )
    out_matmul = semi_lagrangian.interpolate_3d(
        field, method='matmul', **common
    )
    np.testing.assert_allclose(out_matmul, out_gather, rtol=1e-5, atol=1e-5)

    if use_ref:
      # A constant reference profile must be an exact no-op in float32.
      const_ref = jnp.full((layers,), 288.0, dtype=jnp.float32)
      out_const = semi_lagrangian.interpolate_3d(
          field,
          **(common | dict(limiter_bound_reference=const_ref)),
          method='matmul',
      )
      out_none = semi_lagrangian.interpolate_3d(
          field,
          **(common | dict(limiter_bound_reference=None)),
          method='matmul',
      )
      np.testing.assert_allclose(out_const, out_none, rtol=1e-6, atol=1e-6)

  def test_multichannel_batching_and_scan_chunking(self):
    grid = spherical_harmonic.Grid.T21()
    coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(6)
    )
    layers = coords.vertical.layers
    rng = np.random.RandomState(99)
    u = jnp.asarray(0.1 * rng.normal(size=(layers,) + grid.nodal_shape), jnp.float32)
    v = jnp.asarray(0.1 * rng.normal(size=(layers,) + grid.nodal_shape), jnp.float32)
    sigma_dot = jnp.asarray(
        0.02 * rng.normal(size=(layers + 1,) + grid.nodal_shape), jnp.float32
    )
    dep_g = semi_lagrangian.departure_points_3d(
        u, v, sigma_dot, coords, dt=0.2, method='gather'
    )
    dep_m = semi_lagrangian.departure_points_3d(
        u, v, sigma_dot, coords, dt=0.2, method='matmul'
    )
    np.testing.assert_allclose(dep_m.cartesian, dep_g.cartesian, atol=1e-5)
    np.testing.assert_allclose(dep_m.sigma, dep_g.sigma, atol=1e-5)

    # 11 channels exercises C > 8 lax.scan chunking in matmul and C > 4 in gather.
    tracers = jnp.asarray(
        np.maximum(rng.normal(size=(11, layers) + grid.nodal_shape), 0.0),
        jnp.float32,
    )
    interp_g = semi_lagrangian.GridInterpolator(
        grid, 'cubic', 'quasi_monotone', method='gather'
    )
    interp_m = semi_lagrangian.GridInterpolator(
        grid, 'cubic', 'quasi_monotone', method='matmul'
    )
    out_g = semi_lagrangian.transport_scalar(
        tracers, dep_g, coords.vertical, interp_g, vertical_order='cubic'
    )
    out_m_same_dep = semi_lagrangian.transport_scalar(
        tracers, dep_g, coords.vertical, interp_m, vertical_order='cubic'
    )
    np.testing.assert_allclose(out_m_same_dep, out_g, rtol=2e-6, atol=2e-6)
    out_m = semi_lagrangian.transport_scalar(
        tracers, dep_m, coords.vertical, interp_m, vertical_order='cubic'
    )
    np.testing.assert_allclose(out_m, out_g, rtol=2e-5, atol=5e-5)
    self.assertGreaterEqual(float(jnp.min(out_m)), 0.0)
    for idx in (0, 5, 10):
      single = semi_lagrangian.transport_scalar(
          tracers[idx], dep_m, coords.vertical, interp_m, vertical_order='cubic'
      )
      np.testing.assert_allclose(out_m[idx], single, rtol=1e-6, atol=1e-6)

  def test_nodal_padding_and_orography_helpers(self):
    padded_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=22,
        total_wavenumbers=23,
        longitude_nodes=68,
        latitude_nodes=34,
        spherical_harmonics_impl=functools.partial(
            spherical_harmonic.FastSphericalHarmonics,
            nodal_shape_multiple=8,
        ),
    )
    self.assertEqual(padded_grid.nodal_shape, (72, 40))
    self.assertEqual(padded_grid.nodal_padding, (4, 6))
    coords = coordinate_systems.CoordinateSystem(
        padded_grid, sigma_coordinates.SigmaCoordinates.equidistant(4)
    )

    rng = np.random.RandomState(12)
    orography_unpadded = jnp.asarray(rng.normal(size=(68, 34)), jnp.float32)
    orography_padded = jnp.pad(orography_unpadded, [(0, 4), (0, 6)])
    np.testing.assert_allclose(
        primitive_equations.truncated_modal_orography(orography_unpadded, coords),
        primitive_equations.truncated_modal_orography(orography_padded, coords),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        primitive_equations.filtered_modal_orography(orography_unpadded, coords),
        primitive_equations.filtered_modal_orography(orography_padded, coords),
        atol=1e-6,
    )
    with self.assertRaisesRegex(ValueError, 'Expected nodal orography with shape='):
      primitive_equations.truncated_modal_orography(jnp.zeros((10, 10)), coords)

    u = jnp.pad(
        jnp.asarray(0.1 * rng.normal(size=(4, 68, 34)), jnp.float32),
        [(0, 0), (0, 4), (0, 6)],
    )
    v = jnp.pad(
        jnp.asarray(0.1 * rng.normal(size=(4, 68, 34)), jnp.float32),
        [(0, 0), (0, 4), (0, 6)],
    )
    sigma_dot = jnp.zeros((5, 72, 40), dtype=jnp.float32)
    dep = semi_lagrangian.departure_points_3d(
        u, v, sigma_dot, coords, dt=0.2, method='matmul'
    )
    interp_m = semi_lagrangian.GridInterpolator(
        padded_grid, 'cubic', 'quasi_monotone', method='matmul'
    )
    interp_g = semi_lagrangian.GridInterpolator(
        padded_grid, 'cubic', 'quasi_monotone', method='gather'
    )
    u_m, v_m = semi_lagrangian.transport_wind(
        u, v, dep, coords.vertical, interp_m, planetary_rotation_rate=1.0
    )
    u_g, v_g = semi_lagrangian.transport_wind(
        u, v, dep, coords.vertical, interp_g, planetary_rotation_rate=1.0
    )
    np.testing.assert_allclose(u_m[:, :68, :34], u_g[:, :68, :34], atol=1e-5)
    np.testing.assert_allclose(v_m[:, :68, :34], v_g[:, :68, :34], atol=1e-5)
    # Padded entries must remain strictly zero.
    np.testing.assert_array_equal(u_m[:, 68:, :], 0.0)
    np.testing.assert_array_equal(u_m[:, :, 34:], 0.0)
    np.testing.assert_array_equal(v_m[:, 68:, :], 0.0)
    np.testing.assert_array_equal(v_m[:, :, 34:], 0.0)

  def test_matmul_stablehlo_has_no_field_gather_or_scatter(self):
    grid = spherical_harmonic.Grid.T21()
    coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(6)
    )
    layers = coords.vertical.layers
    u = jnp.zeros((layers,) + grid.nodal_shape, dtype=jnp.float32)
    v = jnp.zeros((layers,) + grid.nodal_shape, dtype=jnp.float32)
    sigma_dot = jnp.zeros((layers + 1,) + grid.nodal_shape, dtype=jnp.float32)
    tracers = jnp.zeros((10, layers) + grid.nodal_shape, dtype=jnp.float32)
    interp = semi_lagrangian.GridInterpolator(
        grid, 'cubic', 'quasi_monotone', method='matmul'
    )

    def loss(u_in, v_in, sd_in, tr_in):
      dep = semi_lagrangian.departure_points_3d(
          u_in, v_in, sd_in, coords, dt=0.2, method='matmul'
      )
      u_out, v_out = semi_lagrangian.transport_wind(
          u_in, v_in, dep, coords.vertical, interp, vertical_order='cubic'
      )
      tr_out = semi_lagrangian.transport_scalar(
          tr_in, dep, coords.vertical, interp, vertical_order='cubic'
      )
      return jnp.sum(u_out**2) + jnp.sum(v_out**2) + jnp.sum(tr_out**2)

    hlo = (
        jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
        .lower(u, v, sigma_dot, tracers)
        .as_text()
    )
    self.assertNotIn('stablehlo.scatter', hlo)
    gather_lines = [line for line in hlo.splitlines() if 'stablehlo.gather' in line]
    for line in gather_lines:
      # Only 1-D coordinate lookup tables (sigma_nodes / lat_ext) may be gathered.
      operand_type = line.split(': (')[1].split(',')[0]
      dims_and_dtype = operand_type.split('<')[1].rstrip('>')
      self.assertEqual(len(dims_and_dtype.split('x')), 2, msg=line)

    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      jax.clear_caches()
      tpu_hlo = (
          jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
          .lower(u, v, sigma_dot, tracers)
          .as_text()
      )
      self.assertNotIn('stablehlo.scatter', tpu_hlo)
      self.assertNotIn('stablehlo.gather', tpu_hlo)
    jax.clear_caches()

    with self.assertRaisesRegex(ValueError, 'unknown interpolation method'):
      semi_lagrangian.GridInterpolator(grid, method='bogus')  # pytype: disable=wrong-arg-types

  def test_matmul_t63_tpu_tile_memory_bound(self):
    grid = spherical_harmonic.Grid.T63()
    coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(47)
    )
    layers = coords.vertical.layers
    u = jnp.zeros((layers,) + grid.nodal_shape, dtype=jnp.float32)
    v = jnp.zeros((layers,) + grid.nodal_shape, dtype=jnp.float32)
    sigma_dot = jnp.zeros((layers + 1,) + grid.nodal_shape, dtype=jnp.float32)
    tracers = jnp.zeros((10, layers) + grid.nodal_shape, dtype=jnp.float32)
    interp = semi_lagrangian.GridInterpolator(
        grid, 'cubic', 'quasi_monotone', method='matmul'
    )

    @jax.jit
    def step_transport(u_in, v_in, sd_in, tr_in):
      dep = semi_lagrangian.departure_points_3d(
          u_in, v_in, sd_in, coords, dt=600.0, method='matmul'
      )
      u_out, v_out = semi_lagrangian.transport_wind(
          u_in, v_in, dep, coords.vertical, interp, vertical_order='cubic'
      )
      tr_out = semi_lagrangian.transport_scalar(
          tr_in, dep, coords.vertical, interp, vertical_order='cubic'
      )
      return u_out, v_out, tr_out

    layout_re = re.compile(r'\b(f32|bf16|s32|u32|pred)\[([0-9,]+)\]\{([0-9,]+)\}')
    dtype_bytes = {'f32': 4, 'bf16': 2, 's32': 4, 'u32': 4, 'pred': 1}

    def max_padded_bytes(hlo_text: str) -> int:
      best = 0
      for match in layout_re.finditer(hlo_text):
        dt, dims_s, lay_s = match.groups()
        dims = [int(x) for x in dims_s.split(',')]
        lay = [int(x) for x in lay_s.split(',')]
        item = dtype_bytes[dt]
        if len(dims) == 1:
          tpu_bytes = math.ceil(dims[0] / 128) * 128 * item
        else:
          d_minor = dims[lay[0]]
          d_sub = dims[lay[1]]
          rest = math.prod(
              dims[i] for i in range(len(dims)) if i not in (lay[0], lay[1])
          )
          tpu_bytes = (
              rest
              * (math.ceil(d_sub / 8) * 8)
              * (math.ceil(d_minor / 128) * 128)
              * item
          )
        best = max(best, tpu_bytes)
      return best

    compiled = step_transport.lower(u, v, sigma_dot, tracers).compile()
    mem = compiled.memory_analysis()
    # Unpadded HLO temporary workspace for T63 L47 + 10 tracers must stay <750MB
    # on single-level non-TPU paths, with every buffer <200MB under TPU (8, 128)
    # physical tile padding.
    self.assertLess(mem.temp_size_in_bytes, 750 * 1024 * 1024)
    self.assertLess(max_padded_bytes(compiled.as_text()), 200 * 1024 * 1024)

    # Also verify multi-level TPU MXU chunking stays bounded under (8, 128)
    # tile padding (catches 6-D transpose blowups while trading modest workspace
    # for multi-level MXU batching).
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      jax.clear_caches()
      compiled_tpu = step_transport.lower(u, v, sigma_dot, tracers).compile()
      mem_tpu = compiled_tpu.memory_analysis()
      self.assertLess(mem_tpu.temp_size_in_bytes, 1600 * 1024 * 1024)
      self.assertLess(
          max_padded_bytes(compiled_tpu.as_text()), 650 * 1024 * 1024
      )
    jax.clear_caches()

  @parameterized.parameters('gather', 'matmul')
  def test_interpolate_3d_mixed_no_trilinear_channels(self, method):
    # `trilinear_channel_start == C` leaves no trilinear channels; with more
    # channels than one MXU chunk and a limiter split the matmul method used
    # to have no trilinear chunk to concatenate.
    grid = spherical_harmonic.Grid.T21()
    layers = 8
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    shape = (layers,) + grid.nodal_shape
    lon = jnp.broadcast_to(jnp.asarray(lon_mesh, jnp.float32), shape)
    sin_lat = jnp.broadcast_to(jnp.asarray(sin_lat_mesh, jnp.float32), shape)
    sigma = jnp.broadcast_to(
        jnp.asarray(centers, jnp.float32)[:, np.newaxis, np.newaxis], shape
    )
    num_channels = 10
    fields = jnp.ones((num_channels, layers) + grid.nodal_shape, jnp.float32)
    out, out_lin = semi_lagrangian._interpolate_3d_mixed(
        fields,
        grid,
        lon,
        sin_lat,
        sigma,
        centers,
        order='cubic',
        limiter='quasi_monotone',
        limited_channel_start=4,
        trilinear_channel_start=num_channels,
        vertical_order='cubic',
        method=method,
    )
    self.assertEqual(out.shape, (num_channels,) + shape)
    self.assertEqual(out_lin.shape, (0,) + shape)
    self.assertEqual(out_lin.dtype, fields.dtype)
    np.testing.assert_allclose(out, 1.0, atol=1e-5)

  def test_interpolate_3d_mixed_partial_limiting_and_trilinear(self):
    grid = spherical_harmonic.Grid.T21()
    coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(16)
    )
    layers = coords.vertical.layers
    rng = np.random.RandomState(23)
    u = jnp.asarray(0.2 * rng.normal(size=(layers,) + grid.nodal_shape), jnp.float32)
    v = jnp.asarray(0.2 * rng.normal(size=(layers,) + grid.nodal_shape), jnp.float32)
    sigma_dot = jnp.zeros((layers + 1,) + grid.nodal_shape, dtype=jnp.float32)
    sigma_dot = sigma_dot.at[1:-1].set(
        jnp.asarray(0.05 * rng.normal(size=(layers - 1,) + grid.nodal_shape), jnp.float32)
    )
    fields = jnp.asarray(
        jnp.maximum(0.5 + rng.normal(size=(7, layers) + grid.nodal_shape), 0.0),
        jnp.float32,
    )

    dep_g = semi_lagrangian.departure_points_3d(
        u, v, sigma_dot, coords, dt=0.25, method='gather'
    )
    dep_cpu = semi_lagrangian.departure_points_3d(
        u, v, sigma_dot, coords, dt=0.25, method='matmul'
    )
    np.testing.assert_allclose(dep_cpu.cartesian, dep_g.cartesian, atol=1e-5)
    np.testing.assert_allclose(dep_cpu.sigma, dep_g.sigma, atol=1e-5)
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      jax.clear_caches()
      dep_tpu = semi_lagrangian.departure_points_3d(
          u, v, sigma_dot, coords, dt=0.25, method='matmul'
      )
    jax.clear_caches()
    np.testing.assert_allclose(dep_tpu.cartesian, dep_g.cartesian, atol=5e-5)
    np.testing.assert_allclose(dep_tpu.sigma, dep_g.sigma, atol=1e-5)

    ref_unlim = semi_lagrangian.interpolate_3d(
        fields[:4],
        grid,
        dep_g.lon,
        dep_g.sin_lat,
        dep_g.sigma,
        coords.vertical.centers,
        order='cubic',
        limiter=None,
        vertical_order='cubic',
        method='gather',
    )
    ref_lim = semi_lagrangian.interpolate_3d(
        fields[4:],
        grid,
        dep_g.lon,
        dep_g.sin_lat,
        dep_g.sigma,
        coords.vertical.centers,
        order='cubic',
        limiter='quasi_monotone',
        vertical_order='cubic',
        method='gather',
    )
    ref_trilin = semi_lagrangian.interpolate_3d(
        fields[4:],
        grid,
        dep_g.lon,
        dep_g.sin_lat,
        dep_g.sigma,
        coords.vertical.centers,
        order='linear',
        limiter=None,
        vertical_order='linear',
        method='gather',
    )
    ref_high = jnp.concatenate([ref_unlim, ref_lim], axis=0)

    for backend in ('cpu', 'tpu'):
      with self.subTest(backend=backend):
        with mock.patch.object(jax, 'default_backend', return_value=backend):
          jax.clear_caches()
          fn = jax.jit(
              lambda f: semi_lagrangian._interpolate_3d_mixed(
                  f,
                  grid,
                  dep_g.lon,
                  dep_g.sin_lat,
                  dep_g.sigma,
                  coords.vertical.centers,
                  order='cubic',
                  limiter='quasi_monotone',
                  limited_channel_start=4,
                  trilinear_channel_start=4,
                  vertical_order='cubic',
                  method='matmul',
              )
          )
          out_high, out_low = fn(fields)
          np.testing.assert_allclose(out_high, ref_high, rtol=2e-5, atol=5e-5)
          np.testing.assert_allclose(out_low, ref_trilin, rtol=2e-5, atol=5e-5)
          self.assertGreaterEqual(float(jnp.min(out_high[4:])), 0.0)
          self.assertGreaterEqual(float(jnp.min(out_low)), 0.0)

          if backend == 'tpu':
            loss_fn = jax.jit(
                jax.value_and_grad(
                    lambda f: jnp.sum(fn(f)[0] ** 2) + 0.5 * jnp.sum(fn(f)[1] ** 2)
                )
            )
            hlo = loss_fn.lower(fields).as_text()
            self.assertNotIn('stablehlo.gather', hlo)
            self.assertNotIn('stablehlo.scatter', hlo)
            _, grad_f = loss_fn(fields)
            self.assertTrue(np.all(np.isfinite(np.asarray(grad_f))))
    jax.clear_caches()


# Window half-width of the out-of-window tests, and the displacement range (in
# grid cells) of their departure points: beyond the window, by up to 2S + 3.
_SHIFT = 2
_FAR = (3.0, 7.0)


def _far_queries(grid, layers, seed, dtype=np.float32, cells=_FAR):
  """Grid-aligned [layers, *nodal_shape] queries displaced by `cells`."""
  rng = np.random.RandomState(seed)
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  phi_mesh = np.arcsin(sin_lat_mesh)
  shape = (layers,) + grid.nodal_shape

  def shifts():
    return rng.uniform(*cells, size=shape) * rng.choice([-1.0, 1.0], size=shape)

  phi = np.clip(
      phi_mesh + shifts() * np.pi / grid.latitude_nodes,
      -np.pi / 2 + 1e-4,
      np.pi / 2 - 1e-4,
  )
  lon = (lon_mesh + rng.uniform(-0.3, 0.3, size=shape)) % (2 * np.pi)
  centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
  sigma = np.clip(
      centers[:, np.newaxis, np.newaxis] + shifts() / layers,
      centers[0] - 0.02,
      centers[-1] + 0.02,
  )
  return tuple(jnp.asarray(x, dtype) for x in (lon, np.sin(phi), sigma))


def _library_phi(sin_lat):
  s = jnp.clip(sin_lat, -1, 1)
  return jnp.arctan2(s, jnp.sqrt(jnp.maximum(1 - jnp.square(s), 1e-12)))


def _reference_clamp(grid, nodes, sin_lat, sigma, order, shift):
  """The window clamp, written out independently of the library's code.

  Query [k, i, j] arrives at halo-extended latitude row a = j + 2 and level
  k: its latitude is clamped to [φ_ext[a - S], φ_ext[a + S + 1]) and its σ to
  [z[k - S], z[k + S + 1]) (indices clipped to the axes; the upper bounds one
  ulp below the node). The clamped latitude is handed back as sin(latitude),
  nudged by whole ulps until the arctan2 maps it a few ulps inside the
  interval (the library's arctan2, compiled in another program, may round
  differently by an ulp), so it represents the clamped point, in the same
  bracketing cell, to within the resolution of sin(latitude).
  """
  phi_ext = np.asarray(semi_lagrangian._latitude_halo_axis(grid))
  n_ext = phi_ext.size
  c_min, c_max = (0, n_ext - 2) if order == 'linear' else (1, n_ext - 3)
  a = np.minimum(np.arange(grid.nodal_shape[1]), grid.latitude_nodes - 1) + 2
  dtype = sin_lat.dtype
  lo = phi_ext[np.clip(a - shift, c_min, c_max)].astype(dtype)
  hi = np.nextafter(
      phi_ext[np.clip(a + shift, c_min, c_max) + 1].astype(dtype),
      np.asarray(-np.inf, dtype),
  )
  phi = _library_phi(sin_lat)
  sl = jnp.where((phi < lo) | (phi > hi), jnp.sin(jnp.clip(phi, lo, hi)), sin_lat)
  one = jnp.ones_like(sl)
  lo_in, hi_in = jnp.asarray(lo), jnp.asarray(hi)
  for _ in range(4):
    lo_in = jnp.nextafter(lo_in, jnp.full_like(lo_in, jnp.inf))
    hi_in = jnp.nextafter(hi_in, jnp.full_like(hi_in, -jnp.inf))
  for _ in range(32):
    q = _library_phi(sl)
    sl = jnp.where(
        q > hi_in,
        jnp.nextafter(sl, -one),
        jnp.where(q < lo_in, jnp.nextafter(sl, one), sl),
    )
  z = np.asarray(nodes).astype(dtype)
  k = np.minimum(np.arange(sigma.shape[0]), z.size - 1)
  hi_i = np.minimum(k + shift + 1, z.size - 1)
  z_lo = z[np.maximum(k - shift, 0)]
  z_hi = np.where(
      hi_i < z.size - 1, np.nextafter(z[hi_i], np.asarray(-np.inf, dtype)),
      z[hi_i],
  )
  sg = jnp.clip(sigma, z_lo[:, None, None], z_hi[:, None, None])
  return sl, sg


class WindowClampTest(parameterized.TestCase):
  """Departure points beyond the stencil windows are clamped to them.

  The 'matmul' method builds stencils inside windows around each arrival
  point; whenever windows are given, both methods clamp the query into them,
  so every method/backend interpolates the same clamped query.
  """

  def _tpu(self):
    jax.clear_caches()
    self.addCleanup(jax.clear_caches)
    return mock.patch.object(jax, 'default_backend', return_value='tpu')

  def _assert_close(self, actual, desired, scale, widened=False, rel=1e-5):
    # Mocked-TPU limited float32 interpolation widens the bounds by
    # `_BF16X3_BOUND_WIDENING` (relative), as it does inside the windows.
    atol = rel * scale
    if widened:
      atol += 2 * semi_lagrangian._BF16X3_BOUND_WIDENING * scale
    np.testing.assert_allclose(actual, desired, rtol=0, atol=atol)

  @parameterized.product(
      order=('linear', 'cubic'),
      vertical_order=('linear', 'cubic'),
      limiter=(None, 'quasi_monotone'),
      use_ref=(False, True),
  )
  def test_interpolate_3d_matmul_matches_gather(
      self, order, vertical_order, limiter, use_ref
  ):
    if use_ref and limiter is None:
      return
    grid = spherical_harmonic.Grid.T21()
    layers = 12
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    lon, sin_lat, sigma = _far_queries(grid, layers, seed=1)
    rng = np.random.RandomState(2)
    # strictly positive, so a dropped stencil weight would show up as a zero.
    field = jnp.asarray(
        1.0 + np.abs(rng.normal(size=(2, layers) + grid.nodal_shape)),
        jnp.float32,
    )
    ref = (
        jnp.asarray(np.linspace(220.0, 295.0, layers), jnp.float32)
        if use_ref
        else None
    )
    kw = dict(
        order=order,
        limiter=limiter,
        vertical_order=vertical_order,
        limiter_bound_reference=ref,
        max_lat_shift=_SHIFT,
        max_vertical_shift=_SHIFT,
    )
    interp = lambda method, **k: semi_lagrangian.interpolate_3d(
        field, grid, lon, sin_lat, sigma, centers, method=method, **(kw | k)
    )
    out_g = interp('gather')
    out_m = interp('matmul')
    with self._tpu():
      out_t = interp('matmul')
    scale = float(jnp.max(field)) + (295.0 if use_ref else 0.0)
    self._assert_close(out_m, out_g, scale)
    self._assert_close(
        out_t, out_m, scale, widened=limiter is not None, rel=2e-5
    )
    for out in (out_m, out_t):
      self.assertGreater(float(jnp.min(out)), 0.0)

    # The departures are beyond the windows: the clamp must act.
    out_unclamped = interp('gather', max_lat_shift=None, max_vertical_shift=None)
    self.assertGreater(float(jnp.max(jnp.abs(out_unclamped - out_g))), 1e-2)

    # Gather with windows interpolates at the independently clamped points.
    sl_c, sg_c = _reference_clamp(grid, centers, sin_lat, sigma, order, _SHIFT)
    out_ref = semi_lagrangian.interpolate_3d(
        field, grid, lon, sl_c, sg_c, centers, method='gather',
        **(kw | dict(max_lat_shift=None, max_vertical_shift=None)),
    )
    self._assert_close(out_g, out_ref, scale, rel=2e-5)

  @parameterized.parameters('linear', 'cubic')
  def test_mixed_limiting_and_trilinear(self, order):
    grid = spherical_harmonic.Grid.T21()
    layers = 8
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    lon, sin_lat, sigma = _far_queries(grid, layers, seed=3)
    rng = np.random.RandomState(4)
    fields = jnp.asarray(
        1.0 + np.abs(rng.normal(size=(5, layers) + grid.nodal_shape)),
        jnp.float32,
    )

    def mixed(method):
      return semi_lagrangian._interpolate_3d_mixed(
          fields, grid, lon, sin_lat, sigma, centers, order=order,
          limiter='quasi_monotone', limited_channel_start=2,
          trilinear_channel_start=3, vertical_order='cubic', method=method,
          max_lat_shift=_SHIFT, max_vertical_shift=_SHIFT,
      )

    high_g, low_g = mixed('gather')
    for backend in ('cpu', 'tpu'):
      with self.subTest(backend):
        if backend == 'tpu':
          with self._tpu():
            high_m, low_m = mixed('matmul')
        else:
          high_m, low_m = mixed('matmul')
        scale = float(jnp.max(fields))
        self._assert_close(high_m, high_g, scale, widened=backend == 'tpu')
        self._assert_close(low_m, low_g, scale, rel=2e-5)
        self.assertGreater(float(jnp.min(high_m)), 0.0)
        self.assertGreater(float(jnp.min(low_m)), 0.0)
    # The trilinear output is the trilinear interpolation at the same
    # (clamped) query, also when computed on its own.
    low_alone = semi_lagrangian.interpolate_3d(
        fields[3:], grid, lon, sin_lat, sigma, centers, order='linear',
        vertical_order='linear', method='gather', max_lat_shift=_SHIFT,
        max_vertical_shift=_SHIFT,
    )
    self._assert_close(low_g, low_alone, float(jnp.max(fields)))

  def test_shifts_none_disables_the_clamp(self):
    grid = spherical_harmonic.Grid.T21()
    layers = 8
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    lon, sin_lat, sigma = _far_queries(grid, layers, seed=5)
    field = jnp.asarray(
        np.random.RandomState(6).normal(size=(layers,) + grid.nodal_shape),
        jnp.float32,
    )
    interp = lambda **k: semi_lagrangian.interpolate_3d(
        field, grid, lon, sin_lat, sigma, centers, order='cubic',
        vertical_order='cubic', method='gather', **k
    )
    unclamped = interp(max_lat_shift=None, max_vertical_shift=None)
    # Either shift None disables the clamp in both directions (as it
    # disables the 'matmul' windows).
    np.testing.assert_array_equal(
        interp(max_lat_shift=_SHIFT, max_vertical_shift=None), unclamped
    )
    np.testing.assert_array_equal(
        interp(max_lat_shift=None, max_vertical_shift=_SHIFT), unclamped
    )
    # Windows wider than the displacements leave every query unchanged.
    wide = interp(max_lat_shift=10, max_vertical_shift=10)
    np.testing.assert_array_equal(wide, unclamped)
    # Arbitrary (not grid-aligned) query layouts are never clamped.
    pts = (lon[:, :5, :7], sin_lat[:, :5, :7], sigma[:, :5, :7])
    for method in ('gather', 'matmul'):
      out = semi_lagrangian.interpolate_3d(
          field, grid, *pts, centers, order='cubic', vertical_order='cubic',
          method=method, max_lat_shift=_SHIFT, max_vertical_shift=_SHIFT,
      )
      np.testing.assert_allclose(
          out, unclamped[:, :5, :7], rtol=1e-5, atol=1e-5
      )

  @parameterized.parameters(('linear', None), ('cubic', 'quasi_monotone'))
  def test_transport_matmul_matches_gather(self, order, limiter):
    grid = spherical_harmonic.Grid.T21()
    layers = 8
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    lon, sin_lat, sigma = _far_queries(grid, layers, seed=7)
    departure = semi_lagrangian.DeparturePoints(
        cartesian=semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat),
        sigma=sigma,
    )
    departure_2d = semi_lagrangian.DeparturePoints(
        cartesian=semi_lagrangian.lon_lat_to_cartesian(lon[0], sin_lat[0])
    )
    departure_2d_levels = semi_lagrangian.DeparturePoints(
        cartesian=departure.cartesian
    )
    rng = np.random.RandomState(8)
    tracers = jnp.asarray(
        1.0 + np.abs(rng.normal(size=(3, layers) + grid.nodal_shape)),
        jnp.float32,
    )
    u, v = jnp.asarray(rng.normal(size=(2, layers) + grid.nodal_shape),
                       jnp.float32)
    t_ref = jnp.asarray(np.linspace(220.0, 295.0, layers), jnp.float32)

    def run_all(method):
      interp = semi_lagrangian.GridInterpolator(
          grid, order, limiter, method=method, max_lat_shift=_SHIFT
      )
      kw = dict(vertical_order='cubic', max_vertical_shift=_SHIFT)
      return dict(
          scalar=semi_lagrangian.transport_scalar(
              tracers, departure, vertical, interp, **kw
          ),
          scalar_ref=semi_lagrangian.transport_scalar(
              tracers[0], departure, vertical, interp,
              limiter_bound_reference=t_ref, **kw
          ),
          wind=jnp.stack(semi_lagrangian.transport_wind(
              u, v, departure, vertical, interp, planetary_rotation_rate=0.7,
              **kw
          )),
          scalar_2d=semi_lagrangian.transport_scalar_2d(
              tracers[0, 0], departure_2d, interp
          ),
          scalar_2d_levels=semi_lagrangian.transport_scalar_2d(
              tracers, departure_2d_levels, interp
          ),
          wind_2d=jnp.stack(semi_lagrangian.transport_wind_2d(
              u, v, departure_2d_levels, interp, planetary_rotation_rate=0.7
          )),
          levels=interp(tracers[1], lon, sin_lat),
      )

    out_g = run_all('gather')
    out_m = run_all('matmul')
    with self._tpu():
      out_t = run_all('matmul')
    for name, ref in out_g.items():
      with self.subTest(name):
        scale = float(jnp.max(jnp.abs(ref)))
        self._assert_close(out_m[name], ref, scale)
        self._assert_close(
            out_t[name], out_m[name], scale + (295.0 * (name == 'scalar_ref')),
            widened=limiter is not None, rel=2e-5,
        )
        if name in ('scalar', 'scalar_ref', 'scalar_2d', 'scalar_2d_levels',
                    'levels'):
          self.assertGreater(float(jnp.min(out_m[name])), 0.0)
          self.assertGreater(float(jnp.min(out_t[name])), 0.0)

  def test_departure_points_matmul_matches_gather(self):
    grid = spherical_harmonic.Grid.T21()
    layers = 24
    coords = coordinate_systems.CoordinateSystem(
        grid, sigma_coordinates.SigmaCoordinates.equidistant(layers)
    )
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    cos_lat = np.sqrt(1 - sin_lat_mesh**2)
    k = np.arange(layers)[:, np.newaxis, np.newaxis]
    # winds carrying midpoints several cells beyond the midpoint windows
    # (half-width 1 for shifts of 2).
    u = 0.6 * cos_lat * (1 + 0.3 * np.sin(2 * lon_mesh + k))
    v = 0.6 * cos_lat * np.sin(lon_mesh + 0.5 * k)
    b = coords.vertical.boundaries[:, np.newaxis, np.newaxis]
    sigma_dot = np.broadcast_to(
        0.25 * np.sin(np.pi * b) * (1 + 0.5 * np.cos(lon_mesh)),
        (layers + 1,) + grid.nodal_shape,
    )
    u, v, sigma_dot = (jnp.asarray(x, jnp.float32) for x in (u, v, sigma_dot))
    kw = dict(iterations=3, max_lat_shift=_SHIFT, max_vertical_shift=_SHIFT)

    def solve(method):
      dep = semi_lagrangian.departure_points_3d(
          u, v, sigma_dot, coords, 1.0, method=method, **kw
      )
      dep_2d = semi_lagrangian.horizontal_departure_points(
          u, v, grid, 1.0, method=method, iterations=3, max_lat_shift=_SHIFT
      )
      return dep.cartesian, dep.sigma, dep_2d.cartesian

    out_g = solve('gather')
    out_m = solve('matmul')
    with self._tpu():
      out_t = solve('matmul')
    for a, b_, c in zip(out_g, out_m, out_t):
      np.testing.assert_allclose(b_, a, atol=2e-5)
      np.testing.assert_allclose(c, b_, atol=2e-5)
    # the departures really are far: several cells, beyond the windows.
    unclamped = semi_lagrangian.departure_points_3d(
        u, v, sigma_dot, coords, 1.0, method='gather', iterations=3
    )
    self.assertGreater(
        float(jnp.max(jnp.abs(unclamped.sigma - out_g[1]))), 1e-2
    )
    self.assertGreater(
        float(jnp.max(jnp.abs(unclamped.cartesian - out_g[0]))), 1e-2
    )

  def test_fused_wind_and_sigma_dot_match_separate_interpolations(self):
    grid = spherical_harmonic.Grid.T21()
    layers = 8
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    centers, boundaries = vertical.centers, vertical.boundaries
    lon, sin_lat, sigma = _far_queries(grid, layers, seed=9)
    rng = np.random.RandomState(10)
    wind = jnp.asarray(rng.normal(size=(3, layers) + grid.nodal_shape),
                       jnp.float32)
    sd = jnp.asarray(rng.normal(size=(layers + 1,) + grid.nodal_shape),
                     jnp.float32)
    point = (lon, sin_lat, sigma)
    shifts = dict(max_lat_shift=_SHIFT, max_vertical_shift=_SHIFT)
    separate = lambda method: (
        semi_lagrangian.interpolate_3d(
            wind, grid, *point, centers, order='linear', method=method,
            **shifts,
        ),
        semi_lagrangian.interpolate_3d(
            sd, grid, *point, boundaries, order='linear', method=method,
            **shifts,
        ),
    )
    ref_w, ref_sd = separate('gather')
    fused_g = semi_lagrangian._interpolate_3d_wind_and_sigma_dot_gather(
        wind, sd, grid, *point, centers, boundaries, **shifts
    )
    fused_m = semi_lagrangian._interpolate_3d_wind_and_sigma_dot_matmul(
        wind, sd, grid, *point, centers, boundaries, **shifts
    )
    with self._tpu():
      fused_t = semi_lagrangian._interpolate_3d_wind_and_sigma_dot_matmul(
          wind, sd, grid, *point, centers, boundaries, **shifts
      )
      sep_t = separate('matmul')
    for w, s in (fused_g, fused_m, fused_t, separate('matmul'), sep_t):
      np.testing.assert_allclose(w, ref_w, rtol=0, atol=2e-5)
      np.testing.assert_allclose(s, ref_sd, rtol=0, atol=2e-5)

  @parameterized.parameters('cpu', 'tpu')
  def test_clamped_values_are_bounded_by_the_window(self, backend):
    grid = spherical_harmonic.Grid.T21()
    layers = 10
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    lon, sin_lat, sigma = _far_queries(grid, layers, seed=11)
    rng = np.random.RandomState(12)
    field_np = 1.0 + np.abs(rng.normal(size=(layers,) + grid.nodal_shape))
    field = jnp.asarray(field_np, jnp.float32)
    interp = lambda: semi_lagrangian.interpolate_3d(
        field, grid, lon, sin_lat, sigma, centers, order='linear',
        vertical_order='linear', method='matmul', max_lat_shift=_SHIFT,
        max_vertical_shift=_SHIFT,
    )
    if backend == 'tpu':
      with self._tpu():
        out = np.asarray(interp())
    else:
      out = np.asarray(interp())
    # (Bi/tri)linear interpolation inside the window of arrival (k, j) is a
    # convex combination of nodes on rows j - S - 1 ... j + S + 1 (clipped;
    # the cross-pole halo rows are other longitudes of the polar rows) and
    # levels k - S ... k + S + 1.
    n_lat = grid.latitude_nodes
    for kk in range(layers):
      z = slice(max(kk - _SHIFT, 0), kk + _SHIFT + 2)
      for j in range(n_lat):
        rows = slice(max(j - _SHIFT - 1, 0), min(j + _SHIFT + 2, n_lat))
        window = field_np[z, :, rows]
        col = out[kk, :, j]
        self.assertGreaterEqual(col.min(), window.min() * (1 - 1e-6))
        self.assertLessEqual(col.max(), window.max() * (1 + 1e-6))

  def test_pole_adjacent_and_end_levels(self):
    grid = spherical_harmonic.Grid.T21()
    layers = 8
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    n_lat = grid.latitude_nodes
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    phi_mesh = np.arcsin(sin_lat_mesh)
    phi_ext = semi_lagrangian._latitude_halo_axis(grid)
    shape = (layers,) + grid.nodal_shape
    rows = np.arange(n_lat)
    polar = np.minimum(rows, n_lat - 1 - rows) < 2
    north = rows >= n_lat // 2
    # On the two rows nearest each pole: departures poleward of the row
    # (towards the pole, within the cell over the pole: never clamped) on
    # even longitudes, and 5 rows equatorward (clamped) on odd ones.
    poleward = np.where(
        north, 0.5 * (phi_mesh + np.pi / 2), 0.5 * (phi_mesh - np.pi / 2)
    )
    equatorward = phi_ext[np.clip(rows + 2 + np.where(north, -5, 5), 0, None)]
    even = (np.arange(grid.nodal_shape[0]) % 2 == 0)[:, np.newaxis]
    phi = np.where(polar, np.where(even, poleward, equatorward), phi_mesh)
    phi = np.broadcast_to(phi, shape)
    lon = np.broadcast_to(lon_mesh + 0.05, shape) % (2 * np.pi)
    # Top and bottom levels: departures 5 levels inward (clamped) and beyond
    # the end nodes (clipped to them, as without windows).
    sigma = np.broadcast_to(
        centers[:, np.newaxis, np.newaxis], shape
    ).copy()
    sigma[0] = np.where(even, centers[5], -0.5)
    sigma[-1] = np.where(even, centers[-6], 1.5)
    lon, sin_lat, sigma = (
        jnp.asarray(x, jnp.float32) for x in (lon, np.sin(phi), sigma)
    )
    field = jnp.asarray(
        1.0 + np.abs(np.random.RandomState(13).normal(size=shape)), jnp.float32
    )
    kw = dict(order='cubic', vertical_order='cubic',
              max_lat_shift=_SHIFT, max_vertical_shift=_SHIFT)
    interp = lambda m, s, **k: semi_lagrangian.interpolate_3d(
        field, grid, lon, s, sigma, centers, method=m, **(kw | k)
    )
    out_g = interp('gather', sin_lat)
    out_m = interp('matmul', sin_lat)
    with self._tpu():
      out_t = interp('matmul', sin_lat)
    scale = float(jnp.max(field))
    self._assert_close(out_m, out_g, scale)
    self._assert_close(out_t, out_g, scale, rel=2e-5)
    unclamped = interp(
        'gather', sin_lat, max_lat_shift=None, max_vertical_shift=None
    )
    # Middle levels: polar departures towards the pole are not clamped,
    # equatorward ones are (and differ).
    mid = slice(1, layers - 1)
    pole_cols = np.nonzero(polar)[0]
    np.testing.assert_array_equal(
        out_g[mid][:, ::2][..., pole_cols], unclamped[mid][:, ::2][..., pole_cols]
    )
    self.assertGreater(
        float(jnp.max(jnp.abs(out_g[mid][:, 1::2][..., pole_cols]
                              - unclamped[mid][:, 1::2][..., pole_cols]))),
        1e-3,
    )
    # End levels: beyond the end nodes is the same as at them; 5 levels
    # inward is clamped to the window edge.
    sl_c, sg_c = _reference_clamp(grid, centers, sin_lat, sigma, 'cubic',
                                  _SHIFT)
    self.assertTrue(np.all(np.asarray(sg_c[0, 1::2]) == np.float32(centers[0])))
    self.assertTrue(np.all(np.asarray(sg_c[-1, 1::2])
                           == np.float32(centers[-1])))
    self.assertTrue(np.all(np.asarray(sg_c[0, ::2]) < centers[3]))
    out_ref = semi_lagrangian.interpolate_3d(
        field, grid, lon, sl_c, sg_c, centers, method='gather',
        **(kw | dict(max_lat_shift=None, max_vertical_shift=None)),
    )
    self._assert_close(out_g, out_ref, scale, rel=2e-5)

  def test_gradients_through_clamped_queries(self):
    grid = spherical_harmonic.Grid.T21()
    layers = 6
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    lon, sin_lat, sigma = _far_queries(grid, layers, seed=14)
    # include points exactly on a pole.
    sin_lat = sin_lat.at[:, 0, :2].set(-1.0).at[:, 1, -2:].set(1.0)
    field = jnp.asarray(
        np.random.RandomState(15).normal(size=(layers,) + grid.nodal_shape),
        jnp.float32,
    )

    def loss(method, f, x, y, z):
      out = semi_lagrangian.interpolate_3d(
          f, grid, x, y, z, centers, order='cubic', vertical_order='cubic',
          method=method, max_lat_shift=_SHIFT, max_vertical_shift=_SHIFT,
      )
      return jnp.sum(out**2)

    args = (field, lon, sin_lat, sigma)
    grads = {}
    for method in ('gather', 'matmul'):
      grads[method] = jax.grad(
          functools.partial(loss, method), argnums=(0, 1, 2, 3)
      )(*args)
      for g in grads[method]:
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))
    for g_m, g_g in zip(grads['matmul'], grads['gather']):
      scale = float(jnp.max(jnp.abs(g_g)))
      np.testing.assert_allclose(g_m, g_g, rtol=0, atol=1e-4 * scale)
    # a clamped coordinate has zero gradient.
    _, sg_c = _reference_clamp(grid, centers, sin_lat, sigma, 'cubic', _SHIFT)
    clamped = np.asarray(sg_c != sigma) & (np.asarray(sigma) > centers[0]) & (
        np.asarray(sigma) < centers[-1]
    )
    self.assertGreater(clamped.sum(), 0)
    np.testing.assert_array_equal(np.asarray(grads['gather'][3])[clamped], 0.0)
    np.testing.assert_array_equal(np.asarray(grads['matmul'][3])[clamped], 0.0)


if __name__ == '__main__':
  absltest.main()
