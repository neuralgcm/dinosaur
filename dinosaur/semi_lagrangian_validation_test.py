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

"""Validation tests for passive semi-Lagrangian transport.

Standard 2-D transport test cases with prescribed winds on the unit sphere:

- Williamson et al. (1992) case 1: a cosine bell advected by solid-body
  rotation at several orientations, including directly over the poles.
- Nair & Lauritzen (2010) non-divergent deformational flow (their case 4,
  with background rotation), which stretches tracers into thin filaments and
  returns them to the initial condition at t = T.
- A positivity stress case mirroring the aerosol/chemistry use case: a sharp
  Gaussian hill on a zero background, where spectral transport produces
  Gibbs ringing and negative values while semi-Lagrangian transport with the
  quasi-monotone limiter stays non-negative.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import primitive_equations
from dinosaur import semi_lagrangian
from dinosaur import spherical_harmonic
from dinosaur import time_integration
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


def cosine_bell(r: np.ndarray, center: np.ndarray, radius: float = 1 / 3):
  """Williamson et al. (1992) cosine bell of unit amplitude."""
  distance = np.arccos(np.clip(np.einsum('c...,c->...', r, center), -1, 1))
  return np.where(
      distance < radius, 0.5 * (1 + np.cos(np.pi * distance / radius)), 0.0
  )


def gaussian_hills(r: np.ndarray, width: float = 5.0) -> np.ndarray:
  """Sum of two Gaussian hills (Nair & Lauritzen 2010, Eq. 11-13)."""
  center1 = np.array([np.cos(5 * np.pi / 6), np.sin(5 * np.pi / 6), 0.0])
  center2 = np.array([np.cos(7 * np.pi / 6), np.sin(7 * np.pi / 6), 0.0])
  total = np.zeros(r.shape[1:])
  for center in [center1, center2]:
    center = center.reshape((3,) + (1,) * (r.ndim - 1))
    squared_distance = ((r - center) ** 2).sum(0)
    total += 0.95 * np.exp(-width * squared_distance)
  return total


def solid_body_winds_tilted(grid, alpha: float, u0: float):
  """Williamson case 1 winds: solid-body rotation tilted by angle alpha.

  The rotation axis is (-sin(alpha), 0, cos(alpha)); alpha = π/2 advects
  directly over both poles.
  """
  lon, sin_lat = grid.nodal_mesh
  cos_lat = np.sqrt(1 - sin_lat**2)
  u = u0 * (cos_lat * np.cos(alpha) + np.cos(lon) * sin_lat * np.sin(alpha))
  v = -u0 * np.sin(lon) * np.sin(alpha)
  return u, v


def normalized_errors(grid, actual, expected):
  """Williamson-style normalized l2 and linf error norms."""
  actual = np.asarray(actual)
  expected = np.asarray(expected)
  l2 = np.sqrt(grid.integrate((actual - expected) ** 2)) / np.sqrt(
      grid.integrate(expected**2)
  )
  linf = np.abs(actual - expected).max() / np.abs(expected).max()
  return float(l2), float(linf)


def advect_static_flow(grid, u, v, field, dt, steps, order, monotone):
  """Advects `field` by a steady flow with repeated SL transport steps.

  For a steady flow the departure points are the same every step, so they are
  computed once and the transport operator is applied `steps` times.
  """
  departure = semi_lagrangian.horizontal_departure_points(
      jnp.asarray(u), jnp.asarray(v), grid, dt=dt
  )
  interpolator = semi_lagrangian.GridInterpolator(grid, order, monotone)
  transport = functools.partial(
      semi_lagrangian.transport_scalar_2d,
      departure=departure,
      interpolator=interpolator,
  )
  step = jax.jit(lambda x: transport(x))
  return time_integration.repeated(step, steps)(jnp.asarray(field))


class WilliamsonCase1Test(parameterized.TestCase):
  """Cosine bell advection by solid-body rotation (Williamson case 1)."""

  @parameterized.named_parameters(
      dict(testcase_name='along_equator', alpha=0.0),
      dict(testcase_name='tilted', alpha=np.pi / 4),
      dict(testcase_name='over_poles', alpha=np.pi / 2),
  )
  def test_cosine_bell_one_revolution(self, alpha):
    grid = spherical_harmonic.Grid.T42()
    lon, sin_lat = grid.nodal_mesh
    r = np.stack(semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat))
    center = np.array([np.cos(3 * np.pi / 2), np.sin(3 * np.pi / 2), 0.0])
    field0 = cosine_bell(r, center)

    u0 = 2 * np.pi  # one revolution per unit time
    steps = 75  # three quarters of a revolution
    dt = 1.0 / 100  # max CFL ~1.3 at T42: beyond Eulerian limits
    u, v = solid_body_winds_tilted(grid, alpha, u0)
    result = advect_static_flow(
        grid, u, v, field0, dt, steps, order='cubic', monotone=False
    )
    # Compare against the analytically rotated initial condition (a partial
    # revolution, so a no-op transport cannot pass). The error is dominated
    # by accumulated interpolation (remap) error, so it *decreases* with
    # larger dt — the defining semi-Lagrangian property (measured: l2 = 0.07
    # at 100 steps vs 0.16 at 250 steps for a full revolution).
    axis = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
    exact = cosine_bell(
        np.einsum(
            'ab,b...->a...', rotation_matrix(axis, -u0 * steps * dt), r
        ),
        center,
    )
    l2, linf = normalized_errors(grid, result, exact)
    self.assertLess(l2, 0.1)
    self.assertLess(linf, 0.1)

  def test_error_decreases_with_resolution(self):
    errors = {}
    for name, grid in [('T21', spherical_harmonic.Grid.T21()),
                       ('T42', spherical_harmonic.Grid.T42())]:
      lon, sin_lat = grid.nodal_mesh
      r = np.stack(semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat))
      center = np.array([0.0, -1.0, 0.0])
      field0 = cosine_bell(r, center)
      u, v = solid_body_winds_tilted(grid, np.pi / 2, 2 * np.pi)
      result = advect_static_flow(
          grid, u, v, field0, dt=1 / 100, steps=100, order='cubic',
          monotone=False,
      )
      errors[name], _ = normalized_errors(grid, result, field0)
    # measured ratio ~5x (l2 of 0.37 vs 0.07).
    self.assertLess(errors['T42'], 0.3 * errors['T21'])

  def test_limiter_preserves_bounds_and_accuracy(self):
    grid = spherical_harmonic.Grid.T42()
    lon, sin_lat = grid.nodal_mesh
    r = np.stack(semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat))
    center = np.array([0.0, -1.0, 0.0])
    field0 = cosine_bell(r, center)
    u, v = solid_body_winds_tilted(grid, np.pi / 2, 2 * np.pi)
    steps = 75  # three quarters of a revolution: a no-op transport fails
    result = advect_static_flow(
        grid, u, v, field0, dt=1 / 100, steps=steps, order='cubic',
        monotone=True,
    )
    result = np.asarray(result)
    axis = np.array([-1.0, 0.0, 0.0])
    exact = cosine_bell(
        np.einsum(
            'ab,b...->a...',
            rotation_matrix(axis, -2 * np.pi * steps / 100),
            r,
        ),
        center,
    )
    with self.subTest('bounds preserved'):
      self.assertGreaterEqual(result.min(), 0.0)
      self.assertLessEqual(result.max(), field0.max() + 1e-6)
    with self.subTest('accuracy'):
      l2, _ = normalized_errors(grid, result, exact)
      self.assertLess(l2, 0.12)
    with self.subTest('mass approximately conserved'):
      # semi-Lagrangian transport does not conserve mass exactly; the
      # limiter clips the strong bell edges (measured drift ~3%).
      drift = abs(
          float(grid.integrate(result) - grid.integrate(field0))
      ) / float(grid.integrate(field0))
      self.assertLess(drift, 0.05)


class DeformationalFlowTest(parameterized.TestCase):
  """Nair & Lauritzen (2010) non-divergent deformational flow."""

  def nair_lauritzen_winds(self, grid, t, period, k=2.0):
    """Non-divergent case-4 winds (deformation + background rotation)."""
    lon, sin_lat = grid.nodal_mesh
    cos_lat = np.sqrt(1 - sin_lat**2)
    lat = np.arcsin(sin_lat)
    lon_prime = lon - 2 * np.pi * t / period
    u = (
        k * np.sin(lon_prime) ** 2 * np.sin(2 * lat) * np.cos(np.pi * t / period)
        + 2 * np.pi * cos_lat / period
    )
    v = k * np.sin(2 * lon_prime) * cos_lat * np.cos(np.pi * t / period)
    return u, v

  def test_gaussian_hills_return_to_initial(self):
    grid = spherical_harmonic.Grid.T42()
    lon, sin_lat = grid.nodal_mesh
    r = np.stack(semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat))
    field0 = gaussian_hills(r)

    period = 5.0
    steps = 256
    dt = period / steps
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')

    @jax.jit
    def step(field, u, v):
      departure = semi_lagrangian.horizontal_departure_points(
          u, v, grid, dt=dt
      )
      return semi_lagrangian.transport_scalar_2d(
          field, departure, interpolator
      )

    field = jnp.asarray(field0)
    half_time_field = None
    for n in range(steps):
      # winds at the step midpoint give second-order trajectories for
      # time-dependent flows.
      u, v = self.nair_lauritzen_winds(grid, (n + 0.5) * dt, period)
      field = step(field, jnp.asarray(u), jnp.asarray(v))
      if n == steps // 2 - 1:
        half_time_field = np.asarray(field)

    with self.subTest('flow deforms the field at half time'):
      l2_mid, _ = normalized_errors(grid, half_time_field, field0)
      self.assertGreater(l2_mid, 0.5)
    with self.subTest('field returns to initial condition'):
      # at half time the tracer is drawn into filaments too thin for T42, so
      # the return error is resolution-limited: measured l2 = 0.24 at T42
      # and 0.021 at T85 with the same 256 steps.
      l2, linf = normalized_errors(grid, field, field0)
      self.assertLess(l2, 0.35)
      self.assertLess(linf, 0.45)


class PositivityTest(parameterized.TestCase):
  """Sharp tracer on a zero background: the aerosol/chemistry failure mode."""

  def spectral_advection(self, grid, u, v, field, dt, steps):
    """Pseudo-spectral flux-form advection with an RK3 stepper."""
    cos_lat_u = jnp.asarray(u) * grid.cos_lat
    cos_lat_v = jnp.asarray(v) * grid.cos_lat

    def explicit_terms(modal):
      nodal = grid.to_nodal(modal)
      return -primitive_equations.div_sec_lat(
          cos_lat_u * nodal, cos_lat_v * nodal, grid
      )

    equation = time_integration.ImplicitExplicitODE.from_functions(
        explicit_terms,
        lambda x: jnp.zeros_like(x),
        lambda x, eta: x,
    )
    step = jax.jit(time_integration.crank_nicolson_rk3(equation, dt))
    modal = grid.to_modal(jnp.asarray(field))
    modal = time_integration.repeated(step, steps)(modal)
    return grid.to_nodal(modal)

  def test_sharp_hill_positivity(self):
    grid = spherical_harmonic.Grid.T42()
    lon, sin_lat = grid.nodal_mesh
    r = np.stack(semi_lagrangian.lon_lat_to_cartesian(lon, sin_lat))
    # hills ~1.5 grid cells wide on an exactly zero background, advected
    # along the equator (where the spectral baseline is stable; tilted
    # flows make the unfiltered pseudo-spectral operator pole-unstable).
    width = 400.0
    field0 = np.asarray(gaussian_hills(r, width=width))
    u, v = solid_body_winds_tilted(grid, 0.0, 2 * np.pi)

    revolution_fraction = 0.25
    sl_steps = 16  # CFL ~2: fewer remaps preserve the sharp hill better
    sl_dt = revolution_fraction / sl_steps
    limited = advect_static_flow(
        grid, u, v, field0, sl_dt, sl_steps, order='cubic', monotone=True
    )
    unlimited = advect_static_flow(
        grid, u, v, field0, sl_dt, sl_steps, order='cubic', monotone=False
    )
    spectral_steps = 512
    spectral = self.spectral_advection(
        grid, u, v, field0, revolution_fraction / spectral_steps,
        spectral_steps,
    )
    exact = gaussian_hills(
        np.einsum(
            'ab,b...->a...',
            rotation_matrix([0, 0, 1], -2 * np.pi * revolution_fraction),
            r,
        ),
        width=width,
    )

    peak = field0.max()
    with self.subTest('spectral transport rings negative'):
      # measured -5.8% of peak: genuine Gibbs ringing of the sharp hill
      # (the transport itself is stable and accurate in this configuration).
      self.assertLess(np.asarray(spectral).min(), -0.02 * peak)
    with self.subTest('unlimited cubic undershoots'):
      self.assertLess(np.asarray(unlimited).min(), -1e-4 * peak)
    with self.subTest('limited transport stays non-negative'):
      self.assertGreaterEqual(np.asarray(limited).min(), 0.0)
    with self.subTest('limited transport does not amplify the peak'):
      self.assertLessEqual(np.asarray(limited).max(), peak * (1 + 1e-6))
    with self.subTest('limited transport retains the signal'):
      l2, _ = normalized_errors(grid, limited, exact)
      self.assertLess(l2, 0.5)
    with self.subTest('spectral accuracy for reference'):
      # the spectral core remains more accurate in l2 on this barely
      # resolved field — the SL win is positivity, not accuracy.
      l2_spectral, _ = normalized_errors(grid, spectral, exact)
      self.assertLess(l2_spectral, 0.5)


if __name__ == '__main__':
  absltest.main()
