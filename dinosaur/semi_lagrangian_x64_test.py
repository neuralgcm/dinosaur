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

"""Float64 tests for dinosaur.semi_lagrangian.

Convergence-order measurements and gradient/finite-difference comparisons
need double precision. JAX handles the x64 flag reliably only when set once
at startup (toggling it mid-process interacts badly with cached compilations
and already-created arrays), so — following the convention of
`time_integration_test.py` and `spherical_harmonic_test.py` — these tests
live in their own module that enables x64 at import time, rather than
switching modes per-test inside `semi_lagrangian_test.py`.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import semi_lagrangian
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import units
from dinosaur import xarray_utils
import jax
import jax.numpy as jnp
import numpy as np


jax.config.update('jax_enable_x64', True)
jax.config.parse_flags_with_absl()

s_units = scales.units


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


class DeparturePointsTest(parameterized.TestCase):

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

  def test_departure_points_with_physical_radius(self):
    """The dt/radius scaling must convert physical winds to angular motion."""
    radius = 6.371e6
    grid = spherical_harmonic.Grid.T42(radius=radius)
    omega = 2 * np.pi / (86400.0 * 12)  # one revolution per 12 days
    u, v = solid_body_winds(grid, axis=[0, 0, 1], omega=omega * radius)
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    dt = 3600.0  # one hour, in the same (SI) units as the winds
    departure = semi_lagrangian.horizontal_departure_points(
        jnp.asarray(u), jnp.asarray(v), grid, dt=dt
    )
    r_exact = rotation_matrix([0, 0, 1], -omega * dt) @ np.stack(
        semi_lagrangian.lon_lat_to_cartesian(lon_mesh, sin_lat_mesh)
    ).reshape(3, -1)
    error = np.linalg.norm(
        np.asarray(departure.cartesian).reshape(3, -1) - r_exact, axis=0
    )
    # angular displacement is ~6e-3 radians; errors are interpolation-level.
    self.assertLess(error.max(), 1e-5)

  def test_vertical_departure_convergence(self):
    """σ̇ interpolated at the trajectory midpoint beats frozen-at-arrival."""
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(16),
    )
    grid = coords.horizontal
    layers = coords.vertical.layers
    boundaries = coords.vertical.boundaries
    centers = coords.vertical.centers
    zeros = jnp.zeros((layers,) + grid.nodal_shape)
    # σ̇ = c·σ is linear in σ, so vertical interpolation is exact and all
    # error comes from the trajectory iteration. dσ/dt = cσ has the exact
    # solution σ_d = σ_a·exp(-c·dt).
    rate = 0.5
    sigma_dot = jnp.broadcast_to(
        rate * boundaries[:, np.newaxis, np.newaxis],
        (layers + 1,) + grid.nodal_shape,
    )
    dt = 0.6
    departure = semi_lagrangian.departure_points_3d(
        zeros, zeros, sigma_dot, coords, dt=dt
    )
    exact = centers * np.exp(-rate * dt)
    frozen_at_arrival = centers * (1 - rate * dt)
    # exclude layers whose exact departure point is clipped to the
    # layer-center range.
    unclipped = exact > centers[0]
    error = np.abs(np.asarray(departure.sigma)[:, 0, 0] - exact)[unclipped]
    frozen_error = np.abs(frozen_at_arrival - exact)[unclipped]
    # the midpoint iteration is second order: much closer than frozen winds.
    np.testing.assert_array_less(error, 0.2 * frozen_error)

  def test_vertical_departure_clipping(self):
    """Departure σ beyond the layer-center range is clipped on both sides."""
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(4),
    )
    grid = coords.horizontal
    layers = coords.vertical.layers
    centers = coords.vertical.centers
    zeros = jnp.zeros((layers,) + grid.nodal_shape)
    for rate in [2.0, -2.0]:  # strong downward and upward motion
      sigma_dot = rate * jnp.ones((layers + 1,) + grid.nodal_shape)
      departure = semi_lagrangian.departure_points_3d(
          zeros, zeros, sigma_dot, coords, dt=1.0
      )
      sigma = np.asarray(departure.sigma)
      self.assertGreaterEqual(sigma.min(), centers[0])
      self.assertLessEqual(sigma.max(), centers[-1])
      if rate > 0:
        np.testing.assert_allclose(sigma, centers[0], atol=1e-12)
      else:
        np.testing.assert_allclose(sigma, centers[-1], atol=1e-12)

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


class TransportTest(parameterized.TestCase):

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
    # errors here are only interpolation and departure-point errors
    # (measured ~6e-5; rotate=False fails at ~5e-3).
    np.testing.assert_allclose(u_rot[0], u_expected, atol=5e-4)
    np.testing.assert_allclose(v_rot[0], v_expected, atol=5e-4)

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


class _IdentityTransportODE(
    time_integration.SemiLagrangianImplicitExplicitODE
):
  """A semi-Lagrangian equation with zero velocities (identity transport)."""

  def __init__(self, base: time_integration.ImplicitExplicitODE):
    self.base = base

  def nonadvective_terms(self, state):
    return self.base.explicit_terms(state)

  def implicit_terms(self, state):
    return self.base.implicit_terms(state)

  def implicit_inverse(self, state, step_size):
    return self.base.implicit_inverse(state, step_size)

  def nodal_velocities(self, state):
    return jnp.zeros(())

  def departure_points(self, velocities, dt):
    del velocities, dt  # unused
    return None

  def semi_lagrangian_transport(self, bracket, departure):
    del departure  # unused
    return bracket


class _RingAdvectionODE(time_integration.SemiLagrangianImplicitExplicitODE):
  """DX/Dt = cos(θ) - γ X along dθ/dt = ω on a periodic ring.

  Transport is an exact spectral shift, so all discretization error comes
  from the time stepping. The exact solution is:

    X(θ, T) = e^(-γT) X₀(θ - ωT)
              + Re{ e^(iθ) (1 - e^(-(γ + iω)T)) / (γ + iω) }
  """

  def __init__(self, num_points: int, omega: float, gamma: float):
    self.theta = 2 * np.pi * jnp.arange(num_points) / num_points
    self.omega = omega
    self.gamma = gamma

  def nonadvective_terms(self, state):
    return jnp.cos(self.theta)

  def implicit_terms(self, state):
    return -self.gamma * state

  def implicit_inverse(self, state, step_size):
    return state / (1 + step_size * self.gamma)

  def nodal_velocities(self, state):
    return jnp.asarray(self.omega)

  def departure_points(self, velocities, dt):
    return velocities * dt

  def semi_lagrangian_transport(self, bracket, departure):
    num_points = self.theta.size
    k = jnp.fft.rfftfreq(num_points, d=1 / num_points)
    shifted = jnp.fft.rfft(bracket) * jnp.exp(-1j * k * departure)
    return jnp.fft.irfft(shifted, num_points)

  def exact_solution(self, state0, time):
    theta = np.asarray(self.theta)
    gamma, omega = self.gamma, self.omega
    decay = np.exp(-gamma * time)
    # initial condition advected and decayed (state0 must be band-limited).
    initial_term = decay * np.asarray(
        self.semi_lagrangian_transport(state0, omega * time)
    )
    forced = np.real(
        np.exp(1j * theta)
        * (1 - np.exp(-(gamma + 1j * omega) * time))
        / (gamma + 1j * omega)
    )
    return initial_term + forced


class _StateDependentRingODE(_RingAdvectionODE):
  """Ring advection whose velocity depends on the state.

  dθ/dt = ω₀ + c·mean(X): exercises the stepper's time-centered stage-2
  winds `(V(x) + V(x*)) / 2`, which constant-velocity toys cannot (their
  V(x*) equals V(x) identically).
  """

  def __init__(self, num_points, omega, gamma, coupling):
    super().__init__(num_points, omega, gamma)
    self.coupling = coupling

  def nodal_velocities(self, state):
    return self.omega + self.coupling * jnp.mean(state)


class _StateDependentForcingRingODE(_RingAdvectionODE):
  """Ring advection whose non-advective forcing depends on the state.

  DX/Dt = cos(θ)·(1 + c·mean(X)) − γX: exercises SETTLS's defining tendency
  extrapolation `2N^n − N^{n−1}`, which state-independent forcings cannot
  (they make `N^n ≡ N^{n−1}` identically).
  """

  def __init__(self, num_points, omega, gamma, coupling):
    super().__init__(num_points, omega, gamma)
    self.coupling = coupling

  def nonadvective_terms(self, state):
    return jnp.cos(self.theta) * (1 + self.coupling * jnp.mean(state))


class SemiLagrangianSteppersTest(parameterized.TestCase):

  def test_reduces_to_crank_nicolson_rk2_for_zero_velocities(self):
    rng = np.random.RandomState(0)
    state = {
        'a': jnp.asarray(rng.normal(size=(5, 7))),
        'b': jnp.asarray(rng.normal(size=(3,))),
    }
    base = time_integration.ImplicitExplicitODE.from_functions(
        explicit_terms=lambda x: jax.tree.map(jnp.sin, x),
        implicit_terms=lambda x: jax.tree.map(lambda a: -2.0 * a, x),
        implicit_inverse=lambda x, eta: jax.tree.map(
            lambda a: a / (1 + 2.0 * eta), x
        ),
    )
    dt = 0.3
    expected = time_integration.crank_nicolson_rk2(base, dt)(state)
    actual = time_integration.semi_lagrangian_crank_nicolson_rk2(
        _IdentityTransportODE(base), dt
    )(state)
    jax.tree.map(
        functools.partial(np.testing.assert_allclose, atol=1e-14),
        expected,
        actual,
    )

  def test_second_order_convergence(self):
    equation = _RingAdvectionODE(num_points=64, omega=1.3, gamma=0.7)
    state0 = jnp.sin(2 * equation.theta) + 0.5
    total_time = 1.0
    exact = equation.exact_solution(state0, total_time)

    def global_error(num_steps):
      dt = total_time / num_steps
      step = jax.jit(
          time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
      )
      state = time_integration.repeated(step, num_steps)(state0)
      return np.abs(np.asarray(state) - exact).max()

    errors = [global_error(n) for n in [8, 16, 32]]
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    for order in orders:
      self.assertGreater(order, 1.7)
      self.assertLess(order, 2.3)

  def test_second_order_convergence_state_dependent_velocity(self):
    """Second order requires the time-centered stage-2 winds ½(V(x)+V(x*))."""
    equation = _StateDependentRingODE(
        num_points=64, omega=1.1, gamma=0.4, coupling=2.0
    )
    state0 = jnp.sin(2 * equation.theta) + 0.8
    total_time = 1.0

    def solve(num_steps):
      dt = total_time / num_steps
      step = jax.jit(
          time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
      )
      return time_integration.repeated(step, num_steps)(state0)

    # self-convergence against a fine-step reference (no closed form:
    # trajectories depend on the evolving mean state).
    reference = np.asarray(solve(512))
    errors = [
        np.abs(np.asarray(solve(n)) - reference).max() for n in [8, 16, 32]
    ]
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    for order in orders:
      self.assertGreater(order, 1.7)
      self.assertLess(order, 2.4)

  def test_settls_second_order_convergence(self):
    """SETTLS with the RK2 bootstrap converges at second order."""
    equation = _RingAdvectionODE(num_points=64, omega=1.3, gamma=0.7)
    state0 = jnp.sin(2 * equation.theta) + 0.5
    total_time = 1.0
    exact = equation.exact_solution(state0, total_time)

    def global_error(num_steps):
      dt = total_time / num_steps
      init_fn = time_integration.semi_lagrangian_settls_init(equation, dt)
      step_fn = jax.jit(time_integration.semi_lagrangian_settls(equation, dt))
      carry = time_integration.repeated(step_fn, num_steps - 1)(init_fn(state0))
      return np.abs(np.asarray(carry[0]) - exact).max()

    errors = [global_error(n) for n in [8, 16, 32]]
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    for order in orders:
      self.assertGreater(order, 1.7)
      self.assertLess(order, 2.4)

  def test_settls_second_order_convergence_state_dependent_velocity(self):
    """SETTLS wind extrapolation is second order for evolving flows."""
    equation = _StateDependentRingODE(
        num_points=64, omega=1.1, gamma=0.4, coupling=2.0
    )
    state0 = jnp.sin(2 * equation.theta) + 0.8
    total_time = 1.0

    def solve(num_steps):
      dt = total_time / num_steps
      init_fn = time_integration.semi_lagrangian_settls_init(equation, dt)
      step_fn = jax.jit(time_integration.semi_lagrangian_settls(equation, dt))
      carry = time_integration.repeated(step_fn, num_steps - 1)(init_fn(state0))
      return carry[0]

    reference = np.asarray(solve(512))
    errors = [
        np.abs(np.asarray(solve(n)) - reference).max() for n in [8, 16, 32]
    ]
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    for order in orders:
      self.assertGreater(order, 1.7)
      self.assertLess(order, 2.4)

  def test_settls_second_order_convergence_state_dependent_forcing(self):
    """Pins SETTLS's tendency extrapolation `2N^n − N^{n−1}`.

    With state-dependent forcing, dropping the extrapolation (using `N^n`
    alone) or swapping the time levels reduces measured convergence to
    first order; the correct scheme stays second order.
    """
    equation = _StateDependentForcingRingODE(
        num_points=64, omega=1.1, gamma=0.4, coupling=2.0
    )
    state0 = jnp.sin(2 * equation.theta) + 0.8
    total_time = 1.0

    def solve(num_steps):
      dt = total_time / num_steps
      init_fn = time_integration.semi_lagrangian_settls_init(equation, dt)
      step_fn = jax.jit(time_integration.semi_lagrangian_settls(equation, dt))
      carry = time_integration.repeated(step_fn, num_steps - 1)(init_fn(state0))
      return carry[0]

    reference = np.asarray(solve(512))
    errors = [
        np.abs(np.asarray(solve(n)) - reference).max() for n in [8, 16, 32]
    ]
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    for order in orders:
      self.assertGreater(order, 1.7)
      self.assertLess(order, 2.4)

  def test_time_reversed_matches_eulerian_reversal_for_zero_velocities(self):
    rng = np.random.RandomState(1)
    state = jnp.asarray(rng.normal(size=(8,)))
    base = time_integration.ImplicitExplicitODE.from_functions(
        explicit_terms=lambda x: jnp.sin(x),
        implicit_terms=lambda x: -2.0 * x,
        implicit_inverse=lambda x, eta: x / (1 + 2.0 * eta),
    )
    dt = 0.2
    expected = time_integration.crank_nicolson_rk2(
        time_integration.TimeReversedImExODE(base), dt
    )(state)
    reversed_sl = time_integration.TimeReversedSemiLagrangianODE(
        _IdentityTransportODE(base)
    )
    actual = time_integration.semi_lagrangian_crank_nicolson_rk2(
        reversed_sl, dt
    )(state)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)

  def test_time_reversed_round_trip_is_nearly_exact(self):
    """Forward-then-backward integration returns the initial state.

    For this toy the ring velocity is spatially uniform, which makes the
    reversed step the near-exact algebraic inverse of the forward step
    (measured ~2e-14 in float64); any inconsistency in the time-reversed
    equation (wrong sign, wrong velocity, wrong implicit inverse) would
    instead show up at the O(dt²) ≈ 1e-3 discretization scale.
    """
    equation = _StateDependentRingODE(
        num_points=64, omega=1.1, gamma=0.4, coupling=2.0
    )
    reversed_equation = time_integration.TimeReversedSemiLagrangianODE(
        equation
    )
    state0 = jnp.sin(2 * equation.theta) + 0.8
    num_steps = 16
    dt = 0.5 / num_steps
    forward = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
        ),
        num_steps,
    )
    backward = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(
                reversed_equation, dt
            )
        ),
        num_steps,
    )
    round_trip_error = float(jnp.abs(backward(forward(state0)) - state0).max())
    self.assertLess(round_trip_error, 1e-9)

  def test_settls_step_filter(self):
    equation = _RingAdvectionODE(num_points=16, omega=1.0, gamma=0.5)
    state = jnp.sin(equation.theta)
    aux = (jnp.zeros_like(state), jnp.zeros(()))
    base_filter = time_integration.runge_kutta_step_filter(lambda x: 0.5 * x)
    settls_filter = time_integration.settls_step_filter(base_filter)
    filtered_state, filtered_aux = settls_filter(
        (state, aux), (state, aux)
    )
    np.testing.assert_allclose(filtered_state, 0.5 * state)
    jax.tree.map(np.testing.assert_array_equal, filtered_aux, aux)

  def test_off_centering(self):
    equation = _RingAdvectionODE(num_points=64, omega=1.3, gamma=0.7)
    state0 = jnp.sin(2 * equation.theta) + 0.5
    total_time = 1.0
    exact = equation.exact_solution(state0, total_time)

    def global_error(num_steps, off_centering):
      dt = total_time / num_steps
      step = jax.jit(
          time_integration.semi_lagrangian_crank_nicolson_rk2(
              equation, dt, off_centering=off_centering
          )
      )
      state = time_integration.repeated(step, num_steps)(state0)
      return np.abs(np.asarray(state) - exact).max()

    with self.subTest('off-centering costs accuracy'):
      self.assertGreater(global_error(16, 0.1), global_error(16, 0.0))
    with self.subTest('still converges, at reduced order'):
      errors = [global_error(n, 0.1) for n in [16, 32]]
      order = np.log2(errors[0] / errors[1])
      self.assertGreater(order, 0.7)


class SemiLagrangianGradientTest(parameterized.TestCase):
  """Gradients through full semi-Lagrangian primitive-equation steps."""

  def _setup(self, layers):
    physics_specs = units.SimUnits.from_si()
    grid = spherical_harmonic.Grid.T21()
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical)
    init_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    state = init_fn()
    state = state + primitive_equations_states.baroclinic_perturbation_jw(
        coords, physics_specs
    )
    orography = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords
    )
    equation = primitive_equations.SemiLagrangianPrimitiveEquations(
        aux_features[xarray_utils.REF_TEMP_KEY],
        orography,
        coords,
        physics_specs,
    )
    return equation, state

  def _nondim_minutes(self, physics_specs, minutes):
    return float(physics_specs.nondimensionalize(minutes * 60 * s_units.s))

  @parameterized.parameters(dict(stepper='rk2'), dict(stepper='settls'))
  def test_gradients_match_finite_differences(self, stepper):
    """jax.grad through SL steps agrees with finite differences.

    Differentiates two directional derivatives — a scale on the initial
    vorticity (through departure points and wind transport) and a scale on
    an initial tracer (through scalar transport) — against a loss over both
    temperature and the tracer. Runs in float64 so central differences
    resolve the gradient well below the comparison tolerance.
    """
    equation, state0 = self._setup(layers=4)
    physics_specs = equation.physics_specs
    state0.tracers = {
        'tracer': primitive_equations_states.gaussian_scalar(
            coords=equation.coords, physics_specs=physics_specs
        )
    }
    dt = self._nondim_minutes(physics_specs, 30)
    if stepper == 'rk2':
      step_fn = time_integration.semi_lagrangian_crank_nicolson_rk2(
          equation, dt
      )
      advance = lambda state: step_fn(step_fn(state))
    else:
      init_fn = time_integration.semi_lagrangian_settls_init(equation, dt)
      settls_fn = time_integration.semi_lagrangian_settls(equation, dt)
      advance = lambda state: settls_fn(init_fn(state))[0]

    @jax.jit
    def loss(scales):
      vorticity_scale, tracer_scale = scales
      state = state0.replace(
          vorticity=vorticity_scale * state0.vorticity,
          tracers={'tracer': tracer_scale * state0.tracers['tracer']},
      )
      out = advance(state)
      return jnp.sum(out.temperature_variation**2) + jnp.sum(
          out.tracers['tracer'] ** 2
      )

    ones = jnp.asarray([1.0, 1.0])
    gradient = np.asarray(jax.grad(loss)(ones))
    epsilon = 1e-4
    for direction in range(2):
      unit = jnp.zeros(2).at[direction].set(1.0)
      finite_difference = float(
          (loss(ones + epsilon * unit) - loss(ones - epsilon * unit))
          / (2 * epsilon)
      )
      np.testing.assert_allclose(
          gradient[direction], finite_difference, rtol=1e-4,
          err_msg=f'{direction=}',
      )


if __name__ == '__main__':
  absltest.main()
