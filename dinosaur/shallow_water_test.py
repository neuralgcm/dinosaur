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

"""Tests for shallow_water.

The validation tests below are based on

"A standard test set for numerical approximations to the shallow water equations
in spherical geometry"
David L.Williamson, John B.Drake, James J.Hack, Rüdiger Jakob,
Paul N.Swarztrauber
https://doi.org/10.1016/S0021-9991(05)80016-6

At present, the tests cover only test case 2 from the paper, "Steady State
Nonlinear Zonal Geostrophic Flow." We plan to add additional test cases as we
build out the feature set of the solver.
"""
import unittest

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import associated_legendre
from dinosaur import coordinate_systems
from dinosaur import layer_coordinates
from dinosaur import scales
from dinosaur import shallow_water
from dinosaur import shallow_water_states
from dinosaur import spherical_harmonic

import jax
import jax.numpy as jnp
import numpy as np

units = scales.units
jax.config.parse_flags_with_absl()


def _tpu_or_gpu_available():
  return jax.devices()[0].platform != 'cpu'


def assert_states_close(state0, state1, **kwargs):
  for field in state0.fields:
    np.testing.assert_allclose(getattr(state0, field.name),
                               getattr(state1, field.name),
                               err_msg=f'Mismatch in {field}:',
                               **kwargs)


def _get_mountain(grid, height, physics_specs):
  """Returns the orography for a mountain at (3π / 2, π / 6)."""
  mountain_geopotential = (
      physics_specs.nondimensionalize(height) * physics_specs.g)
  center_lon = 3 * np.pi / 2
  center_lat = np.pi / 6
  lon, sin_lat = grid.nodal_mesh
  lat = np.arcsin(sin_lat)
  r = np.pi / 9
  d = np.sqrt(np.minimum(r**2, (lon - center_lon)**2 + (lat - center_lat)**2))
  mountain_nodal = mountain_geopotential * (1 - d / r)
  mountain = grid.to_modal(mountain_nodal)
  return mountain


def _get_geopotential(grid, max_velocity, thickness, layers, physics_specs):
  """Mean geopotential and fluctuation corresponding to velocity u0 · sin 𝜃."""
  _, sin_lat = grid.nodal_mesh
  gh0 = physics_specs.nondimensionalize(thickness) * physics_specs.g
  max_v = physics_specs.nondimensionalize(max_velocity)
  total_geopotential = gh0 - (
      physics_specs.radius * physics_specs.angular_velocity * max_v +
      max_v ** 2 / 2) * sin_lat ** 2
  geopotential = jnp.stack([total_geopotential / layers] * layers)

  _, w = associated_legendre.gauss_legendre_nodes(grid.latitude_nodes)
  mean_geopotential = (
      (geopotential * w).sum((-1, -2)) / w.sum() / grid.longitude_nodes)
  delta_geopotential = grid.to_modal(
      geopotential - mean_geopotential[..., jnp.newaxis, jnp.newaxis])
  return mean_geopotential, delta_geopotential


def _compute_mass(grid, potential, mean_geopotential, density):
  """Computes the total mass in arbitrary units."""
  layers = density.shape[0]
  _, w = associated_legendre.gauss_legendre_nodes(grid.latitude_nodes)
  total_potential = (grid.to_nodal(potential) +
                     jnp.expand_dims(mean_geopotential, (1, 2)))
  volume = ((total_potential * w).sum((-1, -2)) /
            w.sum() / grid.longitude_nodes / layers)
  return (volume * density).sum(-1)


class ShallowWaterTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(wavenumbers=64,
           layers=1,
           velocity_function=lambda lat: np.cos(3 * lat) / 5,
           dt=1e-3,
           density_ratio=.9,
           mean_potential=1/10,
           inner_steps=1000,
           outer_steps=10),
      dict(wavenumbers=64,
           layers=4,
           velocity_function=(lambda lat: np.cos(lat) / 5),
           dt=1e-3,
           density_ratio=.9,
           mean_potential=1/10,
           inner_steps=1000,
           outer_steps=10),
      dict(wavenumbers=128,
           layers=2,
           velocity_function=(lambda lat: np.cos(lat) ** 2 / 5),
           dt=1e-4,
           density_ratio=.9,
           mean_potential=1/10,
           inner_steps=1000,
           outer_steps=10),
  )
  def testSteadyStateGeostrophicFlow(self, wavenumbers, layers,
                                     velocity_function, dt, density_ratio,
                                     mean_potential, inner_steps, outer_steps):
    """Tests steady state zonal geostrophic flow."""

    if not _tpu_or_gpu_available():
      # TODO(shoyer): speed up these tests, fast enough to include in CI on
      # GitHub!
      raise unittest.SkipTest('test is too slow to run on CPU')

    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    vertical_grid = layer_coordinates.LayerCoordinates(layers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    density = np.array([density_ratio ** n for n in range(layers)][::-1])
    physics_specs = shallow_water.ShallowWaterSpecs.from_si(
        density * scales.WATER_DENSITY)
    mean_potential = np.ones(layers) * mean_potential
    orography = None  # no orography in the geostrophic flow test case.

    # Set up time integration of the shallow water equations.
    default_filters = shallow_water.default_filters(grid, dt)
    trajectory_fn = shallow_water.shallow_water_leapfrog_trajectory(
        coords, dt, physics_specs, inner_steps, outer_steps, mean_potential,
        orography, default_filters)

    # Constructs steady state from a zonal velocity field.
    lat = np.arccos(grid.cos_lat)
    velocity = jnp.stack([velocity_function(lat)] * layers)
    state_0 = state_1 = shallow_water_states.multi_layer(
        velocity, density, coords)
    initial_state = (state_0, state_1)

    # Quantities that will be used to compute relative errors.
    init_potential = grid.to_nodal(state_0.potential)
    init_potential_l2 = np.sqrt(np.square(init_potential).sum())

    # Compute the potentials at several time steps and compare them to the
    # initial potential
    # TODO(jamieas): we need more principled expectations for the deviation
    # from reference values.
    _, trajectory = trajectory_fn(initial_state)
    potentials = grid.to_nodal(trajectory.potential)
    _, w = associated_legendre.gauss_legendre_nodes(grid.latitude_nodes)

    for j, potential in enumerate(potentials):
      step = (j + 1) * inner_steps
      with self.subTest(f'Mean potential conservation, step {step}'):
        # The mean of the fluctuations in geopotential should be zero.
        mean_potential = np.mean(w * potential / np.sum(w), axis=(-1, -2))
        np.testing.assert_array_less(np.abs(mean_potential), 1e-8)

      with self.subTest(f'Steady state L2 error, step {step}'):
        # The geopotential should stay constant over time.
        l2_error = np.sqrt(
            np.square(potential - init_potential).sum()) / init_potential_l2
        self.assertLess(l2_error, 1e-5)

  @parameterized.parameters(
      dict(wavenumbers=128,
           layers=1,
           density_ratio=.9,
           max_velocity=20 * units.meter / units.second,
           mountain_height=0 * units.meter,
           atmosphere_thickness=5960 * units.meter,
           total_time=15 * units.day,
           save_every=6 * units.hour,
           dt=60 * units.second),
      dict(wavenumbers=128,
           layers=1,
           density_ratio=.9,
           max_velocity=20 * units.meter / units.second,
           mountain_height=2000 * units.meter,
           atmosphere_thickness=5960 * units.meter,
           total_time=15 * units.day,
           save_every=6 * units.hour,
           dt=60 * units.second),
      dict(wavenumbers=64,
           layers=3,
           density_ratio=.9,
           max_velocity=25 * units.meter / units.second,
           mountain_height=3000 * units.meter,
           atmosphere_thickness=7000 * units.meter,
           total_time=15 * units.day,
           save_every=6 * units.hour,
           dt=60 * units.second),
  )
  def testFlowOverAMountainMassConservation(
      self, wavenumbers, layers, density_ratio, max_velocity, mountain_height,
      atmosphere_thickness, total_time, save_every, dt):
    """Tests that mass is conserved for a flow over a mountain."""

    if not _tpu_or_gpu_available():
      raise unittest.SkipTest('test is too slow to run on CPU')

    # This test is based on Test Case 5 from
    #  "A standard test set for numerical approximations to the shallow water
    #  equations in spherical geometry"
    #  David L.Williamson, John B.Drake, James J.Hack, Rüdiger Jakob,
    #  Paul N.Swarztrauber
    #  https://doi.org/10.1016/S0021-9991(05)80016-6

    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    density = np.array([density_ratio ** n for n in range(layers)][::-1])
    vertical_grid = layer_coordinates.LayerCoordinates(layers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    physics_specs = shallow_water.ShallowWaterSpecs.from_si(
        densities=density * scales.WATER_DENSITY)

    # Construct initial state.
    max_v = physics_specs.nondimensionalize(max_velocity)
    u_nodal = jnp.array(
        [[max_v * grid.cos_lat] * grid.longitude_nodes] * layers)
    v_nodal = jnp.zeros_like(u_nodal)
    cos_lat_velocity = grid.to_modal(
        jnp.stack([u_nodal, v_nodal]) / grid.cos_lat)
    vorticity = grid.curl_cos_lat(cos_lat_velocity)

    # Orography consists of a single mountain.
    orography = _get_mountain(grid, mountain_height, physics_specs)

    mean_potential, delta_potential = _get_geopotential(
        grid, max_velocity, atmosphere_thickness, layers, physics_specs)

    divergence = jnp.zeros_like(delta_potential)

    state_0 = state_1 = shallow_water.State(
        vorticity=vorticity,
        divergence=divergence,
        potential=delta_potential - orography / layers)
    initial_state = (state_0, state_1)

    # Set up time stepping.
    total_time = physics_specs.nondimensionalize(total_time)
    save_every = physics_specs.nondimensionalize(save_every)
    dt = physics_specs.nondimensionalize(dt)
    inner_steps = int(save_every / dt)
    outer_steps = int(total_time / save_every)

    # Set up time integration of the shallow water equations.
    default_filters = shallow_water.default_filters(grid, dt)
    trajectory_fn = shallow_water.shallow_water_leapfrog_trajectory(
        coords, dt, physics_specs, inner_steps, outer_steps, mean_potential,
        orography, default_filters)

    # Perform integration and check conservation of mass.
    _, trajectory = trajectory_fn(initial_state)
    initial_mass = _compute_mass(
        grid, state_0.potential, mean_potential, density)
    masses = _compute_mass(
        grid, trajectory.potential, mean_potential, density)
    np.testing.assert_allclose(masses, initial_mass, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
