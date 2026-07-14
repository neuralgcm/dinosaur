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
from dinosaur import time_integration
from dinosaur import units
from dinosaur import xarray_utils
import jax
import jax.numpy as jnp
import numpy as np


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
    physics_specs = units.SimUnits.from_si()
    nondim_densities = physics_specs.nondimensionalize(
        density * scales.WATER_DENSITY
    )
    mean_potential = np.ones(layers) * mean_potential
    orography = None  # no orography in the geostrophic flow test case.

    # Set up time integration of the shallow water equations.
    equation = shallow_water.ShallowWaterEquations(
        coords,
        physics_specs,
        orography,
        mean_potential,
        densities=nondim_densities,
    )
    step_fn = time_integration.imex_rk_sil3(equation, dt)
    filters = [
        time_integration.exponential_step_filter(grid, dt),
    ]
    step_fn = time_integration.step_with_filters(step_fn, filters)
    trajectory_fn = time_integration.trajectory_from_step(
        step_fn, outer_steps, inner_steps)

    # Constructs steady state from a zonal velocity field.
    lat = np.arccos(grid.cos_lat)
    velocity = jnp.stack([velocity_function(lat)] * layers)
    initial_state = shallow_water_states.multi_layer(
        velocity, nondim_densities, coords
    )

    # Quantities that will be used to compute relative errors.
    init_potential = grid.to_nodal(initial_state.potential)
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
           max_velocity=20 * scales.units.meter / scales.units.second,
           mountain_height=0 * scales.units.meter,
           atmosphere_thickness=5960 * scales.units.meter,
           total_time=15 * scales.units.day,
           save_every=6 * scales.units.hour,
           dt=60 * scales.units.second),
      dict(wavenumbers=128,
           layers=1,
           density_ratio=.9,
           max_velocity=20 * scales.units.meter / scales.units.second,
           mountain_height=2000 * scales.units.meter,
           atmosphere_thickness=5960 * scales.units.meter,
           total_time=15 * scales.units.day,
           save_every=6 * scales.units.hour,
           dt=60 * scales.units.second),
      dict(wavenumbers=64,
           layers=3,
           density_ratio=.9,
           max_velocity=25 * scales.units.meter / scales.units.second,
           mountain_height=3000 * scales.units.meter,
           atmosphere_thickness=7000 * scales.units.meter,
           total_time=15 * scales.units.day,
           save_every=6 * scales.units.hour,
           dt=60 * scales.units.second),
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
    physics_specs = units.SimUnits.from_si()
    nondim_densities = physics_specs.nondimensionalize(
        density * scales.WATER_DENSITY
    )

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

    initial_state = shallow_water.State(
        vorticity=vorticity,
        divergence=divergence,
        potential=delta_potential - orography / layers)

    # Set up time stepping.
    total_time = physics_specs.nondimensionalize(total_time)
    save_every = physics_specs.nondimensionalize(save_every)
    dt = physics_specs.nondimensionalize(dt)
    inner_steps = int(save_every / dt)
    outer_steps = int(total_time / save_every)

    # Set up time integration of the shallow water equations.
    equation = shallow_water.ShallowWaterEquations(
        coords,
        physics_specs,
        orography,
        mean_potential,
        densities=nondim_densities,
    )
    step_fn = time_integration.imex_rk_sil3(equation, dt)
    filters = [
        time_integration.exponential_step_filter(grid, dt),
    ]
    step_fn = time_integration.step_with_filters(step_fn, filters)
    trajectory_fn = time_integration.trajectory_from_step(
        step_fn, outer_steps, inner_steps)

    # Perform integration and check conservation of mass.
    _, trajectory = trajectory_fn(initial_state)
    initial_mass = _compute_mass(
        grid, initial_state.potential, mean_potential, nondim_densities
    )
    masses = _compute_mass(
        grid, trajectory.potential, mean_potential, nondim_densities
    )
    np.testing.assert_allclose(masses, initial_mass, rtol=1e-6)


class SemiLagrangianShallowWaterTest(parameterized.TestCase):
  """Tests for the semi-Lagrangian shallow water equations."""

  def _coords(self, layers=1):
    # T42 rather than `with_wavenumbers` because cross-pole halos require an
    # even number of longitude nodes.
    return coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T42(), layer_coordinates.LayerCoordinates(layers)
    )

  def _steady_state_setup(self, coriolis_mode, layers=1):
    coords = self._coords(layers)
    grid = coords.horizontal
    physics_specs = units.SimUnits.from_si()
    densities = np.array([0.9 ** n for n in range(layers)][::-1])
    mean_potential = np.ones(layers) / 10
    equation = shallow_water.SemiLagrangianShallowWaterEquations(
        coords,
        physics_specs,
        None,
        mean_potential,
        densities=densities,
        coriolis_mode=coriolis_mode,
    )
    lat = np.arccos(grid.cos_lat)
    velocity = jnp.stack([np.cos(3 * lat) / 5] * layers)
    initial_state = shallow_water_states.multi_layer(
        velocity, densities, coords
    )
    return equation, initial_state

  def _potential_l2_error(self, grid, state, reference):
    potential = grid.to_nodal(state.potential)
    reference = grid.to_nodal(reference.potential)
    return float(
        np.sqrt(
            np.square(potential - reference).sum() / np.square(reference).sum()
        )
    )

  @parameterized.parameters(
      dict(coriolis_mode='planetary_momentum'),
      dict(coriolis_mode='explicit'),
      dict(coriolis_mode='planetary_momentum', layers=2),
  )
  def test_steady_state_geostrophic_flow(self, coriolis_mode, layers=1):
    """Williamson case 2 analog: geostrophic flow remains steady.

    The two-layer case exercises the inter-layer pressure-gradient coupling
    in the non-advective terms (`density_ratios @ potential`), which is
    identically zero for a single layer.
    """
    equation, initial_state = self._steady_state_setup(coriolis_mode, layers)
    grid = equation.coords.horizontal
    dt = 0.01
    step_fn = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
    )
    final = time_integration.repeated(step_fn, 300)(initial_state)
    # measured l2 error ~6e-5 for both Coriolis modes.
    self.assertLess(self._potential_l2_error(grid, final, initial_state), 5e-4)

  def test_eulerian_stepper_use_is_rejected(self):
    """explicit_terms raises so Eulerian steppers cannot silently misuse."""
    equation, initial_state = self._steady_state_setup('planetary_momentum')
    with self.assertRaisesRegex(TypeError, 'semi-Lagrangian'):
      equation.explicit_terms(initial_state)

  def test_consistency_with_eulerian_core_with_orography(self):
    """Flow over a mountain: SL and Eulerian cores track each other.

    Exercises the orography contribution to the non-advective terms, which
    the steady-state tests (flat) never touch.
    """
    coords = self._coords()
    grid = coords.horizontal
    physics_specs = units.SimUnits.from_si()
    densities = np.ones(1)
    mean_potential = np.ones(1) / 10
    lon, sin_lat = grid.nodal_mesh
    lat = np.arcsin(sin_lat)
    mountain = 0.005 * np.exp(
        -((lat - np.pi / 6) ** 2 + (lon - np.pi) ** 2) / 0.1
    )[np.newaxis]
    orography = grid.to_modal(jnp.asarray(mountain))
    velocity_profile = np.cos(3 * np.arccos(grid.cos_lat)) / 5
    initial_state = shallow_water_states.multi_layer(
        jnp.stack([velocity_profile]), densities, coords
    )
    common = dict(densities=densities)
    sl_equation = shallow_water.SemiLagrangianShallowWaterEquations(
        coords, physics_specs, orography, mean_potential, **common
    )
    eulerian = shallow_water.ShallowWaterEquations(
        coords, physics_specs, orography, mean_potential, **common
    )
    dt = 0.01
    steps = 100
    sl_final = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(
                sl_equation, dt
            )
        ),
        steps,
    )(initial_state)
    eulerian_final = time_integration.repeated(
        jax.jit(time_integration.imex_rk_sil3(eulerian, dt)), steps
    )(initial_state)
    with self.subTest('the mountain drives dynamics'):
      self.assertGreater(
          self._potential_l2_error(grid, eulerian_final, initial_state), 1e-3
      )
    with self.subTest('SL tracks the Eulerian core'):
      self.assertLess(
          self._potential_l2_error(grid, sl_final, eulerian_final), 1e-3
      )

  def test_large_time_step_stability(self):
    """SL stays stable and accurate at time steps where the Eulerian core
    blows up."""
    equation, initial_state = self._steady_state_setup('planetary_momentum')
    coords = equation.coords
    grid = coords.horizontal
    physics_specs = equation.physics_specs
    eulerian = shallow_water.ShallowWaterEquations(
        coords,
        physics_specs,
        None,
        equation.reference_potential,
        densities=equation.densities,
    )
    dt = 0.4  # ~8x the largest stable Eulerian step for this configuration
    steps = 50
    sl_final = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
        ),
        steps,
    )(initial_state)
    eulerian_final = time_integration.repeated(
        jax.jit(time_integration.imex_rk_sil3(eulerian, dt)), steps
    )(initial_state)
    with self.subTest('Eulerian core is unstable at this time step'):
      self.assertFalse(
          np.isfinite(np.asarray(eulerian_final.potential)).all()
      )
    with self.subTest('semi-Lagrangian core remains steady'):
      error = self._potential_l2_error(grid, sl_final, initial_state)
      self.assertTrue(np.isfinite(np.asarray(sl_final.potential)).all())
      self.assertLess(error, 0.01)  # measured ~3e-3

  def test_consistency_with_eulerian_core(self):
    """SL and Eulerian cores track each other on an unsteady flow."""
    coords = self._coords()
    grid = coords.horizontal
    physics_specs = units.SimUnits.from_si()
    state_fn, aux_features = shallow_water_states.barotropic_instability_tc(
        coords, physics_specs
    )
    initial_state = state_fn(jax.random.PRNGKey(0))
    reference_potential = aux_features[xarray_utils.REF_POTENTIAL_KEY]
    densities = np.ones(1)
    sl_equation = shallow_water.SemiLagrangianShallowWaterEquations(
        coords, physics_specs, None, reference_potential, densities=densities
    )
    eulerian = shallow_water.ShallowWaterEquations(
        coords, physics_specs, None, reference_potential, densities=densities
    )
    dt = 0.01
    steps = 100
    sl_final = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(
                sl_equation, dt
            )
        ),
        steps,
    )(initial_state)
    eulerian_final = time_integration.repeated(
        jax.jit(time_integration.imex_rk_sil3(eulerian, dt)), steps
    )(initial_state)
    # The differences plateau at the SL interpolation-error floor rather than
    # shrinking indefinitely with dt (see plan §9.6); measured values are
    # ~5e-4 (potential) and ~6e-3 (vorticity) across dt in [0.005, 0.02].
    self.assertLess(
        self._potential_l2_error(grid, sl_final, eulerian_final), 2e-3
    )
    vorticity_sl = grid.to_nodal(sl_final.vorticity)
    vorticity_eu = grid.to_nodal(eulerian_final.vorticity)
    vorticity_error = np.sqrt(
        np.square(vorticity_sl - vorticity_eu).sum()
        / np.square(vorticity_eu).sum()
    )
    self.assertLess(float(vorticity_error), 2e-2)


if __name__ == '__main__':
  absltest.main()
