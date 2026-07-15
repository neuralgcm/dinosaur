# Copyright 2023 Google LLC
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
import functools

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import held_suarez
from dinosaur import hybrid_coordinates
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import semi_lagrangian
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import units
from dinosaur import vertical_interpolation
from dinosaur import xarray_utils
import jax
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()

s_units = scales.units


def random_state(coords, key):
  (
      vorticity_key,
      divergence_key,
      temperature_variation_key,
      log_surface_pressure_key,
  ) = jax.random.split(key, 4)
  # All values are scaled by 1 / total_wavenumber**2
  scale = (coords.horizontal.total_wavenumbers + 1) ** -2
  vorticity = scale * jax.random.normal(vorticity_key, shape=coords.modal_shape)
  divergence = scale * jax.random.normal(
      divergence_key, shape=coords.modal_shape
  )
  temperature_variation = scale * jax.random.normal(
      temperature_variation_key, shape=coords.modal_shape
  )
  log_surface_pressure = scale * jax.random.normal(
      log_surface_pressure_key, shape=coords.surface_modal_shape
  )
  state = primitive_equations.State(
      vorticity, divergence, temperature_variation, log_surface_pressure
  )
  primitive_equations.validate_state_shape(state, coords)
  return state


def assert_states_close(state0, state1, **kwargs):
  for field in state0.fields:
    if field.name == 'tracers':
      for tracer_name in state0.tracers.keys():
        np.testing.assert_allclose(
            state0.tracers[tracer_name],
            state1.tracers[tracer_name],
            err_msg=f'Mismatch in tracer {tracer_name}:',
            **kwargs,
        )
    else:
      if field.name == 'sim_time':
        if state0.sim_time is None != state1.sim_time is None:
          raise AssertionError(
              f'Mismatch is sim_time: {state0.sim_time} != {state1.sim_time}'
          )
        if state0.sim_time is None:  # assert_allclose does not handle None
          continue
      np.testing.assert_allclose(
          getattr(state0, field.name),
          getattr(state1, field.name),
          err_msg=f'Mismatch in {field}:',
          **kwargs,
      )


class PrimitiveEquationsSigmaImplicitTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          wavenumbers=256,
          test_m_fn=lambda lon, lat: jnp.sin(lon) * jnp.cos(lat) ** 2,
          test_n_fn=lambda lon, lat: jnp.cos(lat) ** 2,
      ),
      dict(
          wavenumbers=128,
          test_m_fn=lambda lon, lat: 2.3 * jnp.cos(lon) ** 2 * jnp.cos(lat),
          test_n_fn=lambda lon, lat: 3.6 * jnp.cos(lat) * jnp.sin(2 * lat),
      ),
  )
  def test_div_sec_lat(self, wavenumbers, test_m_fn, test_n_fn):
    """Test that helper function div_sec_lat returns expected values."""
    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    lon, sin_lat = grid.nodal_mesh
    lat = np.arcsin(sin_lat)
    m = test_m_fn(lon, lat)
    n = test_n_fn(lon, lat)
    # should be same as H(M, N) = (1 / cos²θ) * ∂M/∂λ + (1 / cosθ) * ∂N/∂θ
    dm_dlon_fn = jax.vmap(jax.vmap(jax.grad(test_m_fn)))
    dn_dlat_fn = jax.vmap(jax.vmap(jax.grad(test_n_fn, argnums=1)))
    h_mn_expected = grid.to_modal(
        dm_dlon_fn(lon, lat) / (np.cos(lat) ** 2)
        + dn_dlat_fn(lon, lat) / np.cos(lat)
    )
    h_mn_actual = primitive_equations.div_sec_lat(m, n, grid)
    np.testing.assert_allclose(h_mn_actual, h_mn_expected, atol=1e-3)

  @parameterized.parameters(
      dict(coordinates=sigma_coordinates.SigmaCoordinates.equidistant(10)),
      dict(coordinates=sigma_coordinates.SigmaCoordinates.equidistant(111)),
  )
  def test_get_sigma_ratios(self, coordinates):
    """Tests that the values of the sigma ratios 𝛼 are correct."""
    alpha = primitive_equations.get_sigma_ratios(coordinates)
    np.testing.assert_array_equal([coordinates.layers], alpha.shape)
    sigma = coordinates.centers
    for j in range(coordinates.layers):
      if j == coordinates.layers - 1:
        expected_entry = -np.log(sigma[j])
      else:
        expected_entry = (np.log(sigma[j + 1]) - np.log(sigma[j])) / 2
      np.testing.assert_almost_equal(expected_entry, alpha[j])

  @parameterized.parameters(
      dict(wavenumbers=8, layers=3, atol=5e-3),
      dict(wavenumbers=128, layers=16, atol=5e-3),
  )
  def test_get_geopotential_steady_state(self, wavenumbers, layers, atol):
    """Tests that `get_geopotential_on_sigma` works for steady states."""
    physics_specs = units.SimUnits.from_si()
    coords = coordinate_systems.CoordinateSystem(
        horizontal=spherical_harmonic.Grid.with_wavenumbers(wavenumbers),
        vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers),
    )
    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    state = initial_state_fn(jax.random.PRNGKey(0))
    modal_orography = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords
    )
    expected_geopotential = aux_features[xarray_utils.GEOPOTENTIAL_KEY]
    with self.subTest('dry_geopotential'):
      temperature = aux_features[xarray_utils.REF_TEMP_KEY][
          :, np.newaxis, np.newaxis
      ] + coords.horizontal.to_nodal(state.temperature_variation)
      actual = primitive_equations.get_geopotential_on_sigma(
          temperature,
          nodal_orography=coords.horizontal.to_nodal(modal_orography),
          sigma=coords.vertical,
          gravity_acceleration=physics_specs.gravity_acceleration,
          ideal_gas_constant=physics_specs.ideal_gas_constant,
      )
      np.testing.assert_allclose(actual, expected_geopotential, atol=atol)
    with self.subTest('moist_geopotential'):
      temperature = aux_features[xarray_utils.REF_TEMP_KEY][
          :, np.newaxis, np.newaxis
      ] + coords.horizontal.to_nodal(state.temperature_variation)
      specific_humidity = jnp.zeros_like(temperature)
      nodal_orography = coords.horizontal.to_nodal(modal_orography)
      actual = primitive_equations.get_geopotential_on_sigma(
          temperature,
          specific_humidity,
          nodal_orography=nodal_orography,
          sigma=coords.vertical,
          gravity_acceleration=physics_specs.gravity_acceleration,
          ideal_gas_constant=physics_specs.ideal_gas_constant,
          water_vapor_gas_constant=physics_specs.water_vapor_gas_constant,
      )
      np.testing.assert_allclose(actual, expected_geopotential, atol=atol)

  def test_stationary_solution(self):
    """Tests that steady state is stationary for primitive equations."""
    wavenumbers = 42
    layers = 26
    dt_si = 600 * s_units.s
    save_every_si = 4 * s_units.hour
    inner_steps = int(save_every_si / dt_si)
    outer_steps = 6
    physics_specs = units.SimUnits.from_si()
    dt = physics_specs.nondimensionalize(dt_si)
    coords = coordinate_systems.CoordinateSystem(
        horizontal=spherical_harmonic.Grid.with_wavenumbers(wavenumbers),
        vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers),
    )
    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    modal_orography = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords
    )
    state = initial_state_fn()
    tracer_names = ['tracer_a', 'tracer_b']
    tracer_amplitudes = [1.5, 2.5]
    state.tracers = {
        name: primitive_equations_states.gaussian_scalar(
            coords, physics_specs, amplitude=amplitude
        )
        for name, amplitude in zip(tracer_names, tracer_amplitudes)
    }
    primitive = primitive_equations.PrimitiveEquationsSigma(
        ref_temps, modal_orography, coords, physics_specs
    )
    step_fn = time_integration.imex_rk_sil3(primitive, dt)
    filters = [
        time_integration.exponential_step_filter(coords.horizontal, dt),
    ]
    step_fn = time_integration.step_with_filters(step_fn, filters)
    trajectory_fn = time_integration.trajectory_from_step(
        step_fn, outer_steps, inner_steps
    )
    trajectory_fn = jax.jit(trajectory_fn)
    input_state = state
    _, trajectory = trajectory_fn(input_state)
    trajectory = jax.device_get(trajectory)

    def tracer_integral(tracer):
      tracer_nodal = coords.horizontal.to_nodal(tracer)
      tracer_columns = sigma_coordinates.sigma_integral(
          tracer_nodal, coords.vertical, keepdims=False
      )
      return coords.horizontal.integrate(tracer_columns)

    expected_tracer_sums = {
        tracer_name: tracer_integral(state.tracers[tracer_name])
        for tracer_name in tracer_names
    }
    for step in range(outer_steps):
      with self.subTest(f'Divergence remains close to zero, step {step}'):
        np.testing.assert_array_less(abs(trajectory.divergence[step]), 1e-3)

      with self.subTest(f'Vorticity is stationary, step {step}'):
        np.testing.assert_allclose(
            trajectory.vorticity[step], state.vorticity, atol=5e-4
        )

      with self.subTest(f'Temperature is stationary, step {step}'):
        np.testing.assert_allclose(
            trajectory.temperature_variation[step],
            state.temperature_variation,
            atol=5e-2,
        )

      with self.subTest(f'Log surface pressure is stationary, step {step}'):
        np.testing.assert_allclose(
            trajectory.log_surface_pressure[step],
            state.log_surface_pressure,
            atol=5e-4,
        )

      with self.subTest(f'Conservation of tracer, step {step}'):
        # Note: mass is not conserved by construction, but should change rather
        # slowly during smooth evolution.
        for tracer_name in tracer_names:
          actual_tracer_sum = tracer_integral(
              trajectory.tracers[tracer_name][step]
          )
          expected_tracer_sum = expected_tracer_sums[tracer_name]
          np.testing.assert_allclose(
              actual_tracer_sum / expected_tracer_sum, 1, atol=3e-5
          )

  @parameterized.parameters(
      dict(
          coordinates=sigma_coordinates.SigmaCoordinates.equidistant(10),
          ideal_gas_constant=1,
      ),
      dict(
          coordinates=sigma_coordinates.SigmaCoordinates.equidistant(21),
          ideal_gas_constant=12.3,
      ),
  )
  def test_get_geopotential_weights_sigma(
      self, coordinates, ideal_gas_constant
  ):
    """Tests that the entries of geopotential weights `G` are correct."""
    g = primitive_equations.get_geopotential_weights_sigma(
        coordinates, ideal_gas_constant
    )
    np.testing.assert_array_equal(
        [coordinates.layers, coordinates.layers], g.shape
    )
    alpha = primitive_equations.get_sigma_ratios(coordinates)
    for i in range(coordinates.layers):
      for j in range(coordinates.layers):

        #            𝜶[0]    𝜶[0] + 𝜶[1]    𝜶[1] + 𝜶[2]    𝜶[2] + 𝜶[3]    ᠁
        # G / R  =   0       𝜶[1]           𝜶[1] + 𝜶[2]    𝜶[2] + 𝜶[3]    ᠁
        #            0       0              𝜶[2]           𝜶[2] + 𝜶[3]    ᠁
        #            ⋮       ⋮               ⋮              ⋮              ⋱

        if i > j:
          expected_entry = 0
        elif i == j:
          expected_entry = ideal_gas_constant * alpha[j]
        else:
          expected_entry = ideal_gas_constant * (alpha[j] + alpha[j - 1])
        np.testing.assert_almost_equal(
            expected_entry, g[i, j], err_msg=f'Mismatch on entry {[i, j]}.'
        )

  def test_get_geopotential_diff_sigma_both_ways(self):
    temperature = np.random.RandomState(0).randn(12, 1, 1)
    coordinates = sigma_coordinates.SigmaCoordinates.equidistant(12)
    ideal_gas_constant = 1.5
    result_matvec = primitive_equations.get_geopotential_diff_sigma(
        temperature, coordinates, ideal_gas_constant, method='dense'
    )
    result_cumsum = primitive_equations.get_geopotential_diff_sigma(
        temperature, coordinates, ideal_gas_constant, method='sparse'
    )
    np.testing.assert_allclose(result_matvec, result_cumsum, atol=1e-6)

  @parameterized.parameters(
      dict(
          coordinates=sigma_coordinates.SigmaCoordinates.equidistant(
              5, dtype=np.float64
          ),
          reference_temperature=np.linspace(100, 200, 5),
          heat_capacity_ratio=0.5,
      ),
      dict(
          coordinates=sigma_coordinates.SigmaCoordinates.equidistant(
              23, dtype=np.float64
          ),
          reference_temperature=np.linspace(250, 300, 23),
          heat_capacity_ratio=0.2857,
      ),
  )
  def test_get_temperature_implicit_weights_sigma(
      self, coordinates, reference_temperature, heat_capacity_ratio
  ):
    """Tests that the entries of temperature weights `H` are correct."""
    h = primitive_equations.get_temperature_implicit_weights_sigma(
        coordinates, reference_temperature, heat_capacity_ratio
    )
    np.testing.assert_array_equal(
        [coordinates.layers, coordinates.layers], h.shape
    )
    alpha = primitive_equations.get_sigma_ratios(coordinates)

    def k(r, s):
      """Computes the term denoted `K[r, s]` in the code."""
      assert r >= -1
      assert r <= coordinates.layers - 1
      assert s >= 0
      assert s <= coordinates.layers - 1

      # K[r, s] = (T[r + 1] - T[r]) / (Δ𝜎[r + 1] + Δ𝜎[r])
      #           · (P(r - s) - sum(Δ𝜎[:r + 1]))
      # K[r, s] = 0  if r < 0
      # K[r, s] = 0  when `r = coordinates.layers - 1`

      if r < 0:
        return 0
      if r == coordinates.layers - 1:
        return 0

      return (
          ((r - s >= 0) - coordinates.layer_thickness[: r + 1].sum())
          * (reference_temperature[r + 1] - reference_temperature[r])
          / (
              coordinates.layer_thickness[r + 1]
              + coordinates.layer_thickness[r]
          )
      )

    for r in range(coordinates.layers):
      for s in range(coordinates.layers):

        # H[r, s] / Δ𝜎[s] = 𝜅T[r] · (P(r-s) 𝛼[r] + P(r-s-1) 𝛼[r-1]) / Δ𝜎[r]
        #           - ̇K[r, s]
        #           - K[r-1, s]

        expected_entry = coordinates.layer_thickness[s] * (
            heat_capacity_ratio
            * reference_temperature[r]
            * ((r - s >= 0) * alpha[r] + (r - s - 1 >= 0) * alpha[r - 1])
            / coordinates.layer_thickness[r]
            - k(r, s)
            - k(r - 1, s)
        )
        np.testing.assert_almost_equal(
            expected_entry, h[r, s], err_msg=f'Mismatch in entry {[r, s]}.'
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='variable_reference_temperature',
          reference_temperature=np.linspace(100, 200, 5),
      ),
      dict(
          testcase_name='constant_reference_temperature',
          reference_temperature=100 * np.ones(5),
      ),
  )
  def test_get_temperature_implicit_sigma_both_ways(
      self, reference_temperature
  ):
    divergence = np.random.RandomState(0).randn(5, 1, 1)
    coordinates = sigma_coordinates.SigmaCoordinates.equidistant(5)
    kappa = 2 / 7
    result_matvec = primitive_equations.get_temperature_implicit_sigma(
        divergence, coordinates, reference_temperature, kappa, method='dense'
    )
    result_cumsum = primitive_equations.get_temperature_implicit_sigma(
        divergence, coordinates, reference_temperature, kappa, method='sparse'
    )
    np.testing.assert_allclose(result_matvec, result_cumsum, atol=1e-5)

  @parameterized.parameters(
      dict(wavenumbers=32, layers=4),
      dict(wavenumbers=64, layers=10),
  )
  def test_primitive_equations_sigma_explicit_shape(self, wavenumbers, layers):
    """Tests that output of explicit_terms has expected shape."""
    coords = coordinate_systems.CoordinateSystem(
        horizontal=spherical_harmonic.Grid.with_wavenumbers(wavenumbers),
        vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers),
    )
    reference_temperature = 300 * np.ones(layers)
    l, _ = coords.horizontal.modal_mesh
    modal_orography = np.zeros_like(l)
    vorticity = jnp.ones((layers,) + l.shape)
    divergence = jnp.ones((layers,) + l.shape)
    temperature_variation = jnp.ones((layers,) + l.shape)
    log_surface_pressure = jnp.ones((1,) + l.shape)

    physics_specs = units.SimUnits.from_si()
    state = primitive_equations.State(
        vorticity, divergence, temperature_variation, log_surface_pressure
    )
    primitive = primitive_equations.PrimitiveEquationsSigma(
        reference_temperature, modal_orography, coords, physics_specs
    )

    output = primitive.explicit_terms(state)
    with self.subTest('divergence shape'):
      self.assertEqual(state.divergence.shape, output.divergence.shape)
    with self.subTest('vorticity shape'):
      self.assertEqual(state.vorticity.shape, output.vorticity.shape)
    with self.subTest('temperature shape'):
      self.assertEqual(
          state.temperature_variation.shape, output.temperature_variation.shape
      )
    with self.subTest('log_surface_pressure shape'):
      self.assertEqual(
          state.log_surface_pressure.shape, output.log_surface_pressure.shape
      )

  @parameterized.parameters(
      dict(wavenumbers=64, layers=10),
  )
  def test_primitive_equations_sigma_explicit_scales_invariance(
      self, wavenumbers, layers
  ):
    """Tests that tendencies in SI units are not affected by scales."""
    default_scale = scales.DEFAULT_SCALE
    custom_scale = scales.Scale(
        scales.RADIUS / 100,
        55.3 / 2 / scales.OMEGA,
        1 * s_units.kilogram * 16.4,
        1 * s_units.degK * 3.15,
    )
    physics_specs_a = units.SimUnits.from_si(
        scale=default_scale
    )
    grid_a = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, radius=physics_specs_a.radius
    )
    physics_specs_b = units.SimUnits.from_si(
        scale=custom_scale
    )
    grid_b = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, radius=physics_specs_b.radius
    )
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    coords_a = coordinate_systems.CoordinateSystem(grid_a, vertical_grid)
    coords_b = coordinate_systems.CoordinateSystem(grid_b, vertical_grid)

    # defining input states using different grids and scales;
    initial_state_fn, aux_features_a = (
        primitive_equations_states.steady_state_jw(coords_a, physics_specs_a)
    )
    modal_orography_a = primitive_equations.truncated_modal_orography(
        aux_features_a[xarray_utils.OROGRAPHY], coords_a
    )
    state_a = initial_state_fn()
    state_a = state_a + primitive_equations_states.baroclinic_perturbation_jw(
        coordinate_systems.CoordinateSystem(grid_a, vertical_grid),
        physics_specs_a,
    )
    initial_state_fn, aux_features_b = (
        primitive_equations_states.steady_state_jw(coords_b, physics_specs_b)
    )
    modal_orography_b = primitive_equations.truncated_modal_orography(
        aux_features_b[xarray_utils.OROGRAPHY], coords_b
    )
    state_b = initial_state_fn()
    state_b = state_b + primitive_equations_states.baroclinic_perturbation_jw(
        coordinate_systems.CoordinateSystem(grid_b, vertical_grid),
        physics_specs_b,
    )

    # Computing tendencies using both variations.
    primitive_a = primitive_equations.PrimitiveEquationsSigma(
        aux_features_a[xarray_utils.REF_TEMP_KEY],
        modal_orography_a,
        coordinate_systems.CoordinateSystem(grid_a, vertical_grid),
        physics_specs_a,
    )

    primitive_b = primitive_equations.PrimitiveEquationsSigma(
        aux_features_b[xarray_utils.REF_TEMP_KEY],
        modal_orography_b,
        coordinate_systems.CoordinateSystem(grid_b, vertical_grid),
        physics_specs_b,
    )
    tendencies_a = primitive_a.explicit_terms(state_a)
    tendencies_b = primitive_b.explicit_terms(state_b)

    with self.subTest('divergence tendency'):
      divergence_a = physics_specs_a.dimensionalize(
          tendencies_a.divergence, 1 / s_units.s**2
      )
      divergence_b = physics_specs_b.dimensionalize(
          tendencies_b.divergence, 1 / s_units.s**2
      )
      np.testing.assert_allclose(
          divergence_a.magnitude, divergence_b.magnitude, atol=5e-7
      )
    with self.subTest('vorticity tendency'):
      vorticity_a = physics_specs_a.dimensionalize(
          tendencies_a.vorticity, 1 / s_units.s**2
      )
      vorticity_b = physics_specs_b.dimensionalize(
          tendencies_b.vorticity, 1 / s_units.s**2
      )
      np.testing.assert_allclose(
          vorticity_a.magnitude, vorticity_b.magnitude, atol=5e-7
      )
    with self.subTest('temperature tendency'):
      temperature_a = physics_specs_a.dimensionalize(
          tendencies_a.temperature_variation, s_units.degK / s_units.s
      )
      temperature_b = physics_specs_b.dimensionalize(
          tendencies_b.temperature_variation, s_units.degK / s_units.s
      )
      np.testing.assert_allclose(
          temperature_a.magnitude, temperature_b.magnitude, atol=5e-7
      )
    with self.subTest('surface pressure tendency'):
      pressure_a = physics_specs_a.dimensionalize(
          np.exp(tendencies_a.log_surface_pressure), s_units.pascal / s_units.s
      )
      pressure_b = physics_specs_b.dimensionalize(
          np.exp(tendencies_b.log_surface_pressure), s_units.pascal / s_units.s
      )
      np.testing.assert_allclose(
          pressure_a.magnitude, pressure_b.magnitude, atol=1e-7
      )

  @parameterized.parameters(
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(16),
          vertical_grid=sigma_coordinates.SigmaCoordinates.equidistant(5),
          reference_temperature=np.linspace(100, 200, 5),
          kappa=1.4 * s_units.dimensionless,
          ideal_gas_constant=33 * s_units.J / s_units.kilogram / s_units.degK,
          step_size=0.3,
          implicit_inverse_method='split',
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(16),
          vertical_grid=sigma_coordinates.SigmaCoordinates.equidistant(5),
          reference_temperature=np.linspace(100, 200, 5),
          kappa=1.4 * s_units.dimensionless,
          ideal_gas_constant=33 * s_units.J / s_units.kilogram / s_units.degK,
          step_size=0.3,
          implicit_inverse_method='blockwise',
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(16),
          vertical_grid=sigma_coordinates.SigmaCoordinates.equidistant(5),
          reference_temperature=np.linspace(100, 200, 5),
          kappa=1.4 * s_units.dimensionless,
          ideal_gas_constant=33 * s_units.J / s_units.kilogram / s_units.degK,
          step_size=0.3,
          implicit_inverse_method='stacked',
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(128),
          vertical_grid=sigma_coordinates.SigmaCoordinates.equidistant(23),
          reference_temperature=np.linspace(250, 300, 23),
          kappa=111 * s_units.dimensionless,
          ideal_gas_constant=1 * s_units.J / s_units.kilogram / s_units.degK,
          step_size=0.1,
          implicit_inverse_method='split',
          seed=1,
      ),
  )
  def test_primitive_inverse(
      self,
      vertical_grid,
      grid,
      reference_temperature,
      kappa,
      ideal_gas_constant,
      step_size,
      implicit_inverse_method,
      seed,
  ):
    """`primitive_inverse` computes (1 - step_size · primitive_implicit)⁻¹."""
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    physics_specs = units.SimUnits.from_si(
        ideal_gas_constant_si=ideal_gas_constant, kappa_si=kappa
    )
    state = random_state(coords, jax.random.PRNGKey(seed))
    l, _ = coords.horizontal.modal_mesh
    modal_orography = np.zeros_like(l)
    primitive = primitive_equations.PrimitiveEquationsSigma(
        reference_temperature,
        modal_orography,
        coords,
        physics_specs,
        implicit_inverse_method=implicit_inverse_method,
    )
    implicit_terms = primitive.implicit_terms(state)
    primitive_equations.validate_state_shape(implicit_terms, coords)

    with self.subTest('RequiresStaticEta'):
      # Tests that inversion fails if `step_size` is not a static value.
      with self.assertRaisesRegex(TypeError, '`step_size` must be concrete'):
        jitted_inverse = jax.jit(lambda s, t: primitive.implicit_inverse(s, t))  # pylint: disable=unnecessary-lambda
        _ = jitted_inverse(state - step_size * implicit_terms, step_size)

    jitted_inverse = jax.jit(lambda s: primitive.implicit_inverse(s, step_size))
    inverted_state = jitted_inverse(state - step_size * implicit_terms)
    primitive_equations.validate_state_shape(inverted_state, coords)
    assert_states_close(state, inverted_state, atol=1e-5)

  def test_equivalence_of_primitive_equations_with_and_without_humidity(self):
    """Tests that primitive equations + humidity reduces to default for q=0."""
    physics_specs = units.SimUnits.from_si()
    horizontal = spherical_harmonic.Grid.T21()
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(4)
    coords = coordinate_systems.CoordinateSystem(horizontal, vertical)

    # defining input states using different grids and scales;
    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    modal_orography = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords
    )
    state = initial_state_fn()
    state = state + primitive_equations_states.baroclinic_perturbation_jw(
        coords, physics_specs
    )
    state.tracers = {
        'specific_humidity': primitive_equations_states.gaussian_scalar(
            coords, physics_specs, amplitude=0.0
        )
    }
    # Computing tendencies using both variations.
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    primitive_a = primitive_equations.PrimitiveEquationsSigma(
        ref_temps, modal_orography, coords, physics_specs
    )
    primitive_b = primitive_equations.PrimitiveEquationsSigma(
        ref_temps,
        modal_orography,
        coords,
        physics_specs,
        humidity_key='specific_humidity',
    )

    tendencies_a = primitive_a.explicit_terms(state)
    tendencies_b = primitive_b.explicit_terms(state)

    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, atol=1e-7),
        tendencies_a,
        tendencies_b,
    )


class ExplicitTermsSplitTest(parameterized.TestCase):
  """Tests the advective/non-advective split of explicit tendencies."""

  def _make_equation_and_state(
      self,
      humidity,
      variable_t_ref,
      clouds=False,
      include_vertical_advection=True,
  ):
    physics_specs = units.SimUnits.from_si()
    horizontal = spherical_harmonic.Grid.T21()
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(4)
    coords = coordinate_systems.CoordinateSystem(horizontal, vertical)
    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    modal_orography = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords
    )
    state = initial_state_fn()
    state = state + primitive_equations_states.baroclinic_perturbation_jw(
        coords, physics_specs
    )
    state.sim_time = 0.0
    state.tracers = {
        'tracer': primitive_equations_states.gaussian_scalar(
            coords, physics_specs
        )
    }
    if humidity:
      state.tracers['specific_humidity'] = (
          primitive_equations_states.gaussian_scalar(
              coords, physics_specs, amplitude=0.01
          )
      )
    if clouds:
      state.tracers['cloud_water'] = primitive_equations_states.gaussian_scalar(
          coords, physics_specs, amplitude=0.001
      )
    if variable_t_ref:
      ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    else:
      ref_temps = 288.0 * np.ones(coords.vertical.layers)
    primitive = primitive_equations.PrimitiveEquationsSigma(
        ref_temps,
        modal_orography,
        coords,
        physics_specs,
        humidity_key='specific_humidity' if humidity else None,
        cloud_keys=('cloud_water',) if clouds else None,
        include_vertical_advection=include_vertical_advection,
    )
    return primitive, state

  @parameterized.parameters(
      dict(humidity=False, variable_t_ref=False),
      dict(humidity=False, variable_t_ref=True),
      dict(humidity=True, variable_t_ref=True),
      dict(humidity=True, variable_t_ref=True, clouds=True),
      dict(
          humidity=False, variable_t_ref=True, include_vertical_advection=False
      ),
  )
  def test_explicit_terms_split_reconstruction(self, **kwargs):
    """Advective + non-advective terms must reconstruct explicit_terms."""
    primitive, state = self._make_equation_and_state(**kwargs)
    full = primitive.explicit_terms(state)
    advective = primitive.explicit_advective_terms(state)
    nonadvective = primitive.explicit_nonadvective_terms(state)
    reconstructed = jax.tree.map(lambda x, y: x + y, advective, nonadvective)
    # The only differences are floating point rounding from re-associated
    # linear operations.
    tol = dict(rtol=1e-4, atol=1e-6)
    assert_states_close(full, reconstructed, **tol)
    self.assertEqual(full.sim_time, 1.0)
    self.assertEqual(advective.sim_time, 0.0)
    self.assertEqual(nonadvective.sim_time, 1.0)

  def test_nonadvective_terms_have_no_transport(self):
    """Tracers and log surface pressure have zero non-advective tendencies."""
    primitive, state = self._make_equation_and_state(
        humidity=False, variable_t_ref=True
    )
    nonadvective = primitive.explicit_nonadvective_terms(state)
    np.testing.assert_array_equal(
        nonadvective.log_surface_pressure,
        np.zeros_like(state.log_surface_pressure),
    )
    for name, tracer in nonadvective.tracers.items():
      np.testing.assert_array_equal(
          tracer, np.zeros_like(state.tracers[name]), err_msg=name
      )

  def test_nodal_velocities(self):
    """Checks shapes and consistency of nodal velocities."""
    primitive, state = self._make_equation_and_state(
        humidity=False, variable_t_ref=True
    )
    coords = primitive.coords
    velocities = primitive.nodal_velocities(state)
    layers = coords.vertical.layers
    self.assertEqual(velocities.u.shape, coords.nodal_shape)
    self.assertEqual(velocities.v.shape, coords.nodal_shape)
    self.assertEqual(
        velocities.sigma_dot.shape,
        (layers + 1,) + coords.horizontal.nodal_shape,
    )
    self.assertEqual(velocities.u_mean.shape, coords.surface_nodal_shape)
    self.assertEqual(velocities.v_mean.shape, coords.surface_nodal_shape)
    # winds match the standard modal-to-nodal conversion.
    u_expected, v_expected = spherical_harmonic.vor_div_to_uv_nodal(
        coords.horizontal, state.vorticity, state.divergence, clip=False
    )
    np.testing.assert_allclose(velocities.u, u_expected, atol=1e-6)
    np.testing.assert_allclose(velocities.v, v_expected, atol=1e-6)
    # sigma_dot vanishes at the top and bottom boundaries.
    np.testing.assert_array_equal(
        velocities.sigma_dot[0], np.zeros_like(velocities.sigma_dot[0])
    )
    np.testing.assert_array_equal(
        velocities.sigma_dot[-1], np.zeros_like(velocities.sigma_dot[-1])
    )
    # vertical mean matches explicit integration.
    np.testing.assert_allclose(
        velocities.u_mean,
        sigma_coordinates.sigma_integral(velocities.u, coords.vertical),
        atol=1e-6,
    )

  @parameterized.parameters(dict(humidity=False), dict(humidity=True))
  def test_nonadvective_terms_affine_in_winds(self, humidity):
    """Pins the classification: N must be affine in (ζ, δ).

    At fixed (T', ln pₛ, tracers), every non-advective term is affine in the
    winds: Coriolis is linear, the explicit PGF/orography/humidity corrections
    are constant, and the adiabatic and T_ref source terms are linear (via
    σ̇ and ω/p). Advective terms like ζ(k ✕ v) are quadratic, so accidentally
    classifying one as non-advective fails this test.
    """
    primitive, state = self._make_equation_and_state(
        humidity=humidity, variable_t_ref=True
    )

    def with_winds(factor):
      return state.replace(
          vorticity=factor * state.vorticity,
          divergence=factor * state.divergence,
      )

    n0 = primitive.explicit_nonadvective_terms(with_winds(0.0))
    n1 = primitive.explicit_nonadvective_terms(with_winds(1.0))
    n2 = primitive.explicit_nonadvective_terms(with_winds(2.0))
    lhs = jax.tree.map(lambda a, b: a - b, n2, n0)
    rhs = jax.tree.map(lambda a, b: 2 * (a - b), n1, n0)
    assert_states_close(lhs, rhs, rtol=1e-4, atol=1e-6)

  @parameterized.parameters(dict(humidity=False), dict(humidity=True))
  def test_advective_terms_scaling_in_winds(self, humidity):
    """Pins the classification: advection scales polynomially in (ζ, δ).

    At fixed (T', ln pₛ, tracers), scaling the winds by 2 scales momentum
    advection (vorticity flux, kinetic energy, vertical advection) by 4 and
    scalar advection (T', ln pₛ, tracers) by 2, and all advective terms
    vanish for zero winds. Non-advective terms like Coriolis (2✕), the
    pressure gradient, orography or wind-independent humidity corrections
    (1✕) would break these scalings.
    """
    primitive, state = self._make_equation_and_state(
        humidity=humidity, variable_t_ref=True
    )

    def with_winds(factor):
      return state.replace(
          vorticity=factor * state.vorticity,
          divergence=factor * state.divergence,
      )

    a0 = primitive.explicit_advective_terms(with_winds(0.0))
    a1 = primitive.explicit_advective_terms(with_winds(1.0))
    a2 = primitive.explicit_advective_terms(with_winds(2.0))
    with self.subTest('vanishes for zero winds'):
      for name in ['vorticity', 'divergence', 'temperature_variation',
                   'log_surface_pressure']:
        np.testing.assert_array_equal(
            getattr(a0, name), np.zeros_like(getattr(a0, name)), err_msg=name
        )
      for name, tracer in a0.tracers.items():
        np.testing.assert_array_equal(
            tracer, np.zeros_like(tracer), err_msg=name
        )
    tol = dict(rtol=1e-4, atol=1e-6)
    with self.subTest('momentum advection is quadratic'):
      np.testing.assert_allclose(a2.vorticity, 4 * a1.vorticity, **tol)
      np.testing.assert_allclose(a2.divergence, 4 * a1.divergence, **tol)
    with self.subTest('scalar advection is linear'):
      np.testing.assert_allclose(
          a2.temperature_variation, 2 * a1.temperature_variation, **tol
      )
      np.testing.assert_allclose(
          a2.log_surface_pressure, 2 * a1.log_surface_pressure, **tol
      )
      for name in a1.tracers:
        np.testing.assert_allclose(
            a2.tracers[name], 2 * a1.tracers[name], err_msg=name, **tol
        )


class SemiLagrangianPrimitiveEquationsTest(parameterized.TestCase):
  """Tests for the semi-Lagrangian primitive equations (dry, sigma)."""

  def setUp(self):
    # Some test modules (e.g. time_integration_test) enable x64 globally at
    # import time, which leaks into full-suite runs. These tests are
    # calibrated in float32, and test_time_step_extension pins the *onset*
    # of an Eulerian instability, which shifts with precision.
    super().setUp()
    self._x64_was_enabled = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', False)

  def tearDown(self):
    jax.config.update('jax_enable_x64', self._x64_was_enabled)
    super().tearDown()

  def _setup(self, grid, layers=8, perturbation=False, **kwargs):
    physics_specs = units.SimUnits.from_si()
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical)
    init_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    state = init_fn()
    if perturbation:
      state = state + primitive_equations_states.baroclinic_perturbation_jw(
          coords, physics_specs
      )
    orography = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords
    )
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    equation = primitive_equations.SemiLagrangianPrimitiveEquations(
        ref_temps, orography, coords, physics_specs, **kwargs
    )
    eulerian = primitive_equations.PrimitiveEquationsSigma(
        ref_temps, orography, coords, physics_specs
    )
    return equation, eulerian, state

  def _nondim_minutes(self, physics_specs, minutes):
    return float(physics_specs.nondimensionalize(minutes * 60 * s_units.s))

  def _l2(self, grid, x, y):
    x, y = grid.to_nodal(x), grid.to_nodal(y)
    return float(np.sqrt(np.square(x - y).sum() / np.square(y).sum()))

  @parameterized.parameters(
      dict(coriolis_mode='planetary_momentum'),
      dict(coriolis_mode='explicit'),
  )
  def test_jw_steady_state_remains_steady(self, coriolis_mode):
    """The JW steady state stays steady over a day of 30-minute steps.

    This is a sensitive test of the vector transport + pressure-gradient
    residual split: any misclassified momentum term unbalances the jet.
    """
    equation, eulerian, state0 = self._setup(
        spherical_harmonic.Grid.T21(), coriolis_mode=coriolis_mode
    )
    del eulerian  # unused
    grid = equation.coords.horizontal
    dt = self._nondim_minutes(equation.physics_specs, 30)
    step_fn = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
    )
    final = time_integration.repeated(step_fn, 48)(state0)
    # measured drift ~1.1e-3, comparable to the Eulerian core at dt=10min
    # (1.3e-3); dominated by the T21 spatial truncation of the initial
    # balance, not by time stepping.
    self.assertLess(
        self._l2(
            grid, final.temperature_variation, state0.temperature_variation
        ),
        3e-3,
    )
    max_divergence = np.abs(grid.to_nodal(final.divergence)).max()
    self.assertLess(float(max_divergence), 2e-2)

  def test_baroclinic_wave_consistency_with_eulerian(self):
    """SL at dt=30min tracks the Eulerian core at dt=10min."""
    equation, eulerian, state0 = self._setup(
        spherical_harmonic.Grid.T21(), perturbation=True
    )
    grid = equation.coords.horizontal
    physics_specs = equation.physics_specs
    sl_final = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(
                equation, self._nondim_minutes(physics_specs, 30)
            )
        ),
        48,
    )(state0)
    eulerian_final = time_integration.repeated(
        jax.jit(
            time_integration.imex_rk_sil3(
                eulerian, self._nondim_minutes(physics_specs, 10)
            )
        ),
        144,
    )(state0)
    # measured: T' 1.2e-3, ln(ps) 3.3e-6.
    self.assertLess(
        self._l2(
            grid,
            sl_final.temperature_variation,
            eulerian_final.temperature_variation,
        ),
        5e-3,
    )
    self.assertLess(
        self._l2(
            grid,
            sl_final.log_surface_pressure,
            eulerian_final.log_surface_pressure,
        ),
        1e-4,
    )

  def test_time_step_extension(self):
    """SL remains stable at time steps where the Eulerian core blows up."""
    # At this extreme step the trajectory iteration operates near its
    # convergence margin (dt·max‖∇V‖ < 1, plan §5): the default
    # single warm-started iteration is built for operating-point steps and
    # drifts 5x more here, so converged (two-iteration) trajectories are
    # requested explicitly.
    equation, eulerian, state0 = self._setup(
        spherical_harmonic.Grid.T42(), departure_iterations=2
    )
    grid = equation.coords.horizontal
    dt = self._nondim_minutes(equation.physics_specs, 180)
    steps = 16  # two simulated days
    sl_final = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
        ),
        steps,
    )(state0)
    eulerian_final = time_integration.repeated(
        jax.jit(time_integration.imex_rk_sil3(eulerian, dt)), steps
    )(state0)
    with self.subTest('Eulerian core is unstable at this time step'):
      self.assertFalse(
          np.isfinite(
              grid.to_nodal(eulerian_final.temperature_variation)
          ).all()
      )
    with self.subTest('semi-Lagrangian core remains stable'):
      # measured drift 0.067: accuracy degrades at this step (the departure
      # iteration approaches its convergence margin dt·max‖∇V‖ < 1, plan §5)
      # but the solution remains bounded and qualitatively steady where the
      # Eulerian core is NaN.
      temperature = grid.to_nodal(sl_final.temperature_variation)
      self.assertTrue(np.isfinite(temperature).all())
      self.assertLess(
          self._l2(
              grid,
              sl_final.temperature_variation,
              state0.temperature_variation,
          ),
          0.1,
      )

  def test_eulerian_stepper_use_is_rejected(self):
    """explicit_terms raises so Eulerian steppers cannot silently misuse."""
    equation, _, state0 = self._setup(spherical_harmonic.Grid.T21())
    with self.assertRaisesRegex(TypeError, 'semi-Lagrangian'):
      equation.explicit_terms(state0)
    step_fn = time_integration.imex_rk_sil3(equation, 0.01)
    with self.assertRaisesRegex(TypeError, 'semi-Lagrangian'):
      step_fn(state0)

  @parameterized.parameters(
      dict(coriolis_mode='planetary_momentum'),
      dict(coriolis_mode='explicit'),
  )
  def test_jw_steady_state_stays_steady_backward_in_time(self, coriolis_mode):
    """Reversed-time SL integration also holds the steady state.

    A sign error in the time-reversed planetary-momentum bookkeeping or
    Coriolis treatment would destroy steadiness immediately.
    """
    equation, _, state0 = self._setup(
        spherical_harmonic.Grid.T21(), coriolis_mode=coriolis_mode
    )
    grid = equation.coords.horizontal
    reversed_equation = time_integration.TimeReversedSemiLagrangianODE(
        equation
    )
    dt = self._nondim_minutes(equation.physics_specs, 30)
    backward_step = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(
            reversed_equation, dt
        )
    )
    forward_step = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
    )
    backward = time_integration.repeated(backward_step, 24)(state0)
    forward = time_integration.repeated(forward_step, 24)(state0)
    backward_drift = self._l2(
        grid, backward.temperature_variation, state0.temperature_variation
    )
    forward_drift = self._l2(
        grid, forward.temperature_variation, state0.temperature_variation
    )
    # backward integration is exactly as steady as forward (measured drifts
    # agree to 0.5% at 24 steps); a sign error in the reversed planetary
    # momentum would inflate the backward drift by orders of magnitude.
    self.assertLess(backward_drift, 1.5 * forward_drift + 1e-4)

  def test_semi_lagrangian_digital_filter_initialization(self):
    """SL DFI runs and agrees with Eulerian DFI on the same window."""
    sl_equation, eulerian, state0 = self._setup(
        spherical_harmonic.Grid.T21(), perturbation=True
    )
    grid = sl_equation.coords.horizontal
    physics_specs = sl_equation.physics_specs
    dt = self._nondim_minutes(physics_specs, 30)
    time_span = self._nondim_minutes(physics_specs, 6 * 60)
    common = dict(time_span=time_span, cutoff_period=time_span, dt=dt)
    sl_dfi = jax.jit(
        time_integration.digital_filter_initialization(
            equation=sl_equation,
            ode_solver=time_integration.semi_lagrangian_crank_nicolson_rk2,
            filters=[],
            **common,
        )
    )
    eulerian_dfi = jax.jit(
        time_integration.digital_filter_initialization(
            equation=eulerian,
            ode_solver=time_integration.imex_rk_sil3,
            filters=[],
            **common,
        )
    )
    sl_filtered = sl_dfi(state0)
    eulerian_filtered = eulerian_dfi(state0)
    for field in ['temperature_variation', 'log_surface_pressure']:
      actual = getattr(sl_filtered, field)
      self.assertTrue(np.isfinite(np.asarray(actual)).all(), field)
      self.assertLess(
          self._l2(grid, actual, getattr(eulerian_filtered, field)),
          2e-3,
          field,
      )

  def test_warm_started_corrector_is_consistent(self):
    """Warm- and cold-started RK2 correctors give nearly the same step.

    The corrector's warm start only changes the initial guess of a
    convergent fixed-point iteration, so over a day of baroclinic-wave
    steps the two solutions must agree to well within the discretization
    error (the residual difference is O((dt·∇V)²) per solve).
    """
    equation, _, state0 = self._setup(
        spherical_harmonic.Grid.T21(), perturbation=True,
        departure_iterations=2,
    )
    grid = equation.coords.horizontal
    dt = self._nondim_minutes(equation.physics_specs, 30)
    finals = {}
    for warm in (True, False):
      step_fn = jax.jit(
          time_integration.semi_lagrangian_crank_nicolson_rk2(
              equation, dt, warm_start_corrector=warm
          )
      )
      finals[warm] = time_integration.repeated(step_fn, 48)(state0)
    difference = self._l2(
        grid,
        finals[True].temperature_variation,
        finals[False].temperature_variation,
    )
    # measured 2.5e-4: well below the case's ~1.1e-3 steady-state drift.
    self.assertLess(difference, 1e-3)
    self.assertGreater(difference, 0.0)  # the guess is actually used

  def test_settls_warm_started_departures_are_consistent(self):
    """SETTLS with and without carried departure points nearly agree."""
    equation, _, state0 = self._setup(
        spherical_harmonic.Grid.T21(), perturbation=True,
        departure_iterations=2,
    )
    grid = equation.coords.horizontal
    dt = self._nondim_minutes(equation.physics_specs, 30)
    finals = {}
    for warm in (True, False):
      init_fn = time_integration.semi_lagrangian_settls_init(
          equation, dt, warm_start_departures=warm
      )
      step_fn = jax.jit(
          time_integration.semi_lagrangian_settls(
              equation, dt, warm_start_departures=warm
          )
      )
      carry = init_fn(state0)
      self.assertLen(carry[1], 3 if warm else 2)
      finals[warm], _ = time_integration.repeated(step_fn, 47)(carry)
    difference = self._l2(
        grid,
        finals[True].temperature_variation,
        finals[False].temperature_variation,
    )
    # measured 2.4e-4, same scale as the RK2 warm-start difference above.
    self.assertLess(difference, 1e-3)
    self.assertGreater(difference, 0.0)

  def test_step_filter_for_nodal_tracers_touches_only_nodal_tracers(self):
    """The nodal filter applies to named nodal tracers and nothing else."""
    grid = spherical_harmonic.Grid.T21()
    rng = np.random.RandomState(0)
    nodal_shape = (4,) + grid.nodal_shape
    state0 = primitive_equations.State(
        vorticity=jnp.asarray(rng.standard_normal(nodal_shape)),
        divergence=jnp.asarray(rng.standard_normal(nodal_shape)),
        temperature_variation=jnp.asarray(rng.standard_normal(nodal_shape)),
        log_surface_pressure=jnp.asarray(
            rng.standard_normal((1,) + grid.nodal_shape)
        ),
        tracers={
            'modal_tracer': jnp.asarray(rng.standard_normal(nodal_shape)),
            'sharp_tracer': jnp.asarray(rng.uniform(0.0, 1.0, nodal_shape)),
        },
    )
    nodal_filter = semi_lagrangian.nodal_diffusion_filter(
        grid, dt=1.0, tau=1.0, order=2
    )
    step_filter = primitive_equations.step_filter_for_nodal_tracers(
        nodal_filter, nodal_tracers=('sharp_tracer',)
    )
    filtered = step_filter(state0, state0)
    with self.subTest('nodal tracer is filtered'):
      self.assertGreater(
          float(
              np.abs(
                  np.asarray(filtered.tracers['sharp_tracer'])
                  - np.asarray(state0.tracers['sharp_tracer'])
              ).max()
          ),
          0.0,
      )
    with self.subTest('other tracers and state pass through untouched'):
      np.testing.assert_array_equal(
          filtered.tracers['modal_tracer'], state0.tracers['modal_tracer']
      )
      np.testing.assert_array_equal(filtered.vorticity, state0.vorticity)
    with self.subTest('missing tracer name fails fast'):
      bad_filter = primitive_equations.step_filter_for_nodal_tracers(
          nodal_filter, nodal_tracers=('absent',)
      )
      with self.assertRaises(KeyError):
        bad_filter(state0, state0)

  def test_settls_tracks_rk2_on_baroclinic_wave(self):
    """The SETTLS stepper matches the RK2 stepper at half the per-step cost.

    A one-day baroclinic-wave comparison of the two SL steppers, including
    sim_time bookkeeping through the SETTLS bracket.
    """
    equation, _, state0 = self._setup(
        spherical_harmonic.Grid.T21(), perturbation=True
    )
    state0.sim_time = 0.0
    grid = equation.coords.horizontal
    dt = self._nondim_minutes(equation.physics_specs, 30)
    rk2_final = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
        ),
        48,
    )(state0)
    init_fn = time_integration.semi_lagrangian_settls_init(equation, dt)
    step_fn = jax.jit(time_integration.semi_lagrangian_settls(equation, dt))
    settls_final, _ = time_integration.repeated(step_fn, 47)(init_fn(state0))
    self.assertLess(
        self._l2(
            grid,
            settls_final.temperature_variation,
            rk2_final.temperature_variation,
        ),
        2e-3,
    )
    self.assertTrue(
        np.isfinite(grid.to_nodal(settls_final.temperature_variation)).all()
    )
    np.testing.assert_allclose(settls_final.sim_time, 48 * dt, rtol=1e-5)

  def test_sim_time_advances_by_dt_per_step(self):
    equation, _, state0 = self._setup(spherical_harmonic.Grid.T21())
    state0.sim_time = 0.0
    dt = self._nondim_minutes(equation.physics_specs, 30)
    step_fn = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
    )
    state = step_fn(step_fn(state0))
    np.testing.assert_allclose(state.sim_time, 2 * dt, rtol=1e-6)

  def test_transport_preserves_constant_tracer(self):
    """A spatially constant tracer must remain constant under transport."""
    equation, _, state0 = self._setup(
        spherical_harmonic.Grid.T21(), perturbation=True
    )
    grid = equation.coords.horizontal
    ones = jnp.zeros_like(state0.temperature_variation)
    ones = spherical_harmonic.add_constant(ones, 1.0)
    state0.tracers = {'constant': ones}
    dt = self._nondim_minutes(equation.physics_specs, 30)
    step_fn = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
    )
    final = time_integration.repeated(step_fn, 4)(state0)
    nodal_tracer = grid.to_nodal(final.tracers['constant'])
    np.testing.assert_allclose(
        nodal_tracer, np.ones_like(nodal_tracer), rtol=1e-4
    )


class SemiLagrangianMoistAndTracerTest(parameterized.TestCase):
  """Moist dynamics, tracer limiting and differentiability of the SL core."""

  def setUp(self):
    # thresholds calibrated in float32; guard against the module-level x64
    # enablement that other test files leak into full-suite runs.
    super().setUp()
    self._x64_was_enabled = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', False)

  def tearDown(self):
    jax.config.update('jax_enable_x64', self._x64_was_enabled)
    super().tearDown()

  def _setup(self, layers=8, humidity=False, **kwargs):
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
    if humidity:
      state.tracers = {
          'specific_humidity': primitive_equations_states.gaussian_scalar(
              coords, physics_specs, amplitude=0.01
          )
      }
    orography = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords
    )
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    equation = primitive_equations.SemiLagrangianPrimitiveEquations(
        ref_temps,
        orography,
        coords,
        physics_specs,
        humidity_key='specific_humidity' if humidity else None,
        **kwargs,
    )
    return equation, state, ref_temps, orography

  def _nondim_minutes(self, physics_specs, minutes):
    return float(physics_specs.nondimensionalize(minutes * 60 * s_units.s))

  def _l2(self, grid, x, y):
    x, y = grid.to_nodal(x), grid.to_nodal(y)
    return float(np.sqrt(np.square(x - y).sum() / np.square(y).sum()))

  def test_moist_equations_reduce_to_dry_for_zero_humidity(self):
    """With q = 0, moist SL tendencies match treating q as a passive tracer."""
    moist, state, ref_temps, orography = self._setup(humidity=True)
    state.tracers = {
        'specific_humidity': jnp.zeros_like(state.temperature_variation)
    }
    dry = primitive_equations.SemiLagrangianPrimitiveEquations(
        ref_temps, orography, moist.coords, moist.physics_specs
    )
    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, atol=1e-7),
        moist.nonadvective_terms(state),
        dry.nonadvective_terms(state),
    )
    dt = self._nondim_minutes(moist.physics_specs, 30)
    step_moist = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(moist, dt)
    )
    step_dry = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(dry, dt)
    )
    jax.tree.map(
        # jitted steps differ by XLA re-association noise (measured 6e-7).
        lambda x, y: np.testing.assert_allclose(x, y, atol=3e-6),
        step_moist(state),
        step_dry(state),
    )

  def test_moist_baroclinic_consistency_with_eulerian(self):
    """Moist SL at dt=30min tracks the moist Eulerian core at dt=10min."""
    equation, state0, ref_temps, orography = self._setup(humidity=True)
    grid = equation.coords.horizontal
    physics_specs = equation.physics_specs
    eulerian = primitive_equations.PrimitiveEquationsSigma(
        ref_temps,
        orography,
        equation.coords,
        physics_specs,
        humidity_key='specific_humidity',
    )
    sl_final = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(
                equation, self._nondim_minutes(physics_specs, 30)
            )
        ),
        48,
    )(state0)
    eulerian_final = time_integration.repeated(
        jax.jit(
            time_integration.imex_rk_sil3(
                eulerian, self._nondim_minutes(physics_specs, 10)
            )
        ),
        144,
    )(state0)
    self.assertLess(
        self._l2(
            grid,
            sl_final.temperature_variation,
            eulerian_final.temperature_variation,
        ),
        5e-3,
    )
    self.assertLess(
        self._l2(
            grid,
            sl_final.tracers['specific_humidity'],
            eulerian_final.tracers['specific_humidity'],
        ),
        0.1,
    )

  def test_tracer_positivity_with_limiter(self):
    """Pins the modal-storage positivity floor the limiter cannot beat.

    Because the state stays modal, each step's modal round trip reintroduces
    Gibbs ringing regardless of how good the transport is: for this barely
    resolved tracer at T21 the undershoot is ~-6% of peak with or without
    the limiter (measured -0.061 limited vs -0.069 unlimited). This is
    precisely the plan §7 caveat motivating opt-in *nodal* tracer storage,
    where the limiter's exact non-negativity survives (see
    `test_nodal_tracer_positivity`). The limiter must still never make
    things worse, never amplify the peak, and approximately conserve mass.
    """
    results = {}
    for limited in [False, True]:
      equation, state0, _, _ = self._setup(
          monotone_tracers=('tracer',) if limited else ()
      )
      physics_specs = equation.physics_specs
      state0.tracers = {
          'tracer': primitive_equations_states.gaussian_scalar(
              coords=equation.coords,
              physics_specs=physics_specs,
              perturbation_radius=0.1,
          )
      }
      grid = equation.coords.horizontal
      dt = self._nondim_minutes(physics_specs, 30)
      step_fn = jax.jit(
          time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
      )
      final = time_integration.repeated(step_fn, 96)(state0)  # two days
      nodal = np.asarray(grid.to_nodal(final.tracers['tracer']))
      initial_nodal = np.asarray(grid.to_nodal(state0.tracers['tracer']))
      mass = grid.integrate(nodal).sum()
      initial_mass = grid.integrate(initial_nodal).sum()
      results[limited] = dict(
          min=nodal.min(),
          max=nodal.max(),
          mass_drift=abs(float(mass - initial_mass)) / float(initial_mass),
      )
    with self.subTest('limiter does not worsen the modal ringing floor'):
      self.assertLess(results[True]['min'], 0.0)  # modal round trip
      self.assertGreaterEqual(
          results[True]['min'], results[False]['min'] - 1e-3
      )
    with self.subTest('no amplification of the maximum'):
      self.assertLess(results[True]['max'], 1.05)
    with self.subTest('mass approximately conserved'):
      self.assertLess(results[True]['mass_drift'], 0.05)

  def test_nodal_tracer_positivity(self):
    """Nodal tracer storage delivers exact non-negativity for sharp tracers.

    The companion test above measures a ~-6% undershoot floor for the same
    tracer in modal storage; storing it nodally removes the modal round trip
    entirely, so the quasi-monotone limiter's bounds hold exactly.
    """
    equation, state0, _, _ = self._setup(
        monotone_tracers=('tracer',), nodal_tracers=('tracer',)
    )
    grid = equation.coords.horizontal
    physics_specs = equation.physics_specs
    modal_tracer = primitive_equations_states.gaussian_scalar(
        coords=equation.coords,
        physics_specs=physics_specs,
        perturbation_radius=0.1,
    )
    nodal_tracer = jnp.maximum(grid.to_nodal(modal_tracer), 0.0)
    state0.tracers = {'tracer': nodal_tracer}
    dt = self._nondim_minutes(physics_specs, 30)
    step_fn = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
    )
    final = time_integration.repeated(step_fn, 96)(state0)  # two days
    tracer = np.asarray(final.tracers['tracer'])
    self.assertEqual(tracer.shape, equation.coords.nodal_shape)
    with self.subTest('exactly non-negative'):
      self.assertGreaterEqual(tracer.min(), 0.0)
    with self.subTest('no new maximum'):
      self.assertLessEqual(tracer.max(), float(np.asarray(nodal_tracer).max()))
    with self.subTest('mass approximately conserved'):
      mass = grid.integrate(tracer).sum()
      initial_mass = grid.integrate(np.asarray(nodal_tracer)).sum()
      self.assertLess(
          abs(float(mass - initial_mass)) / float(initial_mass), 0.05
      )

  def test_nodal_tracer_consistent_with_modal_tracer(self):
    """For a well-resolved tracer, both storage formats evolve alike."""
    finals = {}
    for nodal in [False, True]:
      equation, state0, _, _ = self._setup(
          nodal_tracers=('tracer',) if nodal else ()
      )
      grid = equation.coords.horizontal
      physics_specs = equation.physics_specs
      modal_tracer = primitive_equations_states.gaussian_scalar(
          coords=equation.coords,
          physics_specs=physics_specs,
          perturbation_radius=0.3,
      )
      state0.tracers = {
          'tracer': grid.to_nodal(modal_tracer) if nodal else modal_tracer
      }
      dt = self._nondim_minutes(physics_specs, 30)
      step_fn = jax.jit(
          time_integration.semi_lagrangian_crank_nicolson_rk2(equation, dt)
      )
      final = time_integration.repeated(step_fn, 48)(state0)
      tracer = final.tracers['tracer']
      finals[nodal] = np.asarray(
          grid.to_nodal(tracer) if not nodal else tracer
      )
    difference = np.sqrt(
        np.square(finals[True] - finals[False]).sum()
        / np.square(finals[False]).sum()
    )
    self.assertLess(float(difference), 0.05)

  def test_nodal_tracers_reject_dynamics_keys(self):
    with self.assertRaisesRegex(ValueError, 'cannot be stored nodally'):
      self._setup(humidity=True, nodal_tracers=('specific_humidity',))

  def test_step_filter_excluding_nodal_tracers(self):
    equation, state0, _, _ = self._setup(nodal_tracers=('tracer',))
    grid = equation.coords.horizontal
    nodal_tracer = jnp.maximum(
        grid.to_nodal(
            primitive_equations_states.gaussian_scalar(
                coords=equation.coords, physics_specs=equation.physics_specs
            )
        ),
        0.0,
    )
    state0.tracers = {'tracer': nodal_tracer}
    dt = self._nondim_minutes(equation.physics_specs, 30)
    base_filter = time_integration.exponential_step_filter(grid, dt)
    step_filter = primitive_equations.step_filter_excluding_nodal_tracers(
        base_filter, ('tracer',)
    )
    filtered = step_filter(state0, state0)
    with self.subTest('nodal tracer untouched'):
      np.testing.assert_array_equal(filtered.tracers['tracer'], nodal_tracer)
    with self.subTest('modal fields filtered'):
      self.assertGreater(
          float(
              np.abs(
                  np.asarray(
                      filtered.temperature_variation
                      - state0.temperature_variation
                  )
              ).max()
          ),
          0.0,
      )

  def test_held_suarez_composition(self):
    """compose_equations preserves the SL interface; a forced run is stable."""
    equation, state0, ref_temps, _ = self._setup()
    coords = equation.coords
    grid = coords.horizontal
    physics_specs = equation.physics_specs
    forcing = held_suarez.HeldSuarezForcingSigma(
        coords, physics_specs, ref_temps
    )
    composed = time_integration.compose_equations([equation, forcing])
    self.assertIsInstance(
        composed, time_integration.SemiLagrangianImplicitExplicitODE
    )
    dt = self._nondim_minutes(physics_specs, 30)
    step_fn = jax.jit(
        time_integration.semi_lagrangian_crank_nicolson_rk2(composed, dt)
    )
    final = time_integration.repeated(step_fn, 144)(state0)  # three days
    temperature = np.asarray(
        grid.to_nodal(final.temperature_variation)
        + ref_temps[:, np.newaxis, np.newaxis]
    )
    self.assertTrue(np.isfinite(temperature).all())
    # temperatures remain physical under Held-Suarez relaxation.
    self.assertGreater(temperature.min(), 150.0)
    self.assertLess(temperature.max(), 350.0)


class SemiLagrangianHybridTest(parameterized.TestCase):
  """Minimal validation of the hybrid-coordinate semi-Lagrangian equations.

  The load-bearing test pins the hybrid class to the (well-validated) sigma
  class in the sigma-like configuration (A = 0); genuinely hybrid levels get
  reconstruction, consistency-with-Eulerian and rejection coverage.
  """

  def setUp(self):
    super().setUp()
    self._x64_was_enabled = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', False)

  def tearDown(self):
    jax.config.update('jax_enable_x64', self._x64_was_enabled)
    super().tearDown()

  def _l2(self, grid, x, y):
    x, y = grid.to_nodal(x), grid.to_nodal(y)
    return float(np.sqrt(np.square(x - y).sum() / np.square(y).sum()))

  def _nondim_minutes(self, physics_specs, minutes):
    return float(physics_specs.nondimensionalize(minutes * 60 * s_units.s))

  def _hybrid_setup(self, hybrid_levels, layers=8, **kwargs):
    physics_specs = units.SimUnits.from_si()
    grid = spherical_harmonic.Grid.T21()
    coords = coordinate_systems.CoordinateSystem(grid, hybrid_levels)
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
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    equation = primitive_equations.SemiLagrangianPrimitiveEquationsHybrid(
        ref_temps, orography, coords, physics_specs, **kwargs
    )
    return equation, state, ref_temps, orography

  def test_explicit_terms_split_reconstruction(self):
    """Advective + non-advective terms reconstruct hybrid explicit_terms."""
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        8, sigma_exponent=1.5, stretch_exponent=0.5
    )
    equation, state, ref_temps, orography = self._hybrid_setup(hybrid_levels)
    # the SL class disables explicit_terms; reconstruct against the Eulerian
    # hybrid class, whose split methods the SL class inherits.
    eulerian = primitive_equations.PrimitiveEquationsHybrid(
        ref_temps, orography, equation.coords, equation.physics_specs
    )
    full = eulerian.explicit_terms(state)
    advective = eulerian.explicit_advective_terms(state)
    nonadvective = eulerian.explicit_nonadvective_terms(state)
    reconstructed = jax.tree.map(lambda x, y: x + y, advective, nonadvective)
    assert_states_close(full, reconstructed, rtol=1e-4, atol=1e-6)

  def test_sigma_like_matches_sigma_semi_lagrangian(self):
    """With A = 0 levels, the hybrid SL class matches the sigma SL class."""
    layers = 8
    physics_specs = units.SimUnits.from_si()
    grid = spherical_harmonic.Grid.T21()
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    hybrid_levels = hybrid_coordinates.HybridCoordinates.from_sigma_levels(
        sigma_levels
    )
    coords_sigma = coordinate_systems.CoordinateSystem(grid, sigma_levels)
    coords_hybrid = coordinate_systems.CoordinateSystem(grid, hybrid_levels)
    init_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords_sigma, physics_specs
    )
    state = init_fn()
    state = state + primitive_equations_states.baroclinic_perturbation_jw(
        coords_sigma, physics_specs
    )
    orography = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords_sigma
    )
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    sl_sigma = primitive_equations.SemiLagrangianPrimitiveEquations(
        ref_temps, orography, coords_sigma, physics_specs
    )
    sl_hybrid = primitive_equations.SemiLagrangianPrimitiveEquationsHybrid(
        ref_temps, orography, coords_hybrid, physics_specs
    )
    with self.subTest('vertical nodes match sigma levels'):
      np.testing.assert_allclose(
          sl_hybrid._vertical_nodes.centers, sigma_levels.centers, atol=1e-6
      )
    with self.subTest('nodal velocities match'):
      v_sigma = sl_sigma.nodal_velocities(state)
      v_hybrid = sl_hybrid.nodal_velocities(state)
      for name in ['u', 'v', 'sigma_dot', 'u_mean', 'v_mean']:
        np.testing.assert_allclose(
            getattr(v_hybrid, name),
            getattr(v_sigma, name),
            atol=2e-6,
            err_msg=name,
        )
    with self.subTest('non-advective terms match'):
      # divergence agrees tightly; the temperature entry carries the known
      # Simmons-Burridge vs sigma vertical-discretization difference in the
      # adiabatic term (the Eulerian sigma-like equivalence test absorbs the
      # same gap with atol=1e-3 on nondimensional tendencies).
      n_sigma = sl_sigma.nonadvective_terms(state)
      n_hybrid = sl_hybrid.nonadvective_terms(state)
      for field, tol in [('divergence', 1e-3), ('temperature_variation', 0.1)]:
        self.assertLess(
            self._l2(grid, getattr(n_hybrid, field), getattr(n_sigma, field)),
            tol,
            field,
        )
    with self.subTest('stepped run matches'):
      dt = self._nondim_minutes(physics_specs, 30)
      step_sigma = jax.jit(
          time_integration.semi_lagrangian_crank_nicolson_rk2(sl_sigma, dt)
      )
      step_hybrid = jax.jit(
          time_integration.semi_lagrangian_crank_nicolson_rk2(sl_hybrid, dt)
      )
      out_sigma = time_integration.repeated(step_sigma, 12)(state)
      out_hybrid = time_integration.repeated(step_hybrid, 12)(state)
      # measured 4.0e-3: the accumulated Simmons-Burridge vs sigma vertical
      # discretization difference (mostly the adiabatic term), matching the
      # Eulerian classes' behavior in the same configuration.
      self.assertLess(
          self._l2(
              grid,
              out_hybrid.temperature_variation,
              out_sigma.temperature_variation,
          ),
          8e-3,
      )

  def test_baroclinic_wave_consistent_with_eulerian_hybrid(self):
    """On genuinely hybrid levels, SL tracks the Eulerian hybrid core."""
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        8, sigma_exponent=1.5, stretch_exponent=0.5
    )
    sl_equation, state, ref_temps, orography = self._hybrid_setup(
        hybrid_levels
    )
    physics_specs = sl_equation.physics_specs
    grid = sl_equation.coords.horizontal
    eulerian = primitive_equations.PrimitiveEquationsHybrid(
        ref_temps, orography, sl_equation.coords, physics_specs
    )
    sl_final = time_integration.repeated(
        jax.jit(
            time_integration.semi_lagrangian_crank_nicolson_rk2(
                sl_equation, self._nondim_minutes(physics_specs, 30)
            )
        ),
        24,
    )(state)
    eulerian_final = time_integration.repeated(
        jax.jit(
            time_integration.imex_rk_sil3(
                eulerian, self._nondim_minutes(physics_specs, 10)
            )
        ),
        72,
    )(state)
    self.assertTrue(
        np.isfinite(grid.to_nodal(sl_final.temperature_variation)).all()
    )
    # measured 1.0e-3 (24 SL steps at dt=30 min vs 72 Eulerian at 10 min).
    self.assertLess(
        self._l2(
            grid,
            sl_final.temperature_variation,
            eulerian_final.temperature_variation,
        ),
        3e-3,
    )

  def test_eulerian_stepper_use_is_rejected(self):
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        8, sigma_exponent=1.5, stretch_exponent=0.5
    )
    equation, state, _, _ = self._hybrid_setup(hybrid_levels)
    with self.assertRaisesRegex(TypeError, 'semi-Lagrangian'):
      equation.explicit_terms(state)


def interpolate_state_hybrid_to_sigma(
    state_hybrid: primitive_equations.State,
    coords_hybrid: coordinate_systems.CoordinateSystem,
    coords_sigma: coordinate_systems.CoordinateSystem,
    surface_pressure_nodal: jax.Array,
) -> primitive_equations.State:
  """Interpolates hybrid state to sigma coordinates."""

  grid = coords_hybrid.horizontal
  hybrid_levels = coords_hybrid.vertical
  sigma_levels = coords_sigma.vertical

  fields_to_interpolate = {
      'vorticity': grid.to_nodal(state_hybrid.vorticity),
      'divergence': grid.to_nodal(state_hybrid.divergence),
      'temperature_variation': grid.to_nodal(
          state_hybrid.temperature_variation
      ),
  }
  if state_hybrid.tracers:
    fields_to_interpolate['tracers'] = jax.tree.map(
        grid.to_nodal, state_hybrid.tracers
    )
  interpolated_fields = vertical_interpolation.interp_hybrid_to_sigma(
      fields_to_interpolate,
      hybrid_levels,
      sigma_levels,
      surface_pressure_nodal.squeeze(),
  )
  modal_interpolated_fields = jax.tree.map(grid.to_modal, interpolated_fields)
  return state_hybrid.replace(**modal_interpolated_fields)


class PrimitiveEquationsHybridTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(wavenumbers=32, layers=4),
      dict(wavenumbers=64, layers=10),
  )
  def test_primitive_equations_hybrid_shape(self, wavenumbers, layers):
    """Tests that output of explicit_terms has expected shape."""
    coords = coordinate_systems.CoordinateSystem(
        horizontal=spherical_harmonic.Grid.with_wavenumbers(wavenumbers),
        vertical=hybrid_coordinates.HybridCoordinates.analytic_levels(layers),
    )
    reference_temperature = 300 * np.ones(layers)
    l, _ = coords.horizontal.modal_mesh
    modal_orography = np.zeros_like(l)
    vorticity = jnp.ones((layers,) + l.shape)
    divergence = jnp.ones((layers,) + l.shape)
    temperature_variation = jnp.ones((layers,) + l.shape)
    log_surface_pressure = jnp.ones((1,) + l.shape)

    physics_specs = units.SimUnits.from_si()
    state = primitive_equations.State(
        vorticity, divergence, temperature_variation, log_surface_pressure
    )
    primitive = primitive_equations.PrimitiveEquationsHybrid(
        reference_temperature, modal_orography, coords, physics_specs
    )

    explicit_output = primitive.explicit_terms(state)
    implicit_output = primitive.implicit_terms(state)
    for output in [explicit_output, implicit_output]:
      with self.subTest('divergence shape'):
        self.assertEqual(state.divergence.shape, output.divergence.shape)
      with self.subTest('vorticity shape'):
        self.assertEqual(state.vorticity.shape, output.vorticity.shape)
      with self.subTest('temperature shape'):
        self.assertEqual(
            state.temperature_variation.shape,
            output.temperature_variation.shape,
        )
      with self.subTest('log_surface_pressure shape'):
        self.assertEqual(
            state.log_surface_pressure.shape,
            output.log_surface_pressure.shape,
        )

  def test_equivalence_sigma_like(self):
    """Tests equivalence to PE when using sigma-like coordinates."""
    wavenumbers = 21
    layers = 40
    physics_specs = units.SimUnits.from_si()
    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    hybrid_levels = hybrid_coordinates.HybridCoordinates.from_sigma_levels(
        sigma_levels
    )
    coords_sigma = coordinate_systems.CoordinateSystem(grid, sigma_levels)
    coords_hybrid = coordinate_systems.CoordinateSystem(grid, hybrid_levels)
    state_fn_sigma, features_sigma = primitive_equations_states.steady_state_jw(
        coords_sigma, physics_specs
    )
    state = state_fn_sigma()
    state_hybrid = primitive_equations.State(**state.asdict())
    ref_temps = features_sigma[xarray_utils.REF_TEMP_KEY]
    modal_orography = primitive_equations.truncated_modal_orography(
        features_sigma[xarray_utils.OROGRAPHY], coords_sigma
    )
    pe_sigma = primitive_equations.PrimitiveEquationsSigma(
        ref_temps,
        modal_orography,
        coords_sigma,
        physics_specs,
    )
    pe_hybrid = primitive_equations.PrimitiveEquationsHybrid(
        ref_temps,
        modal_orography,
        coords_hybrid,
        physics_specs,
    )
    explicit_sigma = pe_sigma.explicit_terms(state)
    explicit_hybrid = pe_hybrid.explicit_terms(state_hybrid)
    implicit_sigma = pe_sigma.implicit_terms(state)
    implicit_hybrid = pe_hybrid.implicit_terms(state_hybrid)

    nodal_surface_pressure = jnp.exp(
        coords_hybrid.horizontal.to_nodal(state_hybrid.log_surface_pressure)
    )
    interp_fn = functools.partial(
        interpolate_state_hybrid_to_sigma,
        coords_hybrid=coords_hybrid,
        coords_sigma=coords_sigma,
        surface_pressure_nodal=nodal_surface_pressure,
    )
    explicit_hybrid_interp = interp_fn(explicit_hybrid)
    implicit_hybrid_interp = interp_fn(implicit_hybrid)

    with self.subTest('explicit_terms'):
      assert_states_close(
          explicit_sigma,
          primitive_equations.State(**explicit_hybrid_interp.asdict()),
          atol=1e-3,
      )
    with self.subTest('implicit_terms'):
      assert_states_close(
          implicit_sigma,
          primitive_equations.State(**implicit_hybrid_interp.asdict()),
          atol=1e-3,
      )

  @parameterized.parameters(
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(16),
          layers=5,
          implicit_inverse_method='split',
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(16),
          layers=5,
          implicit_inverse_method='blockwise',
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(16),
          layers=5,
          implicit_inverse_method='stacked',
          seed=0,
      ),
  )
  def test_implicit_inverse_sigma_like(
      self, grid, layers, implicit_inverse_method, seed
  ):
    """`implicit_inverse` computes (1 - step_size · implicit_terms)⁻¹."""
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    hybrid_levels = hybrid_coordinates.HybridCoordinates.from_sigma_levels(
        sigma_levels
    )
    coords = coordinate_systems.CoordinateSystem(grid, hybrid_levels)
    physics_specs = units.SimUnits.from_si()
    state = random_state(coords, jax.random.PRNGKey(seed))
    reference_temperature = 280 * np.ones(layers)
    l, _ = coords.horizontal.modal_mesh
    modal_orography = np.zeros_like(l)
    step_size = 0.1
    primitive = primitive_equations.PrimitiveEquationsHybrid(
        reference_temperature,
        modal_orography,
        coords,
        physics_specs,
        implicit_inverse_method=implicit_inverse_method,
    )
    implicit_terms = primitive.implicit_terms(state)
    primitive_equations.validate_state_shape(implicit_terms, coords)
    jitted_inverse = jax.jit(lambda s: primitive.implicit_inverse(s, step_size))
    inverted_state = jitted_inverse(state - step_size * implicit_terms)
    primitive_equations.validate_state_shape(inverted_state, coords)
    assert_states_close(state, inverted_state, atol=1e-5)

  def test_stationarity_sigma_like(self):
    """Tests that isothermal rest state is stationary."""
    wavenumbers = 42
    layers = 26
    dt_si = 600 * s_units.s
    save_every_si = 4 * s_units.hour
    inner_steps = int(save_every_si / dt_si)
    outer_steps = 6
    physics_specs = units.SimUnits.from_si()
    dt = physics_specs.nondimensionalize(dt_si)
    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    hybrid_levels = hybrid_coordinates.HybridCoordinates.from_sigma_levels(
        sigma_levels
    )
    coords = coordinate_systems.CoordinateSystem(grid, hybrid_levels)
    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    modal_orography = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords
    )
    state_sigma = initial_state_fn()
    state = primitive_equations.State(**state_sigma.asdict())
    primitive = primitive_equations.PrimitiveEquationsHybrid(
        ref_temps, modal_orography, coords, physics_specs
    )
    step_fn = time_integration.imex_rk_sil3(primitive, dt)
    filters = [
        time_integration.exponential_step_filter(coords.horizontal, dt),
    ]
    step_fn = time_integration.step_with_filters(step_fn, filters)
    trajectory_fn = time_integration.trajectory_from_step(
        step_fn,
        outer_steps,
        inner_steps,
    )
    trajectory_fn = jax.jit(trajectory_fn)
    input_state = state
    _, trajectory = trajectory_fn(input_state)
    trajectory = jax.device_get(trajectory)
    for step in range(outer_steps):
      step_state = jax.tree.map(lambda x: x[step], trajectory)  # pylint: disable=cell-var-from-loop
      with self.subTest(f'Divergence remains close to zero, step {step}'):
        np.testing.assert_array_less(abs(step_state.divergence), 8e-3)

      with self.subTest(f'Vorticity is stationary, step {step}'):
        np.testing.assert_allclose(
            step_state.vorticity, state.vorticity, atol=6e-3
        )

      with self.subTest(f'Temperature is stationary, step {step}'):
        np.testing.assert_allclose(
            step_state.temperature_variation,
            state.temperature_variation,
            atol=0.2,
        )

      with self.subTest(f'Log surface pressure is stationary, step {step}'):
        np.testing.assert_allclose(
            step_state.log_surface_pressure,
            state.log_surface_pressure,
            atol=5e-4,
        )

  def test_simulation_equivalence_sigma_like(self):
    """Tests that simulation with sigma-like hybrid coords matches PE."""
    wavenumbers = 21
    layers = 10
    dt_si = 600 * s_units.s
    # simulation for 3 days
    sim_time_si = 3 * s_units.day
    inner_steps = int(sim_time_si / dt_si)
    outer_steps = 1
    physics_specs = units.SimUnits.from_si()
    dt = physics_specs.nondimensionalize(dt_si)
    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    hybrid_levels = hybrid_coordinates.HybridCoordinates.from_sigma_levels(
        sigma_levels
    )
    coords_sigma = coordinate_systems.CoordinateSystem(grid, sigma_levels)
    coords_hybrid = coordinate_systems.CoordinateSystem(grid, hybrid_levels)
    state_fn_sigma, features_sigma = primitive_equations_states.steady_state_jw(
        coords_sigma, physics_specs
    )
    state_sigma = state_fn_sigma()
    # initial states are numerically identical, but need to convert type for
    # hybrid.
    state_hybrid = primitive_equations.State(**state_sigma.asdict())
    modal_orography = primitive_equations.truncated_modal_orography(
        features_sigma[xarray_utils.OROGRAPHY], coords_sigma
    )

    pe_sigma = primitive_equations.PrimitiveEquationsSigma(
        features_sigma[xarray_utils.REF_TEMP_KEY],
        modal_orography,
        coords_sigma,
        physics_specs,
    )
    pe_hybrid = primitive_equations.PrimitiveEquationsHybrid(
        features_sigma[xarray_utils.REF_TEMP_KEY],
        modal_orography,
        coords_hybrid,
        physics_specs,
    )

    def run_sim(primitive, state):
      step_fn = time_integration.imex_rk_sil3(primitive, dt)
      filters = [
          time_integration.exponential_step_filter(
              primitive.coords.horizontal, dt
          ),
      ]
      step_fn = time_integration.step_with_filters(step_fn, filters)
      trajectory_fn = time_integration.trajectory_from_step(
          step_fn, outer_steps, inner_steps
      )
      trajectory_fn = jax.jit(trajectory_fn)
      final_state, _ = trajectory_fn(state)
      return final_state

    final_state_sigma = run_sim(pe_sigma, state_sigma)
    final_state_hybrid = run_sim(pe_hybrid, state_hybrid)
    nodal_surface_pressure = jnp.exp(
        coords_hybrid.horizontal.to_nodal(
            final_state_hybrid.log_surface_pressure
        )
    )
    interp_fn = functools.partial(
        interpolate_state_hybrid_to_sigma,
        coords_hybrid=coords_hybrid,
        coords_sigma=coords_sigma,
        surface_pressure_nodal=nodal_surface_pressure,
    )
    final_state_hybrid_interp = interp_fn(final_state_hybrid)
    assert_states_close(
        final_state_sigma,
        primitive_equations.State(**final_state_hybrid_interp.asdict()),
        atol=0.2,
    )
    nodal_surface_pressure_sigma = jnp.exp(
        coords_sigma.horizontal.to_nodal(
            final_state_sigma.log_surface_pressure
        )
    )
    np.testing.assert_allclose(
        nodal_surface_pressure_sigma,
        nodal_surface_pressure,
        rtol=1e-3,
    )


  def test_baroclinic_test_case_surface_pressure_similar(self):
    """Tests that simulation with hybrid and simga coords are close."""
    wavenumbers = 21
    layers = 16
    dt_si = 600 * s_units.s
    # simulation for 3 days
    sim_time_si = 1 * s_units.day
    inner_steps = int(sim_time_si / dt_si)
    outer_steps = 1
    physics_specs = units.SimUnits.from_si()
    dt = physics_specs.nondimensionalize(dt_si)
    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        layers, sigma_exponent=1.5, stretch_exponent=0.5
    )
    coords_sigma = coordinate_systems.CoordinateSystem(grid, sigma_levels)
    coords_hybrid = coordinate_systems.CoordinateSystem(grid, hybrid_levels)
    state_fn_sigma, features_sigma = primitive_equations_states.steady_state_jw(
        coords_sigma, physics_specs
    )
    perturbation = primitive_equations_states.baroclinic_perturbation_jw(
        coords_sigma,
        physics_specs,
    )
    state_sigma = state_fn_sigma()
    state_sigma += perturbation
    state_fn_hybrid, features_hybrid = primitive_equations_states.steady_state_jw(
        coords_hybrid, physics_specs
    )
    state_hybrid = state_fn_hybrid()
    state_hybrid += perturbation  # perturbation is level independent.
    modal_orography = primitive_equations.truncated_modal_orography(
        features_sigma[xarray_utils.OROGRAPHY], coords_sigma
    )

    pe_sigma = primitive_equations.PrimitiveEquationsSigma(
        features_sigma[xarray_utils.REF_TEMP_KEY],
        modal_orography,
        coords_sigma,
        physics_specs,
    )
    pe_hybrid = primitive_equations.PrimitiveEquationsHybrid(
        features_hybrid[xarray_utils.REF_TEMP_KEY],
        modal_orography,
        coords_hybrid,
        physics_specs,
    )

    def run_sim(primitive, state):
      step_fn = time_integration.imex_rk_sil3(primitive, dt)
      filters = [
          time_integration.exponential_step_filter(
              primitive.coords.horizontal, dt
          ),
      ]
      step_fn = time_integration.step_with_filters(step_fn, filters)
      trajectory_fn = time_integration.trajectory_from_step(
          step_fn, outer_steps, inner_steps
      )
      trajectory_fn = jax.jit(trajectory_fn)
      final_state, _ = trajectory_fn(state)
      return final_state

    final_state_sigma = run_sim(pe_sigma, state_sigma)
    final_state_hybrid = run_sim(pe_hybrid, state_hybrid)
    nodal_surface_pressure_hybrid = jnp.exp(
        coords_hybrid.horizontal.to_nodal(
            final_state_hybrid.log_surface_pressure
        )
    )
    nodal_surface_pressure_sigma = jnp.exp(
        coords_sigma.horizontal.to_nodal(
            final_state_sigma.log_surface_pressure
        )
    )
    np.testing.assert_allclose(
        nodal_surface_pressure_hybrid,
        nodal_surface_pressure_sigma,
        rtol=1e-3,
    )

  @parameterized.parameters(
      dict(scale=scales.DEFAULT_SCALE),
      dict(scale=scales.ATMOSPHERIC_SCALE),
  )
  def test_tracer_conservation(self, scale):
    """Tests that tracer mass is conserved over a simulation."""
    wavenumbers = 21
    layers = 10
    dt_si = 600 * s_units.s
    # simulation for 1 day
    sim_time_si = 1 * s_units.day
    inner_steps = int(sim_time_si / dt_si)
    outer_steps = 1
    physics_specs = units.SimUnits.from_si(scale=scale)
    dt = physics_specs.nondimensionalize(dt_si)
    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    ref_pressure_in_hpa = 1000
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        layers,
        p0=ref_pressure_in_hpa,
        sigma_exponent=1.5,
        stretch_exponent=0.5,
    )
    coords = coordinate_systems.CoordinateSystem(grid, hybrid_levels)

    state_fn, features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    state_pe = state_fn()
    state = primitive_equations.State(**state_pe.asdict())
    state.tracers = {
        'tracer': primitive_equations_states.gaussian_scalar(
            coords, physics_specs
        )
    }

    modal_orography = primitive_equations.truncated_modal_orography(
        features[xarray_utils.OROGRAPHY], coords
    )
    ref_temps = features[xarray_utils.REF_TEMP_KEY]
    # Primitive equations nondimensionalize reference values and coordinates,
    # so we can pass in coords directly.
    primitive = primitive_equations.PrimitiveEquationsHybrid(
        ref_temps,
        modal_orography,
        coords,
        physics_specs,
        reference_surface_pressure=ref_pressure_in_hpa * scales.units.millibar,
    )

    def tracer_integral(tracer, surface_pressure_nodal):
      tracer_nodal = coords.horizontal.to_nodal(tracer)
      # surface pressure is in nondim units, hence we need to use nondim coords.
      tracer_columns = hybrid_coordinates.integral_over_pressure(
          tracer_nodal,
          surface_pressure_nodal,
          primitive.nondim_coords.vertical,
          keepdims=False,
      )
      return coords.horizontal.integrate(tracer_columns)

    nodal_surface_pressure_initial = jnp.exp(
        coords.horizontal.to_nodal(state.log_surface_pressure)
    )
    initial_tracer_total = tracer_integral(
        state.tracers['tracer'], nodal_surface_pressure_initial
    )

    step_fn = time_integration.imex_rk_sil3(primitive, dt)
    filters = [
        time_integration.exponential_step_filter(
            primitive.coords.horizontal, dt
        ),
    ]
    step_fn = time_integration.step_with_filters(step_fn, filters)
    trajectory_fn = time_integration.trajectory_from_step(
        step_fn, outer_steps, inner_steps
    )
    trajectory_fn = jax.jit(trajectory_fn)
    final_state, _ = trajectory_fn(state)

    nodal_surface_pressure_final = jnp.exp(
        coords.horizontal.to_nodal(final_state.log_surface_pressure)
    )
    final_tracer_total = tracer_integral(
        final_state.tracers['tracer'], nodal_surface_pressure_final
    )
    np.testing.assert_allclose(
        initial_tracer_total, final_tracer_total, rtol=1e-3
    )

  def test_primitive_equations_hybrid_run_with_clouds(self):
    """Tests that the hybrid primitive equations can be run with clouds."""
    wavenumbers = 21
    layers = 10
    dt_si = 600 * s_units.s
    inner_steps = 2
    outer_steps = 1
    physics_specs = units.SimUnits.from_si()
    dt = physics_specs.nondimensionalize(dt_si)
    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_levels(layers)
    coords = coordinate_systems.CoordinateSystem(grid, hybrid_levels)

    state_fn, features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    state_pe = state_fn()
    state = primitive_equations.State(**state_pe.asdict())
    state.tracers = {
        'specific_humidity': primitive_equations_states.gaussian_scalar(
            coords, physics_specs
        ),
        'specific_cloud_liquid_water_content': (
            (primitive_equations_states.gaussian_scalar(coords, physics_specs))
            * 0.1
        ),
        'specific_cloud_ice_water_content': (
            (primitive_equations_states.gaussian_scalar(coords, physics_specs))
            * 0.01
        ),
    }

    modal_orography = primitive_equations.truncated_modal_orography(
        features[xarray_utils.OROGRAPHY], coords
    )
    ref_temps = features[xarray_utils.REF_TEMP_KEY]

    primitive = primitive_equations.PrimitiveEquationsHybrid(
        ref_temps,
        modal_orography,
        coords,
        physics_specs,
        humidity_key='specific_humidity',
        cloud_keys=(
            'specific_cloud_liquid_water_content',
            'specific_cloud_ice_water_content',
        ),
    )
    step_fn = time_integration.imex_rk_sil3(primitive, dt)
    trajectory_fn = time_integration.trajectory_from_step(
        step_fn, outer_steps, inner_steps, start_with_input=True
    )
    trajectory_fn = jax.jit(trajectory_fn)
    _, trajectory = trajectory_fn(state)
    self.assertEqual(
        trajectory.vorticity.shape, (outer_steps,) + coords.modal_shape
    )


class PrimitiveEquationsSpecsTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.timedelta64(1, 'h'),),
      (np.timedelta64(1, 'm'),),
      (np.timedelta64(1, 's'),),
      (np.arange(5).astype('timedelta64[s]'),),
  )
  def test_timedelta_roundtrip(self, timedelta):
    physics_specs = units.SimUnits.from_si()
    dt = physics_specs.nondimensionalize_timedelta64(timedelta)
    actual = physics_specs.dimensionalize_timedelta64(dt)
    np.testing.assert_equal(actual, timedelta)

  @parameterized.parameters(
      dict(value=1.0, expected=6856),  # rounded down from 6856.8294
      dict(value=1e-4, expected=0),  # rounded down from 0.0.68568294
  )
  def test_equivalent_rounding_behavior(self, value, expected):
    physics_specs = units.SimUnits.from_si()
    array_value = np.array(value)
    actual_scalar = physics_specs.dimensionalize_timedelta64(value)
    acutal_array = physics_specs.dimensionalize_timedelta64(array_value)
    self.assertEqual(actual_scalar, expected)
    self.assertEqual(acutal_array, np.array(actual_scalar))


if __name__ == '__main__':
  absltest.main()
