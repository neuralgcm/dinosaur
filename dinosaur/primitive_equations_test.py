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
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import xarray_utils
import jax
from jax import config
import jax.numpy as jnp
import numpy as np


config.parse_flags_with_absl()

units = scales.units


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


class PrimitiveEquationsImplicitTest(parameterized.TestCase):

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
    """Tests that `get_geopotential` works for steady states."""
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
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
      actual = coords.horizontal.to_nodal(
          primitive_equations.get_geopotential(
              state.temperature_variation,
              aux_features[xarray_utils.REF_TEMP_KEY],
              modal_orography,
              coords.vertical,
              physics_specs.gravity_acceleration,
              physics_specs.ideal_gas_constant,
          )
      )
      np.testing.assert_allclose(actual, expected_geopotential, atol=atol)
    with self.subTest('moist_geopotential'):
      temperature = aux_features[xarray_utils.REF_TEMP_KEY][
          :, np.newaxis, np.newaxis
      ] + coords.horizontal.to_nodal(state.temperature_variation)
      specific_humidity = jnp.zeros_like(temperature)
      nodal_orography = coords.horizontal.to_nodal(modal_orography)
      actual = primitive_equations.get_geopotential_with_moisture(
          temperature,
          specific_humidity,
          nodal_orography,
          coords.vertical,
          physics_specs.gravity_acceleration,
          physics_specs.ideal_gas_constant,
          physics_specs.water_vapor_gas_constant,
      )
      np.testing.assert_allclose(actual, expected_geopotential, atol=atol)

  def test_stationary_solution(self):
    """Tests that steady state is stationary for primitive equations."""
    wavenumbers = 42
    layers = 26
    dt_si = 600 * units.s
    save_every_si = 4 * units.hour
    inner_steps = int(save_every_si / dt_si)
    outer_steps = 6
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
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
    primitive = primitive_equations.PrimitiveEquations(
        ref_temps, modal_orography, coords, physics_specs
    )
    step_fn = time_integration.semi_implicit_leapfrog(primitive, dt)
    filters = (
        time_integration.exponential_leapfrog_step_filter(
            coords.horizontal, dt
        ),
        time_integration.robert_asselin_leapfrog_filter(0.05),
    )
    step_fn = time_integration.step_with_filters(step_fn, filters)
    post_process_fn = lambda x: x[0]  # select slices of leapfrog tuple.
    trajectory_fn = time_integration.trajectory_from_step(
        step_fn, outer_steps, inner_steps, post_process_fn=post_process_fn
    )
    trajectory_fn = jax.jit(trajectory_fn)
    input_state = (state, state)
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
  def test_get_geopotential_weights(self, coordinates, ideal_gas_constant):
    """Tests that the entries of geopotential weights `G` are correct."""
    g = primitive_equations.get_geopotential_weights(
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

  def test_get_geopotential_diff_both_ways(self):
    temperature = np.random.RandomState(0).randn(12, 1, 1)
    coordinates = sigma_coordinates.SigmaCoordinates.equidistant(12)
    ideal_gas_constant = 1.5
    result_matvec = primitive_equations.get_geopotential_diff(
        temperature, coordinates, ideal_gas_constant, method='dense'
    )
    result_cumsum = primitive_equations.get_geopotential_diff(
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
  def test_get_temperature_implicit_weights(
      self, coordinates, reference_temperature, heat_capacity_ratio
  ):
    """Tests that the entries of temperature weights `H` are correct."""
    h = primitive_equations.get_temperature_implicit_weights(
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
  def test_get_temperature_implicit_both_ways(self, reference_temperature):
    divergence = np.random.RandomState(0).randn(5, 1, 1)
    coordinates = sigma_coordinates.SigmaCoordinates.equidistant(5)
    kappa = 2 / 7
    result_matvec = primitive_equations.get_temperature_implicit(
        divergence, coordinates, reference_temperature, kappa, method='dense'
    )
    result_cumsum = primitive_equations.get_temperature_implicit(
        divergence, coordinates, reference_temperature, kappa, method='sparse'
    )
    np.testing.assert_allclose(result_matvec, result_cumsum, atol=1e-5)

  @parameterized.parameters(
      dict(wavenumbers=32, layers=4),
      dict(wavenumbers=64, layers=10),
  )
  def test_primitive_equations_explicit_shape(self, wavenumbers, layers):
    """Tests that output of primitive_equations_explicit has expected shape."""
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

    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    state = primitive_equations.State(
        vorticity, divergence, temperature_variation, log_surface_pressure
    )
    primitive = primitive_equations.PrimitiveEquations(
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
  def test_primitive_equations_explicit_scales_invariance(
      self, wavenumbers, layers
  ):
    """Tests that tendencies in SI units are not affected by scales."""
    default_scale = scales.DEFAULT_SCALE
    custom_scale = scales.Scale(
        scales.RADIUS / 100,
        55.3 / 2 / scales.OMEGA,
        1 * units.kilogram * 16.4,
        1 * units.degK * 3.15,
    )
    physics_specs_a = primitive_equations.PrimitiveEquationsSpecs.from_si(
        scale=default_scale
    )
    grid_a = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, radius=physics_specs_a.radius
    )
    physics_specs_b = primitive_equations.PrimitiveEquationsSpecs.from_si(
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
    primitive_a = primitive_equations.PrimitiveEquations(
        aux_features_a[xarray_utils.REF_TEMP_KEY],
        modal_orography_a,
        coordinate_systems.CoordinateSystem(grid_a, vertical_grid),
        physics_specs_a,
    )

    primitive_b = primitive_equations.PrimitiveEquations(
        aux_features_b[xarray_utils.REF_TEMP_KEY],
        modal_orography_b,
        coordinate_systems.CoordinateSystem(grid_b, vertical_grid),
        physics_specs_b,
    )
    tendencies_a = primitive_a.explicit_terms(state_a)
    tendencies_b = primitive_b.explicit_terms(state_b)

    with self.subTest('divergence tendency'):
      divergence_a = physics_specs_a.dimensionalize(
          tendencies_a.divergence, 1 / units.s**2
      )
      divergence_b = physics_specs_b.dimensionalize(
          tendencies_b.divergence, 1 / units.s**2
      )
      np.testing.assert_allclose(
          divergence_a.magnitude, divergence_b.magnitude, atol=5e-7
      )
    with self.subTest('vorticity tendency'):
      vorticity_a = physics_specs_a.dimensionalize(
          tendencies_a.vorticity, 1 / units.s**2
      )
      vorticity_b = physics_specs_b.dimensionalize(
          tendencies_b.vorticity, 1 / units.s**2
      )
      np.testing.assert_allclose(
          vorticity_a.magnitude, vorticity_b.magnitude, atol=5e-7
      )
    with self.subTest('temperature tendency'):
      temperature_a = physics_specs_a.dimensionalize(
          tendencies_a.temperature_variation, units.degK / units.s
      )
      temperature_b = physics_specs_b.dimensionalize(
          tendencies_b.temperature_variation, units.degK / units.s
      )
      np.testing.assert_allclose(
          temperature_a.magnitude, temperature_b.magnitude, atol=5e-7
      )
    with self.subTest('surface pressure tendency'):
      pressure_a = physics_specs_a.dimensionalize(
          np.exp(tendencies_a.log_surface_pressure), units.pascal / units.s
      )
      pressure_b = physics_specs_b.dimensionalize(
          np.exp(tendencies_b.log_surface_pressure), units.pascal / units.s
      )
      np.testing.assert_allclose(
          pressure_a.magnitude, pressure_b.magnitude, atol=1e-7
      )

  @parameterized.parameters(
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(16),
          vertical_grid=sigma_coordinates.SigmaCoordinates.equidistant(5),
          reference_temperature=np.linspace(100, 200, 5),
          kappa=1.4 * units.dimensionless,
          ideal_gas_constant=33 * units.J / units.kilogram / units.degK,
          step_size=0.3,
          implicit_inverse_method='split',
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(16),
          vertical_grid=sigma_coordinates.SigmaCoordinates.equidistant(5),
          reference_temperature=np.linspace(100, 200, 5),
          kappa=1.4 * units.dimensionless,
          ideal_gas_constant=33 * units.J / units.kilogram / units.degK,
          step_size=0.3,
          implicit_inverse_method='blockwise',
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(16),
          vertical_grid=sigma_coordinates.SigmaCoordinates.equidistant(5),
          reference_temperature=np.linspace(100, 200, 5),
          kappa=1.4 * units.dimensionless,
          ideal_gas_constant=33 * units.J / units.kilogram / units.degK,
          step_size=0.3,
          implicit_inverse_method='stacked',
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(128),
          vertical_grid=sigma_coordinates.SigmaCoordinates.equidistant(23),
          reference_temperature=np.linspace(250, 300, 23),
          kappa=111 * units.dimensionless,
          ideal_gas_constant=1 * units.J / units.kilogram / units.degK,
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
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si(
        ideal_gas_constant_si=ideal_gas_constant, kappa_si=kappa
    )
    state = random_state(coords, jax.random.PRNGKey(seed))
    l, _ = coords.horizontal.modal_mesh
    modal_orography = np.zeros_like(l)
    primitive = primitive_equations.PrimitiveEquations(
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
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
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
    primitive_a = primitive_equations.PrimitiveEquationsWithTime(
        ref_temps, modal_orography, coords, physics_specs
    )
    primitive_b = primitive_equations.MoistPrimitiveEquations(
        ref_temps, modal_orography, coords, physics_specs
    )

    tendencies_a = primitive_a.explicit_terms(state)
    tendencies_b = primitive_b.explicit_terms(state)

    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, atol=1e-7),
        tendencies_a,
        tendencies_b,
    )


class PrimitiveEquationsSpecsTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.timedelta64(1, 'h'),),
      (np.timedelta64(1, 'm'),),
      (np.timedelta64(1, 's'),),
      (np.arange(5).astype('timedelta64[s]'),),
  )
  def test_timedelta_roundtrip(self, timedelta):
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    dt = physics_specs.nondimensionalize_timedelta64(timedelta)
    actual = physics_specs.dimensionalize_timedelta64(dt)
    np.testing.assert_equal(actual, timedelta)

  @parameterized.parameters(
      dict(value=1.0, expected=6856),  # rounded down from 6856.8294
      dict(value=1e-4, expected=0),  # rounded down from 0.0.68568294
  )
  def test_equivalent_rounding_behavior(self, value, expected):
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    array_value = np.array(value)
    actual_scalar = physics_specs.dimensionalize_timedelta64(value)
    acutal_array = physics_specs.dimensionalize_timedelta64(array_value)
    self.assertEqual(actual_scalar, expected)
    self.assertEqual(acutal_array, np.array(actual_scalar))


if __name__ == '__main__':
  absltest.main()
