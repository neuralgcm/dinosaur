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
import functools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import horizontal_interpolation
from dinosaur import layer_coordinates
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import shallow_water
from dinosaur import shallow_water_states
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import vertical_interpolation
from dinosaur import xarray_utils
import jax
import numpy as np
import pandas as pd
import xarray


def seed_stream(s):
  k = jax.random.PRNGKey(s)
  while True:
    k, l = jax.random.split(k)
    yield l


def shape_structure(inputs):
  return jax.tree.map(lambda x: x.shape, inputs)


class XarrayUtilsTest(parameterized.TestCase):

  def _check_attrs(self, dataset, grid_attrs, custom_attrs):
    """Helper function that asserts that dataset has expected attrs."""
    for attr in grid_attrs:
      self.assertTrue(hasattr(dataset, attr))
    if custom_attrs is not None:
      for attr in custom_attrs.keys():
        self.assertTrue(hasattr(dataset, attr))

  def assert_values_shape_equal(self, x: Any, y: Any):
    if not isinstance(x, dict):
      x = x.asdict()
    if not isinstance(y, dict):
      y = y.asdict()
    self.assertDictEqual(shape_structure(x), shape_structure(y))

  @parameterized.parameters(
      dict(
          samples=2,
          time_steps=3,
          dt=0.6,
          layers=8,
          wavenumbers=32,
          attrs=dict(g=9.80616),
      ),
      dict(
          samples=4,
          time_steps=2,
          dt=300.0,
          layers=2,
          wavenumbers=32,
          attrs=dict(a=9.80616, mean=10.5),
      ),
  )
  def test_primitive_eq_data_to_xarray(
      self, samples, time_steps, dt, layers, wavenumbers, attrs
  ):
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, radius=physics_specs.radius
    )
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)

    # creating state, trajectory and batch of trajectories to be converted.
    initial_state_fn, _ = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    state = initial_state_fn()
    state.tracers = {
        'test_tracer': primitive_equations_states.gaussian_scalar(
            coords, physics_specs, amplitude=0.1
        )
    }
    trajectory = jax.tree.map(
        lambda *args: np.stack(args), *([state] * time_steps)
    )
    batch_of_trajectories = jax.tree.map(
        lambda *args: np.stack(args), *([trajectory] * samples)
    )
    expected_grid_attrs = coords.asdict()

    with self.subTest('state_to_dataset_modal'):
      expected_coords_sizes = {
          xarray_utils.XR_LON_MODE_NAME: grid.longitude_wavenumbers * 2 - 1,
          xarray_utils.XR_LAT_MODE_NAME: grid.total_wavenumbers,
          xarray_utils.XR_LEVEL_NAME: vertical_grid.layers,
          xarray_utils.XR_SURFACE_NAME: 1,
      }
      ds = xarray_utils.data_to_xarray(
          state.asdict(),
          sample_ids=None,
          times=None,
          coords=coords,
          attrs=attrs,
      )
      self._check_attrs(ds, expected_grid_attrs, attrs)
      self.assertDictEqual(dict(ds.sizes), expected_coords_sizes)
      reconstruction = xarray_utils.xarray_to_primitive_eq_data(
          ds, tracers_to_include=('test_tracer',)
      )
      self.assert_values_shape_equal(state, reconstruction)

    with self.subTest('trajectory_to_dataset_modal'):
      expected_coords_sizes = {
          xarray_utils.XR_LON_MODE_NAME: grid.longitude_wavenumbers * 2 - 1,
          xarray_utils.XR_LAT_MODE_NAME: grid.total_wavenumbers,
          xarray_utils.XR_LEVEL_NAME: vertical_grid.layers,
          xarray_utils.XR_SURFACE_NAME: 1,
          xarray_utils.XR_TIME_NAME: time_steps,
      }
      times = dt * np.arange(time_steps)
      ds = xarray_utils.data_to_xarray(
          trajectory.asdict(),
          sample_ids=None,
          times=times,
          coords=coords,
          attrs=attrs,
      )
      self._check_attrs(ds, expected_grid_attrs, attrs)
      self.assertDictEqual(dict(ds.sizes), expected_coords_sizes)
      reconstruction = xarray_utils.xarray_to_primitive_eq_data(
          ds, tracers_to_include=('test_tracer',)
      )
      self.assert_values_shape_equal(trajectory, reconstruction)

    with self.subTest('trajectory_to_dataset_modal'):
      expected_coords_sizes = {
          xarray_utils.XR_LON_MODE_NAME: grid.longitude_wavenumbers * 2 - 1,
          xarray_utils.XR_LAT_MODE_NAME: grid.total_wavenumbers,
          xarray_utils.XR_LEVEL_NAME: vertical_grid.layers,
          xarray_utils.XR_SURFACE_NAME: 1,
          xarray_utils.XR_TIME_NAME: time_steps,
          xarray_utils.XR_SAMPLE_NAME: samples,
      }
      times = dt * np.arange(time_steps)
      sample_ids = np.arange(samples)
      ds = xarray_utils.data_to_xarray(
          batch_of_trajectories.asdict(),
          sample_ids=sample_ids,
          times=times,
          coords=coords,
          attrs=attrs,
      )
      self._check_attrs(ds, expected_grid_attrs, attrs)
      self.assertDictEqual(dict(ds.sizes), expected_coords_sizes)
      reconstruction = xarray_utils.xarray_to_primitive_eq_data(
          ds, tracers_to_include=('test_tracer',)
      )
      self.assert_values_shape_equal(batch_of_trajectories, reconstruction)

    with self.subTest('trajectory_to_dataset_nodal'):
      expected_coords_sizes = {
          xarray_utils.XR_LON_NAME: grid.longitude_nodes,
          xarray_utils.XR_LAT_NAME: grid.latitude_nodes,
          xarray_utils.XR_LEVEL_NAME: vertical_grid.layers,
          xarray_utils.XR_SURFACE_NAME: 1,
          xarray_utils.XR_TIME_NAME: time_steps,
          xarray_utils.XR_SAMPLE_NAME: samples,
      }
      times = dt * np.arange(time_steps)
      sample_ids = np.arange(samples)
      nodal_batch_of_trajectories = jax.tree.map(
          grid.to_nodal, batch_of_trajectories
      )
      ds = xarray_utils.data_to_xarray(
          nodal_batch_of_trajectories.asdict(),
          sample_ids=sample_ids,
          times=times,
          coords=coords,
          attrs=attrs,
      )
      self._check_attrs(ds, expected_grid_attrs, attrs)
      self.assertDictEqual(dict(ds.sizes), expected_coords_sizes)
      reconstruction = xarray_utils.xarray_to_primitive_eq_data(
          ds, tracers_to_include=('test_tracer',)
      )
      self.assert_values_shape_equal(
          nodal_batch_of_trajectories, reconstruction
      )

  @parameterized.parameters(
      dict(
          samples=2,
          time_steps=3,
          dt=0.6,
          layers=8,
          wavenumbers=32,
          attrs=dict(g=9.80616),
      ),
      dict(
          samples=4,
          time_steps=2,
          dt=300.0,
          layers=2,
          wavenumbers=32,
          attrs=dict(a=9.80616, mean=10.5),
      ),
  )
  def test_shallow_water_eq_data_to_xarray(
      self, samples, time_steps, dt, layers, wavenumbers, attrs
  ):
    physics_specs = shallow_water.ShallowWaterSpecs.from_si(
        np.ones((layers,)) * scales.WATER_DENSITY
    )
    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, radius=physics_specs.radius
    )
    vertical_grid = layer_coordinates.LayerCoordinates(layers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)

    # creating state, trajectory and batch of trajectories to be converted.
    _, sin_lat = grid.nodal_mesh
    lat = np.arcsin(sin_lat)
    velocity_function = lambda lat: np.cos(lat) ** 2 / 5
    velocity = np.stack([velocity_function(lat)] * layers)
    state = shallow_water_states.multi_layer(
        velocity, physics_specs.densities, coords
    )

    trajectory = jax.tree.map(
        lambda *args: np.stack(args), *([state] * time_steps)
    )
    batch_of_trajectories = jax.tree.map(
        lambda *args: np.stack(args), *([trajectory] * samples)
    )
    expected_grid_attrs = coords.asdict()

    with self.subTest('state_to_dataset_modal'):
      expected_coords_sizes = {
          xarray_utils.XR_LON_MODE_NAME: grid.longitude_wavenumbers * 2 - 1,
          xarray_utils.XR_LAT_MODE_NAME: grid.total_wavenumbers,
          xarray_utils.XR_LEVEL_NAME: vertical_grid.layers,
      }
      ds = xarray_utils.data_to_xarray(
          state.asdict(),
          sample_ids=None,
          times=None,
          coords=coords,
          attrs=attrs,
      )
      self._check_attrs(ds, expected_grid_attrs, attrs)
      self.assertDictEqual(dict(ds.sizes), expected_coords_sizes)
      reconstruction = xarray_utils.xarray_to_shallow_water_eq_data(ds)
      self.assert_values_shape_equal(state, reconstruction)

    with self.subTest('trajectory_to_dataset_modal'):
      expected_coords_sizes = {
          xarray_utils.XR_LON_MODE_NAME: grid.longitude_wavenumbers * 2 - 1,
          xarray_utils.XR_LAT_MODE_NAME: grid.total_wavenumbers,
          xarray_utils.XR_LEVEL_NAME: vertical_grid.layers,
          xarray_utils.XR_TIME_NAME: time_steps,
      }
      times = dt * np.arange(time_steps)
      ds = xarray_utils.data_to_xarray(
          trajectory.asdict(),
          sample_ids=None,
          times=times,
          coords=coords,
          attrs=attrs,
      )
      self._check_attrs(ds, expected_grid_attrs, attrs)
      self.assertDictEqual(dict(ds.sizes), expected_coords_sizes)
      reconstruction = xarray_utils.xarray_to_shallow_water_eq_data(ds)
      self.assert_values_shape_equal(trajectory, reconstruction)

    with self.subTest('trajectory_to_dataset_modal'):
      expected_coords_sizes = {
          xarray_utils.XR_LON_MODE_NAME: grid.longitude_wavenumbers * 2 - 1,
          xarray_utils.XR_LAT_MODE_NAME: grid.total_wavenumbers,
          xarray_utils.XR_LEVEL_NAME: vertical_grid.layers,
          xarray_utils.XR_TIME_NAME: time_steps,
          xarray_utils.XR_SAMPLE_NAME: samples,
      }
      times = dt * np.arange(time_steps)
      sample_ids = np.arange(samples)
      ds = xarray_utils.data_to_xarray(
          batch_of_trajectories.asdict(),
          sample_ids=sample_ids,
          times=times,
          coords=coords,
          attrs=attrs,
      )
      self._check_attrs(ds, expected_grid_attrs, attrs)
      self.assertDictEqual(dict(ds.sizes), expected_coords_sizes)
      reconstruction = xarray_utils.xarray_to_shallow_water_eq_data(ds)
      self.assert_values_shape_equal(batch_of_trajectories, reconstruction)

    with self.subTest('trajectory_to_dataset_nodal'):
      expected_coords_sizes = {
          xarray_utils.XR_LON_NAME: grid.longitude_nodes,
          xarray_utils.XR_LAT_NAME: grid.latitude_nodes,
          xarray_utils.XR_LEVEL_NAME: vertical_grid.layers,
          xarray_utils.XR_TIME_NAME: time_steps,
          xarray_utils.XR_SAMPLE_NAME: samples,
      }
      times = dt * np.arange(time_steps)
      sample_ids = np.arange(samples)
      nodal_batch_of_trajectories = jax.tree.map(
          grid.to_nodal, batch_of_trajectories
      )
      ds = xarray_utils.data_to_xarray(
          nodal_batch_of_trajectories.asdict(),
          sample_ids=sample_ids,
          times=times,
          coords=coords,
          attrs=attrs,
      )
      self._check_attrs(ds, expected_grid_attrs, attrs)
      self.assertDictEqual(dict(ds.sizes), expected_coords_sizes)
      reconstruction = xarray_utils.xarray_to_shallow_water_eq_data(ds)
      self.assert_values_shape_equal(
          nodal_batch_of_trajectories, reconstruction
      )

  @parameterized.parameters(
      dict(integrator=time_integration.crank_nicolson_rk2),
      dict(integrator=time_integration.imex_rk_sil3),
      dict(integrator=time_integration.backward_forward_euler),
  )
  def test_xarray_to_primitive_equations_with_time_data(self, integrator):
    wavenumbers = 21
    layers = 6
    inner_steps = 3
    outer_steps = 12
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, radius=physics_specs.radius
    )
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    dt = physics_specs.nondimensionalize(10 * scales.units.minute)

    # creating state, trajectory and batch of trajectories to be converted.
    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    ref_temperatures = aux_features[xarray_utils.REF_TEMP_KEY]
    nodal_orography = aux_features[xarray_utils.OROGRAPHY]
    orography = primitive_equations.truncated_modal_orography(
        nodal_orography, coords
    )
    state = initial_state_fn()
    state.sim_time = 0.0
    equation = primitive_equations.PrimitiveEquations(
        ref_temperatures, orography, coords, physics_specs
    )
    step_fn = integrator(equation, dt)
    trajectory_fn = jax.jit(
        time_integration.trajectory_from_step(
            step_fn, outer_steps, inner_steps, start_with_input=True
        )
    )
    _, state_trajectory = trajectory_fn(state)
    times = dt * inner_steps * np.arange(outer_steps)
    ds = xarray_utils.data_to_xarray(
        state_trajectory.asdict(), sample_ids=None, times=times, coords=coords
    )
    reconstructed = xarray_utils.xarray_to_primitive_equations_with_time_data(
        ds
    )
    with self.subTest('round_trip'):
      for actual, expected in zip(
          jax.tree_util.tree_leaves(reconstructed),
          jax.tree_util.tree_leaves(state_trajectory.asdict()),
      ):
        np.testing.assert_allclose(actual, expected)
    with self.subTest('simulation_time'):
      np.testing.assert_allclose(ds.time.values, ds.sim_time.values, atol=1e-6)

  def test_xarray_to_data_with_renaming(self):
    wavenumbers = 21
    layers = 6
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, radius=physics_specs.radius
    )
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    # creating state to be converted.
    initial_state_fn, _ = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    state = initial_state_fn().asdict()
    state['tracers']['specific_humidity'] = state['divergence']  # mock tracer.
    renaming_dict = {  # dataset names: primitive equation state notation.
        'delta_renamed': 'divergence',
        'nu_renamed': 'vorticity',
        'dt_renamed': 'temperature_variation',
        'lsp_value': 'log_surface_pressure',
        'q_stuff': 'specific_humidity',
    }
    base_to_xarray_fn = xarray_utils.data_to_xarray
    base_from_xarray_fn = functools.partial(
        xarray_utils.xarray_to_primitive_eq_data,
        tracers_to_include=('specific_humidity',),
    )
    ds = xarray_utils.data_to_xarray_with_renaming(
        state,
        to_xarray_fn=base_to_xarray_fn,
        renaming_dict=renaming_dict,
        coords=coords,
        sample_ids=None,
        times=None,
    )
    reconstructed = xarray_utils.xarray_to_data_with_renaming(
        ds, xarray_to_data_fn=base_from_xarray_fn, renaming_dict=renaming_dict
    )
    with self.subTest('dataset_names'):
      self.assertSameElements(ds.keys(), renaming_dict.keys())
    with self.subTest('round_trip'):
      for actual, expected in zip(
          jax.tree_util.tree_leaves(reconstructed),
          jax.tree_util.tree_leaves(state),
      ):
        np.testing.assert_allclose(actual, expected)

  @parameterized.parameters(
      dict(
          coords=coordinate_systems.CoordinateSystem(
              spherical_harmonic.Grid.T21(),
              vertical_interpolation.PressureCoordinates([50, 100, 800]),
          ),
          grid_type='CUBIC',
      ),
      dict(
          coords=coordinate_systems.CoordinateSystem(
              spherical_harmonic.Grid.T42(),
              vertical_interpolation.PressureCoordinates([1, 30, 50, 900]),
          ),
          grid_type='CUBIC',
      ),
      dict(
          coords=coordinate_systems.CoordinateSystem(
              spherical_harmonic.Grid.T21(latitude_spacing='equiangular'),
              vertical_interpolation.PressureCoordinates([10, 1000, 2000]),
          ),
          grid_type='CUBIC',
      ),
      dict(
          coords=coordinate_systems.CoordinateSystem(
              spherical_harmonic.Grid.T85(
                  latitude_spacing='equiangular_with_poles'
              ),
              vertical_interpolation.PressureCoordinates(
                  [10, 50, 500, 600, 900]
              ),
          ),
          grid_type='CUBIC',
      ),
      dict(
          coords=coordinate_systems.CoordinateSystem(
              spherical_harmonic.Grid.TL127(
                  latitude_spacing='equiangular_with_poles'
              ),
              vertical_interpolation.PressureCoordinates(
                  [10, 50, 500, 600, 900]
              ),
          ),
          grid_type='LINEAR',
      ),
      dict(
          coords=coordinate_systems.CoordinateSystem(
              spherical_harmonic.Grid.TL31(),
              vertical_interpolation.PressureCoordinates(
                  [10, 50, 500, 600, 900]
              ),
          ),
          grid_type='LINEAR',
      ),
  )
  def test_coordinate_system_from_dataset(self, coords, grid_type):
    data_array = np.zeros(coords.nodal_shape)
    data_template = {'u': 0, 'v': 0, 'z': 0, 't': 0, 'tracers': {}}
    wb_data = jax.tree_util.tree_map(lambda x: data_array, data_template)
    ds = xarray_utils.data_to_xarray(wb_data, coords=coords, times=None)
    with self.subTest('using_metadata'):
      reconstructed = xarray_utils.coordinate_system_from_dataset(ds)
      self.assertEqual(reconstructed, coords)

    with self.subTest('using_xarray_axes'):
      ds.attrs = {}  # remove metadata.
      reconstructed = xarray_utils.coordinate_system_from_dataset(ds, grid_type)
      self.assertEqual(reconstructed, coords)

  def test_xarray_to_state_and_dynamic_covariate_data(self):
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        vertical_interpolation.PressureCoordinates([50, 100, 800]),
    )
    times = np.datetime64('1990-01-01') + np.arange(5) * np.timedelta64(1, 'h')

    # Construct xarray dataset from mock state data
    data_array = np.zeros((len(times),) + coords.nodal_shape)
    data_template = {'u': 0, 'v': 0, 'z': 0, 't': 0, 'tracers': {}}
    data = jax.tree_util.tree_map(lambda x: data_array, data_template)
    ds_state = xarray_utils.data_to_xarray(data, coords=coords, times=times)

    # Construct xarray dataset from mock surface covariates
    data_array = np.zeros((len(times),) + coords.surface_nodal_shape)
    data_template = {'sea_ice_cover': 0, 'sea_surface_temperature': 0}
    data = jax.tree_util.tree_map(lambda x: data_array, data_template)
    data['sim_time'] = np.arange(5)  # expect nondim time in prepreocessed data
    ds_surface_covariates = xarray_utils.dynamic_covariate_data_to_xarray(
        data, coords=coords, times=times
    )

    # Construct xarray dataset from mock volume covariates
    data_array = np.zeros((len(times),) + coords.nodal_shape)
    data_template = {'cloud_cover': 0}
    data = jax.tree_util.tree_map(lambda x: data_array, data_template)
    data['sim_time'] = np.arange(5)  # expect nondim time in prepreocessed data
    ds_volume_covariates = xarray_utils.dynamic_covariate_data_to_xarray(
        data, coords=coords, times=times
    )

    ds = xarray.merge([ds_state, ds_surface_covariates, ds_volume_covariates])
    xarray_to_dynamic_covariate_data_fn = functools.partial(
        xarray_utils.xarray_to_dynamic_covariate_data,
        covariates_to_include=['sea_ice_cover', 'cloud_cover'],
    )
    state_data, covariate_data = (
        xarray_utils.xarray_to_state_and_dynamic_covariate_data(
            ds,
            xarray_to_state_data_fn=xarray_utils.xarray_to_weatherbench_data,
            xarray_to_dynamic_covariate_data_fn=(
                xarray_to_dynamic_covariate_data_fn
            ),
        )
    )

    with self.subTest('state data'):
      expected_keys = {'u', 'v', 'z', 't', 'tracers', 'diagnostics', 'sim_time'}
      self.assertSetEqual(set(state_data.keys()), expected_keys)
      sim_time = state_data.pop('sim_time')
      _ = state_data.pop('tracers')
      _ = state_data.pop('diagnostics')
      self.assertEqual(sim_time.shape, (5,))
      for var in state_data.values():
        self.assertEqual(var.shape, (5, 3, 64, 32))

    with self.subTest('covariate data'):
      expected_keys = {'cloud_cover', 'sea_ice_cover', 'sim_time'}
      self.assertSetEqual(set(covariate_data.keys()), expected_keys)
      self.assertEqual(covariate_data['cloud_cover'].shape, (5, 3, 64, 32))
      self.assertEqual(covariate_data['sea_ice_cover'].shape, (5, 1, 64, 32))
      self.assertEqual(covariate_data['sim_time'].shape, (5,))

  @parameterized.parameters(
      dict(time_shift_type='str'),
      dict(time_shift_type='np'),
      dict(time_shift_type='pd'),
  )
  def test_selective_temporal_shift(self, time_shift_type):
    time = np.datetime64('2000-01-01') + [
        np.timedelta64(x, 'h') for x in range(5)
    ]
    input_ds = xarray.Dataset(
        data_vars=dict(
            a=(['time'], [40, 41, 42, 43, 44]),
            b=(['time'], [50, 51, 52, 53, 54]),
            c=(['time'], [60.0, 61.0, 62.0, 63.0, 64.0]),
        ),
        coords=dict(time=time),
    )

    with self.subTest('zero shift'):
      match time_shift_type:
        case 'str':
          time_shift = '0 hours'
        case 'np':
          time_shift = np.timedelta64(0, 'h')
        case 'pd':
          time_shift = pd.Timedelta(0, 'h')
      shifted_ds = xarray_utils.selective_temporal_shift(
          input_ds, variables=['b', 'c'], time_shift=time_shift
      )
      xarray.testing.assert_equal(shifted_ds, input_ds)

    with self.subTest('empty variables list'):
      match time_shift_type:
        case 'str':
          time_shift = '1 hours'
        case 'np':
          time_shift = np.timedelta64(1, 'h')
        case 'pd':
          time_shift = pd.Timedelta(1, 'h')
      shifted_ds = xarray_utils.selective_temporal_shift(
          input_ds, variables=[], time_shift=time_shift
      )
      xarray.testing.assert_equal(shifted_ds, input_ds)

    with self.subTest('positive shift'):
      # Shifting ['b', 'c'] by 1 hour relative to the dataset implies that for
      # a given time they have their value from 1 hour earlier.
      match time_shift_type:
        case 'str':
          time_shift = '1 hour'
        case 'np':
          time_shift = np.timedelta64(1, 'h')
        case 'pd':
          time_shift = pd.Timedelta(1, 'h')
      shifted_ds = xarray_utils.selective_temporal_shift(
          input_ds, variables=['b', 'c'], time_shift=time_shift
      )
      expected_shifted_ds = xarray.Dataset(
          data_vars=dict(
              a=(['time'], [41, 42, 43, 44]),  # unshifted
              b=(['time'], [50, 51, 52, 53]),  # shifted right
              c=(['time'], [60.0, 61.0, 62.0, 63.0]),  # shifted right
          ),
          coords=dict(time=time[1:]),  # remove first entry
      )
      xarray.testing.assert_equal(shifted_ds, expected_shifted_ds)

    with self.subTest('negative shift'):
      # Shifting ['b', 'c'] by -2 hours relative to the dataset implies that for
      # a given time they have their value from 2 hours later.
      match time_shift_type:
        case 'str':
          time_shift = '-2 hours'
        case 'np':
          time_shift = np.timedelta64(-2, 'h')
        case 'pd':
          time_shift = pd.Timedelta(-2, 'h')
      shifted_ds = xarray_utils.selective_temporal_shift(
          input_ds, variables=['b', 'c'], time_shift=time_shift
      )
      expected_shifted_ds = xarray.Dataset(
          data_vars=dict(
              a=(['time'], [40, 41, 42]),  # unshifted
              b=(['time'], [52, 53, 54]),  # shifted left
              c=(['time'], [62.0, 63.0, 64.0]),  # shifted left
          ),
          coords=dict(time=time[:-2]),  # removed last two entries
      )
      xarray.testing.assert_equal(shifted_ds, expected_shifted_ds)

  def test_fill_nan_nearest(self):

    times = np.arange(2)
    rng = np.random.default_rng(seed=0)
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        vertical_interpolation.PressureCoordinates([50, 100, 800]),
    )

    def to_xarray(data_dict):
      return xarray_utils.data_to_xarray(
          data_dict, coords=coords, times=times
      ).rename({'lon': 'longitude', 'lat': 'latitude'})

    with_valid_nan = rng.random(times.shape + coords.horizontal.nodal_shape)
    with_valid_nan[:, 10:15, 3] = np.nan  # longitude nan slice.
    data_dict = {
        'u': rng.random(times.shape + coords.nodal_shape),
        'v': rng.random(times.shape + coords.nodal_shape),
        'with_nan': with_valid_nan,
    }
    ds_with_nan = to_xarray(data_dict)
    filled_ds = xarray_utils.fill_nan_with_nearest(ds_with_nan)

    with self.subTest('correct_along_longitude'):
      # longitude neighbors are closest on the outside of the nan slice.
      np.testing.assert_allclose(
          filled_ds.with_nan.values[:, 10, 3], with_valid_nan[:, 9, 3]
      )
      np.testing.assert_allclose(
          filled_ds.with_nan.values[:, 14, 3], with_valid_nan[:, 15, 3]
      )

    with self.subTest('correct_along_latitude'):
      # in the center upper latitude neighbor is closest.
      np.testing.assert_allclose(
          filled_ds.with_nan.values[:, 12, 3], with_valid_nan[:, 12, 2]
      )

    with self.subTest('extra_scalar_variable_okay'):
      filled_ds_with_extra = xarray_utils.fill_nan_with_nearest(
          ds_with_nan.assign(sim_time=1.0)
      )
      expected = filled_ds.assign(sim_time=1.0)
      xarray.testing.assert_identical(filled_ds_with_extra, expected)

    with self.subTest('allows_no_time_dimension'):
      filled_ds_no_time = xarray_utils.fill_nan_with_nearest(
          ds_with_nan.isel(time=0)
      )
      xarray.testing.assert_identical(filled_ds_no_time, filled_ds.isel(time=0))

    with self.subTest('raises_on_missing_lat_lon'):
      with self.assertRaisesRegex(
          ValueError, 'did not find latitude and longitude dimensions'
      ):
        xarray_utils.fill_nan_with_nearest(
            ds_with_nan.rename({'latitude': 'lat'})
        )

    with self.subTest('raises_on_non_stationary_mask'):
      with_invalid_nan = rng.random(times.shape + coords.horizontal.nodal_shape)
      with_invalid_nan[:, 10, 3] = np.array([0, np.nan])
      data_dict = {
          'u': rng.random(times.shape + coords.nodal_shape),
          'v': rng.random(times.shape + coords.nodal_shape),
          'with_nan': with_invalid_nan,
      }
      ds_with_invalid_nan = to_xarray(data_dict)
      with self.assertRaisesRegex(ValueError, 'NaN mask is not fixed'):
        xarray_utils.fill_nan_with_nearest(ds_with_invalid_nan)

    with self.subTest('raises_on_all_nan'):
      data_dict = {
          'u': rng.random(times.shape + coords.nodal_shape),
          'v': rng.random(times.shape + coords.nodal_shape),
          'with_nan': with_invalid_nan * np.nan,
      }
      ds_with_invalid_nan = to_xarray(data_dict)
      with self.assertRaisesRegex(ValueError, 'all values are NaN'):
        xarray_utils.fill_nan_with_nearest(ds_with_invalid_nan)

    with self.subTest('works_on_dataarray'):
      filled_u = xarray_utils.fill_nan_with_nearest(ds_with_nan.u)
      xarray.testing.assert_identical(filled_u, filled_ds.u)

  @parameterized.parameters(
      dict(regridder_cls=horizontal_interpolation.BilinearRegridder),
      dict(regridder_cls=horizontal_interpolation.ConservativeRegridder),
      dict(regridder_cls=horizontal_interpolation.NearestRegridder),
  )
  def test_regrid_horizontal(self, regridder_cls):
    old_coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.TL31(),
        vertical_interpolation.PressureCoordinates([50, 100, 150]),
    )
    rng = np.random.default_rng(seed=0)
    data_dict = {
        'u': rng.random(old_coords.nodal_shape),
        'v': rng.random(old_coords.surface_nodal_shape),
    }
    ds = (
        xarray_utils.data_to_xarray(data_dict, coords=old_coords, times=None)
        .rename({'lon': 'longitude', 'lat': 'latitude'})
        .squeeze('surface')
    )

    regridder = regridder_cls(
        old_coords.horizontal, spherical_harmonic.Grid.TL63()
    )
    ds_regridded = xarray_utils.regrid_horizontal(ds, regridder)
    expected_sizes = {'latitude': 64, 'longitude': 128, 'level': 3}
    self.assertEqual(ds_regridded.sizes, expected_sizes)

    u_regridded = xarray_utils.regrid_horizontal(ds.u, regridder)
    xarray.testing.assert_identical(u_regridded, ds_regridded.u)

    ds_flipped = ds.isel(latitude=slice(None, None, -1))
    ds_flipped_regridded = xarray_utils.regrid_horizontal(ds_flipped, regridder)
    xarray.testing.assert_identical(ds_flipped_regridded, ds_regridded)

    with self.assertRaisesRegex(ValueError, 'inconsistent latitude'):
      xarray_utils.regrid_horizontal(
          ds.assign(latitude=ds.latitude - 180), regridder
      )

  @parameterized.parameters(
      dict(regridder_cls=vertical_interpolation.BilinearRegridder),
      dict(regridder_cls=vertical_interpolation.ConservativeRegridder),
  )
  def test_regrid_vertical(self, regridder_cls):
    old_coords = vertical_interpolation.HybridCoordinates(
        a_boundaries=np.array([0, 400, 300, 0]),
        b_boundaries=np.array([0, 0, 0.5, 1.0]),
    )
    new_coords = sigma_coordinates.SigmaCoordinates.equidistant(4)

    with self.subTest('single time slice'):
      surface_pressure = xarray.DataArray(
          1000 * np.ones((10, 12)), dims=['longitude', 'latitude']
      )
      ds = xarray.Dataset(
          {'u': (('z', 'longitude', 'latitude'), np.ones((3, 10, 12)))}
      )

      regridder = regridder_cls(old_coords, new_coords)
      ds_regridded = xarray_utils.regrid_vertical(
          ds, surface_pressure, regridder, in_dim='z', out_dim='sigma'
      )

      expected_sizes = {'sigma': 4, 'longitude': 10, 'latitude': 12}
      self.assertEqual(ds_regridded.sizes, expected_sizes)

      expected_sigma = np.array([0.125, 0.375, 0.625, 0.875])
      np.testing.assert_array_equal(ds_regridded['sigma'].data, expected_sigma)

    with self.subTest('with leading time dimension'):
      surface_pressure = xarray.DataArray(
          1000 * np.ones((2, 10, 12)), dims=['time', 'longitude', 'latitude']
      )
      ds = xarray.Dataset({
          'u': (('time', 'z', 'longitude', 'latitude'), np.ones((2, 3, 10, 12)))
      })
      regridder = regridder_cls(old_coords, new_coords)
      ds_regridded = xarray_utils.regrid_vertical(
          ds, surface_pressure, regridder, in_dim='z', out_dim='sigma'
      )
      expected_sizes = {'time': 2, 'sigma': 4, 'longitude': 10, 'latitude': 12}
      self.assertEqual(ds_regridded.sizes, expected_sizes)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
