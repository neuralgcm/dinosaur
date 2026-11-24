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

"""Tests for coordinate_systems."""

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import coordinate_systems
from dinosaur import layer_coordinates
from dinosaur import shallow_water
from dinosaur import shallow_water_states
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

import jax
import numpy as np


class CoordinateSystemTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          horizontal=spherical_harmonic.Grid.T21(),
          vertical=sigma_coordinates.SigmaCoordinates.equidistant(6)),
      dict(
          horizontal=spherical_harmonic.Grid.T21(),
          vertical=layer_coordinates.LayerCoordinates(5)),
  )
  def test_coordinate_system_shape(self, horizontal, vertical):
    coordinate_system = coordinate_systems.CoordinateSystem(
        horizontal, vertical)
    self.assertTupleEqual(
        coordinate_system.nodal_shape,
        ((vertical.layers,) + horizontal.nodal_shape))
    self.assertTupleEqual(
        coordinate_system.modal_shape,
        ((vertical.layers,) + horizontal.modal_shape))
    self.assertTupleEqual(
        coordinate_system.surface_nodal_shape,
        ((1,) + horizontal.nodal_shape))
    self.assertTupleEqual(
        coordinate_system.surface_modal_shape,
        ((1,) + horizontal.modal_shape))

  @parameterized.parameters(
      dict(
          horizontal=spherical_harmonic.Grid.T21(),
          vertical=sigma_coordinates.SigmaCoordinates.equidistant(8),
      ),
      dict(
          horizontal=spherical_harmonic.Grid.with_wavenumbers(34),
          vertical=sigma_coordinates.SigmaCoordinates.equidistant(8),
      ),
      dict(
          horizontal=spherical_harmonic.Grid.T21(),
          vertical=layer_coordinates.LayerCoordinates(5),
      ),
      dict(
          horizontal=spherical_harmonic.Grid.T21(
              transform_method='least_squares'
          ),
          vertical=layer_coordinates.LayerCoordinates(5),
      ),
  )
  def test_xarray_attrs_roundtrip(self, horizontal, vertical):
    """Tests that `CoordinateSystem` converts to attrs and back."""
    coordinate_system = coordinate_systems.CoordinateSystem(
        horizontal, vertical)
    coord_attrs = coordinate_system.asdict()
    reconstructed = xarray_utils.coordinate_system_from_attrs(coord_attrs)
    actual_leaves, actual_def = jax.tree_util.tree_flatten(
        reconstructed.asdict())
    expected_leaves, expected_def = jax.tree_util.tree_flatten(
        coordinate_system.asdict())
    self.assertEqual(actual_def, expected_def)
    for actual, expected in zip(actual_leaves, expected_leaves):
      if isinstance(actual, np.ndarray):
        np.testing.assert_allclose(actual, expected)
      else:
        self.assertEqual(actual, expected)

  def test_xarray_attrs_backwards_compatibility(self):
    """Tests deserializing legacy attrs missing optional `Grid` fields."""
    coordinate_system = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        layer_coordinates.LayerCoordinates(5),
    )
    legacy_attrs = coordinate_system.asdict()
    legacy_attrs.pop('transform_method')
    legacy_attrs.pop('basis_as_jax_arrays')
    reconstructed = xarray_utils.coordinate_system_from_attrs(legacy_attrs)
    self.assertEqual(reconstructed, coordinate_system)

  @parameterized.parameters(
      dict(wavenumbers=128,
           save_wavenumbers=120,
           vertical_coordinate=layer_coordinates.LayerCoordinates(1),
           physics_specs=shallow_water.ShallowWaterSpecs.from_si(),
           initial_conditions=shallow_water_states.barotropic_instability_tc,
           should_raise_on_downsample=False,
           should_raise_on_upsample=True,
           atol=1e-4),
      dict(wavenumbers=64,
           save_wavenumbers=32,
           vertical_coordinate=layer_coordinates.LayerCoordinates(1),
           physics_specs=shallow_water.ShallowWaterSpecs.from_si(),
           initial_conditions=shallow_water_states.barotropic_instability_tc,
           should_raise_on_downsample=False,
           should_raise_on_upsample=True,
           atol=2e-2),
      dict(wavenumbers=64,
           save_wavenumbers=16,
           vertical_coordinate=layer_coordinates.LayerCoordinates(2),
           physics_specs=shallow_water.ShallowWaterSpecs.from_si(),
           initial_conditions=shallow_water_states.barotropic_instability_tc,
           should_raise_on_downsample=False,
           should_raise_on_upsample=True,
           atol=0.15),
      dict(wavenumbers=32,
           save_wavenumbers=64,
           vertical_coordinate=layer_coordinates.LayerCoordinates(1),
           physics_specs=shallow_water.ShallowWaterSpecs.from_si(),
           initial_conditions=shallow_water_states.barotropic_instability_tc,
           should_raise_on_downsample=True,
           should_raise_on_upsample=False,
           atol=4e-2),
      dict(wavenumbers=80,
           save_wavenumbers=90,
           vertical_coordinate=layer_coordinates.LayerCoordinates(1),
           physics_specs=shallow_water.ShallowWaterSpecs.from_si(),
           initial_conditions=shallow_water_states.barotropic_instability_tc,
           should_raise_on_downsample=True,
           should_raise_on_upsample=False,
           atol=2e-4),
      dict(wavenumbers=60,
           save_wavenumbers=60,
           vertical_coordinate=layer_coordinates.LayerCoordinates(1),
           physics_specs=shallow_water.ShallowWaterSpecs.from_si(),
           initial_conditions=shallow_water_states.barotropic_instability_tc,
           should_raise_on_downsample=False,
           should_raise_on_upsample=False,
           atol=0.0),
  )
  def test_spectral_intrpolate_fn(
      self,
      wavenumbers,
      save_wavenumbers,
      vertical_coordinate,
      physics_specs,
      initial_conditions,
      should_raise_on_downsample,
      should_raise_on_upsample,
      atol,
  ):
    """Tests that spectral interpolation works as expected."""
    grid = spherical_harmonic.Grid.with_wavenumbers(wavenumbers)
    save_grid = spherical_harmonic.Grid.with_wavenumbers(save_wavenumbers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_coordinate)
    save_coords = coordinate_systems.CoordinateSystem(
        save_grid, vertical_coordinate)

    rng_key = jax.random.PRNGKey(42)
    input_state_generator_fn, _ = initial_conditions(
        coords=coords, physics_specs=physics_specs)
    target_state_generator_fn, _ = initial_conditions(
        coords=save_coords, physics_specs=physics_specs)
    input_state = input_state_generator_fn(rng_key)
    expected_state = target_state_generator_fn(rng_key)

    if should_raise_on_downsample:
      with self.assertRaisesRegex(ValueError, 'save_coords.horizontal .*'):
        _ = coordinate_systems.get_spectral_downsample_fn(coords, save_coords)
    if should_raise_on_upsample:
      with self.assertRaisesRegex(ValueError, 'save_coords.horizontal .*'):
        _ = coordinate_systems.get_spectral_upsample_fn(coords, save_coords)
    interpolate_fn = coordinate_systems.get_spectral_interpolate_fn(
        coords, save_coords)
    actual = interpolate_fn(input_state)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected_state)):
      np.testing.assert_allclose(
          save_coords.horizontal.to_nodal(x),
          save_coords.horizontal.to_nodal(y), atol=atol)

  @parameterized.parameters(
      dict(input_representations=('nodal', 'modal', 'modal', 't')),
      dict(input_representations=('t', 'modal', 'nodal', 'modal')),
  )
  def test_maybe_to_nodal_shapes(self, input_representations):
    """Tests that maybe_to_nodal produces expected outputs."""
    grid = spherical_harmonic.Grid.T21()
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(6)
    coords = coordinate_systems.CoordinateSystem(grid, vertical)
    input_pytree = {}
    expected_output = {}
    modal_data = jax.random.uniform(jax.random.PRNGKey(4), grid.modal_shape)
    data = {
        'modal': modal_data,
        'nodal': grid.to_nodal(modal_data),
        't': np.asarray(1.34),
    }
    for i, representation in enumerate(input_representations):
      input_pytree[i] = data[representation]
      expected_output[i] = data['t'] if representation == 't' else data['nodal']
    actual = coordinate_systems.maybe_to_nodal(input_pytree, coords)
    for x, y in zip(jax.tree_util.tree_leaves(actual),
                    jax.tree_util.tree_leaves(expected_output)):
      np.testing.assert_allclose(x, y)

  @parameterized.parameters(
      dict(input_representations=('nodal', 'modal', 'modal', 't')),
      dict(input_representations=('t', 'modal', 'nodal', 'modal')),
  )
  def test_maybe_to_modal_shapes(self, input_representations):
    """Tests that maybe_to_modal produces expected outputs."""
    grid = spherical_harmonic.Grid.T21()
    vertical = sigma_coordinates.SigmaCoordinates.equidistant(6)
    coords = coordinate_systems.CoordinateSystem(grid, vertical)
    input_pytree = {}
    expected_output = {}
    modal_data = jax.random.uniform(jax.random.PRNGKey(4), grid.modal_shape)
    nodal_data = grid.to_nodal(modal_data),
    data = {
        'modal': grid.to_modal(nodal_data),
        'nodal': nodal_data,
        't': np.asarray(1.34),
    }
    for i, representation in enumerate(input_representations):
      input_pytree[i] = data[representation]
      expected_output[i] = data['t'] if representation == 't' else data['modal']
    actual = coordinate_systems.maybe_to_modal(input_pytree, coords)
    for x, y in zip(jax.tree_util.tree_leaves(actual),
                    jax.tree_util.tree_leaves(expected_output)):
      np.testing.assert_allclose(x, y)

  def test_scale_levels_for_matching_keys(self):
    """Tests that level scaling works as expected."""
    n_layers = 6
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(n_layers)
    )
    vertical_values = np.arange(n_layers)[:, np.newaxis, np.newaxis]
    inputs = {
        'vorticity': np.ones(coords.nodal_shape) * vertical_values,
        'x': np.ones(coords.nodal_shape) * vertical_values,
        'tracers': {
            'specific_humidity': np.ones(coords.nodal_shape) * vertical_values,
            'y': np.ones(coords.nodal_shape) * vertical_values,
        },
    }
    level_scales = np.asarray([1., 1., 1., 10., 100., 3.])
    keys_to_scale = ['x', 'z', 'specific_humidity']
    actual = coordinate_systems.scale_levels_for_matching_keys(
        inputs, level_scales, keys_to_scale)
    with self.subTest('where_scaling_is_applied'):
      expected_level_scale = np.asarray(level_scales)[:, np.newaxis, np.newaxis]
      np.testing.assert_allclose(
          actual['tracers']['specific_humidity'],
          inputs['tracers']['specific_humidity'] * expected_level_scale)
      np.testing.assert_allclose(
          actual['x'], inputs['x'] * expected_level_scale)
    with self.subTest('where_scaling_is_omitted'):
      np.testing.assert_allclose(actual['vorticity'], inputs['vorticity'])
      np.testing.assert_allclose(actual['tracers']['y'], inputs['tracers']['y'])


if __name__ == '__main__':
  absltest.main()
