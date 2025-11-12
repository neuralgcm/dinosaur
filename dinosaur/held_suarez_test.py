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

"""Tests for held_suarez."""

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import coordinate_systems
from dinosaur import held_suarez
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

import jax
import numpy as np

jax.config.parse_flags_with_absl()


def _subtract_longitudes(a, b):
  """Computes the difference `a - b`, accounting for periodicity."""
  return np.mod(a - b + np.pi, 2 * np.pi) - np.pi


def _gaussian_blob(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    lon_loc: float = np.pi * 120 / 180,
    lat_loc: float = np.pi * 30 / 180,
    lon_scale: float = np.pi * 10 / 180,
    lat_scale: float = np.pi * 10 / 180,
):
  return np.cos(latitudes) * np.exp(
      -(_subtract_longitudes(lon_loc, longitudes) / lon_scale) ** 2
      -((lat_loc - latitudes) / lat_scale) ** 2
  )


def get_test_orography(
    grid: spherical_harmonic.Grid,
    min_height_in_meters: float = -100.0,
    max_height_in_meters: float = 5000.0,
) -> np.ndarray:
  """Returns random orography with values in the same range as data."""
  lons, sin_lats = grid.nodal_mesh
  lats = np.arcsin(sin_lats)
  return (
      min_height_in_meters * _gaussian_blob(lons, lats, lon_loc=0.0) +
      max_height_in_meters * _gaussian_blob(lons, lats, lat_loc=-np.pi / 6))


class HeldSuarezTest(parameterized.TestCase):

  def test_isothermal_rest_atmosphere_no_orography(self):
    units = scales.units
    layers = 26
    coords = coordinate_systems.CoordinateSystem(
        horizontal=spherical_harmonic.Grid.T42(),
        vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers))
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()

    initial_state_fn, aux_features = (
        primitive_equations_states.isothermal_rest_atmosphere(
            coords,
            physics_specs,
            p0=1e5 * units.pascal,
            p1=5e3 * units.pascal,
        ))

    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    orography = aux_features[xarray_utils.OROGRAPHY]
    state = initial_state_fn(rng_key=jax.random.PRNGKey(0))

    with self.subTest('state'):
      np.testing.assert_allclose(state.vorticity, 0)
      np.testing.assert_allclose(state.divergence, 0)
      np.testing.assert_allclose(state.temperature_variation, 0)
      self.assertEqual(state.vorticity.shape, coords.modal_shape)
      self.assertEqual(state.divergence.shape, coords.modal_shape)
      self.assertEqual(state.temperature_variation.shape, coords.modal_shape)
      self.assertEqual(state.log_surface_pressure.shape,
                       coords.surface_modal_shape)

    with self.subTest('surface pressure'):
      surface_pressure = physics_specs.dimensionalize(np.exp(
          coords.horizontal.to_nodal(state.log_surface_pressure)), units.pascal)
      np.testing.assert_array_less(surface_pressure, 102000)
      np.testing.assert_array_less(98000, surface_pressure)

    with self.subTest('orography'):
      np.testing.assert_allclose(orography, 0)
      self.assertEqual(orography.shape, coords.horizontal.nodal_shape)

    with self.subTest('reference temperature'):
      np.testing.assert_allclose(ref_temps, 288)
      self.assertEqual(ref_temps.shape, (coords.vertical.layers,))

  def test_isothermal_rest_atmosphere_with_orography(self):
    units = scales.units
    layers = 26
    coords = coordinate_systems.CoordinateSystem(
        horizontal=spherical_harmonic.Grid.T85(),
        vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers))
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()

    # WeatherBench data is on an equiangular grid.
    orography = get_test_orography(coords.horizontal)

    initial_state_fn, aux_features = (
        primitive_equations_states.isothermal_rest_atmosphere(
            coords,
            physics_specs,
            p0=1e5 * units.pascal,
            p1=5e3 * units.pascal,
            surface_height=orography * units.meter,
        ))

    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    orography = aux_features[xarray_utils.OROGRAPHY]
    state = initial_state_fn(rng_key=jax.random.PRNGKey(0))

    with self.subTest('state'):
      np.testing.assert_allclose(state.vorticity, 0)
      np.testing.assert_allclose(state.divergence, 0)
      np.testing.assert_allclose(state.temperature_variation, 0)
      self.assertEqual(state.vorticity.shape, coords.modal_shape)
      self.assertEqual(state.divergence.shape, coords.modal_shape)
      self.assertEqual(state.temperature_variation.shape, coords.modal_shape)
      self.assertEqual(state.log_surface_pressure.shape,
                       coords.surface_modal_shape)

    with self.subTest('aux_features'):
      for v in aux_features.values():
        self.assertIsInstance(v, np.ndarray)

    with self.subTest('surface pressure'):
      surface_pressure_pa = physics_specs.dimensionalize(np.exp(
          coords.horizontal.to_nodal(state.log_surface_pressure)), units.pascal)
      np.testing.assert_array_less(surface_pressure_pa, 108_000)
      np.testing.assert_array_less(45_000, surface_pressure_pa)

    with self.subTest('orography'):
      self.assertEqual(orography.shape, coords.horizontal.nodal_shape)
      nodal_orography_m = physics_specs.dimensionalize(
          orography, units.meter).magnitude
      np.testing.assert_array_less(nodal_orography_m, 6_000)
      np.testing.assert_array_less(-500, nodal_orography_m)

    with self.subTest('reference temperature'):
      np.testing.assert_allclose(ref_temps, 288)
      self.assertEqual(ref_temps.shape, (coords.vertical.layers,))

  def test_held_suarez_forcing(self):
    units = scales.units
    layers = 26
    coords = coordinate_systems.CoordinateSystem(
        horizontal=spherical_harmonic.Grid.T42(),
        vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers))
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()

    initial_state_fn, aux_features = (
        primitive_equations_states.isothermal_rest_atmosphere(
            coords, physics_specs, p0=1e5 * units.pascal,
            p1=5e3 * units.pascal,))
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    state = initial_state_fn(rng_key=jax.random.PRNGKey(0))

    hs = held_suarez.HeldSuarezForcing(
        coords=coords,
        physics_specs=physics_specs,
        reference_temperature=ref_temps)

    self.assertEqual(hs.kv().shape, (coords.vertical.layers, 1, 1))
    self.assertEqual(hs.kt().shape, coords.nodal_shape)

    surface_pressure = np.ones(coords.nodal_shape)
    self.assertEqual(hs.equilibrium_temperature(surface_pressure).shape,
                     coords.nodal_shape)

    explicit_terms = hs.explicit_terms(state)
    np.testing.assert_allclose(explicit_terms.log_surface_pressure, 0)


if __name__ == '__main__':
  absltest.main()
