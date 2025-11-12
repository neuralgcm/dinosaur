# Copyright 2024 Google LLC
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
# ==============================================================================
from collections import abc

from absl.testing import absltest
from absl.testing import flagsaver
import apache_beam as beam
from dinosaur import horizontal_interpolation
from dinosaur import hybrid_coordinates
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
from dinosaur.pipelines import regrid
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray
import xarray_beam


def mock_reanalysis_data(
    *,
    variables_3d: abc.Sequence[str],
    variables_2d: abc.Sequence[str],
    spatial_resolution_in_degrees: float,
    levels: abc.Sequence[int] = (),
    time_start: str = '2020-01-01',
    time_stop: str = '2021-01-01',
    time_resolution: str = '1 day',
    dtype: npt.DTypeLike = np.float32,
) -> xarray.Dataset:
  num_latitudes = round(180 / spatial_resolution_in_degrees) + 1
  num_longitudes = round(360 / spatial_resolution_in_degrees)
  freq = pd.Timedelta(time_resolution)
  coords = {
      'time': pd.date_range(time_start, time_stop, freq=freq, inclusive='left'),
      'latitude': np.linspace(-90, 90, num_latitudes),
      'longitude': np.linspace(0, 360, num_longitudes, endpoint=False),
      'level': np.array(levels),
  }
  dims_3d = ('time', 'level', 'longitude', 'latitude')
  shape_3d = tuple(coords[dim].size for dim in dims_3d)
  data_vars_3d = {k: (dims_3d, np.zeros(shape_3d, dtype)) for k in variables_3d}
  if not data_vars_3d:
    del coords['level']

  dims_2d = ('time', 'longitude', 'latitude')
  shape_2d = tuple(coords[dim].size for dim in dims_2d)
  data_vars_2d = {k: (dims_2d, np.zeros(shape_2d, dtype)) for k in variables_2d}

  data_vars = {**data_vars_3d, **data_vars_2d}
  return xarray.Dataset(data_vars, coords)


class RegridTest(absltest.TestCase):

  def test_regridding_era5_pressure_level_data(self):
    input_ds = mock_reanalysis_data(
        variables_3d=['geopotential'],
        variables_2d=['2m_temperature'],
        spatial_resolution_in_degrees=10,
        levels=[500, 850, 1000],
        time_start='2020-01-01',
        time_stop='2020-02-01',
    )

    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds.chunk({'time': 8}).to_zarr(input_path)

    with self.subTest('invalid regridder'):
      horizontal_regridder = horizontal_interpolation.ConservativeRegridder(
          source_grid=spherical_harmonic.Grid(
              longitude_nodes=100,
              latitude_nodes=100,
              latitude_spacing='equiangular_with_poles',
          ),
          target_grid=spherical_harmonic.Grid.TL31(),
      )
      with beam.Pipeline('DirectRunner') as p:
        with self.assertRaisesRegex(
            ValueError,
            'horizontal regridder source_grid does not match inferred source',
        ):
          _ = p | regrid.MultiRegridTransform(
              source=regrid.Source(input_zarr_path=input_path),
              regrid_targets=[
                  regrid.RegridTarget(
                      output_path=output_path,
                      horizontal_regridder=horizontal_regridder,
                      vertical_regridder=None,
                      output_chunks={'time': -1},
                  )
              ],
          )

    with self.subTest('valid regridder'):
      horizontal_regridder = horizontal_interpolation.ConservativeRegridder(
          source_grid=spherical_harmonic.Grid(
              longitude_nodes=36,
              latitude_nodes=19,
              latitude_spacing='equiangular_with_poles',
          ),
          target_grid=spherical_harmonic.Grid.TL31(),
      )

      with beam.Pipeline('DirectRunner') as p:
        _ = p | regrid.MultiRegridTransform(
            source=regrid.Source(input_zarr_path=input_path),
            regrid_targets=[
                regrid.RegridTarget(
                    output_path=output_path,
                    horizontal_regridder=horizontal_regridder,
                    vertical_regridder=None,
                    output_chunks={'time': 16},
                )
            ],
        )

      actual_ds, actual_chunks = xarray_beam.open_zarr(output_path)
      self.assertEqual(actual_ds.keys(), input_ds.keys())
      self.assertEqual(
          dict(actual_ds.sizes),
          {'time': 31, 'level': 3, 'longitude': 64, 'latitude': 32},
      )
      self.assertEqual(
          actual_chunks,
          {'time': 16, 'level': 3, 'longitude': 64, 'latitude': 32},
      )

  def test_regridding_era5_hybrid_level_data(self):
    input_path = self.create_tempdir('source').full_path
    surface_pressure_path = self.create_tempdir('surface_pressure').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = mock_reanalysis_data(
        variables_3d=['temperature'],
        variables_2d=[],
        spatial_resolution_in_degrees=10,
        levels=np.arange(1, 138),
        time_start='2020-01-01',
        time_stop='2020-01-03',
    ).rename({'level': 'hybrid'})
    input_ds.chunk().to_zarr(input_path)

    surface_pressure_ds = 100_000 + mock_reanalysis_data(
        variables_3d=[],
        variables_2d=['surface_pressure'],
        spatial_resolution_in_degrees=10,
        time_start='2020-01-01',
        time_stop='2020-01-03',
    )
    surface_pressure_ds['surface_pressure'].attrs['units'] = 'Pa'
    surface_pressure_ds.chunk().to_zarr(surface_pressure_path)

    horizontal_regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid=spherical_harmonic.Grid(
            longitude_nodes=36,
            latitude_nodes=19,
            latitude_spacing='equiangular_with_poles',
        ),
        target_grid=spherical_harmonic.Grid.TL31(),
    )
    ecmwf_coords = hybrid_coordinates.HybridCoordinates.ECMWF137()
    ufs_coords = hybrid_coordinates.HybridCoordinates.UFS127()
    vertical_regridder = vertical_interpolation.ConservativeRegridder(
        source_grid=ecmwf_coords,
        target_grid=ufs_coords.to_approx_sigma_coords(16),
    )

    with beam.Pipeline('DirectRunner') as p:
      _ = p | regrid.MultiRegridTransform(
          source=regrid.Source(
              input_zarr_path=input_path,
              surface_pressure_zarr_path=surface_pressure_path,
              name_mapping=regrid.NameMapping(level='hybrid'),
          ),
          regrid_targets=[
              regrid.RegridTarget(
                  output_path=output_path,
                  horizontal_regridder=horizontal_regridder,
                  vertical_regridder=vertical_regridder,
                  output_chunks={'time': -1},
              )
          ],
      )

    actual_ds, actual_chunks = xarray_beam.open_zarr(output_path)
    self.assertEqual(actual_ds.keys(), input_ds.keys())
    self.assertEqual(
        dict(actual_ds.sizes),
        # vs {'hybrid': 137, 'longitude': 36, 'latitude': 19} on inputs
        {'time': 2, 'sigma': 16, 'longitude': 64, 'latitude': 32},
    )
    self.assertEqual(
        actual_chunks,
        {'time': 2, 'sigma': 16, 'longitude': 64, 'latitude': 32},
    )


if __name__ == '__main__':
  absltest.main()
