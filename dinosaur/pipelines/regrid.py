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
r"""Regrid a Zarr dataset to Dinosaur grids.

Only rectilinear grids (one dimensional lat/lon coordinates) on the input Zarr
file are supported, but irregular spacing is OK.
"""

from concurrent import futures
import dataclasses
import logging
from typing import Any, Callable

import apache_beam as beam
from dinosaur import horizontal_interpolation
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
from dinosaur import xarray_utils
import numpy as np
import xarray
import xarray_beam


# pylint: disable=logging-fstring-interpolation

HorizontalRegridderFactory = Callable[
    [spherical_harmonic.Grid, spherical_harmonic.Grid],
    horizontal_interpolation.Regridder,
]


class NaNFiller:
  """Strategy for handling NaN values in regridded data."""

  def __call__(
      self, key: xarray_beam.Key, chunk: xarray.Dataset
  ) -> tuple[xarray_beam.Key, xarray.Dataset]:
    raise NotImplementedError


class NearestNaNFiller(NaNFiller):
  """Fill NaN values with nearest neighbors.

  This is useful for handling variables with partially invalid values such as
  sea surface temperature, which is not defined over land.
  """

  def __call__(
      self, key: xarray_beam.Key, chunk: xarray.Dataset
  ) -> tuple[xarray_beam.Key, xarray.Dataset]:
    try:
      return key, xarray_utils.fill_nan_with_nearest(chunk)
    except ValueError as e:
      raise ValueError(f'error while filling NaN values for {key=}') from e


@dataclasses.dataclass
class RaiseIfNaNFiller(NaNFiller):
  """Raises an error if any NaN values are found.

  This is a useful sanity check for missing source data (e.g., reading a date
  like 1900 in ARCO-ERA5 for which data is not available) or invalid vertical
  regridding (e.g., to a level that has no overlap with the source data).

  Attributes:
    allowed_incomplete_dims: dimensions allowed to be incomplete but not
      entirely missing, e.g., allowed_incomplete_dims=('latitude', 'longitude')
      would allow for missing values over land in an ocean variable. If no
      dimensions are listed here, then all dimensions must be complete.
  """

  allowed_incomplete_dims: tuple[str, ...] = ()

  def __call__(
      self, key: xarray_beam.Key, chunk: xarray.Dataset
  ) -> tuple[xarray_beam.Key, xarray.Dataset]:
    for var, array in chunk.items():
      is_valid = array.notnull().any(dim=self.allowed_incomplete_dims)
      if not is_valid.all():
        raise ValueError(f'NaN value found in {var} for {key=}')
    return key, chunk


@dataclasses.dataclass
class NameMapping:
  """Names of variables to standardize in the source dataset."""

  surface_pressure: str = 'surface_pressure'
  longitude: str = 'longitude'
  latitude: str = 'latitude'
  time: str = 'time'
  level: str = 'level'
  additional_renames: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Source:
  """Parameters describing a source dataset."""

  input_zarr_path: str
  surface_pressure_zarr_path: str | None = None

  name_mapping: NameMapping = dataclasses.field(default_factory=NameMapping)
  static_vars: list[str] | None = None
  static_time: str | None = None
  dynamic_vars: list[str] | None = None
  variables_to_shift: list[str] = dataclasses.field(default_factory=list)
  time_shift: str = '0 hours'

  time_start: str | None = None
  time_stop: str | None = None
  level_selection: list[float] | None = None


def get_source_dataset_and_chunks(
    source: Source,
) -> tuple[xarray.Dataset, dict[str, int]]:
  """Get the source dataset and its chunks, based on flags."""
  source_ds, input_chunks = xarray_beam.open_zarr(source.input_zarr_path)

  # insert surface pressure
  if source.surface_pressure_zarr_path is not None:
    # We need surface pressure for vertical interpolation from hybrid to sigma
    # coordinates. Sometimes (e.g., in ARCO-ERA5) this is stored in separate
    # Zarr files (for surface vs level variables).
    surface_pressure_ds = xarray.open_zarr(
        source.surface_pressure_zarr_path, chunks=None
    )
    # Put surface pressure in coordinates so it gets stored on each chunk, even
    # after using split_vars=True in DatasetToChunks.
    source_ds.coords['surface_pressure'] = surface_pressure_ds[
        source.name_mapping.surface_pressure
    ]

  # rename to standard names
  renames = {
      source.name_mapping.longitude: 'longitude',
      source.name_mapping.latitude: 'latitude',
      source.name_mapping.time: 'time',
  }
  if source.name_mapping.level != 'level':
    renames[source.name_mapping.level] = 'level'
  renames.update(source.name_mapping.additional_renames)

  source_ds = source_ds.rename(renames)
  input_chunks = {renames.get(k, k): v for k, v in input_chunks.items()}

  # lat/lon/level must be single chunk for regridding.
  input_chunks['longitude'] = -1
  input_chunks['latitude'] = -1
  if 'level' in input_chunks:
    input_chunks['level'] = -1

  # variable selection
  if source.static_vars is not None:
    static_vars = source.static_vars
  else:
    static_vars = [k for k, v in source_ds.items() if 'time' not in v.dims]
  static_ds = source_ds[static_vars]

  if source.static_time is not None and 'time' in static_ds.dims:
    # drop time dimension from static variables (they are duplicated along time
    # in some datasets such as ERA5)
    static_ds = static_ds.sel(time=source.static_time, drop=True)

  if source.dynamic_vars is not None:
    dynamic_vars = source.dynamic_vars
  else:
    dynamic_vars = [k for k, v in source_ds.items() if 'time' in v.dims]
  dynamic_ds = source_ds[dynamic_vars]

  source_ds = xarray.merge([dynamic_ds, static_ds], join='exact')

  # temporal shift
  source_ds = xarray_utils.xarray_selective_shift(
      dataset=source_ds,
      variables=source.variables_to_shift,
      time_shift=source.time_shift,
  )

  # time slice
  source_ds = source_ds.sel(time=slice(source.time_start, source.time_stop))
  input_chunks['time'] = min(source_ds.sizes['time'], input_chunks['time'])

  # level selection
  if source.level_selection is not None:
    source_ds = source_ds.sel(level=source.level_selection)

  input_chunks = {k: v for k, v in input_chunks.items() if k in source_ds.dims}

  return source_ds, input_chunks


def infer_source_horizontal_grid(
    source_ds: xarray.Dataset,
) -> spherical_harmonic.Grid:
  """Infer the horizontal grid of the source dataset."""
  grid = spherical_harmonic.Grid(
      latitude_nodes=source_ds.sizes['latitude'],
      longitude_nodes=source_ds.sizes['longitude'],
      latitude_spacing=xarray_utils.infer_latitude_spacing(source_ds.latitude),
      longitude_offset=xarray_utils.infer_longitude_offset(source_ds.longitude),
  )
  xarray_utils.verify_grid_consistency(
      longitude=source_ds.longitude,
      latitude=xarray_utils.ensure_ascending_latitude(source_ds).latitude,
      grid=grid,
  )
  return grid


def validate_horizontal_regridder(
    source_ds: xarray.Dataset,
    horizontal_regridder: horizontal_interpolation.Regridder,
) -> None:
  """Validate that the horizontal regridder is consistent with the source data."""
  source_horizontal_grid = infer_source_horizontal_grid(source_ds)
  if horizontal_regridder.source_grid != source_horizontal_grid:
    raise ValueError(
        'horizontal regridder source_grid does not match inferred source '
        'grid from input data:'
        f'\n{horizontal_regridder.source_grid=}\nvs'
        f'\n{source_horizontal_grid=}.'
    )


def validate_vertical_regridder(
    source_ds: xarray.Dataset,
    vertical_regridder: vertical_interpolation.Regridder | None,
) -> None:
  """Validate that the vertical regridder is consistent with the source data."""
  if vertical_regridder is not None and 'level' in source_ds.dims:
    if vertical_regridder.source_grid.layers != source_ds.sizes['level']:
      raise ValueError(
          'vertical regridder source_grid has an inconsistent number of '
          'layers with the source data:'
          f'\n{vertical_regridder.source_grid.layers=}\nvs'
          f'\n{source_ds.sizes["level"]=}'
      )


def get_regrid_func(
    horizontal_regridder: horizontal_interpolation.Regridder,
    vertical_regridder: vertical_interpolation.Regridder | None,
) -> Callable[
    [xarray_beam.Key, xarray.Dataset], tuple[xarray_beam.Key, xarray.Dataset]
]:
  """Get a function to regrid a chunk of data."""

  def regrid(
      key: xarray_beam.Key, chunk: xarray.Dataset
  ) -> tuple[xarray_beam.Key, xarray.Dataset]:

    if 'level' in chunk.dims and vertical_regridder is not None:
      match (
          vertical_regridder.source_grid,
          vertical_regridder.target_grid,
      ):
        case (
            vertical_interpolation.HybridCoordinates(),
            sigma_coordinates.SigmaCoordinates(),
        ):
          surface_pressure = chunk.coords['surface_pressure']
          if surface_pressure.attrs.get('units') != 'Pa':
            raise ValueError(
                f'surface_pressure units must be "Pa", but got '
                f'{surface_pressure.attrs.get("units")=}'
            )
          surface_pressure_in_hPa = surface_pressure / 100  # pylint: disable=invalid-name
          chunk = chunk.drop_vars('surface_pressure')
          chunk = xarray_utils.regrid_vertical(
              chunk,
              surface_pressure_in_hPa,
              vertical_regridder,
              in_dim='level',
              out_dim='sigma',
          )
          # Switch dimension name from 'level' to 'sigma'.
          assert 'level' not in chunk.dims and 'sigma' in chunk.dims
          key = key.with_offsets(level=None, sigma=0)  # no vertical chunking

        case (
            vertical_interpolation.PressureCoordinates(),
            vertical_interpolation.PressureCoordinates(),
        ):
          # Surface pressure is not needed for pressure-to-pressure
          # interpolation.
          chunk = xarray_utils.regrid_vertical(
              chunk,
              surface_pressure=None,
              regridder=vertical_regridder,
              in_dim='level',
              out_dim='level',
          )
          # Vertical dimension name remains 'level'.
          assert 'sigma' not in chunk.dims and 'level' in chunk.dims
          key = key.with_offsets(level=0)  # keep level, no vertical chunking

        case _:
          raise ValueError(
              'Unsupported vertical regridding combination: '
              f'source={type(vertical_regridder.source_grid)}, '
              f'target={type(vertical_regridder.target_grid)}'
          )

    chunk = xarray_utils.regrid_horizontal(chunk, horizontal_regridder)
    return key, chunk

  return regrid


def get_template(
    source_ds: xarray.Dataset,
    horizontal_regridder: horizontal_interpolation.Regridder,
    vertical_regridder: vertical_interpolation.Regridder | None,
) -> xarray.Dataset:
  """Build the Xarray-Beam template for the regridded dataset."""
  new_lon = np.rad2deg(horizontal_regridder.target_grid.longitudes)
  new_lat = np.rad2deg(horizontal_regridder.target_grid.latitudes)

  if vertical_regridder is not None and isinstance(
      vertical_regridder.source_grid, vertical_interpolation.HybridCoordinates
  ):
    # drop surface pressure that we added into coordinates for hybrid regridding
    if 'surface_pressure' in source_ds.coords:
      source_ds = source_ds.drop_vars(['surface_pressure'])

  # Data variables are lazy, coordinates are not
  template = (
      xarray_beam.make_template(source_ds, lazy_vars=source_ds.data_vars.keys())
      .isel(longitude=0, latitude=0, drop=True)
      .expand_dims(longitude=new_lon, latitude=new_lat)
      .transpose(..., 'longitude', 'latitude')
  )

  if vertical_regridder is not None:
    match (
        vertical_regridder.source_grid,
        vertical_regridder.target_grid,
    ):
      case (
          vertical_interpolation.HybridCoordinates(),
          sigma_coordinates.SigmaCoordinates(),
      ):

        def replace_level_with_sigma(x):
          if 'level' not in x.dims:
            return x
          axis = x.get_axis_num('level')
          sigmas = vertical_regridder.target_grid.centers
          return x.isel(level=0, drop=True).expand_dims(sigma=sigmas, axis=axis)

        template = template.map(replace_level_with_sigma)

      case (
          vertical_interpolation.PressureCoordinates(),
          vertical_interpolation.PressureCoordinates(),
      ):

        def update_pressure_levels(x):
          if 'level' not in x.dims:
            return x
          axis = x.get_axis_num('level')
          new_levels = vertical_regridder.target_grid.centers
          return x.isel(level=0, drop=True).expand_dims(
              level=new_levels, axis=axis
          )

        template = template.map(update_pressure_levels)

      case _:
        raise ValueError(
            'Unsupported vertical regridding combination: '
            f'source={type(vertical_regridder.source_grid)}, '
            f'target={type(vertical_regridder.target_grid)}'
        )

  return template


def get_output_chunks(
    input_chunks: dict[str, int],
    vertical_regridder: vertical_interpolation.Regridder | None,
    explicit_output_chunks: dict[str, int] | None,
) -> dict[str, int]:
  """Get the output chunks for the regridded dataset."""
  if explicit_output_chunks is None:
    return input_chunks

  output_chunks = input_chunks.copy()

  for dim, size in explicit_output_chunks.items():
    if dim not in input_chunks:
      output_chunks[dim] = size
    else:
      input_size = input_chunks[dim]
      if input_size == -1:
        raise ValueError(f'cannot resize unchunked dimension {dim!r}')
      output_chunks[dim] = max(input_size, size // input_size * input_size)

  if vertical_regridder is not None:
    del output_chunks['level']

    if isinstance(
        vertical_regridder.target_grid, sigma_coordinates.SigmaCoordinates
    ):
      output_chunks['sigma'] = -1
    elif isinstance(
        vertical_regridder.target_grid,
        vertical_interpolation.PressureCoordinates,
    ):
      output_chunks['level'] = -1
    else:
      raise ValueError(
          'Unsupported vertical regridding target grid: '
          f'{type(vertical_regridder.target_grid)}'
      )

  return output_chunks


@dataclasses.dataclass
class RegridTarget:
  """Parameters describing a regridding target.

  Parameters:
    output_path: path to the output Zarr file.
    horizontal_regridder: regridder for horizontal interpolation.
    vertical_regridder: optional regridder for vertical interpolation.
    nan_filler: optional method for filling NaNs in the regridded data.
    output_chunks: optional new output chunks for the regridded data.
  """

  output_path: str
  horizontal_regridder: horizontal_interpolation.Regridder
  vertical_regridder: vertical_interpolation.Regridder | None
  nan_filler: NaNFiller | None = None
  output_chunks: dict[str, int] | None = None
  zarr_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class _RegridTransform(beam.PTransform):
  """PTransform for regridding to a single target grid."""

  source_template: xarray.Dataset
  input_chunks: dict[str, int]
  target: RegridTarget
  io_num_threads: int | None
  target_template: xarray.Dataset = dataclasses.field(init=False)
  output_chunks: dict[str, int] = dataclasses.field(init=False)

  def __post_init__(self):
    validate_horizontal_regridder(
        self.source_template, self.target.horizontal_regridder
    )
    validate_vertical_regridder(
        self.source_template, self.target.vertical_regridder
    )

    self.target_template = get_template(
        self.source_template,
        self.target.horizontal_regridder,
        self.target.vertical_regridder,
    )
    if self.target.zarr_metadata:
      self.target_template.attrs.update(self.target.zarr_metadata)

    self.output_chunks = get_output_chunks(
        self.input_chunks,
        self.target.vertical_regridder,
        self.target.output_chunks,
    )

  def setup_output_zarr(self):
    xarray_beam.setup_zarr(
        self.target_template, self.target.output_path, self.output_chunks
    )

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    regrid = get_regrid_func(
        self.target.horizontal_regridder, self.target.vertical_regridder
    )
    pcoll |= beam.MapTuple(regrid)

    if self.output_chunks is not None:
      pcoll |= xarray_beam.ConsolidateChunks(self.output_chunks)

    if self.target.nan_filler is not None:
      pcoll |= beam.MapTuple(self.target.nan_filler)

    pcoll |= xarray_beam.ChunksToZarr(
        self.target.output_path,
        self.target_template,
        self.output_chunks,
        num_threads=self.io_num_threads,
        needs_setup=False,
    )
    return pcoll


@dataclasses.dataclass
class MultiRegridTransform(beam.PTransform):
  """PTransform for regridding to multiple target grids.

  The most expensive part of regridding (to coarser resolutions) is typically
  reading the source dataset from disk, so this transform reads source data
  once and outputs all regridding targets simultaneously.

  Parameters:
    source: specification of how to load the source dataset.
    regrid_targets: specifications for the desired output grids.
    io_num_threads: number of threads to use for reading/writing Zarr chunks.
  """

  source: Source
  regrid_targets: list[RegridTarget]
  io_num_threads: int | None = None

  def expand(self, pcoll: beam.PCollection) -> list[beam.PCollection]:

    source_ds, input_chunks = get_source_dataset_and_chunks(self.source)
    source_pcoll = pcoll | xarray_beam.DatasetToChunks(
        source_ds,
        input_chunks,
        split_vars=True,
        num_threads=self.io_num_threads,
    )

    # We setup Zarr stores using separate threads, which otherwise takes ~1
    # minute per regridding target.
    executor = futures.ThreadPoolExecutor(max_workers=len(self.regrid_targets))

    output_pcollections = []
    setup_futures = []
    for i, target in enumerate(self.regrid_targets):
      transform = _RegridTransform(
          source_template=source_ds,
          input_chunks=input_chunks,
          target=target,
          io_num_threads=self.io_num_threads,
      )
      pcoll = source_pcoll | f'Regrid{i}' >> transform
      setup_futures.append(executor.submit(transform.setup_output_zarr))
      output_pcollections.append(pcoll)

    # Wait for any setup errors to be raised.
    for future in futures.as_completed(setup_futures):
      future.result()
    logging.info('zarr setup complete')

    return output_pcollections
