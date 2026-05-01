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

"""Defines `Grid` that holds vertical and horizontal grids."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

from dinosaur import layer_coordinates
from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing

import jax
import jax.numpy as jnp
import numpy as np

# keys that correspond to the grid/coordinate type in asdict.
HORIZONTAL_COORD_TYPE_KEY = 'horizontal_grid_type'
VERTICAL_COORD_TYPE_KEY = 'vertical_grid_type'
HorizontalGridTypes = spherical_harmonic.Grid
VerticalCoordinateTypes = (
    layer_coordinates.LayerCoordinates
    | sigma_coordinates.SigmaCoordinates
    | Any
)


P = jax.sharding.PartitionSpec


def _with_sharding_constraint(
    x: typing.Pytree,
    sharding: jax.sharding.NamedSharding | None,
) -> typing.Pytree:
  """Ensure a sharing constraint on all non-scalar arrays in a pytree."""
  if sharding is None:
    return x  # unsharded

  if len(sharding.spec) != 3:
    raise ValueError(f'partition spec does not have length 3: {sharding.spec}')

  def f(y: jax.Array) -> jax.Array:
    if y.ndim == 1 and y.dtype == jnp.uint32:
      return y  # prng key

    if y.ndim not in {2, 3}:
      raise ValueError(f'can only shard 2D or 3D arrays: {y.shape=}')

    # TODO(shoyer): refactor our ML codebase so we only need to shard 3D arrays.
    # This currently is only needed for stochastic perturbations.
    if y.ndim == 2:
      spec = P(*sharding.spec[1:])
      sharding_ = jax.sharding.NamedSharding(sharding.mesh, spec)
    elif y.shape[0] == 1:
      # surface level variable
      spec = P(None, *sharding.spec[1:])
      sharding_ = jax.sharding.NamedSharding(sharding.mesh, spec)
    else:
      sharding_ = sharding
    return jax.lax.with_sharding_constraint(y, sharding_)

  try:
    return pytree_utils.tree_map_over_nonscalars(f, x)
  except ValueError as e:
    shapes = jax.tree_util.tree_map(jnp.shape, x)
    raise ValueError(f'failed to shard pytree with shapes {shapes}') from e


@dataclasses.dataclass(frozen=True)
class CoordinateSystem:
  """Contains horizontal and vertical grid data.

  Attributes:
    horizontal: object describing discretization of the horizontal plane.
    vertical: object describing discretization of the vertical coordinate.
    spmd_mesh: mesh to use for parallelism in the single program multiple device
      (SPMD) paradigm with distributed JAX arrays, if any. Required if using
      model parallelism.
    physics_partition_spec: partition spec for "physics" calculations that act
      only in the vertical direction (e.g., ML corrections and vertical
      interpolation). We also use this partitioning for loading data from disk
      and calculating losses.
    dycore_partition_spec: partition spec for "dycore" calculations across all
      spatial dimensions (e.g., spherical harmonic transforms and solving the
      primitive equations).
  """

  horizontal: HorizontalGridTypes
  vertical: VerticalCoordinateTypes
  spmd_mesh: jax.sharding.Mesh | None = None

  # In principle, these partition spec arguments can be customized, but it is
  # not recommended.

  # The logical dimension names on `spmd_mesh` correspond to those used in the
  # dycore.
  dycore_partition_spec: jax.sharding.PartitionSpec = P('z', 'x', 'y')

  # For cases where we don't want to use vertical partitioning (e.g., because
  # we are applying a column-wise neural net), we merge 'z' into one of the
  # spatial dimensions instead. In principle, it shouldn't matter whether we
  # merge 'z' into 'x' or 'y', but in practice XLA SPMD generates more efficient
  # code (avoiding "full rematerialization") if we merge into the adjacent
  # dimension. Both ('x', 'z') and ('z', 'x') work, but training with ('x', 'z')
  # on TL255 is slightly faster (0.4% overall).
  physics_partition_spec: jax.sharding.PartitionSpec = P(None, ('x', 'z'), 'y')

  def __post_init__(self):
    if self.spmd_mesh is not None:
      if not {'x', 'y', 'z'} <= set(self.spmd_mesh.axis_names):
        raise ValueError(
            "mesh is missing one or more of the required axis names 'x', 'y' "
            f"and 'z': {self.spmd_mesh}"
        )
    # ensure a consistent mesh
    horizontal = dataclasses.replace(self.horizontal, spmd_mesh=self.spmd_mesh)
    object.__setattr__(self, 'horizontal', horizontal)

  def _get_sharding(
      self, partition_spec: jax.sharding.PartitionSpec
  ) -> jax.sharding.NamedSharding | None:
    if self.spmd_mesh is None:
      return None
    return jax.sharding.NamedSharding(self.spmd_mesh, partition_spec)

  @property
  def physics_sharding(self) -> jax.sharding.NamedSharding | None:
    """How to shard arrays for "physics" calculations."""
    return self._get_sharding(self.physics_partition_spec)

  def with_physics_sharding(self, x: typing.PyTreeState) -> typing.PyTreeState:
    """Enforce a "physics" sharding constraint on a pytree."""
    return _with_sharding_constraint(x, self.physics_sharding)

  @property
  def dycore_sharding(self) -> jax.sharding.NamedSharding | None:
    """How to shard arrays for nodal or modal dycore calculations."""
    return self._get_sharding(self.dycore_partition_spec)

  def with_dycore_sharding(self, x: typing.PyTreeState) -> typing.PyTreeState:
    """Enforce a dycore sharding constraint on a pytree."""
    return _with_sharding_constraint(x, self.dycore_sharding)

  def dycore_to_physics_sharding(
      self, x: typing.PyTreeState
  ) -> typing.PyTreeState:
    """Transpose from dycore to physics sharding."""
    return self.with_physics_sharding(self.with_dycore_sharding(x))

  def physics_to_dycore_sharding(
      self, x: typing.PyTreeState
  ) -> typing.PyTreeState:
    """Transpose from physics to dycore sharding."""
    return self.with_dycore_sharding(self.with_physics_sharding(x))

  def asdict(self):
    horizontal_keys = set(self.horizontal.asdict().keys())
    vertical_keys = set(self.vertical.asdict().keys())
    if horizontal_keys.intersection(vertical_keys):
      raise ValueError('keys in horizontal and vertical grids collide.')
    out = {**self.horizontal.asdict(), **self.vertical.asdict()}
    out[HORIZONTAL_COORD_TYPE_KEY] = type(self.horizontal).__name__
    out[VERTICAL_COORD_TYPE_KEY] = type(self.vertical).__name__
    return out

  @property
  def nodal_shape(self) -> tuple[int, int, int]:
    """Returns 3d nodal grid shape, vertical.shape + horizontal.shape."""
    return (self.vertical.layers,) + self.horizontal.nodal_shape

  @property
  def modal_shape(self) -> tuple[int, int, int]:
    """Returns 3d modal grid shape, vertical.shape + horizontal.shape."""
    return (self.vertical.layers,) + self.horizontal.modal_shape

  @property
  def surface_nodal_shape(self) -> tuple[int, int, int]:
    """Returns surface nodal grid shape, (1,) + horizontal.shape."""
    return (1,) + self.horizontal.nodal_shape

  @property
  def surface_modal_shape(self) -> tuple[int, int, int]:
    """Returns surface modal grid shape, (1,) + horizontal.shape."""
    return (1,) + self.horizontal.modal_shape


def get_spectral_downsample_fn(
    coords: CoordinateSystem,
    save_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
  """Returns function that downsample state from `coords` to `save_coords`."""
  if expect_same_vertical and (coords.vertical != save_coords.vertical):
    raise ValueError('downsampling vertical resolution is not supported.')
  lon_wavenumber_slice = slice(0, save_coords.horizontal.modal_shape[0])
  total_wavenumber_slice = slice(0, save_coords.horizontal.modal_shape[1])
  if (
      coords.horizontal.total_wavenumbers
      < save_coords.horizontal.total_wavenumbers
  ) or (
      coords.horizontal.longitude_wavenumbers
      < save_coords.horizontal.longitude_wavenumbers
  ):
    raise ValueError('save_coords.horizontal larger than coords.horizontal')

  def downsample_fn(state: typing.PyTreeState) -> typing.PyTreeState:
    slice_fn = lambda x: x[..., lon_wavenumber_slice, total_wavenumber_slice]
    return pytree_utils.tree_map_over_nonscalars(slice_fn, state)

  return downsample_fn


def get_spectral_upsample_fn(
    coords: CoordinateSystem,
    save_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
  """Returns function that upsamples state from `coords` to `save_coords`."""
  if expect_same_vertical and (coords.vertical != save_coords.vertical):
    raise ValueError('upsampling vertical resolution is not supported.')
  save_shape = save_coords.horizontal.modal_shape
  coords_shape = coords.horizontal.modal_shape
  lon_wavenumber_pad = (0, save_shape[0] - coords_shape[0])
  total_wavenumber_pad = (0, save_shape[1] - coords_shape[1])
  if (min(lon_wavenumber_pad) != 0) or (min(total_wavenumber_pad) != 0):
    raise ValueError('save_coords.horizontal smaller than coords.horizontal')
  tail_pad = (lon_wavenumber_pad, total_wavenumber_pad)

  def upsample_fn(state: typing.PyTreeState) -> typing.PyTreeState:
    pad_fn = lambda x: jnp.pad(x, ((0, 0),) * (x.ndim - 2) + tail_pad)
    return pytree_utils.tree_map_over_nonscalars(pad_fn, state)

  return upsample_fn


def get_spectral_interpolate_fn(
    source_coords: CoordinateSystem,
    target_coords: CoordinateSystem,
    expect_same_vertical: bool = True,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
  """Returns modal interpolation_fn from `source_coords` to `target_coords`."""
  if (
      source_coords.horizontal.total_wavenumbers
      < target_coords.horizontal.total_wavenumbers
  ) and (
      source_coords.horizontal.longitude_wavenumbers
      < target_coords.horizontal.longitude_wavenumbers
  ):
    return get_spectral_upsample_fn(
        source_coords, target_coords, expect_same_vertical
    )
  elif (
      source_coords.horizontal.total_wavenumbers
      >= target_coords.horizontal.total_wavenumbers
  ) and (
      source_coords.horizontal.longitude_wavenumbers
      >= target_coords.horizontal.longitude_wavenumbers
  ):
    return get_spectral_downsample_fn(
        source_coords, target_coords, expect_same_vertical
    )
  else:
    raise ValueError(
        'Incompatible horizontal coordinates with shapes '
        f'{source_coords.horizontal.modal_shape}, '
        f'{target_coords.horizontal.modal_shape}'
    )


def get_nodal_shapes(
    inputs: typing.Pytree,
    coords: CoordinateSystem,
) -> typing.Pytree:
  """Returns a pytree with nodal shapes of arrays in `inputs`."""
  nodal_shape = coords.horizontal.nodal_shape
  array_shape_fn = lambda x: np.asarray(x.shape[:-2] + nodal_shape)
  scalar_shape_fn = lambda x: np.array([], dtype=int)
  return pytree_utils.tree_map_over_nonscalars(
      array_shape_fn, inputs, scalar_fn=scalar_shape_fn
  )


def get_modal_shapes(
    inputs: typing.Pytree,
    coords: CoordinateSystem,
) -> typing.Pytree:
  """Returns a pytree with modal shapes of arrays in `inputs`."""
  modal_shape = coords.horizontal.modal_shape
  array_shape_fn = lambda x: np.asarray(x.shape[:-2] + modal_shape)
  scalar_shape_fn = lambda x: np.array([], dtype=int)
  return pytree_utils.tree_map_over_nonscalars(
      array_shape_fn, inputs, scalar_fn=scalar_shape_fn
  )


# TODO(dkochkov) Consider moving these functions to spherical_harmonic module.


def maybe_to_nodal(
    fields: typing.Pytree,
    coords: CoordinateSystem,
) -> typing.Pytree:
  """Converts non-scalar elements in `fields` that are not nodal to nodal."""
  nodal_shapes = get_nodal_shapes(fields, coords)

  def to_nodal_fn(x):
    return coords.with_dycore_sharding(
        coords.horizontal.to_nodal(coords.with_dycore_sharding(x))
    )

  fn = lambda x, nodal: x if x.shape == tuple(nodal) else to_nodal_fn(x)
  return jax.tree_util.tree_map(fn, fields, nodal_shapes)


def maybe_to_modal(
    fields: typing.Pytree,
    coords: CoordinateSystem,
) -> typing.Pytree:
  """Converts non-scalar elements in `fields` that are not modal to modal."""
  modal_shapes = get_modal_shapes(fields, coords)

  def to_modal_fn(x):
    return coords.with_dycore_sharding(
        coords.horizontal.to_modal(coords.with_dycore_sharding(x))
    )

  fn = lambda x, modal: x if x.shape == tuple(modal) else to_modal_fn(x)
  return jax.tree_util.tree_map(fn, fields, modal_shapes)


def _map_over_matching_keys(
    inputs: dict[str, Any],
    fn: Callable[[typing.Array], typing.Array],
    keys_to_map_over: Sequence[str],
) -> dict[str, Any]:
  """Applies `fn` to values in `inputs` or sub-dictionaries for matching keys.

  Args:
    inputs: potentially nested dictionary to map over.
    fn: function to apply to elements in the `inputs` that have matching keys.
    keys_to_map_over: keys to which `fn` should be applied to.

  Returns:
    `inputs` with all elements that have keys matching an entry in
    `keys_to_map_over` transformed by function `fn`.
  """
  outputs = {}
  for k, v in inputs.items():
    if not isinstance(v, dict):
      outputs[k] = fn(v) if k in keys_to_map_over else v
    else:
      outputs[k] = _map_over_matching_keys(v, fn, keys_to_map_over)
  return outputs


def scale_levels_for_matching_keys(
    inputs: typing.Pytree,
    scales: typing.Array,
    keys_to_scale: Sequence[str] = tuple(),
) -> typing.Pytree:
  """Transforms `inputs` by scaling levels for keys that are in `keys_to_scale.

  Args:
    inputs: pytree of values that will be selectively scaled along levels.
    scales: scaling weights that will be applied.
    keys_to_scale: keys for which scaling operation is applied.

  Returns:
    pytree of the same structure of `inputs` with keys matching `keys_to_scale`
    scaled by `scales` along the level axis.
  """
  if scales.ndim != 1:
    raise ValueError(
        'scales must be 1d array of weights per level, got '
        f'array with shape {scales.shape}'
    )
  scales = scales[:, np.newaxis, np.newaxis]  # broadcasting shape.
  inputs, from_dict_fn = pytree_utils.as_dict(inputs)
  scale_fn = lambda x: x * scales
  inputs = _map_over_matching_keys(inputs, scale_fn, keys_to_scale)
  return from_dict_fn(inputs)
