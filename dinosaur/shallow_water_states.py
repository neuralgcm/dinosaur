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

"""Code for generating conditions for shallow water system.

Here we provide steady geostrophic flow and initial state from barotropic
instability test case [1].

  [1]: Galewsky, Joseph, Richard K. Scott, and Lorenzo M. Polvani.
       "An initial-value problem for testing numerical models of the global
       shallow-water equations."
       Tellus A: Dynamic Meteorology and Oceanography 56, no. 5 (2004): 429-440.
"""

import functools
from typing import Any, Callable, NamedTuple

from dinosaur import coordinate_systems
from dinosaur import scales
from dinosaur import shallow_water
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import xarray_utils

import jax
import jax.numpy as jnp
import numpy as np


units = scales.units
Array = typing.Array
Numeric = typing.Numeric


def subtract_longitudes(a, b):
  """Computes the difference `a - b`, accounting for periodicity."""
  return jnp.mod(a - b + jnp.pi, 2 * jnp.pi) - jnp.pi


def one_layer(
    u: Array,
    grid: spherical_harmonic.Grid
) -> shallow_water.State:
  """Simple steady state for one-layer rotating SWE.

  Args:
    u: longitudinal velocities u(θ) in nodal representation.
    grid: object describing horizontal spatial discretization.

  Returns:
    A steady state solution to the shallow water equations. The steady state
    satisfies

      ∂(Φ + ½u²)/∂θ + (ζ + f) u = 0

    where ζ is vorticity and f is the Coriolis force.
  """
  # Construct a velocity field that has zonal (eastward) velocity specified
  # by `u` at each Gaussian quadrature point. The meridional (northward)
  # velocity is uniformly zero.
  sec_lat = 1 / grid.cos_lat
  lon, _ = grid.nodal_axes
  u = jnp.ones_like(lon)[..., jnp.newaxis] * u
  sec_lat_u = u * sec_lat
  vorticity = -grid.sec_lat_d_dlat_cos2(grid.to_modal(sec_lat_u))
  f = shallow_water.get_coriolis(grid)
  total_vorticity = grid.to_nodal(vorticity) + f
  potential_plus_energy = -grid.inverse_laplacian(
      grid.sec_lat_d_dlat_cos2(grid.to_modal(sec_lat_u * total_vorticity)))
  potential = potential_plus_energy - grid.to_modal(u**2 / 2)
  potential = potential.at[0, 0].set(0)
  return shallow_water.State(vorticity=vorticity,
                             divergence=jnp.zeros_like(vorticity),
                             potential=potential)


def multi_layer(
    u: Array,
    density: Array,
    coords: coordinate_systems.CoordinateSystem,
) -> shallow_water.State:
  """Simple steady state for n-layer rotating SWE.

  Args:
    u: longitudinal velocities u(θ) in nodal representation.
    density: a non-decreasing vector of densities, starting from the top layer.
    coords: horizontal and vertical descritization.

  Returns:
    The steady state solution to the multi-layer SWE, with vᵢ = 0, which
    satisfies

      ∂(RΦ + ½u²)/∂θ + (ζ + f) u = 0

    where `R[i, j] = min(⍴[j] / ⍴[i], 1)`, ζ is vorticity and f is the
    Coriolis force.
  """
  s = jax.vmap(functools.partial(one_layer, grid=coords.horizontal))(u)
  n_layers = coords.vertical.layers
  density_ratios = shallow_water.get_density_ratios(density) + np.eye(n_layers)
  potential = s.potential
  if n_layers > 1:
    # TODO(jamieas): consider just inverting `density_ratios`, as this should be
    # a small matrix.
    flat_potential = jnp.reshape(s.potential, (n_layers, -1))
    potential = jnp.reshape(jnp.linalg.solve(density_ratios, flat_potential),
                            s.potential.shape)
  return shallow_water.State(s.vorticity, s.divergence, potential)


class BarotropicInstabilityParameters(NamedTuple):
  """Parameters of initial state for a barotropic instability test case."""

  # Parameters describing the location and magnitude of the velocity 'jet'.
  jet_northern_lat: Numeric
  jet_southern_lat: Numeric
  jet_max_velocity: Numeric

  # The mean thickness of the atmosphere.
  mean_height: Numeric

  # Parameters describing the latitude, longitude, size and size of the 'bump'
  # that is added to the height field.
  bump_lon_location: Numeric
  bump_lon_scale: Numeric
  bump_lat_location: Numeric
  bump_lat_scale: Numeric
  bump_height_scale: Numeric


def get_default_parameters() -> BarotropicInstabilityParameters:
  """Returns the parameters from the original paper, cited above."""
  return BarotropicInstabilityParameters(
      jet_northern_lat=np.array(np.pi / 2 - np.pi / 7) * units.radian,
      jet_southern_lat=np.array(np.pi / 7) * units.radian,
      jet_max_velocity=np.array(80.) * units.m / units.s,
      mean_height=np.array(10_000.) * units.m,
      bump_lon_location=np.array(0.) * units.radian,
      bump_lon_scale=np.array(1 / 3) * units.radian,
      bump_lat_location=np.array(np.pi / 4) * units.radian,
      bump_lat_scale=np.array(1 / 15) * units.radian,
      bump_height_scale=np.array(120.) * units.m)


def get_random_parameters(
    key: jnp.ndarray,
    default_parameters: BarotropicInstabilityParameters,
) -> BarotropicInstabilityParameters:
  """Return random parameters for the barotropic instability test case."""
  # Currently, we only randomize the latitude of the 'bump', and it is required
  # to be located 'within' the jet.
  jet_width = (default_parameters.jet_northern_lat
               - default_parameters.jet_southern_lat)
  random_bump_lat = (default_parameters.jet_southern_lat
                     + jax.random.beta(key, 4, 4) * jet_width)
  return default_parameters._replace(bump_lat_location=random_bump_lat)


def get_zonal_velocity(
    latitude: Numeric,
    parameters: BarotropicInstabilityParameters,
) -> Array:
  """Returns the zonal velocity based on `parameters` for `latitude`."""
  inside_jet = ((latitude > parameters.jet_southern_lat)
                * (latitude < parameters.jet_northern_lat))
  velocity_shape = jnp.exp(1 / (latitude - parameters.jet_southern_lat)
                           / (latitude - parameters.jet_northern_lat))
  normalizer = jnp.exp(
      -4 / (parameters.jet_northern_lat - parameters.jet_southern_lat)**2)
  velocity_inside_jet = (
      parameters.jet_max_velocity / normalizer * velocity_shape)

  return jnp.where(inside_jet, velocity_inside_jet, 0)


def get_height(
    longitude: Numeric,
    latitude: Numeric,
    parameters: BarotropicInstabilityParameters
):
  """Returns the height profile of the bump specified in `parameters`."""
  bump_shape = jnp.cos(latitude) * jnp.exp(
      -(subtract_longitudes(parameters.bump_lon_location, longitude)
        / parameters.bump_lon_scale) ** 2
      -((parameters.bump_lat_location - latitude)
        / parameters.bump_lat_scale) ** 2)
  return parameters.bump_height_scale * bump_shape


def barotropic_instability_tc(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: Any,
    densities: Array = np.ones((1,)),
) -> tuple[Callable[..., shallow_water.State], typing.AuxFeatures]:
  """Returns a function that generates random states and static features.

  This function implements initial states for barotropic instability test case.
  The returned features includes mean potential at the corresponding levels.

  Args:
    coords: horizontal and vertical descritization.
    physics_specs: physics specs describing global physical parameters.
    densities: a non-decreasing vector of densities, starting from the top.

  Returns:
    random_state_fn: function that generates a randomized initial state.
    aux_features: static auxiliary data to be provided to dynamical models.
  """
  _, sin_lat = coords.horizontal.nodal_mesh
  lat = np.arcsin(sin_lat)
  default_parameters = get_default_parameters()

  def random_state_fn(rng_key: jnp.ndarray) -> shallow_water.State:
    parameters = get_random_parameters(rng_key, default_parameters)
    # make sure that all parameters are non-dimensionalized.
    parameters = jax.tree.map(physics_specs.nondimensionalize, parameters)
    # The initial condition is computed by findind a steady state solution and
    # then adding a small 'bump' to the potential.
    zonal_velocity = jnp.stack([get_zonal_velocity(lat, parameters)
                                for _ in range(coords.vertical.layers)])
    steady = multi_layer(zonal_velocity, densities, coords)
    bump_potential = coords.horizontal.to_modal(
        get_height(*coords.horizontal.nodal_mesh, parameters) * physics_specs.g)
    initial_state = shallow_water.State(
        vorticity=steady.vorticity,
        potential=(steady.potential + bump_potential),
        divergence=steady.divergence)
    return initial_state

  top_potential = physics_specs.nondimensionalize(
      default_parameters.mean_height) * physics_specs.g
  reference_potential = np.linspace(
      0, top_potential, 1 + coords.vertical.layers, endpoint=True)[1:]
  aux_features = {xarray_utils.REF_POTENTIAL_KEY: reference_potential}
  return random_state_fn, aux_features
