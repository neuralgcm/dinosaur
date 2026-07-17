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

"""Multi-layer shallow water equations."""

from __future__ import annotations

import dataclasses
import functools
from typing import Sequence

from dinosaur import coordinate_systems
from dinosaur import scales
from dinosaur import semi_lagrangian
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import units
import jax
import jax.numpy as jnp
import numpy as np
import tree_math


Array = typing.Array
Numeric = typing.Numeric
Quantity = typing.Quantity

FilterFn = typing.FilterFn
InverseFn = typing.InverseFn
StateFn = typing.StateFn
StepFn = typing.StepFn

SCALE = scales.DEFAULT_SCALE


# All `einsum`s should be done at highest available precision.
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)

#  =============================================================================
#  Data Structures
#
#  Data classes that describe the state, scale and parameters of the system.
#  =============================================================================


@tree_math.struct
class State:
  """Records the state of a system described by the shallow water equations."""

  vorticity: Array
  divergence: Array
  potential: Array


# For backwards compatibility.
ShallowWaterSpecs = units.SimUnits


#  =============================================================================
#  Helper Functions
#
#  Functions used to compute individual terms and intermediate values for the
#  primitive equations.
#  =============================================================================


def state_to_nodal(state: State, grid: spherical_harmonic.Grid) -> State:
  """Converts a state to the spatial/nodal basis."""
  return jax.tree.map(lambda x: grid.to_nodal(grid.clip_wavenumbers(x)), state)


def state_to_modal(state: State, grid: spherical_harmonic.Grid) -> State:
  """Converts a state to the spectral/modal basis."""
  return jax.tree.map(grid.to_modal, state)


def get_density_ratios(density: Array) -> np.ndarray:
  """Computes density ratios used to compute interactions between layers.

  Args:
    density: a vector of layer densities, beginning from the top. These values
      must be non-decreasing.

  Returns:
    An array `D` such that

                 density[i] / density[j]  if i < j
      D[i, j] =  0                        if i = j
                 1                        if i > j
  """
  ratios = np.minimum(density / density[..., np.newaxis], 1)
  np.fill_diagonal(ratios, 0)
  return ratios


def get_coriolis(grid: spherical_harmonic.Grid) -> np.ndarray:
  """Returns an array of coriolis forces in the spatial basis."""
  _, sin_lat = grid.nodal_mesh
  return sin_lat


#  =============================================================================
#  The `ShallowWaterEquations` Class
#
#  The `ShallowWaterEquations` class expresses the shallow water equations in a
#  form that is appropriate for semi-implicit time stepping.
#  =============================================================================


@dataclasses.dataclass
class ShallowWaterEquations(time_integration.ImplicitExplicitODE):
  """A semi-implicit description of the shallow water equations.

  See go/shallow-water for more details.

  Attributes:
    coords: horizontal and vertical descritization.
    physics_specs: an object describing the scales and physical constants.
    orography: an array of shape [latitudinal_wavenumbers, total_wavenumbers]
      describing the topography.
    reference_potential: an array of shape [layers] holding mean geopotential.
  """

  coords: coordinate_systems.CoordinateSystem
  physics_specs: units.SimUnitsProtocol
  orography: Array
  reference_potential: Array
  densities: Array = dataclasses.field(default_factory=lambda: np.ones((1,)))

  @property
  def coriolis_parameter(self) -> Array:
    """Returns the value `2Ω sin(θ)` associated with Coriolis force."""
    _, sin_lat = self.coords.horizontal.nodal_mesh
    return 2 * self.physics_specs.angular_velocity * sin_lat

  @property
  def density_ratios(self) -> Array:
    """Returns `density_ratios` with spatial dimensions appended."""
    return get_density_ratios(self.densities)

  @property
  def ref_potential(self) -> Array:
    """Returns `reference_potential` with spatial dimensions appended."""
    return self.reference_potential[..., np.newaxis, np.newaxis]

  def explicit_terms(self, state: State) -> State:
    """Computes explicit tendencies of the shallow water equations."""
    # we stack two components of the velocity to transform them together.
    u = jnp.stack(
        spherical_harmonic.get_cos_lat_vector(
            state.vorticity, state.divergence, self.coords.horizontal
        )
    )

    # Switch to physical coordinates for spatial point-wise operations
    nodal_u = self.coords.horizontal.to_nodal(u)
    nodal_state = state_to_nodal(state, self.coords.horizontal)

    total_vorticity = nodal_state.vorticity + self.coriolis_parameter

    sec2_lat = self.coords.horizontal.sec2_lat
    nodal_b = nodal_u * total_vorticity * sec2_lat
    nodal_g = nodal_u * nodal_state.potential * sec2_lat
    nodal_e = (nodal_u * nodal_u).sum(0) * sec2_lat / 2

    # Stack and unstack values to perform a single transform
    bge_nodal = jnp.concatenate(
        [nodal_b, nodal_g, jnp.expand_dims(nodal_e, axis=0)], axis=0
    )
    bge = self.coords.horizontal.to_modal(bge_nodal)
    b, g, e = jnp.split(bge, [2, 4], axis=0)
    e = jnp.squeeze(e, axis=0)

    # Pressure gradients are computed as weighted sums across layers.
    # Note that this is the only interaction between layers.
    p = einsum('ab,...bml->...aml', self.density_ratios, state.potential)
    if self.orography is not None:
      p = p + self.orography

    explicit_vorticity = self.coords.horizontal.clip_wavenumbers(
        -self.coords.horizontal.div_cos_lat(b)
    )
    explicit_divergence = self.coords.horizontal.clip_wavenumbers(
        -self.coords.horizontal.laplacian(p + e)
        + self.coords.horizontal.curl_cos_lat(b)
    )
    explicit_potential = self.coords.horizontal.clip_wavenumbers(
        -self.coords.horizontal.div_cos_lat(g)
    )
    return State(explicit_vorticity, explicit_divergence, explicit_potential)  # pyrefly: ignore[bad-argument-count]

  def implicit_terms(self, state: State) -> State:
    """Returns the implicit terms of the shallow water equations."""
    return State(
        vorticity=jnp.zeros_like(state.vorticity),  # pyrefly: ignore[unexpected-keyword]
        divergence=-self.coords.horizontal.laplacian(state.potential),  # pyrefly: ignore[unexpected-keyword]
        potential=-self.ref_potential * state.divergence,  # pyrefly: ignore[unexpected-keyword]
    )

  def implicit_inverse(self, state: State, step_size: float) -> State:
    """Computes the inverse `(1 - step_size * implicit_terms)⁻¹."""
    inverse_schur_complement = 1 / (
        1
        - step_size**2
        * self.ref_potential
        * self.coords.horizontal.laplacian_eigenvalues
    )
    return State(
        vorticity=state.vorticity,  # pyrefly: ignore[unexpected-keyword]
        divergence=inverse_schur_complement  # pyrefly: ignore[unexpected-keyword]
        * (
            state.divergence
            - step_size * self.coords.horizontal.laplacian(state.potential)
        ),
        potential=inverse_schur_complement  # pyrefly: ignore[unexpected-keyword]
        * (
            -step_size * self.ref_potential * state.divergence + state.potential
        ),
    )


@dataclasses.dataclass
class SemiLagrangianShallowWaterEquations(
    ShallowWaterEquations,
    time_integration.SemiLagrangianImplicitExplicitODE,
):
  """Shallow water equations in semi-Lagrangian form.

  The state layout (modal vorticity, divergence and potential) and the
  implicit terms/inverse are unchanged from `ShallowWaterEquations`, so this
  class plugs into `time_integration.semi_lagrangian_crank_nicolson_rk2`.
  `explicit_terms` raises TypeError so that Eulerian steppers
  (`imex_rk_sil3` etc.), which would silently integrate advection-free
  dynamics, are rejected. All advection is handled by trajectories: momentum
  is transported as grid-point winds (converted from/to modal vorticity and
  divergence at the transport boundaries) and the potential perturbation as
  a scalar, so `nonadvective_terms` returns only the non-advective forcing:

  - the pressure-gradient coupling between layers and orography,
  - the potential's stretching source `-Φ'δ` (the flux-form Eulerian
    tendency `-∇·(vΦ')` minus advection),
  - and, only for `coriolis_mode='explicit'`, the Coriolis term `-f k✕v`.

  Attributes:
    coriolis_mode: 'planetary_momentum' (default) transports the planetary
      momentum `v + 2Ω✕R`, which obeys a momentum equation with no Coriolis
      force — the standard configuration for long time steps. 'explicit'
      keeps `-f k✕v` as an explicit tendency, which is only suitable for
      small `f·dt` (Heun's method has no imaginary-axis stability).
    interpolation_order: horizontal interpolation order for transport
      ('cubic' or 'linear'); trajectories always use linear interpolation.
    departure_iterations: number of fixed-point iterations in the
      departure-point solve (see
      `semi_lagrangian.horizontal_departure_points`). The default single
      iteration relies on the steppers' default warm starts; use at least
      2 if those are disabled.
    monotone_dynamics: if True, the potential and the wind Cartesian
      components are transported with the quasi-monotone limiter (IFS
      applies quasi-monotone interpolation to its dynamical variables);
      off by default, since it adds diffusion near sharp features.
  """

  coriolis_mode: str = 'planetary_momentum'
  interpolation_order: str = 'cubic'
  departure_iterations: int = 1
  monotone_dynamics: bool = False

  def __post_init__(self):
    if self.coriolis_mode not in ('planetary_momentum', 'explicit'):
      raise ValueError(f'unknown {self.coriolis_mode=}')

  @property
  def _interpolator(self) -> semi_lagrangian.GridInterpolator:
    limiter = 'quasi_monotone' if self.monotone_dynamics else None
    return semi_lagrangian.GridInterpolator(
        self.coords.horizontal, self.interpolation_order, limiter
    )

  @property
  def _planetary_rotation_rate(self) -> float | None:
    if self.coriolis_mode == 'planetary_momentum':
      return self.physics_specs.angular_velocity
    return None

  # Reject Eulerian-stepper misuse (see the class docstring): the inherited
  # ShallowWaterEquations.explicit_terms would otherwise take precedence
  # over the interface's raising version in the MRO.
  explicit_terms = (
      time_integration.SemiLagrangianImplicitExplicitODE.explicit_terms
  )

  def nonadvective_terms(self, state: State) -> State:
    """Computes non-advective explicit tendencies ("N")."""
    grid = self.coords.horizontal
    # Pressure gradients from other layers and orography; the own-layer
    # potential gradient is the implicit term.
    p = einsum('ab,...bml->...aml', self.density_ratios, state.potential)
    if self.orography is not None:
      p = p + self.orography
    explicit_vorticity = jnp.zeros_like(state.vorticity)
    explicit_divergence = -grid.laplacian(p)
    if self.coriolis_mode == 'explicit':
      u = jnp.stack(
          spherical_harmonic.get_cos_lat_vector(
              state.vorticity, state.divergence, grid
          )
      )
      nodal_u = grid.to_nodal(u)
      nodal_b = nodal_u * self.coriolis_parameter * grid.sec2_lat
      b = grid.to_modal(nodal_b)
      explicit_vorticity = explicit_vorticity - grid.div_cos_lat(b)
      explicit_divergence = explicit_divergence + grid.curl_cos_lat(b)
    # DΦ'/Dt = -Φ'δ - Φ_ref δ; the first term is the explicit source and the
    # second is the implicit term. Clipping before the nodal conversion
    # matches the Eulerian class's state_to_nodal convention.
    nodal_divergence = grid.to_nodal(grid.clip_wavenumbers(state.divergence))
    nodal_potential = grid.to_nodal(grid.clip_wavenumbers(state.potential))
    explicit_potential = -grid.to_modal(nodal_potential * nodal_divergence)
    return grid.clip_wavenumbers(
        State(explicit_vorticity, explicit_divergence, explicit_potential)
    )

  def nodal_velocities(self, state: State) -> tuple[Array, Array]:
    return spherical_harmonic.vor_div_to_uv_nodal(
        self.coords.horizontal, state.vorticity, state.divergence, clip=False
    )

  def departure_points(
      self,
      velocities: tuple[Array, Array],
      dt: float,
      initial_guess: semi_lagrangian.DeparturePoints | None = None,
  ) -> semi_lagrangian.DeparturePoints:
    u, v = velocities
    return semi_lagrangian.horizontal_departure_points(
        u,
        v,
        self.coords.horizontal,
        dt=dt,
        iterations=self.departure_iterations,
        initial_guess=initial_guess,
    )

  def semi_lagrangian_transport(
      self, state: State, departure: semi_lagrangian.DeparturePoints
  ) -> State:
    grid = self.coords.horizontal
    interpolator = self._interpolator
    u, v = spherical_harmonic.vor_div_to_uv_nodal(
        grid, state.vorticity, state.divergence, clip=False
    )
    u, v = semi_lagrangian.transport_wind_2d(
        u,
        v,
        departure,
        interpolator,
        planetary_rotation_rate=self._planetary_rotation_rate,
        limiter=interpolator.limiter,
    )
    nodal_potential = grid.to_nodal(state.potential)
    transported_potential = semi_lagrangian.transport_scalar_2d(
        nodal_potential, departure, interpolator
    )
    vorticity, divergence = spherical_harmonic.uv_nodal_to_vor_div_modal(
        grid, u, v, clip=False
    )
    return grid.clip_wavenumbers(
        State(vorticity, divergence, grid.to_modal(transported_potential))
    )


def shallow_water_leapfrog_step(
    coords: coordinate_systems.CoordinateSystem,
    dt: float,
    physics_specs: units.SimUnitsProtocol,
    mean_potential: np.ndarray,
    orography: Array | None = None,
    densities: np.ndarray = np.ones((1,)),
    alpha: float = 0.5,
) -> typing.TimeStepFn:
  """Returns a step function based on semi-implicit leapfrog integrator.

  Args:
    coords: horizontal and vertical descritization.
    dt: the size of the timestep used for integration.
    physics_specs: an `PrimitiveEquationSpecs` object describing the scales and
      physical constants used in the primitive equations.
    mean_potential: a vector of mean geopotentials g · h for each layer,
      starting from the top.
    orography: the geopotential g · h corresponding to the orography underlying
      the simulation. Must be in the spectral/modal basis.
    densities: a vector of densities for each layer.
    alpha: a parameter used to weight previous and future terms in the implicit
      portion of the equation: `f_i(alpha * future + (1 - alpha) * previous)`

  Returns:
    A function that computes a single time step of the shallow water equations.
    The returned function takes `state_0` and `state_1` states and returns the
    next state.
  """
  shallow_water_ode = ShallowWaterEquations(
      coords, physics_specs, orography, mean_potential, densities=densities  # pyrefly: ignore[bad-argument-type]
  )
  return time_integration.semi_implicit_leapfrog(shallow_water_ode, dt, alpha)


def shallow_water_leapfrog_trajectory(
    coords: coordinate_systems.CoordinateSystem,
    dt: float,
    physics_specs: units.SimUnitsProtocol,
    inner_steps: int,
    outer_steps: int,
    mean_potential: np.ndarray,
    orography: Array | None = None,
    densities: np.ndarray = np.ones((1,)),
    filters: Sequence[typing.PyTreeStepFilterFn] = (),
    alpha: float = 0.5,
) -> typing.TrajectoryFn:
  """Returns a trajectory function for shallow water equations."""
  step_fn = shallow_water_leapfrog_step(
      coords, dt, physics_specs, mean_potential, orography, densities, alpha
  )
  step_fn = time_integration.step_with_filters(step_fn, filters)
  post_process_fn = lambda x: x[0]
  trajectory_fn = time_integration.trajectory_from_step(
      step_fn, outer_steps, inner_steps, post_process_fn=post_process_fn
  )
  return trajectory_fn


def default_filters(
    grid: spherical_harmonic.Grid,
    dt: float,
) -> Sequence[typing.PyTreeStepFilterFn]:
  """Returns standard filters for leapfrog integration of shallow water Eqs."""
  return (
      time_integration.exponential_leapfrog_step_filter(grid, dt),
      time_integration.robert_asselin_leapfrog_filter(0.05),
  )
