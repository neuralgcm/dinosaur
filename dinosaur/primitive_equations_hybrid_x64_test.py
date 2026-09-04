# Copyright 2025 Google LLC
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
"""Double-precision checks of the primitive equations on hybrid levels.

The tests pin `PrimitiveEquationsHybrid` to the vertical discretization of
Simmons & Burridge (1981): an independent transcription of their tendencies
and the discrete total energy they conserve. Both checks compare small
residuals of large terms, so they need double precision. JAX handles the x64
flag reliably only when set once at import time, so these tests live in
their own module (see `semi_lagrangian_x64_test.py`).
"""

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import hybrid_coordinates
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import units
from dinosaur import xarray_utils
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

FIELDS = (
    'vorticity',
    'divergence',
    'temperature_variation',
    'log_surface_pressure',
)


def simmons_burridge_reference_tendencies(
    state: primitive_equations.State,
    equation: primitive_equations.PrimitiveEquationsHybrid,
) -> primitive_equations.State:
  """Total dry tendencies on hybrid levels, written directly from S&B (1981).

  Every vertical term (interface mass flux, vertical advection, hydrostatic
  geopotential, pressure-gradient and energy-conversion terms) is evaluated
  at the actual surface pressure with no reference-state linearization, so
  the sum of the model's explicit and implicit terms must reproduce it. The
  horizontal operators and the horizontally uniform part of the
  pressure-gradient term (reference temperature at the reference surface
  pressure, applied spectrally as in the model) are shared with the model,
  so the comparison isolates the vertical discretization.
  """
  grid = equation.coords.horizontal
  levels = equation.nondim_levels
  physics_specs = equation.physics_specs
  R, kappa = physics_specs.R, physics_specs.kappa
  a, b = levels.a_boundaries, levels.b_boundaries
  db = np.diff(b)[:, np.newaxis, np.newaxis]
  aux = primitive_equations.compute_diagnostic_state_hybrid(
      state, equation.nondim_coords
  )
  surface_pressure = jnp.exp(grid.to_nodal(state.log_surface_pressure))

  def coefficients(ps):
    """Returns the S&B layer coefficients for surface pressure `ps`."""
    # layer thickness, ln(p_{k+1/2} / p_{k-1/2}), alpha_k, and the weight of
    # grad(ln ps) in the discrete grad(ln p_k) (S&B eqs. 2.10-2.11)
    p_half = a[:, np.newaxis, np.newaxis] + b[:, np.newaxis, np.newaxis] * ps
    dp = p_half[1:] - p_half[:-1]
    p_top = p_half[:-1]
    # The log ratio is unused for a p = 0 model top, where this adopts the
    # model's `alpha_1 = 1` convention (S&B and the IFS use ln 2 instead), so
    # the top-layer convention itself is not checked by this reference.
    safe_ratio = p_half[1:] / jnp.where(p_top > 0, p_top, 1.0)
    dlnp = jnp.where(p_top > 0, jnp.log(safe_ratio), 0.0)
    alpha = 1 - p_top / dp * dlnp
    weight = dlnp * b[:-1, np.newaxis, np.newaxis] + alpha * db
    return dp, dlnp, alpha, weight

  dp, dlnp, alpha, weight = coefficients(surface_pressure)
  divergence = aux.divergence
  u_dot_grad_log_sp = aux.u_dot_grad_log_sp
  # div(v dp) per layer, its cumulative sum and the interface mass flux
  # eta_dot dp/deta (S&B eqs. 2.5-2.6).
  mass_divergence = dp * divergence + db * surface_pressure * u_dot_grad_log_sp
  cumulative = jnp.cumsum(mass_divergence, axis=0)
  total = cumulative[-1:]
  mass_flux = b[:, np.newaxis, np.newaxis] * total - jnp.concatenate(
      [jnp.zeros_like(total), cumulative]
  )

  def vertical_advection(x):
    flux = mass_flux[1:-1] * (x[1:] - x[:-1])
    zero = jnp.zeros_like(flux[:1])
    padded = jnp.concatenate([zero, flux, zero])
    return -0.5 * (padded[1:] + padded[:-1]) / dp

  # omega / p (S&B eq. 2.13), thermodynamic equation
  omega_over_p = (
      surface_pressure / dp * weight * u_dot_grad_log_sp
      - (dlnp * (cumulative - mass_divergence) + alpha * mass_divergence) / dp
  )
  temperature = equation.T_ref + aux.temperature_variation
  if equation.humidity_key is None:
    virtual_temperature = heating_temperature = temperature
  else:
    # virtual temperature (less condensate) in the hydrostatic and
    # pressure-gradient terms; the moist thermodynamic equation heats with
    # the virtual temperature and the moist heat capacity.
    q = aux.tracers[equation.humidity_key]
    condensate = sum(aux.tracers[k] for k in equation.cloud_keys or ())
    gas_const_ratio = physics_specs.R_vapor / physics_specs.R
    heat_capacity_ratio = physics_specs.Cp_vapor / physics_specs.Cp
    virtual_temperature = temperature * (
        1 + (gas_const_ratio - 1) * q - condensate
    )
    heating_temperature = (
        temperature
        * (1 + (gas_const_ratio - 1) * q)
        / (1 + (heat_capacity_ratio - 1) * q)
    )
  nodal, modal = equation.horizontal_scalar_advection(
      aux.temperature_variation, aux_state=aux
  )
  temperature_tendency = (
      grid.to_modal(
          nodal
          + vertical_advection(temperature)
          + kappa * heating_temperature * omega_over_p
      )
      + modal
  )
  # hydrostatic geopotential above the surface (S&B eq. 2.9) and the
  # coefficient of grad(ln ps) in the pressure-gradient force (eq. 2.11)
  t_dlnp = virtual_temperature * dlnp
  geopotential = R * (
      jnp.cumsum(t_dlnp[::-1], axis=0)[::-1]
      - t_dlnp
      + alpha * virtual_temperature
  )
  pgf_coefficient = R * virtual_temperature / dp * weight * surface_pressure
  dp_ref, _, _, weight_ref = coefficients(equation.p_s_ref)
  pgf_coefficient_ref = (
      R * equation.T_ref / dp_ref * weight_ref * equation.p_s_ref
  )
  # momentum equations in vector-invariant form
  sec2_lat = grid.sec2_lat
  cos_lat_u, cos_lat_v = aux.cos_lat_u
  grad_x, grad_y = aux.cos_lat_grad_log_sp
  absolute_vorticity = aux.vorticity + equation.coriolis_parameter
  pgf_explicit = pgf_coefficient - pgf_coefficient_ref
  vector_u = (
      -cos_lat_v * absolute_vorticity
      - vertical_advection(cos_lat_u)
      + pgf_explicit * grad_x
  ) * sec2_lat
  vector_v = (
      cos_lat_u * absolute_vorticity
      - vertical_advection(cos_lat_v)
      + pgf_explicit * grad_y
  ) * sec2_lat
  modal_vector = (grid.to_modal(vector_u), grid.to_modal(vector_v))
  kinetic_energy = (cos_lat_u**2 + cos_lat_v**2) * sec2_lat / 2
  vorticity_tendency = -grid.curl_cos_lat(modal_vector, clip=False)
  divergence_tendency = (
      -grid.div_cos_lat(modal_vector, clip=False)
      - grid.laplacian(grid.to_modal(kinetic_energy + geopotential))
      - grid.laplacian(pgf_coefficient_ref * state.log_surface_pressure)
      - physics_specs.g * grid.laplacian(equation.orography)
  )
  tendency = primitive_equations.State(
      vorticity=vorticity_tendency,
      divergence=divergence_tendency,
      temperature_variation=temperature_tendency,
      log_surface_pressure=grid.to_modal(-total / surface_pressure),
      tracers={k: jnp.zeros_like(v) for k, v in state.tracers.items()},
  )
  return grid.clip_wavenumbers(tendency)


def total_energy_tendency(
    state: primitive_equations.State,
    tendency: primitive_equations.State,
    equation: primitive_equations.PrimitiveEquationsBase,
) -> tuple[float, float]:
  """Returns (dE/dt, E) for the total energy E = ∫ Σ_k Δp_k (K + cp T) + ps Φs.

  This is the discrete total energy that the vertical discretization of
  Simmons & Burridge (1981) conserves exactly, evaluated from the model's
  own tendencies (so no time-integration error enters).
  """
  grid = equation.coords.horizontal
  physics_specs = equation.physics_specs
  levels = equation.coords.vertical
  if isinstance(levels, hybrid_coordinates.HybridCoordinates):
    levels = equation.nondim_levels
    a, b = levels.a_boundaries, levels.b_boundaries
  else:
    a, b = np.zeros(levels.layers + 1), levels.boundaries
  da = np.diff(a)[:, np.newaxis, np.newaxis]
  db = np.diff(b)[:, np.newaxis, np.newaxis]
  u, v = spherical_harmonic.vor_div_to_uv_nodal(
      grid, state.vorticity, state.divergence, clip=False
  )
  du, dv = spherical_harmonic.vor_div_to_uv_nodal(
      grid, tendency.vorticity, tendency.divergence, clip=False
  )
  temperature = grid.to_nodal(state.temperature_variation) + equation.T_ref
  temperature_tendency = grid.to_nodal(tendency.temperature_variation)
  surface_pressure = jnp.exp(grid.to_nodal(state.log_surface_pressure))
  surface_pressure_tendency = surface_pressure * grid.to_nodal(
      tendency.log_surface_pressure
  )
  dp = da + db * surface_pressure
  cp = physics_specs.Cp
  surface_geopotential = physics_specs.g * grid.to_nodal(equation.orography)
  kinetic = (u**2 + v**2) / 2
  static = kinetic + cp * temperature
  column = (
      db * surface_pressure_tendency * static
      + dp * (u * du + v * dv + cp * temperature_tendency)
  ).sum(axis=0) + surface_pressure_tendency[0] * surface_geopotential
  energy = (dp * static).sum(axis=0) + surface_pressure[0] * surface_geopotential
  return float(grid.integrate(column)), float(grid.integrate(energy))


MOISTURE_KEYS = (
    'specific_humidity',
    'specific_cloud_liquid_water_content',
    'specific_cloud_ice_water_content',
)


def hybrid_test_state(
    coords, physics_specs, smooth_divergence=False, moist=False
):
  """A baroclinic-wave state with a mountain in ln(ps) and divergent flow.

  Args:
    coords: coordinate system.
    physics_specs: physical constants.
    smooth_divergence: if True, the added divergence is a large-scale zonally
      symmetric pattern, so that the cubic products in the energy budget are
      integrated exactly by the Gaussian quadrature; otherwise it is
      proportional to the (spectrally rich) vorticity.
    moist: if True, the state carries specific humidity and cloud tracers
      (smooth, positive, decreasing with height).

  Returns:
    A tuple (state, features) with a modal `State` whose surface pressure
    drops to ~740 hPa over a mountain at (90E, 30N).
  """
  grid = coords.horizontal
  state_fn, features = primitive_equations_states.steady_state_jw(
      coords, physics_specs
  )
  state = state_fn() + primitive_equations_states.baroclinic_perturbation_jw(
      coords, physics_specs
  )
  lon, sin_lat = grid.nodal_mesh
  lat = np.arcsin(sin_lat)
  mountain = 0.3 * np.exp(
      -((lon - np.pi / 2) ** 2 + (lat - np.pi / 6) ** 2) / 0.3**2
  )
  log_sp = grid.to_nodal(state.log_surface_pressure) - mountain
  if smooth_divergence:
    layers = coords.vertical.layers
    # mean-free (a divergence integrates to zero over the sphere), with
    # both symmetric and antisymmetric parts so that it correlates with the
    # temperature structure of the baroclinic-wave state.
    pattern = np.sin(lat) * np.cos(lat) ** 2 + (3 * np.sin(lat) ** 2 - 1) / 4
    nodal = np.stack(
        [(0.5 + k / layers) * pattern for k in range(layers)]
    )
    amplitude = physics_specs.nondimensionalize(1e-5 / scales.units.second)
    divergence = state.divergence + amplitude * grid.to_modal(nodal)
  else:
    divergence = state.divergence + 0.2 * state.vorticity
  tracers = {}
  if moist:
    layers = coords.vertical.layers
    profile = np.linspace(0.05, 1.0, layers)[:, np.newaxis, np.newaxis]
    humidity = 0.02 * profile * np.exp(-((lat / 0.5) ** 2)) * (
        1 + 0.3 * np.cos(2 * lon)
    )
    tracers = {
        'specific_humidity': grid.to_modal(humidity),
        'specific_cloud_liquid_water_content': grid.to_modal(0.1 * humidity),
        'specific_cloud_ice_water_content': grid.to_modal(0.01 * humidity),
    }
  state = state.replace(
      log_surface_pressure=grid.to_modal(log_sp),
      divergence=divergence,
      tracers=tracers,
  )
  return grid.clip_wavenumbers(state), features


def relative_error(actual, expected):
  actual, expected = np.asarray(actual), np.asarray(expected)
  return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


LEVELS = dict(
    ecmwf_like=lambda: hybrid_coordinates.HybridCoordinates.ecmwf137_interpolated(
        12
    ),
    analytic=lambda: hybrid_coordinates.HybridCoordinates.analytic_levels(
        12, sigma_exponent=1.5, stretch_exponent=0.5
    ),
    sigma_like=lambda: hybrid_coordinates.HybridCoordinates.from_sigma_levels(
        sigma_coordinates.SigmaCoordinates.equidistant(12)
    ),
)


class SimmonsBurridgeReferenceTest(parameterized.TestCase):

  def _equation(self, levels, moist=False):
    grid = spherical_harmonic.Grid.with_wavenumbers(21)
    coords = coordinate_systems.CoordinateSystem(grid, levels)
    physics_specs = units.SimUnits.from_si()
    state, features = hybrid_test_state(coords, physics_specs, moist=moist)
    moisture = (
        dict(humidity_key=MOISTURE_KEYS[0], cloud_keys=MOISTURE_KEYS[1:])
        if moist
        else {}
    )
    equation = primitive_equations.PrimitiveEquationsHybrid(
        features[xarray_utils.REF_TEMP_KEY],
        primitive_equations.truncated_modal_orography(
            features[xarray_utils.OROGRAPHY], coords
        ),
        coords,
        physics_specs,
        **moisture,
    )
    return equation, state

  @parameterized.named_parameters(
      *(dict(testcase_name=name, levels_fn=fn) for name, fn in LEVELS.items()),
      dict(
          testcase_name='ecmwf_like_moist',
          levels_fn=LEVELS['ecmwf_like'],
          moist=True,
      ),
  )
  def test_total_tendency_matches_reference(self, levels_fn, moist=False):
    """explicit_terms + implicit_terms reproduces the S&B (1981) tendencies.

    The state has a mountain in the surface pressure (so the reference-
    pressure linearization residuals are exercised), divergent flow and a
    level-dependent reference temperature; the moist case couples humidity
    and cloud tracers to the dynamics.
    """
    equation, state = self._equation(levels_fn(), moist=moist)
    grid = equation.coords.horizontal
    expected = simmons_burridge_reference_tendencies(state, equation)
    actual = grid.clip_wavenumbers(
        equation.explicit_terms(state) + equation.implicit_terms(state)
    )
    for field in FIELDS:
      with self.subTest(field):
        self.assertLess(
            relative_error(getattr(actual, field), getattr(expected, field)),
            1e-10,
        )

  def test_implicit_terms_linearize_reference_tendency(self):
    """implicit_terms is the S&B tendency linearized about the reference state.

    About a resting atmosphere at the reference temperature and surface
    pressure, the temperature and log-surface-pressure tendencies are linear
    in the divergence up to terms quadratic in its (small) amplitude.
    """
    grid = spherical_harmonic.Grid.with_wavenumbers(21)
    levels = LEVELS['ecmwf_like']()
    coords = coordinate_systems.CoordinateSystem(grid, levels)
    physics_specs = units.SimUnits.from_si()
    reference_temperature = physics_specs.nondimensionalize(
        (220 + 60 * np.linspace(0, 1, levels.layers)) * scales.units.degK
    )
    equation = primitive_equations.PrimitiveEquationsHybrid(
        reference_temperature,
        np.zeros(grid.modal_shape),
        coords,
        physics_specs,
    )
    lon, sin_lat = grid.nodal_mesh
    pattern = np.cos(3 * lon) * (1 - sin_lat**2) * sin_lat
    nodal_divergence = np.stack(
        [(0.5 + k / levels.layers) * pattern for k in range(levels.layers)]
    )
    amplitude = 1e-6 * physics_specs.nondimensionalize(1 / scales.units.day)
    state = primitive_equations.State(
        vorticity=np.zeros(coords.modal_shape),
        divergence=amplitude * grid.to_modal(nodal_divergence),
        temperature_variation=np.zeros(coords.modal_shape),
        log_surface_pressure=grid.to_modal(
            np.full((1,) + grid.nodal_shape, np.log(equation.p_s_ref))
        ),
    )
    expected = simmons_burridge_reference_tendencies(state, equation)
    actual = equation.implicit_terms(state)
    for field in ['temperature_variation', 'log_surface_pressure']:
      with self.subTest(field):
        self.assertLess(
            relative_error(getattr(actual, field), getattr(expected, field)),
            1e-5,
        )


class VerticalMatmulMethodTest(absltest.TestCase):

  def test_sparse_matches_dense_implicit_terms(self):
    """The matrix-free ('sparse') implicit terms equal the dense ones.

    On hybrid levels the layer thickness varies with height, so the
    cumulative sums of the sparse method must weight the divergence by the
    reference layer thickness.
    """
    levels = LEVELS['ecmwf_like']()
    physics_specs = units.SimUnits.from_si()
    p_s_ref = physics_specs.nondimensionalize(1013.25 * scales.units.hPa)
    nondim_levels = hybrid_coordinates.HybridCoordinates(
        physics_specs.nondimensionalize(
            levels.a_boundaries * scales.units.hPa
        ),
        levels.b_boundaries,
    )
    reference_temperature = physics_specs.nondimensionalize(
        (220 + 60 * np.linspace(0, 1, levels.layers)) * scales.units.degK
    )
    rng = np.random.RandomState(0)
    field = rng.randn(levels.layers, 8, 4)
    with self.subTest('temperature'):
      dense, sparse = (
          primitive_equations.get_temperature_implicit_hybrid(
              field,
              nondim_levels,
              reference_temperature,
              physics_specs.kappa,
              p_s_ref,
              method=method,
          )
          for method in ('dense', 'sparse')
      )
      np.testing.assert_allclose(sparse, dense, rtol=1e-12, atol=0)
    with self.subTest('geopotential'):
      dense, sparse = (
          primitive_equations.get_geopotential_diff_hybrid(
              field, nondim_levels, physics_specs.R, p_s_ref, method=method
          )
          for method in ('dense', 'sparse')
      )
      np.testing.assert_allclose(sparse, dense, rtol=1e-12, atol=0)


class TotalEnergyConservationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='sigma', levels_fn=None),
      *(dict(testcase_name='hybrid_' + name, levels_fn=fn)
        for name, fn in LEVELS.items()),
  )
  def test_tendencies_conserve_total_energy(self, levels_fn):
    """The total tendencies conserve Σ_k Δp_k (K + cp T) + ps Φs.

    Both vertical discretizations (Simmons & Burridge on hybrid levels and
    the sigma-coordinate scheme) conserve this discrete total energy, and the
    spectral transform method integrates its tendency to quadrature error
    for a state whose nonlinear products are smooth, so it must nearly vanish
    (about 1e-9 per day relative to the total energy at this resolution,
    versus 1e-7 to 1e-4 for an inconsistent discretization).
    """
    grid = spherical_harmonic.Grid.with_wavenumbers(21)
    physics_specs = units.SimUnits.from_si()
    if levels_fn is None:
      coords = coordinate_systems.CoordinateSystem(
          grid, sigma_coordinates.SigmaCoordinates.equidistant(12)
      )
      equation_cls = primitive_equations.PrimitiveEquationsSigma
    else:
      coords = coordinate_systems.CoordinateSystem(grid, levels_fn())
      equation_cls = primitive_equations.PrimitiveEquationsHybrid
    state, features = hybrid_test_state(
        coords, physics_specs, smooth_divergence=True
    )
    equation = equation_cls(
        features[xarray_utils.REF_TEMP_KEY],
        primitive_equations.truncated_modal_orography(
            features[xarray_utils.OROGRAPHY], coords
        ),
        coords,
        physics_specs,
    )
    tendency = grid.clip_wavenumbers(
        equation.explicit_terms(state) + equation.implicit_terms(state)
    )
    energy_tendency, energy = total_energy_tendency(state, tendency, equation)
    one_day = physics_specs.nondimensionalize(1 * scales.units.day)
    self.assertLess(abs(energy_tendency) * one_day / energy, 1e-8)


if __name__ == '__main__':
  absltest.main()
