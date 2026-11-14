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

"""Integration tests for primitive_equations."""
import functools

from absl.testing import absltest
import chex
from dinosaur import coordinate_systems
from dinosaur import hybrid_coordinates
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import pytree_utils
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import units
from dinosaur import xarray_utils
import jax
from jax import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
import numpy as np

try:
  chex.set_n_cpu_devices(8)  # set at top level so standalone pytest works
except RuntimeError:
  pass


def make_coords(
    max_wavenumber: int,
    num_layers: int,
    spmd_mesh: jax.sharding.Mesh | None,
    spherical_harmonics_impl,
) -> coordinate_systems.CoordinateSystem:
  return coordinate_systems.CoordinateSystem(
      spherical_harmonic.Grid.with_wavenumbers(
          longitude_wavenumbers=max_wavenumber + 1,
          spherical_harmonics_impl=spherical_harmonics_impl,
      ),
      sigma_coordinates.SigmaCoordinates.equidistant(num_layers),
      spmd_mesh=spmd_mesh,
  )


def make_initial_state(coords, physics_specs):
  initial_state_fn, _ = primitive_equations_states.steady_state_jw(
      coords, physics_specs
  )
  state = initial_state_fn()
  state = state + primitive_equations_states.baroclinic_perturbation_jw(
      coords, physics_specs
  )
  state.tracers = {
      'specific_humidity': primitive_equations_states.gaussian_scalar(
          coords, physics_specs
      )
  }
  state.sim_time = 0.0
  return state


def make_dycore_sim_fn(coords, physics_specs, num_hours):
  _, aux_features = primitive_equations_states.steady_state_jw(
      coords, physics_specs
  )
  ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
  orography = np.zeros(coords.horizontal.modal_shape, np.float32)
  eq = primitive_equations.PrimitiveEquationsSigma(
      ref_temps,
      orography,
      coords,
      physics_specs,
      humidity_key='specific_humidity',
  )
  dt = physics_specs.nondimensionalize(30 * scales.units.minute)
  step_fn = time_integration.imex_rk_sil3(eq, time_step=dt)
  sim_fn = time_integration.repeated(step_fn, 2 * num_hours)
  return jax.jit(sim_fn)


def assert_allclose(actual, desired, *, atol=0, range_tol=0, **kwargs):
  atol = max(atol, range_tol * (desired.max() - desired.min()))
  return np.testing.assert_allclose(actual, desired, atol=atol, **kwargs)


def assert_states_close(state0, state1, *, atol=0, range_tol=0, **kwargs):
  for field in state0.fields:
    if field.name == 'tracers':
      for tracer_name in state0.tracers.keys():
        assert_allclose(
            state0.tracers[tracer_name],
            state1.tracers[tracer_name],
            atol=atol,
            range_tol=range_tol,
            err_msg=f'Mismatch in tracer {tracer_name}:',
            **kwargs,
        )
    else:
      assert_allclose(
          getattr(state0, field.name),
          getattr(state1, field.name),
          atol=atol,
          range_tol=range_tol,
          err_msg=f'Mismatch in {field}:',
          **kwargs,
      )


def pad_state(state, distributed_coords):
  def f(x):
    pad_x, pad_y = (
        distributed_coords.horizontal.spherical_harmonics.modal_padding
    )
    return jnp.pad(x, [(0, 0), (0, pad_x), (0, pad_y)])

  return pytree_utils.tree_map_over_nonscalars(f, state)


def trim_state(state, distributed_coords):
  def f(x):
    pad_x, pad_y = (
        distributed_coords.horizontal.spherical_harmonics.modal_padding
    )
    return x[..., : -pad_x or None, : -pad_y or None]

  return pytree_utils.tree_map_over_nonscalars(f, state)


class IntegrationTest(absltest.TestCase):

  def test_distributed_simulation_consistency(self):
    physics_specs = units.SimUnits.from_si()
    coords = make_coords(
        max_wavenumber=31,
        num_layers=8,
        spmd_mesh=None,
        spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
    )
    init_state = make_initial_state(coords, physics_specs)
    sim_fn = make_dycore_sim_fn(coords, physics_specs, num_hours=1)
    non_distributed_state = sim_fn(init_state)
    non_distributed_nodal = coords.horizontal.to_nodal(non_distributed_state)

    with self.subTest('vertical sharding'):
      devices = np.array(jax.devices()[:2]).reshape((2, 1, 1))
      mesh = jax.sharding.Mesh(devices, axis_names=['z', 'x', 'y'])
      distributed_coords = make_coords(
          max_wavenumber=31,
          num_layers=8,
          spmd_mesh=mesh,
          spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
      )
      distributed_init_state = pad_state(init_state, distributed_coords)
      distributed_sim_fn = make_dycore_sim_fn(
          distributed_coords, physics_specs, num_hours=1
      )
      distributed_state = distributed_sim_fn(distributed_init_state)
      distributed_state = trim_state(distributed_state, distributed_coords)
      distributed_nodal = coords.horizontal.to_nodal(distributed_state)

      assert_states_close(
          non_distributed_nodal, distributed_nodal, rtol=1e-6, range_tol=1e-6
      )

    with self.subTest('horizontal sharding'):
      devices = np.array(jax.devices()[:4]).reshape((1, 2, 2))
      mesh = jax.sharding.Mesh(devices, axis_names=['z', 'x', 'y'])
      distributed_coords = make_coords(
          max_wavenumber=31,
          num_layers=8,
          spmd_mesh=mesh,
          spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
      )
      distributed_init_state = pad_state(init_state, distributed_coords)
      distributed_sim_fn = make_dycore_sim_fn(
          distributed_coords, physics_specs, num_hours=1
      )
      distributed_state = distributed_sim_fn(distributed_init_state)
      distributed_state = trim_state(distributed_state, distributed_coords)
      distributed_nodal = coords.horizontal.to_nodal(distributed_state)

      assert_states_close(
          non_distributed_nodal, distributed_nodal, rtol=1e-6, range_tol=1e-6
      )

  def test_real_vs_fast_spherical_harmonics(self):
    physics_specs = units.SimUnits.from_si()

    real_coords = make_coords(
        max_wavenumber=31,
        num_layers=8,
        spmd_mesh=None,
        spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics,
    )
    fast_coords = make_coords(
        max_wavenumber=31,
        num_layers=8,
        spmd_mesh=None,
        spherical_harmonics_impl=functools.partial(
            spherical_harmonic.FastSphericalHarmonics,
            transform_precision='float32',
        ),
    )

    real_init_state = make_initial_state(real_coords, physics_specs)
    fast_init_state = make_initial_state(fast_coords, physics_specs)

    real_init_nodal = real_coords.horizontal.to_nodal(real_init_state)
    fast_init_nodal = fast_coords.horizontal.to_nodal(fast_init_state)

    with self.subTest('initial conditions'):
      assert_states_close(
          real_init_nodal, fast_init_nodal, rtol=1e-6, range_tol=1e-6
      )

    sim_fn = make_dycore_sim_fn(real_coords, physics_specs, num_hours=1)
    real_out_state = sim_fn(real_init_state)
    real_out_nodal = real_coords.horizontal.to_nodal(real_out_state)

    sim_fn = make_dycore_sim_fn(fast_coords, physics_specs, num_hours=1)
    fast_out_state = sim_fn(fast_init_state)
    fast_out_nodal = fast_coords.horizontal.to_nodal(fast_out_state)

    with self.subTest('evolved state'):
      assert_states_close(
          real_out_nodal, fast_out_nodal, rtol=1e-4, range_tol=1e-5
      )

  def test_semi_lagrangian_distributed_consistency(self):
    physics_specs = units.SimUnits.from_si()
    num_layers = 16
    nodal_tracer_names = ('nodal_a', 'nodal_b', 'nodal_c')

    def build_sl_coords(mesh):
      return coordinate_systems.CoordinateSystem(
          spherical_harmonic.Grid.T31(
              spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
              radius=physics_specs.radius,
          ),
          sigma_coordinates.SigmaCoordinates.equidistant(num_layers),
          spmd_mesh=mesh,
      )

    coords = build_sl_coords(None)
    init_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    init_state = init_state_fn() + (
        primitive_equations_states.baroclinic_perturbation_jw(
            coords, physics_specs
        )
    )
    base_modal_tracer = primitive_equations_states.gaussian_scalar(
        coords, physics_specs
    )
    base_nodal_tracer = jnp.maximum(
        coords.horizontal.to_nodal(base_modal_tracer), 0.0
    )
    init_state.tracers = {
        'specific_humidity': base_modal_tracer,
        'modal_passive': 0.5 * base_modal_tracer,
        'nodal_a': base_nodal_tracer,
        'nodal_b': 0.3 * jnp.roll(base_nodal_tracer, shift=8, axis=1),
        'nodal_c': 0.7 * jnp.roll(base_nodal_tracer, shift=4, axis=2),
    }
    init_state.sim_time = 0.0

    # Non-zero unpadded nodal orography exercises terrain_smoothed_log_sp and
    # automatic nodal padding handling.
    lon_m, sin_lat_m = coords.horizontal.nodal_mesh
    mountain_height = physics_specs.nondimensionalize(500 * scales.units.meter)
    nodal_orography = (
        mountain_height
        * np.exp(
            -(
                ((lon_m[:96, :48] - 2.0) / 0.5) ** 2
                + (sin_lat_m[:96, :48] / 0.3) ** 2
            )
        )
    ).astype(np.float32)
    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    dt = physics_specs.nondimensionalize(30 * scales.units.minute)

    def make_sl_sim_fn(c, method=None):
      orog = primitive_equations.truncated_modal_orography(nodal_orography, c)
      eq = primitive_equations.SemiLagrangianPrimitiveEquations(
          ref_temps,
          orog,
          c,
          physics_specs,
          humidity_key='specific_humidity',
          nodal_tracers=nodal_tracer_names,
          monotone_tracers=True,
          monotone_dynamics=True,
          vertical_interpolation_order='cubic',
          interpolation_method=method,
      )
      step_fn = time_integration.semi_lagrangian_crank_nicolson_rk2(
          eq, time_step=dt
      )
      sim_fn = time_integration.repeated(step_fn, 2)
      return jax.jit(lambda s: c.with_dycore_sharding(sim_fn(c.with_dycore_sharding(s))))

    def pad_sl_state(state, dist_coords):
      mpad_x, mpad_y = dist_coords.horizontal.modal_padding
      npad_x, npad_y = dist_coords.horizontal.nodal_padding
      pad_modal = lambda x: jnp.pad(x, [(0, 0), (0, mpad_x), (0, mpad_y)])
      pad_nodal = lambda x: jnp.pad(x, [(0, 0), (0, npad_x), (0, npad_y)])
      modal_part = state.replace(
          tracers={
              k: v for k, v in state.tracers.items() if k not in nodal_tracer_names
          }
      )
      padded = pytree_utils.tree_map_over_nonscalars(pad_modal, modal_part)
      padded_nodal = {
          k: pad_nodal(state.tracers[k]) for k in nodal_tracer_names
      }
      return padded.replace(tracers={**padded.tracers, **padded_nodal})

    def to_unpadded_nodal(state, dist_coords=None):
      state = jax.device_get(state)
      if dist_coords is not None:
        mpad_x, mpad_y = dist_coords.horizontal.modal_padding
        npad_x, npad_y = dist_coords.horizontal.nodal_padding
        trim_m = lambda x: x[..., : -mpad_x or None, : -mpad_y or None]
        trim_n = lambda x: x[..., : -npad_x or None, : -npad_y or None]
      else:
        trim_m = lambda x: x
        trim_n = lambda x: x
      modal_only = state.replace(
          tracers={
              k: v for k, v in state.tracers.items() if k not in nodal_tracer_names
          }
      )
      nodal_state = coords.horizontal.to_nodal(
          pytree_utils.tree_map_over_nonscalars(trim_m, modal_only)
      )
      nodal_tr = {
          k: trim_n(state.tracers[k]) for k in nodal_tracer_names
      }
      return nodal_state.replace(tracers={**nodal_state.tracers, **nodal_tr})

    ref_gather_out = to_unpadded_nodal(
        make_sl_sim_fn(coords, method='gather')(init_state)
    )
    ref_matmul_out = to_unpadded_nodal(
        make_sl_sim_fn(coords, method='matmul')(init_state)
    )
    with self.subTest('gather vs matmul unsharded'):
      assert_states_close(
          ref_gather_out, ref_matmul_out, rtol=2e-4, range_tol=2e-4, atol=1e-5
      )

    mesh_shapes = [
        ('vertical_halo_ppermute', (2, 1, 1)),
        ('horizontal_halo_ppermute', (1, 2, 4)),
        ('full_3d_sharding', (2, 2, 2)),
        ('lat_8_way_sharding', (1, 1, 8)),
    ]
    for name, shape in mesh_shapes:
      with self.subTest(name):
        n_dev = int(np.prod(shape))
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()[:n_dev]).reshape(shape),
            axis_names=['z', 'x', 'y'],
        )
        dist_coords = build_sl_coords(mesh)
        dist_init = pad_sl_state(init_state, dist_coords)
        dist_out = make_sl_sim_fn(dist_coords)(dist_init)
        dist_nodal = to_unpadded_nodal(dist_out, dist_coords)
        assert_states_close(
            ref_matmul_out, dist_nodal, rtol=1e-4, range_tol=1e-4, atol=1e-5
        )
        for tr_name in nodal_tracer_names:
          self.assertGreaterEqual(
              float(np.min(np.asarray(dist_nodal.tracers[tr_name]))), -1e-7
          )

  def test_semi_lagrangian_hybrid_distributed_and_grad(self):
    physics_specs = units.SimUnits.from_si()
    num_layers = 16
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        num_layers, p_top=10.0
    )

    def build_hybrid_coords(mesh):
      return coordinate_systems.CoordinateSystem(
          spherical_harmonic.Grid.T31(
              spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
              radius=physics_specs.radius,
          ),
          hybrid_levels,
          spmd_mesh=mesh,
      )

    coords = build_hybrid_coords(None)
    init_state_fn, features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    init_state = init_state_fn() + (
        primitive_equations_states.baroclinic_perturbation_jw(
            coords, physics_specs
        )
    )
    q_modal = primitive_equations_states.gaussian_scalar(coords, physics_specs)
    aerosol = jnp.maximum(coords.horizontal.to_nodal(q_modal), 0.0)
    init_state.tracers = {
        'specific_humidity': q_modal,
        'aerosol_0': aerosol,
        'aerosol_1': 0.5 * aerosol,
    }
    init_state.sim_time = 0.0
    nodal_orography = features[xarray_utils.OROGRAPHY][:96, :48]
    ref_temps = features[xarray_utils.REF_TEMP_KEY]
    dt = physics_specs.nondimensionalize(30 * scales.units.minute)

    def make_step_and_loss(c):
      orog = primitive_equations.truncated_modal_orography(nodal_orography, c)
      eq = primitive_equations.SemiLagrangianPrimitiveEquationsHybrid(
          ref_temps,
          orog,
          c,
          physics_specs,
          humidity_key='specific_humidity',
          nodal_tracers=('aerosol_0', 'aerosol_1'),
          monotone_tracers=True,
          vertical_interpolation_order='cubic',
          interpolation_method='matmul',
      )
      step_fn = time_integration.semi_lagrangian_crank_nicolson_rk2(
          eq, time_step=dt
      )

      def run_step(s):
        return c.with_dycore_sharding(step_fn(c.with_dycore_sharding(s)))

      def loss_fn(aerosol_in):
        s = init_state if c.spmd_mesh is None else dist_init
        s = s.replace(tracers={**s.tracers, 'aerosol_0': aerosol_in})
        out = run_step(s)
        return jnp.sum(out.tracers['aerosol_0'] ** 2)

      return jax.jit(run_step), jax.jit(jax.grad(loss_fn))

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:8]).reshape((2, 2, 2)),
        axis_names=['z', 'x', 'y'],
    )
    dist_coords = build_hybrid_coords(mesh)
    mpad_x, mpad_y = dist_coords.horizontal.modal_padding
    pad_m = lambda x: jnp.pad(x, [(0, 0), (0, mpad_x), (0, mpad_y)])
    modal_only = init_state.replace(
        tracers={'specific_humidity': init_state.tracers['specific_humidity']}
    )
    dist_init = pytree_utils.tree_map_over_nonscalars(pad_m, modal_only)
    dist_init = dist_init.replace(
        tracers={
            **dist_init.tracers,
            'aerosol_0': init_state.tracers['aerosol_0'],
            'aerosol_1': init_state.tracers['aerosol_1'],
        }
    )

    ref_step, ref_grad_fn = make_step_and_loss(coords)
    dist_step, dist_grad_fn = make_step_and_loss(dist_coords)

    ref_out = jax.device_get(ref_step(init_state))
    dist_out = jax.device_get(dist_step(dist_init))
    assert_allclose(
        np.asarray(coords.horizontal.to_nodal(ref_out.temperature_variation)),
        np.asarray(
            coords.horizontal.to_nodal(
                dist_out.temperature_variation[
                    ..., : -mpad_x or None, : -mpad_y or None
                ]
            )
        ),
        rtol=2e-5,
        range_tol=3e-5,
    )
    np.testing.assert_allclose(
        ref_out.tracers['aerosol_0'],
        dist_out.tracers['aerosol_0'],
        rtol=1e-5,
        atol=1e-6,
    )

    grad_ref = ref_grad_fn(init_state.tracers['aerosol_0'])
    grad_dist = dist_grad_fn(dist_init.tracers['aerosol_0'])
    self.assertTrue(np.all(np.isfinite(np.asarray(grad_dist))))
    self.assertGreater(float(np.max(np.abs(np.asarray(grad_dist)))), 0.0)
    np.testing.assert_allclose(grad_dist, grad_ref, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
