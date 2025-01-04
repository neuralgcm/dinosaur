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
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import pytree_utils
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import xarray_utils
import jax
from jax import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
import numpy as np


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
  eq = primitive_equations.MoistPrimitiveEquations(
      ref_temps, orography, coords, physics_specs
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
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
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
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()

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


if __name__ == '__main__':
  chex.set_n_cpu_devices(8)
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
