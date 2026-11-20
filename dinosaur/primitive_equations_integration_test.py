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
from unittest import mock

from absl.testing import absltest
import chex
from dinosaur import coordinate_systems
from dinosaur import hybrid_coordinates
from dinosaur import primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import pytree_utils
from dinosaur import scales
from dinosaur import semi_lagrangian
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import time_integration
from dinosaur import units
from dinosaur import xarray_utils
import jax
from jax import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
import numpy as np

_NUM_DEVICES = 8

try:
  chex.set_n_cpu_devices(_NUM_DEVICES)  # at top level for standalone pytest
except RuntimeError:
  # The backend was initialized before this module was imported. That is fine
  # if it already has enough devices; `IntegrationTest.setUp` checks.
  pass


# A sharded grid pads its nodal axes by default (to a multiple of 8 per shard),
# and the semi-Lagrangian interpolation then all-gathers the latitude axis. The
# nearest-neighbour halo exchange needs the unpadded grid, which is requested
# with `nodal_shape_multiple=1`.
_UNPADDED_FAST_HARMONICS = functools.partial(
    spherical_harmonic.FastSphericalHarmonics, nodal_shape_multiple=1
)

# Window half-width of the out-of-window test, and the displacement range (in
# grid cells) of its departure points: beyond the window, by up to 2S + 3.
_SHIFT = 2
_FAR = (3.0, 7.0)


def _displaced_queries(grid, layers, seed, cells, dtype=np.float32):
  """Grid-aligned [layers, *nodal_shape] queries displaced by `cells`.

  Each query is moved from its arrival point by a number of latitude rows and
  of levels drawn uniformly from the interval `cells`, in either direction.
  """
  rng = np.random.RandomState(seed)
  lon_mesh, sin_lat_mesh = grid.nodal_mesh
  phi_mesh = np.arcsin(sin_lat_mesh)
  shape = (layers,) + grid.nodal_shape

  def shifts():
    return rng.uniform(*cells, size=shape) * rng.choice([-1.0, 1.0], size=shape)

  phi = np.clip(
      phi_mesh + shifts() * np.pi / grid.latitude_nodes,
      -np.pi / 2 + 1e-4,
      np.pi / 2 - 1e-4,
  )
  lon = (lon_mesh + rng.uniform(-0.3, 0.3, size=shape)) % (2 * np.pi)
  centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
  sigma = np.clip(
      centers[:, np.newaxis, np.newaxis] + shifts() / layers,
      centers[0] - 0.02,
      centers[-1] + 0.02,
  )
  return tuple(jnp.asarray(x, dtype) for x in (lon, np.sin(phi), sigma))


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

  def setUp(self):
    super().setUp()
    self.assertGreaterEqual(
        len(jax.devices()),
        _NUM_DEVICES,
        f'these tests need {_NUM_DEVICES} devices; if the JAX backend was'
        ' initialized before this module was imported, set'
        f' XLA_FLAGS=--xla_force_host_platform_device_count={_NUM_DEVICES}',
    )

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
              spherical_harmonics_impl=_UNPADDED_FAST_HARMONICS,
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

  def test_semi_lagrangian_window_clamp_sharded(self):
    """Departures beyond the windows are clamped identically under a mesh."""
    num_layers = 16
    shift = _SHIFT

    def build_coords(mesh):
      return coordinate_systems.CoordinateSystem(
          spherical_harmonic.Grid.T31(
              spherical_harmonics_impl=_UNPADDED_FAST_HARMONICS
          ),
          sigma_coordinates.SigmaCoordinates.equidistant(num_layers),
          spmd_mesh=mesh,
      )

    coords = build_coords(None)
    grid = coords.horizontal
    nx, ny = grid.nodal_shape
    centers = coords.vertical.centers
    lon, sin_lat, sigma = _displaced_queries(
        grid, num_layers, seed=3, cells=_FAR
    )
    rng = np.random.RandomState(4)
    fields = jnp.asarray(
        1.0 + np.abs(rng.normal(size=(2, num_layers) + grid.nodal_shape)),
        jnp.float32,
    )
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    cos_lat = np.sqrt(1 - sin_lat_mesh**2)
    k = np.arange(num_layers)[:, np.newaxis, np.newaxis]
    # winds carrying the trajectory midpoints beyond their windows.
    u = jnp.asarray(
        0.6 * cos_lat * (1 + 0.3 * np.sin(2 * lon_mesh + k)), jnp.float32
    )
    v = jnp.asarray(0.6 * cos_lat * np.sin(lon_mesh + 0.5 * k), jnp.float32)
    b = coords.vertical.boundaries[:, np.newaxis, np.newaxis]
    sigma_dot = jnp.asarray(
        np.broadcast_to(
            0.25 * np.sin(np.pi * b) * (1 + 0.5 * np.cos(lon_mesh)),
            (num_layers + 1,) + grid.nodal_shape,
        ),
        jnp.float32,
    )
    kw = dict(
        order='cubic',
        limiter='quasi_monotone',
        vertical_order='cubic',
        max_lat_shift=shift,
        max_vertical_shift=shift,
    )

    def run(c, method, pad):
      g = c.horizontal
      interp_3d = semi_lagrangian.interpolate_3d(
          pad(fields), g, pad(lon), pad(sin_lat), pad(sigma), centers,
          method=method, **kw,
      )
      interp_2d = semi_lagrangian.GridInterpolator(
          g, 'cubic', 'quasi_monotone', method=method, max_lat_shift=shift
      )(pad(fields[0]), pad(lon), pad(sin_lat))
      dep = semi_lagrangian.departure_points_3d(
          pad(u), pad(v), pad(sigma_dot), c, 1.0, iterations=2,
          method=method, max_lat_shift=shift, max_vertical_shift=shift,
      )
      outs = (interp_3d, interp_2d, dep.cartesian, dep.sigma)
      return [np.asarray(x)[..., :nx, :ny] for x in outs]

    ref = run(coords, 'gather', lambda x: x)
    unclamped = semi_lagrangian.interpolate_3d(
        fields, grid, lon, sin_lat, sigma, centers, method='gather',
        **(kw | dict(max_lat_shift=None, max_vertical_shift=None)),
    )
    self.assertGreater(float(np.max(np.abs(unclamped - ref[0]))), 1e-2)
    for name, shape in [
        ('vertical_halo_ppermute', (2, 1, 1)),
        ('horizontal_halo_ppermute', (1, 2, 4)),
        ('full_3d_sharding', (2, 2, 2)),
    ]:
      with self.subTest(name):
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()[: int(np.prod(shape))]).reshape(shape),
            axis_names=['z', 'x', 'y'],
        )
        dist_coords = build_coords(mesh)
        px, py = dist_coords.horizontal.nodal_padding

        def pad(x, px=px, py=py):
          widths = [(0, 0)] * (x.ndim - 2) + [(0, px), (0, py)]
          return jnp.pad(x, widths, mode='edge')

        out = run(dist_coords, None, pad)
        for o, r in zip(out, ref):
          np.testing.assert_allclose(
              o, r, rtol=0, atol=2e-5 * float(np.max(np.abs(r)))
          )

  def test_semi_lagrangian_z_sharded_level_blocks(self):
    """A z-sharded 3-D interpolation matches the unsharded one on TPU layouts.

    On TPU the 'matmul' method evaluates several query levels per block, with
    one vertical window for the block. That window is wider than a single
    level's, and on the first and last z-shard it must stay off the halo
    levels the cyclic exchange fills from the opposite end of the column;
    with a level count that is not a multiple of the block length the last
    block also reaches past the shard's own levels.
    """
    grid_kwargs = dict(spherical_harmonics_impl=_UNPADDED_FAST_HARMONICS)
    grid = spherical_harmonic.Grid.T21(**grid_kwargs)
    cases = [
        # (layers, z-shards, max_vertical_shift, vertical order)
        (32, 4, 6, 'cubic'),  # 8 levels per shard: exactly the halo width.
        (20, 2, 6, 'cubic'),  # 10 levels per shard: not a multiple of 4.
        (40, 4, 6, 'cubic'),  # interior shards with a ragged last block.
        (32, 4, 6, 'linear'),
        (32, 4, 2, 'cubic'),
    ]
    for layers, z_shards, shift, vertical_order in cases:
      centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
      lon, sin_lat, sigma = _displaced_queries(
          grid, layers, seed=layers, cells=(0.0, shift - 0.25)
      )
      rng = np.random.RandomState(layers + z_shards)
      fields = jnp.asarray(
          1.0 + np.abs(rng.normal(size=(3, layers) + grid.nodal_shape)),
          jnp.float32,
      )
      kw = dict(
          order='cubic',
          vertical_order=vertical_order,
          max_lat_shift=6,
          max_vertical_shift=shift,
      )
      mesh = jax.sharding.Mesh(
          np.array(jax.devices()[:z_shards]).reshape((z_shards, 1, 1)),
          axis_names=['z', 'x', 'y'],
      )
      dist_grid = spherical_harmonic.Grid.T21(spmd_mesh=mesh, **grid_kwargs)
      self.assertEqual(dist_grid.nodal_shape, grid.nodal_shape)
      for limiter in (None, 'quasi_monotone'):
        reference = (
            jnp.asarray(np.linspace(220.0, 295.0, layers), jnp.float32)
            if limiter
            else None
        )
        expected = semi_lagrangian.interpolate_3d(
            fields, grid, lon, sin_lat, sigma, centers, method='gather',
            limiter=limiter, limiter_bound_reference=reference, **kw,
        )
        for backend in ('cpu', 'tpu'):
          with self.subTest(
              layers=layers, z_shards=z_shards, shift=shift,
              vertical_order=vertical_order, limiter=limiter, backend=backend,
          ):
            # The platform is read off the mesh's devices, which are CPUs
            # here: `_target_platform` is what selects the TPU layouts.
            with mock.patch.object(
                semi_lagrangian, '_target_platform', return_value=backend
            ):
              jax.clear_caches()
              actual = semi_lagrangian.interpolate_3d(
                  fields, dist_grid, lon, sin_lat, sigma, centers,
                  method='matmul', limiter=limiter,
                  limiter_bound_reference=reference, **kw,
              )
            np.testing.assert_allclose(
                actual, expected, rtol=0,
                atol=2e-5 * float(np.max(np.abs(expected))),
            )
    jax.clear_caches()

  def test_semi_lagrangian_mesh_with_non_spatial_axes(self):
    """Only spatial sharding selects the sharded interpolation."""
    layers = 8
    grid_kwargs = dict(spherical_harmonics_impl=_UNPADDED_FAST_HARMONICS)
    grid = spherical_harmonic.Grid.T21(**grid_kwargs)
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    lon, sin_lat, sigma = _displaced_queries(
        grid, layers, seed=11, cells=(0.0, 1.75)
    )
    fields = jnp.asarray(
        np.random.RandomState(12).normal(size=(2, layers) + grid.nodal_shape),
        jnp.float32,
    )
    kw = dict(vertical_order='cubic', max_lat_shift=2, max_vertical_shift=2)
    expected = semi_lagrangian.interpolate_3d(
        fields, grid, lon, sin_lat, sigma, centers, method='gather', **kw
    )

    def mesh_of(shape):
      return jax.sharding.Mesh(
          np.array(jax.devices()[: int(np.prod(shape))]).reshape(shape),
          axis_names=['batch', 'z', 'x', 'y'],
      )

    with self.subTest('batch axis only'):
      batch_grid = spherical_harmonic.Grid.T21(
          spmd_mesh=mesh_of((2, 1, 1, 1)), **grid_kwargs
      )
      self.assertIsNone(semi_lagrangian._spatial_mesh(batch_grid))
      self.assertEqual(
          semi_lagrangian._resolve_interpolation_method(None, batch_grid),
          semi_lagrangian._resolve_interpolation_method(None, grid),
      )
      actual = semi_lagrangian.interpolate_3d(
          fields, batch_grid, lon, sin_lat, sigma, centers, **kw
      )
      np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)

    with self.subTest('batch and spatial axes'):
      mixed_grid = spherical_harmonic.Grid.T21(
          spmd_mesh=mesh_of((2, 1, 1, 2)), **grid_kwargs
      )
      self.assertIsNotNone(semi_lagrangian._spatial_mesh(mixed_grid))
      self.assertEqual(
          semi_lagrangian._resolve_interpolation_method(None, mixed_grid),
          'matmul',
      )
      # The shard_map is manual over 'x' and 'y' only: the batch axis of the
      # mesh stays under automatic sharding.
      actual = semi_lagrangian.interpolate_3d(
          fields, mixed_grid, lon, sin_lat, sigma, centers, **kw
      )
      np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
      actual_2d = semi_lagrangian.GridInterpolator(
          mixed_grid, 'cubic', max_lat_shift=2
      )(fields[0], lon, sin_lat)
      expected_2d = semi_lagrangian.GridInterpolator(
          grid, 'cubic', method='gather', max_lat_shift=2
      )(fields[0], lon, sin_lat)
      np.testing.assert_allclose(actual_2d, expected_2d, rtol=0, atol=2e-5)

  def test_semi_lagrangian_step_with_batch_only_mesh(self):
    """A mesh that shards no spatial axis leaves the model step as it is."""
    physics_specs = units.SimUnits.from_si()

    def build(mesh):
      coords = coordinate_systems.CoordinateSystem(
          spherical_harmonic.Grid.T21(
              spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
              radius=physics_specs.radius,
          ),
          sigma_coordinates.SigmaCoordinates.equidistant(8),
          spmd_mesh=mesh,
      )
      init_state_fn, aux_features = primitive_equations_states.steady_state_jw(
          coords, physics_specs
      )
      lon, sin_lat = coords.horizontal.nodal_mesh
      height = physics_specs.nondimensionalize(500 * scales.units.meter)
      orography = (
          height * np.exp(-(((lon - 2.0) / 0.5) ** 2) - (sin_lat / 0.3) ** 2)
      ).astype(np.float32)
      equation = primitive_equations.SemiLagrangianPrimitiveEquations(
          aux_features[xarray_utils.REF_TEMP_KEY],
          primitive_equations.truncated_modal_orography(orography, coords),
          coords,
          physics_specs,
      )
      state = init_state_fn() + (
          primitive_equations_states.baroclinic_perturbation_jw(
              coords, physics_specs
          )
      )
      state.sim_time = 0.0
      return equation, state

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:2]).reshape((2, 1, 1, 1)),
        axis_names=['batch', 'z', 'x', 'y'],
    )
    equation, state = build(None)
    batch_equation, batch_state = build(mesh)
    # The static terrain correction is still evaluated at trace time.
    self.assertTrue(primitive_equations._can_eval_static_terrain(equation))
    self.assertTrue(
        primitive_equations._can_eval_static_terrain(batch_equation)
    )
    dt = physics_specs.nondimensionalize(30 * scales.units.minute)

    def step(eq, x):
      return jax.jit(
          time_integration.semi_lagrangian_crank_nicolson_rk2(eq, dt)
      )(x)

    assert_states_close(
        step(batch_equation, batch_state), step(equation, state),
        range_tol=1e-5,
    )

  def test_semi_lagrangian_platform_follows_the_mesh(self):
    """The mesh's devices, not the default backend, pick the layouts."""
    grid_kwargs = dict(spherical_harmonics_impl=_UNPADDED_FAST_HARMONICS)
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:2]).reshape((1, 1, 2)),
        axis_names=['z', 'x', 'y'],
    )
    device_platform = jax.devices()[0].platform
    other = 'tpu' if device_platform != 'tpu' else 'cpu'
    with mock.patch.object(jax, 'default_backend', return_value=other):
      self.assertEqual(
          semi_lagrangian._target_platform(
              spherical_harmonic.Grid.T21(spmd_mesh=mesh, **grid_kwargs)
          ),
          device_platform,
      )
      self.assertEqual(
          semi_lagrangian._target_platform(
              spherical_harmonic.Grid.T21(**grid_kwargs)
          ),
          other,
      )

  def test_semi_lagrangian_zero_shift_sharded(self):
    """`max_*_shift=0` (a one-cell window) exchanges a wide enough halo."""
    grid_kwargs = dict(spherical_harmonics_impl=_UNPADDED_FAST_HARMONICS)
    layers = 16
    grid = spherical_harmonic.Grid.T31(**grid_kwargs)
    centers = sigma_coordinates.SigmaCoordinates.equidistant(layers).centers
    # A shift of 0 is a window of one cell, so displace by less than a cell.
    lon, sin_lat, sigma = _displaced_queries(
        grid, layers, seed=7, cells=(0.0, 0.75)
    )
    rng = np.random.RandomState(8)
    fields = jnp.asarray(
        rng.normal(size=(2, layers) + grid.nodal_shape), jnp.float32
    )
    expected_3d = semi_lagrangian.interpolate_3d(
        fields, grid, lon, sin_lat, sigma, centers, method='gather',
        vertical_order='cubic', max_lat_shift=0, max_vertical_shift=0,
    )
    expected_2d = semi_lagrangian.GridInterpolator(
        grid, 'cubic', method='gather', max_lat_shift=0
    )(fields[0], lon, sin_lat)
    for name, shape in [('y', (1, 1, 4)), ('z', (4, 1, 1)), ('zy', (2, 1, 4))]:
      with self.subTest(name):
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()[: int(np.prod(shape))]).reshape(shape),
            axis_names=['z', 'x', 'y'],
        )
        dist_grid = spherical_harmonic.Grid.T31(spmd_mesh=mesh, **grid_kwargs)
        self.assertEqual(dist_grid.nodal_shape, grid.nodal_shape)
        actual_3d = semi_lagrangian.interpolate_3d(
            fields, dist_grid, lon, sin_lat, sigma, centers, method='matmul',
            vertical_order='cubic', max_lat_shift=0, max_vertical_shift=0,
        )
        actual_2d = semi_lagrangian.GridInterpolator(
            dist_grid, 'cubic', method='matmul', max_lat_shift=0
        )(fields[0], lon, sin_lat)
        np.testing.assert_allclose(actual_3d, expected_3d, rtol=0, atol=2e-5)
        np.testing.assert_allclose(actual_2d, expected_2d, rtol=0, atol=2e-5)


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
