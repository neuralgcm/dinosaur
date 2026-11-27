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

"""Tests for spherical_harmonic."""

import functools
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import spherical_harmonic
import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update('jax_enable_x64', True)


def _function_0(lat, lon):
  return jnp.cos(lat) ** 4 * jnp.sin(3 * lon)


def _function_1(lat, lon):
  return jnp.cos(lat) ** 4 * jnp.sin(5 * lon) * jnp.cos(5 * lat)


def random_modal_state(grid, seed=0):
  _, l = grid.modal_mesh
  rs = np.random.RandomState(seed)
  array = rs.normal(size=grid.mask.shape)
  if np.issubdtype(grid.spherical_harmonics.modal_dtype, complex):
    array = array + 1j * rs.normal(size=grid.mask.shape)
  array *= grid.mask / (l + 1) ** 2
  return array


class SphericalHarmonicTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          longitude_nodes=64,
          latitude_nodes=32,
          longitude_wavenumbers=32,
          total_wavenumbers=32,
          latitude_spacing='gauss',
      ),
      dict(
          longitude_nodes=117,
          latitude_nodes=13,
          longitude_wavenumbers=45,
          total_wavenumbers=123,
          latitude_spacing='equiangular',
      ),
      dict(
          longitude_nodes=117,
          latitude_nodes=13,
          longitude_wavenumbers=45,
          total_wavenumbers=123,
          latitude_spacing='equiangular_with_poles',
      ),
  )
  def testBasisShapes(self, **params):
    """Tests that the arrays provided by `basis` have the expected shape."""
    spherical_harmonics = spherical_harmonic.RealSphericalHarmonics(**params)
    basis = spherical_harmonics.basis
    longitude_nodes = params['longitude_nodes']
    latitude_nodes = params['latitude_nodes']
    lon_waves, tot_waves = spherical_harmonics.modal_shape
    self.assertEqual((longitude_nodes, lon_waves), basis.f.shape)
    self.assertEqual(
        (lon_waves, latitude_nodes, tot_waves),
        basis.p.shape,
    )
    self.assertIs(basis.p_fwd, basis.p)

    ls_harmonics = spherical_harmonic.RealSphericalHarmonics(
        **params, transform_method='least_squares'
    )
    ls_basis = ls_harmonics.basis
    self.assertEqual(basis.p.shape, ls_basis.p_fwd.shape)
    self.assertTrue(np.all(np.isfinite(ls_basis.p_fwd)))

  def testBasisJaxArrayDeduplicationForQuadrature(self):
    harmonics = spherical_harmonic.FastSphericalHarmonics(
        longitude_wavenumbers=16,
        total_wavenumbers=17,
        longitude_nodes=32,
        latitude_nodes=17,
        latitude_spacing='equiangular_with_poles',
        basis_as_jax_arrays=True,
    )
    self.assertIs(harmonics.basis.p_fwd, harmonics.basis.p)

  def testInvalidTransformMethodRaises(self):
    invalid_method: Any = 'invalid'
    with self.assertRaisesRegex(ValueError, 'Unsupported `transform_method`'):
      spherical_harmonic.RealSphericalHarmonics(
          longitude_wavenumbers=8,
          total_wavenumbers=9,
          longitude_nodes=16,
          latitude_nodes=8,
          transform_method=invalid_method,
      )
    with self.assertRaisesRegex(ValueError, 'Unsupported `transform_method`'):
      spherical_harmonic.Grid.T21(transform_method=invalid_method)

  def testTransformMethodSingleSourceOfTruthAndCustomImpl(self):
    with self.assertRaisesRegex(
        ValueError, 'Pass `transform_method` directly to `Grid`'
    ):
      spherical_harmonic.Grid.T21(
          spherical_harmonics_impl=functools.partial(
              spherical_harmonic.RealSphericalHarmonics,
              transform_method='least_squares',
          )
      )

    def custom_impl(
        longitude_wavenumbers,
        total_wavenumbers,
        longitude_nodes,
        latitude_nodes,
        latitude_spacing='gauss',
        basis_as_jax_arrays=None,
    ):
      return spherical_harmonic.RealSphericalHarmonics(
          longitude_wavenumbers=longitude_wavenumbers,
          total_wavenumbers=total_wavenumbers,
          longitude_nodes=longitude_nodes,
          latitude_nodes=latitude_nodes,
          latitude_spacing=latitude_spacing,
          basis_as_jax_arrays=basis_as_jax_arrays,
      )

    grid = spherical_harmonic.Grid.T21(spherical_harmonics_impl=custom_impl)
    self.assertIsInstance(
        grid.spherical_harmonics, spherical_harmonic.RealSphericalHarmonics
    )


class GridTest(parameterized.TestCase):

  @parameterized.product(
      wavenumbers=(32, 137),
      latitude_spacing=('gauss', 'equiangular'),
      impl=[
          spherical_harmonic.RealSphericalHarmonics,
          spherical_harmonic.FastSphericalHarmonics,
      ],
  )
  def testGridShape(self, wavenumbers, latitude_spacing, impl):
    """Check the nodal and modal shape attributes."""
    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers,
        latitude_spacing=latitude_spacing,
        spherical_harmonics_impl=impl,
    )
    with self.subTest('nodal shape'):
      self.assertTupleEqual(grid.nodal_shape, grid.nodal_mesh[0].shape)
      self.assertTupleEqual(grid.nodal_shape, grid.nodal_mesh[1].shape)
      self.assertTupleEqual(
          grid.nodal_shape, (len(grid.nodal_axes[0]), len(grid.nodal_axes[1]))
      )

    with self.subTest('modal shape'):
      self.assertTupleEqual(grid.modal_shape, grid.modal_mesh[0].shape)
      self.assertTupleEqual(grid.modal_shape, grid.modal_mesh[1].shape)
      self.assertTupleEqual(
          grid.modal_shape, (len(grid.modal_axes[0]), len(grid.modal_axes[1]))
      )

  def testModalAxes(self):
    grid = spherical_harmonic.Grid(
        longitude_wavenumbers=4,
        total_wavenumbers=4,
        longitude_nodes=8,
        latitude_nodes=4,
    )
    m_expected = np.array([0, 1, -1, 2, -2, 3, -3])
    l_expected = np.array([0, 1, 2, 3])
    m_actual, l_actual = grid.modal_axes
    np.testing.assert_array_equal(m_expected, m_actual)
    np.testing.assert_array_equal(l_expected, l_actual)

  @parameterized.parameters(
      dict(longitude_offset=0.0),
      dict(longitude_offset=np.pi / 180),
      dict(longitude_offset=-np.pi / 180),
  )
  def test_longitudes(self, longitude_offset):
    grid = spherical_harmonic.Grid(
        longitude_wavenumbers=4,
        total_wavenumbers=4,
        longitude_nodes=8,
        latitude_nodes=4,
        longitude_offset=longitude_offset,
    )
    longitudes_expected = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    longitudes_expected += grid.longitude_offset
    longitudes_actual, _ = grid.nodal_axes
    np.testing.assert_array_equal(longitudes_expected, longitudes_actual)

  def testConstructors(self):
    grid = spherical_harmonic.Grid.T21()
    self.assertEqual(grid.nodal_shape, (64, 32))
    self.assertEqual(grid.modal_shape, (43, 23))

    grid = spherical_harmonic.Grid.TL31()
    self.assertEqual(grid.nodal_shape, (64, 32))
    self.assertEqual(grid.modal_shape, (63, 33))

  @parameterized.parameters(
      dict(
          longitude_wavenumbers=32,
          total_wavenumbers=32,
          latitude_spacing='gauss',
          jit=False,
          seed=0,
          spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics,
      ),
      dict(
          longitude_wavenumbers=64,
          total_wavenumbers=64,
          latitude_spacing='equiangular',
          jit=True,
          seed=0,
          spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
      ),
      dict(
          longitude_wavenumbers=64,
          total_wavenumbers=64,
          latitude_spacing='equiangular',
          jit=True,
          seed=0,
          spherical_harmonics_impl=functools.partial(
              spherical_harmonic.FastSphericalHarmonics,
              base_shape_multiple=8,
              reverse_einsum_arg_order=True,
          ),
      ),
  )
  def testRoundTrip(
      self,
      longitude_wavenumbers,
      total_wavenumbers,
      latitude_spacing,
      jit,
      seed,
      spherical_harmonics_impl,
  ):
    """Tests that the modal -> nodal -> modal round trip is the identity."""
    longitude_nodes = 4 * longitude_wavenumbers
    latitude_nodes = 2 * total_wavenumbers

    grid = spherical_harmonic.Grid(
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=total_wavenumbers,
        latitude_spacing=latitude_spacing,
        spherical_harmonics_impl=spherical_harmonics_impl,
    )
    modal = random_modal_state(grid, seed=seed)
    modal[0, 0] = 0

    inverse_transform = grid.to_nodal
    transform = grid.to_modal
    if jit:
      inverse_transform = jax.jit(inverse_transform)
      transform = jax.jit(transform)

    nodal = inverse_transform(modal)
    reconstructed_modal = transform(nodal)
    np.testing.assert_allclose(modal, reconstructed_modal, atol=1e-5)

  def testFourierMethodDefaultResolution(self):
    """The default fourier_method always resolves to a concrete method."""
    harmonics = spherical_harmonic.FastSphericalHarmonics(32, 33, 96, 48)
    self.assertIn(harmonics.resolved_fourier_method, ['matmul', 'fft'])
    # the fft path is incompatible with nodal padding, so padded shapes must
    # never resolve to it by default
    padded = spherical_harmonic.FastSphericalHarmonics(
        32, 33, 96, 48, base_shape_multiple=8
    )
    self.assertEqual(padded.resolved_fourier_method, 'matmul')

  @parameterized.product(
      wavenumbers=(32, 137),
      latitude_spacing=('gauss', 'equiangular'),
  )
  def testFourierMethodsAgree(self, wavenumbers, latitude_spacing):
    """FFT and matmul Fourier transforms produce identical results."""
    grids = {
        method: spherical_harmonic.Grid.with_wavenumbers(
            wavenumbers,
            latitude_spacing=latitude_spacing,
            spherical_harmonics_impl=functools.partial(
                spherical_harmonic.FastSphericalHarmonics,
                transform_precision='float32',
                fourier_method=method,
            ),
        )
        for method in ['matmul', 'fft']
    }
    rng = np.random.default_rng(0)
    nodal = rng.standard_normal((3,) + grids['matmul'].nodal_shape).astype(
        np.float32
    )
    modal = {k: grid.to_modal(nodal) for k, grid in grids.items()}
    np.testing.assert_allclose(
        modal['matmul'], modal['fft'], atol=1e-5, rtol=1e-5
    )
    nodal_out = {
        k: grid.to_nodal(modal['matmul']) for k, grid in grids.items()
    }
    np.testing.assert_allclose(
        nodal_out['matmul'], nodal_out['fft'], atol=1e-5, rtol=1e-5
    )

  @parameterized.product(
      wavenumbers=(32, 137, 255),
      latitude_spacing=('gauss', 'equiangular'),
      seed=(0,),
      impl=[
          spherical_harmonic.RealSphericalHarmonics,
          spherical_harmonic.FastSphericalHarmonics,
      ],
  )
  def testLaplacianRoundTrip(self, wavenumbers, latitude_spacing, seed, impl):
    """Similar that `inverse_laplacian` is the inverse of `laplacian`."""
    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers,
        latitude_spacing=latitude_spacing,
        spherical_harmonics_impl=impl,
    )
    x = random_modal_state(grid, seed)
    x[0, 0] = 0
    y = grid.inverse_laplacian(grid.laplacian(x))
    np.testing.assert_allclose(x, y)

  @parameterized.product(
      (
          dict(latitude_spacing='gauss', atol=1e-5),
          dict(latitude_spacing='equiangular', atol=1e-5),
      ),
      wavenumbers=(64, 128),
      test_function=(_function_0, _function_1),
      impl=[
          spherical_harmonic.RealSphericalHarmonics,
          spherical_harmonic.FastSphericalHarmonics,
          functools.partial(
              spherical_harmonic.FastSphericalHarmonics, fourier_method='fft'
          ),
      ],
  )
  def testDerivatives(
      self, latitude_spacing, atol, wavenumbers, test_function, impl
  ):
    """Tests that `Grid` accurately computes derivatives."""
    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers,
        latitude_spacing=latitude_spacing,
        spherical_harmonics_impl=impl,
    )
    lon, sin_lat = grid.nodal_mesh
    lat = np.arcsin(sin_lat)
    fx = test_function(lat, lon)

    with self.subTest(msg='sec_lat_d_dlat_cos2'):

      def cos2latf(lat, lon):
        return jnp.cos(lat) ** 2 * test_function(lat, lon)

      dcos2latf_dlat = jax.vmap(jax.vmap(jax.grad(cos2latf)))
      expected = dcos2latf_dlat(lat, lon) / grid.cos_lat
      actual = grid.to_nodal(grid.sec_lat_d_dlat_cos2(grid.to_modal(fx)))
      np.testing.assert_allclose(expected, actual, atol=atol)

    with self.subTest(msg='cos_lat_d_dlat'):
      df_dlat = jax.vmap(jax.vmap(jax.grad(test_function)))
      expected = df_dlat(lat, lon) * grid.cos_lat
      actual = grid.to_nodal(grid.cos_lat_d_dlat(grid.to_modal(fx)))
      np.testing.assert_allclose(expected, actual, atol=atol)

    with self.subTest(msg='d_dlon'):
      df_dlon = jax.vmap(jax.vmap(jax.grad(test_function, argnums=1)))
      expected = df_dlon(lat, lon)
      actual = grid.to_nodal(grid.d_dlon(grid.to_modal(fx)))
      np.testing.assert_allclose(expected, actual, atol=atol)

  @parameterized.parameters(
      dict(
          wavenumbers=85,
          latitude_spacing='equiangular',
          acceptable_norm_diff=10,
      ),
      dict(
          wavenumbers=85,
          latitude_spacing='gauss',
          acceptable_norm_diff=10,
      ),
      dict(
          wavenumbers=42,
          latitude_spacing='gauss',
          acceptable_norm_diff=5,
      ),
  )
  def testDerivativeArtifacts(
      self,
      wavenumbers,
      latitude_spacing,
      acceptable_norm_diff,
  ):
    """Tests that `Grid` computes derivatives without strong artifacts."""

    def test_function(lat, lon):
      """A hand-picked function that exposes derivative artifacts."""
      xs = jnp.linspace(0, 1.9, wavenumbers)
      ys = jnp.exp(jnp.sin(5 * xs) * 1.1 * xs**2 - 0.8 * xs**4)
      return sum(
          y * jnp.cos(lat * 4) ** 2 * jnp.cos(lon * (n % 4))
          for n, y in zip(np.arange(wavenumbers), (ys))
      )

    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, latitude_spacing=latitude_spacing
    )
    lon, sin_lat = grid.nodal_mesh
    lat = np.arcsin(sin_lat)
    fx = test_function(lat, lon)
    df_dlat = jax.vmap(jax.vmap(jax.grad(test_function)))
    expected = df_dlat(lat, lon) * grid.cos_lat

    with self.subTest(msg='artifact_in_cos_lat_d_dlat'):
      actual = grid.to_nodal(grid.cos_lat_d_dlat(grid.to_modal(fx)))
      error_norm = np.linalg.norm(actual - expected)
      self.assertGreater(error_norm, 10 * acceptable_norm_diff)

    with self.subTest(msg='no_artifact_in_cos_lat_grad'):
      _, actual = grid.to_nodal(grid.cos_lat_grad(grid.to_modal(fx)))
      error_norm = np.linalg.norm(actual - expected)
      self.assertLess(error_norm, acceptable_norm_diff)

  @parameterized.parameters(
      dict(grid=spherical_harmonic.Grid.with_wavenumbers(128), seed=0),
      dict(
          grid=spherical_harmonic.Grid(
              longitude_wavenumbers=64,
              total_wavenumbers=64,
              longitude_nodes=192,
              latitude_nodes=128,
              radius=2.6,
              latitude_spacing='equiangular',
          ),
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(128, radius=0.3), seed=0
      ),
  )
  def testLaplacian(self, grid, seed):
    """Tests that computing the Laplacian in 2 ways gives identical results."""
    # The test runs on random input with high wavenumber components damped.
    x = random_modal_state(grid, seed)
    x[0, 0] = 0
    # Taking the derivative twice will give inaccurate results for the highest
    # total wavenumber, so we trim these values for testing.
    x[:, -1] = 0

    # Compute Laplacian using eigenvalues.
    laplacian0 = grid.laplacian(x)

    # Compute Laplacian using
    # Δx =  ∇ · [cosθ ((cosθ ∇x) / cos²θ)]
    # where θ is latitude.
    # `x` has no top wavenumbers, so it's safe to skip clipping once.
    cos_lat_grad = grid.cos_lat_grad(x, clip=False)
    sec_lat_grad = grid.to_modal(grid.to_nodal(cos_lat_grad) * grid.sec2_lat)
    laplacian1 = grid.div_cos_lat(sec_lat_grad)

    np.testing.assert_allclose(laplacian0, laplacian1, atol=1e-10)

  @parameterized.parameters(
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(128), atol=1e-10, seed=0
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(
              128,
              spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
          ),
          atol=1e-11,
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(
              128,
              spherical_harmonics_impl=functools.partial(
                  spherical_harmonic.FastSphericalHarmonics,
                  transform_precision='float32',
              ),
          ),
          atol=1e-11,
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid(
              longitude_wavenumbers=64,
              total_wavenumbers=64,
              longitude_nodes=192,
              latitude_nodes=128,
              radius=3.2,
              latitude_spacing='equiangular',
          ),
          atol=1e-5,
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(128, radius=0.54),
          atol=1e-10,
          seed=0,
      ),
  )
  def testVorticityStreamVelocityRoundTrip(self, grid, atol, seed):
    """Tests the vorticity -> stream -> velocity round trip is the identity."""
    # Choose a random vorticity 𝜻.
    vorticity = random_modal_state(grid, seed)
    vorticity[0, 0] = 0
    vorticity[:, -1] = 0

    # Solve for the stream function ∇²ѱ = 𝜻
    stream = grid.inverse_laplacian(vorticity)

    # Compute the velocity v = k x 𝝯ѱ
    # `stream` has no top wavenumbers, so it's safe to skip clipping once.
    cos_lat_v = jnp.stack(grid.k_cross(grid.cos_lat_grad(stream, clip=False)))
    sec_lat_v = grid.to_modal(grid.sec2_lat * grid.to_nodal(cos_lat_v))
    div_v = grid.div_cos_lat(sec_lat_v)

    # We expect the velocity to be divergence-free.
    np.testing.assert_allclose(div_v, 0, atol=atol)

    # Reconstruct the vorticity 𝜻 = ∇ x v.
    reconstructed_vorticity = grid.curl_cos_lat(sec_lat_v)

    np.testing.assert_allclose(vorticity, reconstructed_vorticity, atol=atol)

  @parameterized.parameters(
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(128), atol=1e-10, seed=0
      ),
      dict(
          grid=spherical_harmonic.Grid(
              longitude_wavenumbers=64,
              total_wavenumbers=64,
              longitude_nodes=192,
              latitude_nodes=128,
              radius=3.2,
              latitude_spacing='equiangular',
          ),
          atol=1e-5,
          seed=0,
      ),
      dict(
          grid=spherical_harmonic.Grid.with_wavenumbers(128, radius=0.54),
          atol=1e-10,
          seed=0,
      ),
  )
  def testDivergencePotentialVelocityRoundTrip(self, grid, atol, seed):
    """Tests the div -> potential -> velocity round trip is the identity."""
    # Choose a random divergence D.
    divergence = random_modal_state(grid, seed)
    divergence[0, 0] = 0
    divergence[:, -1] = 0

    # Solve for the velocity potential ∇²ɸ = D.
    potential = grid.inverse_laplacian(divergence)

    # Compute the velocity v = 𝝯ɸ
    # `potential` has no top wavenumbers, so it's safe to skip clipping once.
    cos_lat_v = grid.cos_lat_grad(potential, clip=False)
    sec_lat_v = grid.to_modal(grid.sec2_lat * grid.to_nodal(cos_lat_v))
    curl_v = grid.curl_cos_lat(sec_lat_v)

    # We expect the velocity to be curl-free.
    np.testing.assert_allclose(curl_v, 0, atol=atol)

    # Reconstruct the divergence D = ∇ · v.
    reconstructed_divergence = grid.div_cos_lat(sec_lat_v).at[:, -1].set(0)

    np.testing.assert_allclose(divergence, reconstructed_divergence, atol=atol)

  @parameterized.parameters(
      dict(wavenumbers=64, latitude_spacing='gauss', radius=1.3),
      dict(wavenumbers=128, latitude_spacing='equiangular', radius=2.5),
      dict(wavenumbers=256, latitude_spacing='gauss', radius=1.0),
  )
  def testIntegrationSurfaceArea(self, wavenumbers, latitude_spacing, radius):
    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, latitude_spacing=latitude_spacing, radius=radius
    )
    ones = jnp.ones([grid.longitude_nodes, grid.latitude_nodes])
    quadrature_surface_area = grid.integrate(ones)
    expected_surface_area = 4 * np.pi * radius**2  # A = 4πr²
    np.testing.assert_array_almost_equal(
        quadrature_surface_area, expected_surface_area
    )

  @parameterized.product(
      params=[
          dict(wavenumbers=64, l=0, m=0),
          dict(wavenumbers=128, l=32, m=17),
          dict(wavenumbers=256, l=174, m=95),
      ],
      impl=[
          spherical_harmonic.RealSphericalHarmonics,
          spherical_harmonic.FastSphericalHarmonics,
      ],
  )
  def testIntegrationSphericalHarmonics(self, params, impl):
    wavenumbers = params['wavenumbers']
    l = params['l']
    m = params['m']
    grid = spherical_harmonic.Grid.with_wavenumbers(
        wavenumbers, spherical_harmonics_impl=impl
    )
    x = jnp.zeros_like(grid.mask).at[m, l].set(1)
    z = grid.to_nodal(x)
    integral = grid.integrate(z**2)
    np.testing.assert_allclose(integral, 1, atol=1e-10)

  def testClipWavenumber(self):
    state = {'u': np.ones((4, 3)), 'time': 1.0}
    grid = spherical_harmonic.Grid(longitude_wavenumbers=2, total_wavenumbers=3)
    clipped = grid.clip_wavenumbers(state)
    expected_u = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    np.testing.assert_array_equal(clipped['u'], expected_u)
    self.assertEqual(clipped['time'], 1.0)

  def testNodalPaddingHelpers(self):
    padded = spherical_harmonic.Grid(
        longitude_wavenumbers=22,
        total_wavenumbers=23,
        longitude_nodes=68,
        latitude_nodes=34,
        spherical_harmonics_impl=functools.partial(
            spherical_harmonic.FastSphericalHarmonics, nodal_shape_multiple=8
        ),
    )
    self.assertEqual(padded.nodal_shape, (72, 40))
    self.assertEqual(padded.nodal_padding, (4, 6))
    rng = np.random.RandomState(0)
    unpadded_values = rng.normal(size=(3, 68, 34)).astype(np.float32)

    with self.subTest('pad_nodal'):
      actual = padded.pad_nodal(jnp.asarray(unpadded_values))
      self.assertEqual(actual.shape, (3, 72, 40))
      np.testing.assert_array_equal(actual[:, :68, :34], unpadded_values)
      np.testing.assert_array_equal(actual[:, 68:, :], 0.0)
      np.testing.assert_array_equal(actual[:, :, 34:], 0.0)
      # An array that already has the nodal shape is returned as it is.
      self.assertIs(padded.pad_nodal(actual), actual)
      np.testing.assert_array_equal(
          padded.pad_nodal(unpadded_values[0]), actual[0]
      )
      with self.assertRaisesRegex(ValueError, 'expected a nodal array'):
        padded.pad_nodal(jnp.zeros((10, 10)))

    with self.subTest('zero_nodal_padding'):
      values = jnp.asarray(rng.normal(size=(2, 72, 40)), jnp.float32)
      actual = padded.zero_nodal_padding(values)
      np.testing.assert_array_equal(actual[:, :68, :34], values[:, :68, :34])
      np.testing.assert_array_equal(actual[:, 68:, :], 0.0)
      np.testing.assert_array_equal(actual[:, :, 34:], 0.0)
      # Values at points other than the nodes have no padding.
      points = jnp.ones((5, 7))
      np.testing.assert_array_equal(padded.zero_nodal_padding(points), points)

    with self.subTest('no padding'):
      grid = spherical_harmonic.Grid.T21()
      values = jnp.asarray(rng.normal(size=grid.nodal_shape), jnp.float32)
      self.assertIs(grid.pad_nodal(values), values)
      np.testing.assert_array_equal(grid.zero_nodal_padding(values), values)

  def testFastSphericalHarmonicsNodalShapeMultiple(self):
    x_shards = 2 if len(jax.devices()) >= 2 else 1
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:x_shards]).reshape(1, x_shards, 1),
        ['z', 'x', 'y'],
    )
    t63 = spherical_harmonic.Grid.T63(
        spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
        spmd_mesh=mesh,
    )
    self.assertEqual(t63.nodal_shape, (192, 96))
    self.assertEqual(t63.nodal_padding, (0, 0))
    expected_modal = (128, 72) if x_shards > 1 else (128, 65)
    self.assertEqual(t63.modal_shape, expected_modal)

    # A sharded grid keeps its default nodal padding (a multiple of 8 per
    # shard, like the modal shape) unless `nodal_shape_multiple` is given.
    sharded = dict(
        longitude_nodes=68,
        latitude_nodes=34,
        longitude_wavenumbers=22,
        total_wavenumbers=23,
        spmd_mesh=mesh,
    )
    default = spherical_harmonic.FastSphericalHarmonics(**sharded)
    unpadded = spherical_harmonic.FastSphericalHarmonics(
        nodal_shape_multiple=1, **sharded
    )
    if x_shards > 1:
      self.assertEqual(default.base_shape_multiple, 8)
      self.assertEqual(default.nodal_shape_multiple, 8)
      self.assertEqual(default.nodal_shape, (80, 40))
      self.assertEqual(unpadded.nodal_shape, (68, 34))
      self.assertEqual(unpadded.nodal_padding, (0, 0))
    else:
      self.assertEqual(default.nodal_shape, (68, 34))
    self.assertEqual(unpadded.modal_shape, default.modal_shape)

    # The nodal padding follows an explicit base_shape_multiple.
    padded_legacy = spherical_harmonic.FastSphericalHarmonics(
        longitude_nodes=94,
        latitude_nodes=47,
        longitude_wavenumbers=32,
        total_wavenumbers=32,
        base_shape_multiple=8,
    )
    self.assertEqual(padded_legacy.nodal_shape, (96, 48))
    self.assertEqual(padded_legacy.nodal_padding, (2, 1))

    # Explicit nodal_shape_multiple controls nodal rounding independently.
    padded_nodal = spherical_harmonic.FastSphericalHarmonics(
        longitude_nodes=94,
        latitude_nodes=47,
        longitude_wavenumbers=32,
        total_wavenumbers=32,
        nodal_shape_multiple=8,
    )
    self.assertEqual(padded_nodal.nodal_shape, (96, 48))
    self.assertEqual(padded_nodal.modal_shape, (64, 32))

  @parameterized.product(
      grid_ctor=[
          spherical_harmonic.Grid.T21,
          spherical_harmonic.Grid.TL31,
          spherical_harmonic.Grid.TL179,
          # Also cover `longitude_nodes // 2 < longitude_wavenumbers` (Nyquist).
          functools.partial(
              spherical_harmonic.Grid,
              longitude_wavenumbers=9,
              total_wavenumbers=10,
              longitude_nodes=16,
              latitude_nodes=10,
              latitude_spacing='gauss',
          ),
      ],
      impl=[
          spherical_harmonic.RealSphericalHarmonics,
          spherical_harmonic.FastSphericalHarmonics,
      ],
  )
  def testLeastSquaresMatchesQuadratureOnGaussianGrids(self, grid_ctor, impl):
    quad_grid = grid_ctor(
        spherical_harmonics_impl=impl, transform_method='quadrature'
    )
    ls_grid = grid_ctor(
        spherical_harmonics_impl=impl, transform_method='least_squares'
    )
    # All resolved wavenumbers (up to total_wavenumbers - 1) have exact
    # Gauss-Legendre quadrature on Gaussian grids.
    np.testing.assert_allclose(
        ls_grid.spherical_harmonics.basis.p_fwd[:, :, :-1],
        quad_grid.spherical_harmonics.basis.p[:, :, :-1],
        atol=1e-12,
    )
    rng = np.random.default_rng(0)
    nodal = rng.standard_normal(quad_grid.nodal_shape)
    np.testing.assert_allclose(
        ls_grid.clip_wavenumbers(ls_grid.to_modal(nodal)),
        quad_grid.clip_wavenumbers(quad_grid.to_modal(nodal)),
        atol=1e-11,
    )

  @parameterized.product(
      (
          dict(
              impl=spherical_harmonic.RealSphericalHarmonics,
              f64_atol=1e-11,
          ),
          dict(
              impl=functools.partial(
                  spherical_harmonic.FastSphericalHarmonics,
                  fourier_method='matmul',
              ),
              f64_atol=1e-11,
          ),
          dict(
              impl=functools.partial(
                  spherical_harmonic.FastSphericalHarmonics,
                  fourier_method='fft',
              ),
              # _fft_forward_scale / _fft_inverse_scale store float32 scales.
              f64_atol=2e-5,
          ),
          dict(
              impl=functools.partial(
                  spherical_harmonic.FastSphericalHarmonics,
                  base_shape_multiple=8,
                  reverse_einsum_arg_order=True,
              ),
              f64_atol=1e-11,
          ),
      ),
      grid_params=[
          dict(
              longitude_wavenumbers=64,
              total_wavenumbers=65,
              longitude_nodes=128,
              latitude_nodes=64,
              latitude_spacing='equiangular',
          ),
          dict(
              longitude_wavenumbers=120,
              total_wavenumbers=121,
              longitude_nodes=240,
              latitude_nodes=121,
              latitude_spacing='equiangular_with_poles',
          ),
          dict(
              longitude_wavenumbers=180,
              total_wavenumbers=181,
              longitude_nodes=360,
              latitude_nodes=181,
              latitude_spacing='equiangular_with_poles',
          ),
      ],
  )
  def testLeastSquaresRoundTripOnEquiangularLinearGrids(
      self, impl, f64_atol, grid_params
  ):
    ls_grid = spherical_harmonic.Grid(
        **grid_params,
        spherical_harmonics_impl=impl,
        transform_method='least_squares',
    )
    modal = ls_grid.clip_wavenumbers(random_modal_state(ls_grid, seed=0))
    # Also place unit amplitude in a high-degree zonal mode where quadrature
    # fails badly on linear equiangular grids.
    high_l = grid_params['total_wavenumbers'] - 2
    modal = modal.at[0, high_l].set(1.0)

    nodal_f64 = ls_grid.to_nodal(modal)
    recovered_f64 = ls_grid.clip_wavenumbers(ls_grid.to_modal(nodal_f64))
    np.testing.assert_allclose(recovered_f64, modal, atol=f64_atol)

    with jax.enable_x64(False):
      ls_grid_f32 = spherical_harmonic.Grid(
          **grid_params,
          spherical_harmonics_impl=impl,
          basis_as_jax_arrays=True,
          transform_method='least_squares',
      )
      nodal_f32 = jnp.asarray(nodal_f64, dtype=jnp.float32)
      recovered_f32 = ls_grid_f32.clip_wavenumbers(
          ls_grid_f32.to_modal(nodal_f32)
      )
      np.testing.assert_allclose(
          recovered_f32, jnp.asarray(modal, dtype=jnp.float32), atol=5e-5
      )

  def testLeastSquares360x181MatchesTL179AndIsWellConditioned(self):
    tl179 = spherical_harmonic.Grid.TL179(
        spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics
    )
    el180_quad = spherical_harmonic.Grid(
        longitude_wavenumbers=180,
        total_wavenumbers=181,
        longitude_nodes=360,
        latitude_nodes=181,
        latitude_spacing='equiangular_with_poles',
        spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics,
        transform_method='quadrature',
    )
    el180_ls = spherical_harmonic.Grid(
        longitude_wavenumbers=180,
        total_wavenumbers=181,
        longitude_nodes=360,
        latitude_nodes=181,
        latitude_spacing='equiangular_with_poles',
        spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics,
        transform_method='least_squares',
    )

    modal = tl179.clip_wavenumbers(random_modal_state(tl179, seed=42))
    modal = modal.at[0, 179].set(1.0).at[1, 179].set(1.0)

    nodal_tl179 = tl179.to_nodal(modal)
    nodal_el180 = el180_ls.to_nodal(modal)

    modal_tl179 = tl179.clip_wavenumbers(tl179.to_modal(nodal_tl179))
    modal_quad = el180_quad.clip_wavenumbers(el180_quad.to_modal(nodal_el180))
    modal_ls = el180_ls.clip_wavenumbers(el180_ls.to_modal(nodal_el180))

    # Quadrature has O(1) error on high-degree low-m modes on 360x181.
    self.assertGreater(np.max(np.abs(modal_quad - modal_tl179)), 0.5)
    # Least-squares matches TL179 to machine precision across all l <= 179.
    np.testing.assert_allclose(modal_ls, modal_tl179, atol=1e-11)

    # Check conditioning of the weighted analysis operator sqrt(wp) * p_fwd
    # across all zonal wavenumbers m (on 181 Clenshaw-Curtis nodes, polar cap
    # weights scale as ~1/n_lat^2 so max ||sqrt(wp) * p_fwd||_2 is ~28.7).
    _, wp = spherical_harmonic.get_latitude_nodes(181, 'equiangular_with_poles')
    p_fwd = el180_ls.spherical_harmonics.basis.p_fwd
    weighted_ops = np.sqrt(wp)[np.newaxis, :, np.newaxis] * p_fwd
    op_norms = np.linalg.norm(weighted_ops, ord=2, axis=(1, 2))
    self.assertLess(float(np.max(op_norms)), 35.0)

    # Check that out-of-band spherical harmonic power (l >= 181) and gridpoint
    # white noise produce bounded modal coefficients up to the top degrees.
    oob_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=180,
        total_wavenumbers=220,
        longitude_nodes=360,
        latitude_nodes=181,
        latitude_spacing='equiangular_with_poles',
        spherical_harmonics_impl=spherical_harmonic.RealSphericalHarmonics,
    )
    rng = np.random.default_rng(123)
    oob_modal = np.zeros(oob_grid.modal_shape)
    oob_modal[:, 181:] = rng.standard_normal(oob_modal[:, 181:].shape)
    oob_modal *= oob_grid.mask
    oob_modal /= np.linalg.norm(oob_modal)
    nodal_oob = oob_grid.to_nodal(oob_modal)
    aliased_modal = np.asarray(el180_ls.to_modal(nodal_oob))
    self.assertLess(float(np.linalg.norm(aliased_modal)), 5.0)
    self.assertLess(float(np.max(np.abs(aliased_modal[:, -5:]))), 2.0)

    nodal_noise = rng.standard_normal(el180_ls.nodal_shape)
    noise_modal = np.asarray(el180_ls.to_modal(nodal_noise))
    self.assertLess(float(np.max(np.abs(noise_modal[:, -5:]))), 1.0)


if __name__ == '__main__':
  absltest.main()
