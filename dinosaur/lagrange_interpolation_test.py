# Copyright 2026 Google LLC
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

"""Tests for lagrange_interpolation."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import lagrange_interpolation
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
import jax
import jax.numpy as jnp
import numpy as np


def _queries(grid, shape, seed):
  rng = np.random.RandomState(seed)
  lon = jnp.asarray(rng.uniform(0, 2 * np.pi, size=shape), jnp.float32)
  sin_lat = jnp.asarray(np.sin(rng.uniform(-1.5, 1.5, size=shape)), jnp.float32)
  return lon, sin_lat


class StencilTest(parameterized.TestCase):

  @parameterized.product(order=('linear', 'cubic'), backend=('cpu', 'tpu'))
  def test_horizontal_stencil(self, order, backend):
    grid = spherical_harmonic.Grid.T21()
    lon, sin_lat = _queries(grid, (50,), seed=0)
    with mock.patch.object(
        lagrange_interpolation, 'target_platform', return_value=backend
    ):
      plain = lagrange_interpolation.horizontal_stencil(
          grid, lon, sin_lat, order
      )
      stencil = lagrange_interpolation.horizontal_stencil(
          grid, lon, sin_lat, order, include_linear_fractions=True
      )
    points = 2 if order == 'linear' else 4
    self.assertLen(stencil.lon_indices, points)
    self.assertLen(stencil.lat_weights, points)
    # The stencil has the same fields whether or not fractions are asked for.
    self.assertIsNone(plain.linear_fractions)
    np.testing.assert_array_equal(plain.lat_cell, stencil.lat_cell)
    np.testing.assert_allclose(sum(stencil.lon_weights), 1.0, atol=1e-5)
    np.testing.assert_allclose(sum(stencil.lat_weights), 1.0, atol=1e-5)
    # The weights reproduce the query coordinates (exact for linear data).
    phi_ext = lagrange_interpolation.latitude_halo_axis(grid)
    self.assertEqual(stencil.extended_latitude_size, phi_ext.size)
    phi = sum(
        w * phi_ext[np.asarray(i)]
        for i, w in zip(stencil.lat_indices, stencil.lat_weights)
    )
    np.testing.assert_allclose(phi, np.arcsin(sin_lat), atol=2e-5)
    # The bracketing cell contains the query.
    cell = np.asarray(stencil.lat_cell)
    self.assertTrue(np.all(phi_ext[cell] <= np.arcsin(sin_lat) + 1e-5))
    self.assertTrue(np.all(np.arcsin(sin_lat) <= phi_ext[cell + 1] + 1e-5))
    alpha_lon, alpha_lat = stencil.linear_fractions
    for alpha in (alpha_lon, alpha_lat):
      self.assertTrue(np.all(alpha >= -1e-5) and np.all(alpha <= 1 + 1e-5))

  @parameterized.product(order=('linear', 'cubic'), backend=('cpu', 'tpu'))
  def test_vertical_stencil(self, order, backend):
    nodes = jnp.asarray(
        sigma_coordinates.SigmaCoordinates.equidistant(8).centers, jnp.float32
    )
    rng = np.random.RandomState(1)
    sigma = jnp.asarray(rng.uniform(-0.1, 1.1, size=(60,)), jnp.float32)
    stencil = lagrange_interpolation.vertical_stencil(
        sigma,
        nodes,
        order,
        is_tpu=backend == 'tpu',
        include_linear_fractions=True,
    )
    plain = lagrange_interpolation.vertical_stencil(
        sigma, nodes, order, is_tpu=backend == 'tpu'
    )
    self.assertIsNone(plain.linear_fraction)
    np.testing.assert_allclose(sum(stencil.weights), 1.0, atol=1e-5)
    clipped = np.clip(sigma, nodes[0], nodes[-1])
    interpolated = sum(
        w * nodes[i] for i, w in zip(stencil.indices, stencil.weights)
    )
    np.testing.assert_allclose(interpolated, clipped, atol=1e-5)
    cell = np.asarray(stencil.cell)
    self.assertTrue(np.all((cell >= 0) & (cell <= nodes.shape[0] - 2)))
    np.testing.assert_allclose(
        np.asarray(nodes)[cell]
        + stencil.linear_fraction * np.diff(np.asarray(nodes))[cell],
        clipped,
        atol=1e-5,
    )
    # The bracketing cell sits at `bracket_offset` within the stencil.
    start = np.asarray(stencil.indices[0])
    np.testing.assert_array_equal(start + stencil.bracket_offset, cell)

  def test_pole_halo(self):
    grid = spherical_harmonic.Grid.T21()
    nx, ny = grid.nodal_shape
    halo = lagrange_interpolation.HALO_WIDTH
    field = jnp.asarray(
        np.random.RandomState(2).normal(size=(3, nx, ny)), jnp.float32
    )
    extended = lagrange_interpolation.extend_with_pole_halo(field, grid)
    self.assertEqual(extended.shape, (3, nx, ny + 2 * halo))
    np.testing.assert_array_equal(extended[..., halo:-halo], field)
    # A halo row is the row mirrored about the pole, half a turn away.
    rolled = np.roll(np.asarray(field), nx // 2, axis=-2)
    for k in range(halo):
      np.testing.assert_array_equal(extended[..., halo - 1 - k], rolled[..., k])
      np.testing.assert_array_equal(
          extended[..., ny + halo + k], rolled[..., ny - 1 - k]
      )
    phi_ext = lagrange_interpolation.latitude_halo_axis(grid)
    self.assertEqual(phi_ext.size, ny + 2 * halo)
    self.assertTrue(np.all(np.diff(phi_ext) > 0))


class WindowClampTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 2, 6)
  def test_clamp_bounds_cover_the_window_cells(self, shift):
    grid = spherical_harmonic.Grid.T21()
    ny = grid.nodal_shape[1]
    halo = lagrange_interpolation.HALO_WIDTH
    effective = lagrange_interpolation.effective_window_shift(shift)
    self.assertEqual(effective, max(1, shift))
    phi_ext = lagrange_interpolation.latitude_halo_axis(grid)
    lo, hi = lagrange_interpolation.lat_clamp_bounds(
        grid, shift, 'linear', np.float32
    )
    self.assertEqual(lo.shape, (ny,))
    arrival = np.arange(ny) + halo
    # Cells `a - S ... a + S`: from node `a - S` up to (but not including)
    # node `a + S + 1`, clipped to the axis.
    np.testing.assert_array_equal(
        lo, phi_ext[np.clip(arrival - effective, 0, None)].astype(np.float32)
    )
    upper = phi_ext[np.clip(arrival + effective + 1, None, phi_ext.size - 1)]
    self.assertTrue(np.all(hi < upper.astype(np.float32)))
    np.testing.assert_allclose(hi, upper, rtol=1e-6)

    nodes = jnp.asarray(
        sigma_coordinates.SigmaCoordinates.equidistant(10).centers, jnp.float32
    )
    z_lo, z_hi = lagrange_interpolation.level_clamp_bounds(
        nodes, 10, shift, jnp.float32
    )
    k = np.arange(10)
    np.testing.assert_array_equal(
        z_lo, np.asarray(nodes)[np.clip(k - effective, 0, None)]
    )
    np.testing.assert_allclose(
        z_hi, np.asarray(nodes)[np.clip(k + effective + 1, None, 9)], rtol=1e-6
    )
    self.assertTrue(np.all(z_hi <= nodes[-1]))


class PlatformTest(absltest.TestCase):

  def test_platform_without_a_mesh_is_the_default_backend(self):
    grid = spherical_harmonic.Grid.T21()
    for backend in ('cpu', 'gpu', 'tpu'):
      with mock.patch.object(jax, 'default_backend', return_value=backend):
        self.assertEqual(lagrange_interpolation.target_platform(grid), backend)
        self.assertEqual(
            lagrange_interpolation.targets_tpu(grid), backend == 'tpu'
        )


class ContractStencilTest(parameterized.TestCase):

  @parameterized.parameters(dict(n=2), dict(n=3))
  def test_fused_form_matches_einsum(self, n):
    """The accelerator (fused) contraction equals the einsum form.

    The dispatch in `_contract_stencil` picks by backend, so CPU test runs
    exercise only the einsum branch; this pins the fused branch directly.
    """
    rng = np.random.RandomState(0)
    batch = (3, 5, 7)
    sizes = (2, 4, 4)[-n:]
    values = jnp.asarray(rng.standard_normal(batch + sizes))
    weights = [jnp.asarray(rng.standard_normal(batch + (s,))) for s in sizes]
    letters = 'ijkl'[:n]
    expected = jnp.einsum(
        ','.join(f'...{c}' for c in letters) + f',...{letters}->...',
        *weights,
        values,
    )
    actual = lagrange_interpolation._contract_stencil_fused(values, *weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
