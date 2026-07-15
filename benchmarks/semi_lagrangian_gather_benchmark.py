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

r"""Microbenchmarks for semi-Lagrangian gather-based interpolation.

Two questions this script answers on a given backend (CPU/GPU/TPU):

1. Gather formulation: `semi_lagrangian.interpolate_3d` fetches its
   point-dependent 2✕4✕4 stencils with a single 1-D `jnp.take` on the
   raveled field and precomputed linear indices. The natural alternative is
   multi-array advanced indexing (`field[k, i, j]`). Both lower to one XLA
   gather (the ravel is a zero-cost bitcast of a contiguous array); this
   benchmark measures which gather the backend runs faster. On CPU the
   linearized form measured ~2x faster; GPU numbers are the open question.

   The synthetic index pattern is uniformly random — *worse* memory locality
   than real departure points (which are spatially coherent) — so treat
   these numbers as a pessimistic bound and the library-level benchmark
   below as the realistic one.

2. Library-level cost: the real `departure_points_3d` (trajectory solve,
   trilinear wind interpolation) and `transport_scalar` (cubic interpolation
   of one 3-D field) at semi-Lagrangian displacements of a few grid cells,
   which is the per-field cost inside a model step.

Usage on a fresh GPU box (e.g. Lambda/RunPod instance or a Modal function)::

  pip install -U "jax[cuda12]"
  git clone -b semi-lagrangian https://github.com/shoyer/dinosaur.git
  pip install -e ./dinosaur
  python dinosaur/benchmarks/semi_lagrangian_gather_benchmark.py

Optional flags: --iters=50 --big (adds a T340/L64 case, needs ~10 GB of
device memory) --float64.
"""

import argparse
import functools
import time

from dinosaur import coordinate_systems
from dinosaur import semi_lagrangian
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
import jax
import jax.numpy as jnp
import numpy as np


def _time(fn, *args, iters: int) -> float:
  """Returns mean seconds per call after a compile + warmup call."""
  jax.block_until_ready(fn(*args))
  start = time.perf_counter()
  for _ in range(iters):
    result = fn(*args)
  jax.block_until_ready(result)
  return (time.perf_counter() - start) / iters


def _gather_inputs(layers, num_lon, num_lat, dtype, seed=0):
  """Synthetic field, stencil indices and weights for the gather variants."""
  rng = np.random.RandomState(seed)
  extended_lat = num_lat + 4
  field = jnp.asarray(
      rng.normal(size=(layers, num_lon, extended_lat)).astype(dtype)
  )
  points = (layers, num_lon, num_lat)
  k = jnp.asarray(
      rng.randint(0, layers - 1, size=points + (2, 1, 1)).astype(np.int32)
  )
  i = jnp.asarray(
      rng.randint(0, num_lon, size=points + (1, 4, 1)).astype(np.int32)
  )
  j = jnp.asarray(
      rng.randint(0, extended_lat, size=points + (1, 1, 4)).astype(np.int32)
  )
  weights = jnp.asarray(
      rng.normal(size=points + (2, 4, 4)).astype(dtype)
  )
  return field, k, i, j, weights


@jax.jit
def _linearized_take(field, k, i, j, weights):
  """The formulation used by `semi_lagrangian.interpolate_3d`."""
  _, num_lon, extended_lat = field.shape
  flat_index = (k * num_lon + i) * extended_lat + j
  values = jnp.take(field.ravel(), flat_index, mode='clip')
  return (values * weights).sum(axis=(-3, -2, -1))


@jax.jit
def _advanced_indexing(field, k, i, j, weights):
  """The equivalent multi-array advanced-indexing formulation."""
  values = field[k, i, j]
  return (values * weights).sum(axis=(-3, -2, -1))


def _library_inputs(grid, layers, dtype, seed=0):
  """Realistic winds and a field for the library-level benchmark."""
  rng = np.random.RandomState(seed)
  coords = coordinate_systems.CoordinateSystem(
      grid, sigma_coordinates.SigmaCoordinates.equidistant(layers)
  )
  shape = (layers,) + grid.nodal_shape
  _, sin_lat = grid.nodal_mesh
  # A jet-like zonal flow plus noise; dt below gives displacements of a few
  # grid cells (the semi-Lagrangian regime).
  u = jnp.asarray(
      (0.1 * (1 - sin_lat**2) + 0.01 * rng.normal(size=shape)).astype(dtype)
  )
  v = jnp.asarray((0.01 * rng.normal(size=shape)).astype(dtype))
  sigma_dot = jnp.asarray(
      np.pad(
          1e-3 * rng.normal(size=(layers - 1,) + grid.nodal_shape),
          [(1, 1), (0, 0), (0, 0)],
      ).astype(dtype)
  )
  field = jnp.asarray(rng.normal(size=shape).astype(dtype))
  dt = 3.0 * (2 * np.pi / grid.longitude_nodes) / 0.1  # ~3 cells of zonal CFL
  return coords, u, v, sigma_dot, field, dt


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iters', type=int, default=50)
  parser.add_argument('--big', action='store_true')
  parser.add_argument('--float64', action='store_true')
  args = parser.parse_args()

  if args.float64:
    jax.config.update('jax_enable_x64', True)
  dtype = np.float64 if args.float64 else np.float32

  print(f'jax {jax.__version__}, devices: {jax.devices()}, dtype: {dtype.__name__}')

  cases = [
      ('T85/L32', spherical_harmonic.Grid.T85(), 32),
      ('T170/L48', spherical_harmonic.Grid.T170(), 48),
  ]
  if args.big:
    cases.append(('T340/L64', spherical_harmonic.Grid.T340(), 64))

  for name, grid, layers in cases:
    num_lon, num_lat = grid.nodal_shape
    num_points = layers * num_lon * num_lat
    print(f'\n=== {name}: {layers} x {num_lon} x {num_lat} '
          f'({num_points / 1e6:.1f}M points, 32-point stencils) ===')

    inputs = _gather_inputs(layers, num_lon, num_lat, dtype)
    np.testing.assert_allclose(
        _linearized_take(*inputs),
        _advanced_indexing(*inputs),
        rtol=1e-5,
        atol=1e-5,
    )
    for label, fn in [
        ('linearized 1-D take ', _linearized_take),
        ('advanced indexing   ', _advanced_indexing),
    ]:
      seconds = _time(fn, *inputs, iters=args.iters)
      gathered_gb = num_points * 32 * np.dtype(dtype).itemsize / 1e9
      print(f'  {label}: {seconds * 1e3:8.2f} ms '
            f'({gathered_gb / seconds:6.1f} GB/s gathered)')

    coords, u, v, sigma_dot, field, dt = _library_inputs(grid, layers, dtype)
    interpolator = semi_lagrangian.GridInterpolator(grid, order='cubic')

    departure_fn = jax.jit(
        functools.partial(
            semi_lagrangian.departure_points_3d, coords=coords, dt=dt
        )
    )
    seconds = _time(departure_fn, u, v, sigma_dot, iters=args.iters)
    print(f'  departure_points_3d : {seconds * 1e3:8.2f} ms')

    departure = departure_fn(u, v, sigma_dot)
    transport_fn = jax.jit(
        functools.partial(
            semi_lagrangian.transport_scalar,
            vertical=coords.vertical,
            interpolator=interpolator,
        )
    )
    seconds = _time(transport_fn, field, departure, iters=args.iters)
    print(f'  transport_scalar    : {seconds * 1e3:8.2f} ms')


if __name__ == '__main__':
  main()
