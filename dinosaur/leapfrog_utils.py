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

"""Leapfrog utilities for time integration and filtering."""
from __future__ import annotations

from typing import TYPE_CHECKING

from dinosaur import filtering
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
import tree_math

if TYPE_CHECKING:
  # pylint: disable=g-bad-import-order
  from dinosaur import time_integration

tree_map = jax.tree_util.tree_map

PyTreeState = typing.PyTreeState
PyTreeTermsFn = typing.PyTreeTermsFn
TimeStepFn = typing.TimeStepFn
PyTreeStepFilterFn = typing.PyTreeStepFilterFn


def semi_implicit_leapfrog(
    equation: time_integration.ImplicitExplicitODE,
    time_step: float,
    alpha: float = 0.5,
) -> TimeStepFn:
  """Constructs a function that performs a semi-implicit leapfrog timestep.

  Leapfrog integrator works with a pair of interleaved time snapshots.
  See go/semi-implicit-leapfrog for more details.

  Args:
    equation: equation to solve.
    time_step: time step.
    alpha: a parameter used to weight previous and future states in the implicit
      terms of the equation.

  Returns:
    Function that performs a time step.
  """
  explicit_fn = tree_math.unwrap(equation.explicit_terms)
  implicit_fn = tree_math.unwrap(equation.implicit_terms)
  inverse_fn = tree_math.unwrap(equation.implicit_inverse, vector_argnums=0)

  def step_fn(u: PyTreeState) -> PyTreeState:
    # leapfrog assumes that `u` is a Tuple of snapshots at t-1; t
    previous, current = u
    previous, current = tree_math.Vector(previous), tree_math.Vector(current)
    explicit_current = explicit_fn(current)
    implicit_previous = implicit_fn(previous)
    intermediate = previous + 2 * time_step * (
        explicit_current + (1 - alpha) * implicit_previous)
    eta = 2 * time_step * alpha
    future = inverse_fn(intermediate, eta)
    return (current.tree, future.tree)

  return step_fn


def robert_asselin_leapfrog_filter(r: float) -> PyTreeStepFilterFn:
  """Returns a Robert-Asselin filter compatible with leapfrog time steppers.

  This filter introduces mixing in the leapfrog steps by smoothing 3-point
  predictions (previous, current, future).

  See http://weather.ou.edu/~ekalnay/NWPChapter3/Ch3_2_4.html for details.

  Args:
    r: the strength of the filter, a value in the interval [0, 1], typically
      0.01 to 0.05.

  Returns:
    A filter function that accepts `u` and `u_next` leapfrog state
    tuples and returns a filtered output of `u_next`.
  """

  def _filter(u: PyTreeState, u_next: PyTreeState) -> PyTreeState:
    previous, current = u
    _, future = u_next
    filtered_current = tree_map(
        lambda p, c, f: (1 - 2 * r) * c + r * (p + f), previous, current, future
    )
    return (filtered_current, future)

  return _filter


def leapfrog_step_filter(
    state_filter: PyTreeTermsFn,
) -> PyTreeStepFilterFn:
  """Convert a state filter into a leapfrog time integration filter."""

  def _filter(u: PyTreeState, u_next: PyTreeState) -> PyTreeState:
    del u  # unused
    current, future = u_next  # leapfrog state is a tuple of 2 time slices.
    future = state_filter(future)
    return (current, future)

  return _filter


def exponential_leapfrog_step_filter(
    grid: spherical_harmonic.Grid,
    dt: float,
    tau: float = 0.010938,
    order: int = 18,
    cutoff: float = 0,
):
  """Returns an exponential leapfrog step filter.

  This filter simulates dampening on modes according to:

    (∂u_k / ∂t) ≈ -(u_k / 𝜏) * ((k - cutoff) / (1 - cutoff)) ** (2 * order)

  For more details see `filtering.exponential_filter`.

  Args:
    grid: the `spherical_harmonic.Grid` to use for the computation.
    dt: size of the time step to be used for each filter application.
    tau: timescale over which modes are reduced by the corresponding exponential
      factors determined by the wavenumbers, `order` and `cutoff`. Default value
      represents attenuation of `16` for a time step of 20 minutes.
    order: controls the polynomial order of the exponential filter.
    cutoff: a hard threshold with which to start attenuation.

  Returns:
    A function that accepts a state and returns a filtered state.
  """
  filter_fn = filtering.exponential_filter(grid, dt / tau, order, cutoff)
  return leapfrog_step_filter(filter_fn)
