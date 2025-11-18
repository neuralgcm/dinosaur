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

"""Tests for leapfrog integrators and step filters."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import leapfrog_utils
from dinosaur import time_integration
import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np


jax.config.update('jax_enable_x64', True)


def assert_allclose(x, y, rtol=1e-7, atol=0, err_msg=''):
  jax.tree_util.tree_map(
      functools.partial(
          np.testing.assert_allclose, atol=atol, rtol=rtol, err_msg=err_msg
      ),
      x,
      y,
  )


def harmonic_oscillator(x0, t):
  theta = jnp.arctan(x0[0] / x0[1])
  r = jnp.linalg.norm(x0, ord=2, axis=0)
  return r * jnp.stack([jnp.sin(t + theta), jnp.cos(t + theta)])


class CustomODE(time_integration.ImplicitExplicitODE):

  def __init__(self, explicit_terms, implicit_terms, implicit_inverse):
    self.explicit_terms = explicit_terms
    self.implicit_terms = implicit_terms
    self.implicit_inverse = implicit_inverse


class CustomExplicitODE(time_integration.ExplicitODE):

  def __init__(self, explicit_terms):
    self.explicit_terms = explicit_terms


ALL_TEST_PROBLEMS = [
    # x(t) = np.ones(10)
    dict(
        testcase_name='_zero_derivative',
        explicit_terms=lambda x: 0 * x,
        implicit_terms=lambda x: 0 * x,
        implicit_inverse=lambda x, eta: x,
        dt=1e-2,
        inner_steps=10,
        outer_steps=5,
        initial_state=np.ones(10),
        closed_form=lambda x0, t: x0,
        tolerances=[1e-12],
    ),
    # x(t) = 5 * t * np.ones(3)
    dict(
        testcase_name='_constant_derivative',
        explicit_terms=lambda x: 5 * jnp.ones_like(x),
        implicit_terms=lambda x: 0 * x,
        implicit_inverse=lambda x, eta: x,
        dt=1e-2,
        inner_steps=10,
        outer_steps=5,
        initial_state=np.ones(3),
        closed_form=lambda x0, t: x0 + 5 * t,
        tolerances=[1e-12],
    ),
    # x(t) = np.arange(3) * np.exp(t)
    # Uses explicit terms only.
    dict(
        testcase_name='_linear_derivative_explicit',
        explicit_terms=lambda x: x,
        implicit_terms=lambda x: 0 * x,
        implicit_inverse=lambda x, eta: x,
        dt=1e-2,
        inner_steps=20,
        outer_steps=5,
        initial_state=np.arange(3.0),
        closed_form=lambda x0, t: np.arange(3) * jnp.exp(t),
        tolerances=[5e-2],
    ),
    # x(t) = np.arange(3) * np.exp(t)
    # Uses implicit terms only.
    dict(
        testcase_name='_linear_derivative_implicit',
        explicit_terms=lambda x: 0 * x,
        implicit_terms=lambda x: x,
        implicit_inverse=lambda x, eta: x / (1 - eta),
        dt=1e-2,
        inner_steps=20,
        outer_steps=5,
        initial_state=np.arange(3.0),
        closed_form=lambda x0, t: np.arange(3) * jnp.exp(t),
        tolerances=[5e-2],
    ),
    # x(t) = np.arange(3) * np.exp(t)
    # Splits the equation into an implicit and explicit term.
    dict(
        testcase_name='_linear_derivative_semi_implicit',
        explicit_terms=lambda x: x / 2,
        implicit_terms=lambda x: x / 2,
        implicit_inverse=lambda x, eta: x / (1 - eta / 2),
        dt=1e-2,
        inner_steps=20,
        outer_steps=5,
        initial_state=np.arange(3) * np.exp(0),
        closed_form=lambda x0, t: np.arange(3.0) * jnp.exp(t),
        tolerances=[1e-4],
    ),
    dict(
        testcase_name='_harmonic_oscillator_explicit',
        explicit_terms=lambda x: jnp.stack([x[1], -x[0]]),
        implicit_terms=jnp.zeros_like,
        implicit_inverse=lambda x, eta: x,
        dt=1e-2,
        inner_steps=20,
        outer_steps=5,
        initial_state=np.ones(2),
        closed_form=harmonic_oscillator,
        tolerances=[1e-2],
    ),
    dict(
        testcase_name='_harmonic_oscillator_implicit',
        explicit_terms=jnp.zeros_like,
        implicit_terms=lambda x: jnp.stack([x[1], -x[0]]),
        implicit_inverse=lambda x, eta: jnp.stack(  # pylint: disable=g-long-lambda
            [x[0] + eta * x[1], x[1] - eta * x[0]]
        )
        / (1 + eta**2),
        dt=1e-2,
        inner_steps=20,
        outer_steps=5,
        initial_state=np.ones(2),
        closed_form=harmonic_oscillator,
        tolerances=[1e-2],
    ),
]


class LeapfrogUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(ALL_TEST_PROBLEMS)
  def test_integration(
      self,
      explicit_terms,
      implicit_terms,
      implicit_inverse,
      dt,
      inner_steps,
      outer_steps,
      initial_state,
      closed_form,
      tolerances,
  ):
    # Compute closed-form solution.
    time = dt * inner_steps * (1 + np.arange(outer_steps))
    expected = jax.vmap(closed_form, in_axes=(None, 0))(initial_state, time)

    # Compute trajectory using time-stepper.
    atol = tolerances[0]
    time_stepper = leapfrog_utils.semi_implicit_leapfrog
    with self.subTest(time_stepper.__name__):
      equation = CustomODE(explicit_terms, implicit_terms, implicit_inverse)
      semi_implicit_step = time_stepper(equation, dt)
      input_state = (
          closed_form(initial_state, 0 * dt),
          closed_form(initial_state, 1 * dt),
      )
      post_process_fn = lambda x: x[0]
      trajectory_fn = time_integration.trajectory_from_step(
          semi_implicit_step,
          outer_steps,
          inner_steps,
          post_process_fn=post_process_fn,
      )
      _, actual = trajectory_fn(input_state)
      np.testing.assert_allclose(expected, actual, atol=atol, rtol=0)

  def test_pytree_state(self):
    equation = CustomODE(
        explicit_terms=lambda x: tree_util.tree_map(jnp.zeros_like, x),
        implicit_terms=lambda x: tree_util.tree_map(jnp.zeros_like, x),
        implicit_inverse=lambda x, eta: x,
    )
    u0 = {'x': 1.0, 'y': 1.0}
    time_stepper = leapfrog_utils.semi_implicit_leapfrog
    with self.subTest(time_stepper.__name__):
      u0_leapfrog = (u0, u0)
      u1_leapfrog = time_stepper(equation, 1.0)(u0_leapfrog)
      self.assertEqual(u0_leapfrog[0], u1_leapfrog[0])
      self.assertEqual(u0_leapfrog[1], u1_leapfrog[1])

  def test_multiple_equations(self):
    tolerances = [1e-4]
    dt = 1e-2
    inner_steps = 20
    outer_steps = 5
    with self.subTest('array_inputs'):
      initial_state = np.arange(3) * np.exp(0)
      closed_form = lambda x0, t: np.arange(3.0) * jnp.exp(t)
      equation_a = CustomODE(
          explicit_terms=lambda x: 3 * x / 8,
          implicit_terms=lambda x: x / 2,
          implicit_inverse=lambda x, eta: x / (1 - eta / 2),
      )
      equation_b = CustomExplicitODE(explicit_terms=lambda x: x / 8)
      equation = time_integration.compose_equations([equation_a, equation_b])
      # Compute closed-form solution.
      time = dt * inner_steps * (1 + np.arange(outer_steps))
      expected = jax.vmap(closed_form, in_axes=(None, 0))(initial_state, time)

      # Compute trajectory using time-stepper.
      atol = tolerances[0]
      time_stepper = leapfrog_utils.semi_implicit_leapfrog
      with self.subTest(time_stepper.__name__):
        semi_implicit_step = time_stepper(equation, dt)
        input_state = (
            closed_form(initial_state, 0 * dt),
            closed_form(initial_state, 1 * dt),
        )
        post_process_fn = lambda x: x[0]
        trajectory_fn = time_integration.trajectory_from_step(
            semi_implicit_step,
            outer_steps,
            inner_steps,
            post_process_fn=post_process_fn,
        )
        _, actual = trajectory_fn(input_state)
        np.testing.assert_allclose(expected, actual, atol=atol, rtol=0)

    with self.subTest('pytree_inputs'):
      initial_state = {'s': np.arange(3) * np.exp(0)}
      closed_form = lambda x0, t: {'s': np.arange(3.0) * jnp.exp(t)}
      equation_a = CustomODE(
          explicit_terms=lambda x: {'s': x['s'] / 8},
          implicit_terms=lambda x: {'s': x['s'] / 2},
          implicit_inverse=lambda x, eta: {'s': x['s'] / (1 - eta / 2)},
      )
      equation_b = CustomExplicitODE(
          explicit_terms=lambda x: {'s': 3 * x['s'] / 8},
      )
      equation = time_integration.compose_equations([equation_a, equation_b])
      # Compute closed-form solution.
      time = dt * inner_steps * (1 + np.arange(outer_steps))
      expected = jax.vmap(closed_form, in_axes=(None, 0))(initial_state, time)

      # Compute trajectory using time-stepper.
      atol = tolerances[0]
      time_stepper = leapfrog_utils.semi_implicit_leapfrog
      with self.subTest(time_stepper.__name__):
        semi_implicit_step = time_stepper(equation, dt)
        input_state = (
            closed_form(initial_state, 0 * dt),
            closed_form(initial_state, 1 * dt),
        )
        post_process_fn = lambda x: x[0]
        trajectory_fn = time_integration.trajectory_from_step(
            semi_implicit_step,
            outer_steps,
            inner_steps,
            post_process_fn=post_process_fn,
        )
        _, actual = trajectory_fn(input_state)
        np.testing.assert_allclose(expected['s'], actual['s'], atol=atol)


if __name__ == '__main__':
  absltest.main()
