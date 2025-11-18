# Copyright 2023 Google LLC
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

"""Tests for time_integration."""

import functools
import math

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import time_integration
import jax
from jax import tree_util

import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


def assert_allclose(x, y, rtol=1e-7, atol=0, err_msg=''):
  jax.tree_util.tree_map(
      functools.partial(
          np.testing.assert_allclose, atol=atol, rtol=rtol, err_msg=err_msg),
      x, y)


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
    dict(testcase_name='_zero_derivative',
         explicit_terms=lambda x: 0 * x,
         implicit_terms=lambda x: 0 * x,
         implicit_inverse=lambda x, eta: x,
         dt=1e-2,
         inner_steps=10,
         outer_steps=5,
         initial_state=np.ones(10),
         closed_form=lambda x0, t: x0,
         tolerances=[1e-12] * 5),
    # x(t) = 5 * t * np.ones(3)
    dict(testcase_name='_constant_derivative',
         explicit_terms=lambda x: 5 * jnp.ones_like(x),
         implicit_terms=lambda x: 0 * x,
         implicit_inverse=lambda x, eta: x,
         dt=1e-2,
         inner_steps=10,
         outer_steps=5,
         initial_state=np.ones(3),
         closed_form=lambda x0, t: x0 + 5 * t,
         tolerances=[1e-12] * 5),
    # x(t) = np.arange(3) * np.exp(t)
    # Uses explicit terms only.
    dict(testcase_name='_linear_derivative_explicit',
         explicit_terms=lambda x: x,
         implicit_terms=lambda x: 0 * x,
         implicit_inverse=lambda x, eta: x,
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.arange(3.0),
         closed_form=lambda x0, t: np.arange(3) * jnp.exp(t),
         tolerances=[5e-2, 1e-4, 1e-6, 1e-9, 1e-6]),
    # x(t) = np.arange(3) * np.exp(t)
    # Uses implicit terms only.
    dict(testcase_name='_linear_derivative_implicit',
         explicit_terms=lambda x: 0 * x,
         implicit_terms=lambda x: x,
         implicit_inverse=lambda x, eta: x / (1 - eta),
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.arange(3.0),
         closed_form=lambda x0, t: np.arange(3) * jnp.exp(t),
         tolerances=[5e-2, 5e-5, 1e-5, 1e-5, 3e-5]),
    # x(t) = np.arange(3) * np.exp(t)
    # Splits the equation into an implicit and explicit term.
    dict(testcase_name='_linear_derivative_semi_implicit',
         explicit_terms=lambda x: x / 2,
         implicit_terms=lambda x: x / 2,
         implicit_inverse=lambda x, eta: x / (1 - eta / 2),
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.arange(3) * np.exp(0),
         closed_form=lambda x0, t: np.arange(3.0) * jnp.exp(t),
         tolerances=[1e-4, 2e-5, 2e-6, 1e-6, 2e-5]),
    dict(testcase_name='_harmonic_oscillator_explicit',
         explicit_terms=lambda x: jnp.stack([x[1], -x[0]]),
         implicit_terms=jnp.zeros_like,
         implicit_inverse=lambda x, eta: x,
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.ones(2),
         closed_form=harmonic_oscillator,
         tolerances=[1e-2, 3e-5, 6e-8, 5e-11, 6e-8]),
    dict(testcase_name='_harmonic_oscillator_implicit',
         explicit_terms=jnp.zeros_like,
         implicit_terms=lambda x: jnp.stack([x[1], -x[0]]),
         implicit_inverse=lambda x, eta: jnp.stack(  # pylint: disable=g-long-lambda
             [x[0] + eta * x[1], x[1] - eta * x[0]]) / (1 + eta ** 2),
         dt=1e-2,
         inner_steps=20,
         outer_steps=5,
         initial_state=np.ones(2),
         closed_form=harmonic_oscillator,
         tolerances=[1e-2, 2e-5, 2e-6, 1e-6, 6e-6]),
]


ALL_TIME_STEPPERS = [
    time_integration.backward_forward_euler,
    time_integration.crank_nicolson_rk2,
    time_integration.crank_nicolson_rk3,
    time_integration.crank_nicolson_rk4,
    time_integration.imex_rk_sil3,
]


class TimeIntegrationTest(parameterized.TestCase):

  @parameterized.named_parameters(ALL_TEST_PROBLEMS)
  def test_implicit_inverse(
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
    """Tests that time integration is accurate for a range of test cases."""
    del dt, explicit_terms, inner_steps, outer_steps, closed_form  # unused
    del tolerances  # unused

    # Verifies that `implicit_inverse` solves (y - eta * F(y)) = x
    # This does not test the integrator, but rather verifies that the test
    # case is valid.
    eta = 0.3
    solved_state = implicit_inverse(initial_state, eta)
    reconstructed_state = solved_state - eta * implicit_terms(solved_state)
    np.testing.assert_allclose(reconstructed_state, initial_state)

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
    expected = jax.vmap(closed_form, in_axes=(None, 0))(
        initial_state, time)

    # Compute trajectory using time-stepper.
    for atol, time_stepper in zip(tolerances, ALL_TIME_STEPPERS):
      with self.subTest(time_stepper.__name__):
        equation = CustomODE(explicit_terms, implicit_terms, implicit_inverse)
        semi_implicit_step = time_stepper(equation, dt)
        input_state = initial_state
        trajectory_fn = time_integration.trajectory_from_step(
            semi_implicit_step, outer_steps, inner_steps)
        _, actual = trajectory_fn(input_state)
        np.testing.assert_allclose(expected, actual, atol=atol, rtol=0)

  def test_pytree_state(self):
    equation = CustomODE(
        explicit_terms=lambda x: tree_util.tree_map(jnp.zeros_like, x),
        implicit_terms=lambda x: tree_util.tree_map(jnp.zeros_like, x),
        implicit_inverse=lambda x, eta: x,
    )
    u0 = {'x': 1.0, 'y': 1.0}
    for time_stepper in ALL_TIME_STEPPERS:
      with self.subTest(time_stepper.__name__):
        u1 = time_stepper(equation, 1.0)(u0)
        self.assertEqual(u0, u1)

  def test_multiple_equations(self):
    tolerances = [1e-4, 2e-5, 2e-6, 1e-6, 2e-5]
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
      expected = jax.vmap(closed_form, in_axes=(None, 0))(
          initial_state, time)

      # Compute trajectory using time-stepper.
      for atol, time_stepper in zip(tolerances, ALL_TIME_STEPPERS):
        with self.subTest(time_stepper.__name__):
          semi_implicit_step = time_stepper(equation, dt)
          input_state = initial_state
          trajectory_fn = time_integration.trajectory_from_step(
              semi_implicit_step, outer_steps, inner_steps)
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
      expected = jax.vmap(closed_form, in_axes=(None, 0))(
          initial_state, time)

      # Compute trajectory using time-stepper.
      for atol, time_stepper in zip(tolerances, ALL_TIME_STEPPERS):
        with self.subTest(time_stepper.__name__):
          semi_implicit_step = time_stepper(equation, dt)
          input_state = initial_state
          trajectory_fn = time_integration.trajectory_from_step(
              semi_implicit_step, outer_steps, inner_steps)
          _, actual = trajectory_fn(input_state)
          np.testing.assert_allclose(expected['s'], actual['s'], atol=atol)


class NestedCheckpointScanTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(f=lambda x, _: (x + 1, x ** 2), init=1, xs=None,
           lengths=[8]),
      dict(f=lambda x, _: (x + 1, x ** 2), init=1, xs=None,
           lengths=[2, 4]),
      dict(f=lambda x, _: (x + 1, x ** 2), init=1, xs=None,
           lengths=[1, 2, 2, 2]),
      dict(f=lambda x, y: (2 * x, x + y), init=1, xs=np.arange(8),
           lengths=[8]),
      dict(f=lambda x, y: (2 * x, x + y), init=1, xs=np.arange(8),
           lengths=[2, 4]),
      dict(f=lambda x, y: (2 * x, x + y), init=1, xs=np.arange(8),
           lengths=[2, 2, 2, 1]),
  )
  def test(self, f, init, xs, lengths):
    length = math.prod(lengths)
    expected_c, expected_ys = jax.lax.scan(f, init, xs, length)
    actual_c, actual_ys = time_integration.nested_checkpoint_scan(
        f, init, xs, length, nested_lengths=lengths)
    np.testing.assert_allclose(expected_c, actual_c)
    np.testing.assert_allclose(expected_ys, actual_ys)


class DigitalFilterInitiailizationTest(absltest.TestCase):

  def test_accumulate_repeated(self):
    result = time_integration.accumulate_repeated(
        lambda x: 2*x, jnp.arange(4), 1)
    self.assertEqual(result, 0*2 + 1*4 + 2*8 + 3*16)

  def test_dfi_lanczos_weights(self):
    weights = time_integration._dfi_lanczos_weights(10, 10, 0.1)
    self.assertEqual(weights.size, 50)
    self.assertTrue((np.diff(weights) < 0).all())
    self.assertAlmostEqual(weights[0], 1.0, places=2)
    self.assertAlmostEqual(weights[-1], 0.0, places=2)

  def test_digital_filter_initialization(self):
    eq = time_integration.ImplicitExplicitODE.from_functions(
        # x[0] is linear growth
        # x[1] and x[2] are a simple harmonic oscillator
        explicit_terms=lambda x: jnp.stack([0.1, x[2], -x[1]]),
        implicit_terms=jnp.zeros_like,
        implicit_inverse=lambda x, eta: x,
    )
    dfi = time_integration.digital_filter_initialization(
        equation=eq,
        ode_solver=time_integration.imex_rk_sil3,
        filters=[],
        time_span=20.0,
        cutoff_period=20.0,
        dt=0.01,
    )
    result = dfi(np.array([1.0, 1.0, 1.0]))
    expected = np.array([1.0, 0.0, 0.0])  # oscillating terms are filtered
    np.testing.assert_allclose(expected, result, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
