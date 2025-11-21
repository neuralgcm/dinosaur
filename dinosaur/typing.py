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

"""Defined commonly used types in the codebase."""

import dataclasses
from typing import Any, Callable, Generic, Mapping, TypeVar

from dinosaur import scales

import jax.numpy as jnp
import numpy as np
import tree_math


Array = np.ndarray | jnp.ndarray
ArrayOrArrayTuple = Array | tuple[Array, ...]
Numeric = float | int | Array
Unit = scales.Unit
Quantity = scales.Quantity
PRNGKeyArray = Any

# TODO(dkochkov): Rename PyTreeXXX to XXX once old types are pruned.
PyTreeState = TypeVar('PyTreeState')
Pytree = Any
PyTreeMemory = Pytree
PyTreeDiagnostics = Pytree

AuxFeatures = dict[str, Any]
DataState = dict[str, Any]
ForcingData = dict[str, Any]


@dataclasses.dataclass(eq=True, order=True, frozen=True)
class KeyWithCosLatFactor:
  """Class describing a key by `name` and an integer `factor_order`."""
  name: str
  factor_order: int
  filter_strength: float = 0.0


@tree_math.struct
class RandomnessState:
  """Representation of random states on the sphere.

  Attributes:
    core: internal representation of the random state.
    nodal_value: random field values in the nodal representation.
    modal_value: random field values in the modal representation.
    prng_key: underlying PRNG key.
    prng_step: optional iteration counter associated with PRNG key access, to
      allow for avoiding the pattern of iterative splitting the same key, which
      has poor statistical properties. The recommended pattern for generating a
      new PRNG key is `jax.random.fold_in(state.prng_key, state.prng_step)`.
  """
  core: Pytree | None = None
  nodal_value: Pytree | None = None
  modal_value: Pytree | None = None
  prng_key: PRNGKeyArray | None = None
  prng_step: int | None = None


@tree_math.struct
class ModelState(Generic[PyTreeState]):
  """PyTreeState decomposed into deterministic and perturbation components.

  A stochastic model advances by maintaining a model state and perturbations.

  Attributes:
    state: Prognostic variables describing the state of the atmosphere.
    memory: Optional model fields/predictions providing past time context.
    diagnostics: Optional diagnostic values computed in the model space.
    randomness: An optional random field that is used to stochastically perturb
      the advance step of the model.
  """
  state: PyTreeState
  memory: Pytree = dataclasses.field(default=None)
  diagnostics: Pytree = dataclasses.field(default_factory=dict)
  randomness: RandomnessState = dataclasses.field(
      default_factory=RandomnessState
  )


@tree_math.struct
class TrajectoryRepresentations:
  """Dataclass that holds trajectories in all default representations."""
  data_nodal_trajectory: Pytree
  data_modal_trajectory: Pytree
  model_nodal_trajectory: Pytree
  model_modal_trajectory: Pytree

  def get_representation(self, *, is_nodal: bool, is_encoded: bool) -> Pytree:
    """Retrieves representation based on `is_nodal`, `is_encoded` booleans."""
    binary_nodal_encoded_dict = {
        (True, True): self.model_nodal_trajectory,
        (True, False): self.data_nodal_trajectory,
        (False, True): self.model_modal_trajectory,
        (False, False): self.data_modal_trajectory,
    }
    return binary_nodal_encoded_dict[(is_nodal, is_encoded)]

# TODO(dkochkov) unify State and PyTreeState and integrators.
State = TypeVar('State')
StateFn = Callable[[State], State]
InverseFn = Callable[[State, jnp.ndarray], State]
StepFn = Callable[[State, State], State]
FilterFn = Callable[[State, State, State], tuple[State, State]]

ScanFn = Callable[..., Any]
PytreeFn = Callable[[Pytree], Pytree]
PyTreeTermsFn = Callable[[PyTreeState], PyTreeState]
PyTreeInverseFn = Callable[[PyTreeState, Numeric], PyTreeState]
TimeStepFn = Callable[[PyTreeState], PyTreeState]
PyTreeFilterFn = Callable[[PyTreeState], PyTreeState]
PyTreeStepFilterFn = Callable[[PyTreeState, PyTreeState], PyTreeState]
PyTreeStepFilterModule = Callable[..., PyTreeStepFilterFn]

Forcing = Pytree
ForcingFn = Callable[[ForcingData, float], Forcing]
ForcingModule = Callable[..., ForcingFn]

PostProcessFn = Callable[..., Any]
Params = Mapping[str, Mapping[str, Array]] | None

StepFn = Callable[[PyTreeState, Forcing | None], PyTreeState]
StepModule = Callable[..., StepFn]
CorrectorFn = Callable[
    [PyTreeState, PyTreeState | None, Forcing | None], PyTreeState
]
CorrectorModule = Callable[..., CorrectorFn]
ParameterizationFn = Callable[
    [
        PyTreeState,
        PyTreeMemory | None,
        PyTreeDiagnostics | None,
        RandomnessState | None,
        Forcing | None,
    ],
    PyTreeState,
]
ParameterizationModule = Callable[..., ParameterizationFn]
TrajectoryFn = Callable[..., tuple[Any, Any]]
TransformFn = Callable[[Pytree], Pytree]
TransformModule = Callable[..., TransformFn]

GatingFactory = Callable[..., Callable[[Array, Array], Array]]
TowerFactory = Callable[..., Callable[..., Any]]
LayerFactory = Callable[..., Callable[..., Any]]

EmbeddingFn = Callable[
    [
        Pytree,
        PyTreeMemory | None,
        PyTreeDiagnostics | None,
        RandomnessState | None,
        Forcing | None,
    ],
    Pytree,
]
EmbeddingModule = Callable[..., EmbeddingFn]
