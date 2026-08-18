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

"""Defines structures and functions relevant to modeling WeatherBench data."""

import dataclasses

from dinosaur import typing
import jax
import tree_math


@tree_math.struct
class State:
  """A WeatherBench state described using velocity components."""
  u: typing.Array
  v: typing.Array
  t: typing.Array
  z: typing.Array
  sim_time: float | typing.Array | jax.sharding.PartitionSpec | None
  tracers: dict[str, typing.Array] = dataclasses.field(default_factory=dict)
  diagnostics: dict[str, typing.Array] = dataclasses.field(default_factory=dict)
