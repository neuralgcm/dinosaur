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

"""Defines the dinosaur module."""

import dinosaur.coordinate_systems
import dinosaur.filtering
import dinosaur.held_suarez
import dinosaur.horizontal_interpolation
import dinosaur.jax_numpy_utils
import dinosaur.layer_coordinates
import dinosaur.primitive_equations
import dinosaur.primitive_equations_states
import dinosaur.pytree_utils
import dinosaur.radiation
import dinosaur.scales
import dinosaur.shallow_water
import dinosaur.shallow_water_states
import dinosaur.sigma_coordinates
import dinosaur.spherical_harmonic
import dinosaur.time_integration
import dinosaur.typing
import dinosaur.vertical_interpolation
import dinosaur.weatherbench_utils
import dinosaur.xarray_utils

__version__ = "1.3.2"  # keep sync with pyproject.toml
