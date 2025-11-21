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
"""A class and protocol that holds physical constants and scaling routines."""

from __future__ import annotations

import dataclasses
from typing import Protocol

from dinosaur import scales
from dinosaur import typing
import numpy as np

# For consistency with commonly accepted notation, we use Greek letters within
# some of the functions below.
# pylint: disable=invalid-name

Quantity = typing.Quantity
Numeric = typing.Numeric


class SimUnitsProtocol(Protocol):
  """Protocol for a class that handles dimensionalization of quantities."""

  radius: float
  angular_velocity: float
  gravity_acceleration: float
  ideal_gas_constant: float
  water_vapor_gas_constant: float
  water_vapor_isobaric_heat_capacity: float
  kappa: float
  scale: scales.ScaleProtocol

  @property
  def R(self) -> float:
    ...

  @property
  def R_vapor(self) -> float:
    ...

  @property
  def g(self) -> float:
    ...

  @property
  def Cp(self) -> float:
    ...

  @property
  def Cp_vapor(self) -> float:
    ...

  def nondimensionalize(self, quantity: Quantity) -> Numeric:
    ...

  def nondimensionalize_timedelta64(self, timedelta: np.timedelta64) -> Numeric:
    ...

  def dimensionalize(self, value: Numeric, unit: typing.Unit) -> Quantity:
    ...

  def dimensionalize_timedelta64(self, value: Numeric) -> np.timedelta64:
    ...

  @classmethod
  def from_si(
      cls,
      radius_si: Quantity = scales.RADIUS,
      angular_velocity_si: Quantity = scales.ANGULAR_VELOCITY,
      gravity_acceleration_si: Quantity = scales.GRAVITY_ACCELERATION,
      ideal_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT,
      water_vapor_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT_H20,
      water_vapor_isobaric_heat_capacity_si: Quantity = scales.WATER_VAPOR_CP,
      kappa_si: Quantity = scales.KAPPA,
      scale: scales.ScaleProtocol = scales.DEFAULT_SCALE,
  ) -> 'SimUnits':
    ...


@dataclasses.dataclass(frozen=True)
class SimUnits:
  """A class that holds physical constants and scaling routines.

  This class stores non-dimensional physical constants that are used in
  simulations and provides routines for dimensionalization and
  non-dimensionalization of quantities.

  Attributes:
    radius: the non-dimensionalized radius of the domain.
    angular_velocity: the non-dimensionalized angular velocity of the rotating
      domain.
    gravity_acceleration: the non-dimensionalized value of gravitational
      acceleration.
    ideal_gas_constant: the non-dimensionalized gas constant.
    water_vapor_gas_constant: the non-dimensionalized gas constant for vapor.
    water_vapor_isobaric_heat_capacity: isobaric heat capacity of vapor.
    kappa: `ideal_gas_constant / Cp` where  Cp is the isobaric heat capacity.
    scale: an instance implementing `ScaleProtocol` that will be used to
      (non-)dimensionalize quantities.
  """

  radius: float
  angular_velocity: float
  gravity_acceleration: float
  ideal_gas_constant: float
  water_vapor_gas_constant: float
  water_vapor_isobaric_heat_capacity: float
  kappa: float
  scale: scales.ScaleProtocol

  @property
  def R(self) -> float:
    """Alias for `ideal_gas_constant`."""
    return self.ideal_gas_constant

  @property
  def R_vapor(self) -> float:
    """Alias for `ideal_gas_constant`."""
    return self.water_vapor_gas_constant

  @property
  def g(self) -> float:
    """Alias for `gravity_acceleration`."""
    return self.gravity_acceleration

  @property
  def Cp(self) -> float:
    """Isobaric heat capacity."""
    return self.ideal_gas_constant / self.kappa

  @property
  def Cp_vapor(self) -> float:
    """Alias for `water_vapor_isobaric_heat_capacity`."""
    return self.water_vapor_isobaric_heat_capacity

  def nondimensionalize(self, quantity: Quantity) -> Numeric:
    """Non-dimensionalizes and rescales `quantity`."""
    return self.scale.nondimensionalize(quantity)

  def nondimensionalize_timedelta64(self, timedelta: np.timedelta64) -> Numeric:
    """Non-dimensionalizes and rescales a numpy timedelta."""
    base_unit = 's'
    return self.scale.nondimensionalize(
        timedelta / np.timedelta64(1, base_unit) * scales.units(base_unit)
    )

  def dimensionalize(self, value: Numeric, unit: typing.Unit) -> Quantity:
    """Rescales and adds units to the given non-dimensional value."""
    return self.scale.dimensionalize(value, unit)

  def dimensionalize_timedelta64(self, value: Numeric) -> np.timedelta64:
    """Rescales and casts the given non-dimensional value to timedelta64."""
    base_unit = 's'  # return value is rounded down to nearest base_unit
    dt = self.scale.dimensionalize(value, scales.units(base_unit)).m
    if isinstance(dt, np.ndarray):
      return dt.astype(f'timedelta64[{base_unit}]')
    else:
      return np.timedelta64(int(dt), base_unit)

  @classmethod
  def from_si(
      cls,
      radius_si: Quantity = scales.RADIUS,
      angular_velocity_si: Quantity = scales.ANGULAR_VELOCITY,
      gravity_acceleration_si: Quantity = scales.GRAVITY_ACCELERATION,
      ideal_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT,
      water_vapor_gas_constant_si: Quantity = scales.IDEAL_GAS_CONSTANT_H20,
      water_vapor_isobaric_heat_capacity_si: Quantity = scales.WATER_VAPOR_CP,
      kappa_si: Quantity = scales.KAPPA,
      scale: scales.ScaleProtocol = scales.DEFAULT_SCALE,
  ) -> SimUnits:
    # pylint: disable=g-doc-args,g-doc-return-or-yield
    """Constructs `SimUnits` from constants with units.

    By default uses units in which the radius and angular_velocity are set to
    one.
    """
    return cls(
        scale.nondimensionalize(radius_si),
        scale.nondimensionalize(angular_velocity_si),
        scale.nondimensionalize(gravity_acceleration_si),
        scale.nondimensionalize(ideal_gas_constant_si),
        scale.nondimensionalize(water_vapor_gas_constant_si),
        scale.nondimensionalize(water_vapor_isobaric_heat_capacity_si),
        scale.nondimensionalize(kappa_si),
        scale,
    )
