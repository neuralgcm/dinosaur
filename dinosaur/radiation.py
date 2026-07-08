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

"""Top of atmosphere incident solar radiation.

This code was inspired by third_party/py/pysolar, however that library is
intended as a surface of Earth solar radiation model that includes atmospheric
scattering effects. Of particularl note, the constants used to compute solar
irradiance are different. References included below.

Another major difference is that this module represents time using radians,
which greatly reduces the number of conversions by factors of pi.
"""

from __future__ import annotations

import datetime

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np
import tree_math

units = scales.units

Array = typing.Array
Numeric = typing.Numeric

DAYS_PER_YEAR = 365.25
MINUTES_PER_DAY = 1440
SECONDS_PER_DAY = 86400
# TSI: Energy input to the top of the Earth's atmosphere
# https://www.ncei.noaa.gov/products/climate-data-records/total-solar-irradiance
TOTAL_SOLAR_IRRADIANCE = 1361 * units.W / units.meter**2  # pyrefly: ignore[unsupported-operation]
# Seasonal variation in apparent solar irradiance due to Earth-Sun distance
SOLAR_IRRADIANCE_VARIATION = 47 * units.W / units.meter**2  # .5 * 6.9% * TSI  # pyrefly: ignore[unsupported-operation]
# Approximate perihelion, when Earth is closest to the sun (Jan 3rd)
PERIHELION = 3 * 2 * jnp.pi / DAYS_PER_YEAR  # radians
# Approximate equinox (March 20 on non-leap year), when dihedral is zero
SPRING_EQUINOX = 79 * 2 * jnp.pi / DAYS_PER_YEAR  # radians
# Angle between Earth's rotational axis and its orbital axis
EARTH_AXIS_INCLINATION = 23.45 * jnp.pi / 180  # radians

# Reference datetime for WeatherBench
WB_REFERENCE_DATETIME = datetime.datetime(1979, 1, 1, 0, 0)


@tree_math.struct
class OrbitalTime:
  """Nondimensional time based on orbital dynamics.

  Attributes:
    orbital_phase: phase of the Earth's orbit around the Sun in radians. The
      values 0, 2pi correspond to January 1st, midnight UTC.
    synodic_phase: phase of the Earth's rotation around its axis in radians,
      relative to the Sun. The values 0, 2pi correspond to midnight UTC.
  """

  orbital_phase: float
  synodic_phase: float


def datetime64_to_datetime(when: np.datetime64) -> datetime.datetime:
  """Returns datetime corresponding to the provided numpy datetime64."""
  ts = (when - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
  return datetime.datetime.utcfromtimestamp(ts)


def days_in_year(when: datetime.datetime) -> int:
  """Returns the number of days in the year of the provided datetime."""
  return datetime.datetime(year=when.year, month=12, day=31).timetuple().tm_yday


def datetime_to_orbital_time(when: datetime.datetime) -> OrbitalTime:
  """Returns the OrbitalTime associated with the provided datetime."""
  days_this_year = days_in_year(when)
  full_days = when.timetuple().tm_yday - 1
  fraction_of_day = (60 * when.hour + when.minute) / MINUTES_PER_DAY
  fraction_of_year = (full_days + fraction_of_day) / days_this_year
  return OrbitalTime(
      orbital_phase=2 * jnp.pi * fraction_of_year,  # pyrefly: ignore[unexpected-keyword]
      synodic_phase=2 * jnp.pi * fraction_of_day,  # pyrefly: ignore[unexpected-keyword]
  )


def datetime_to_time(
    when: datetime.datetime | np.datetime64,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    reference_datetime: datetime.datetime | np.datetime64,
) -> float:
  """Returns nondimensional time corresponding to the specified datetime.

  Args:
    when: datetime for which to compute nondimensional time.
    physics_specs: object holding physical constants and definition of custom
      units to use for initialization of the state.
    reference_datetime: datetime corresponding to nondimensionalized time = 0.
  """
  if isinstance(when, np.datetime64):
    when = datetime64_to_datetime(when)
  if isinstance(reference_datetime, np.datetime64):
    reference_datetime = datetime64_to_datetime(reference_datetime)
  difference = when - reference_datetime
  days = difference.days + difference.seconds / SECONDS_PER_DAY
  return physics_specs.nondimensionalize(days * units.day)  # pytype: disable=bad-return-type  # jax-ndarray


def get_direct_solar_irradiance(
    orbital_phase: Numeric,
    mean_irradiance: Numeric = TOTAL_SOLAR_IRRADIANCE,
    variation: Numeric = SOLAR_IRRADIANCE_VARIATION,
    perihelion: Numeric = PERIHELION,
) -> jnp.ndarray:
  """Returns solar radiation flux incident on the top of the atmosphere.

  Formula includes 6.9% seasonal variation due to Earth's elliptical orbit, but
  neglects the 0.1% variation of the 11-year solar cycle (Schwabe cycle).
  https://earth.gsfc.nasa.gov/climate/research/solar-radiation
  https://en.wikipedia.org/wiki/Solar_constant

  Note that the default values here are different from third_party/py/pysolar,
  which uses `1160 + (75 * math.sin(2 * math.pi / 365 * (day - 275)))`. This is
  presumably because pysolar is intended for modeling radiation incident on
  Earth's surface, including atmospheric scattering.

  Args:
    orbital_phase: phase of the Earth's orbit around the Sun in radians. The
      values 0, 2pi correspond to January 1st, midnight UTC.
    mean_irradiance: average annual solar flux just outside Earth's atmosphere.
      Default is the "total solar irradiance", or "solar consant", in W/m^2.
    variation: amplitude of fluctuation in solar flux due to Earth's elliptical
      orbit. Default is 0.5 * 6.9% of TSI. Units should match mean_irradiance.
    perihelion: orbital phase in radians where the Earth is closest to the Sun.
  """
  return mean_irradiance + variation * jnp.cos(orbital_phase - perihelion)  # pyrefly: ignore[bad-return]


def get_declination(orbital_phase: Numeric) -> jnp.ndarray:
  """Returns angle between the Earth-Sun line and the Earth equitorial plane."""
  # https://en.wikipedia.org/wiki/Declination
  return EARTH_AXIS_INCLINATION * jnp.sin(orbital_phase - SPRING_EQUINOX)


def equation_of_time(orbital_phase: Numeric) -> jnp.ndarray:
  """Returns the value to add to mean solar time to get actual solar time."""
  # https://en.wikipedia.org/wiki/Equation_of_time
  b = orbital_phase - SPRING_EQUINOX
  added_minutes = 9.87 * jnp.sin(2 * b) - 7.53 * jnp.cos(b) - 1.5 * jnp.sin(b)
  # Output normalized as a correction to synodic_phase
  return 2 * jnp.pi * added_minutes / MINUTES_PER_DAY


def get_hour_angle(
    orbital_phase: Numeric, synodic_phase: Numeric, longitude: Array
) -> jnp.ndarray:
  """Angular displacement of the sun east or west of the local meridian."""
  # https://en.wikipedia.org/wiki/Hour_angle
  solar_time = synodic_phase + equation_of_time(orbital_phase) + longitude
  return solar_time - jnp.pi  # pyrefly: ignore[bad-return]


def get_solar_sin_altitude(
    orbital_phase: Numeric,
    synodic_phase: Numeric,
    longitude: Array,
    latitude: Array,
) -> jnp.ndarray:
  """Returns sine of the solar altitude angle."""
  # https://en.wikipedia.org/wiki/Solar_zenith_angle
  declination = get_declination(orbital_phase)
  hour_angle = get_hour_angle(orbital_phase, synodic_phase, longitude)
  first_term = jnp.cos(latitude) * jnp.cos(declination) * jnp.cos(hour_angle)
  second_term = jnp.sin(latitude) * jnp.sin(declination)
  return first_term + second_term


def get_radiation_flux(
    orbital_time: OrbitalTime,
    longitude: Array,
    latitude: Array,
    mean_irradiance: Numeric = TOTAL_SOLAR_IRRADIANCE,
    variation: Numeric = SOLAR_IRRADIANCE_VARIATION,
) -> jnp.ndarray:
  """Returns TOA incident radiation flux."""
  sin_altitude = get_solar_sin_altitude(
      orbital_phase=orbital_time.orbital_phase,
      synodic_phase=orbital_time.synodic_phase,
      longitude=longitude,
      latitude=latitude,
  )
  is_daytime = sin_altitude > 0
  flux = get_direct_solar_irradiance(
      orbital_phase=orbital_time.orbital_phase,
      mean_irradiance=mean_irradiance,
      variation=variation,
  )
  return flux * is_daytime * sin_altitude


def get_normalized_radiation_flux(
    orbital_time: OrbitalTime,
    longitude: Array,
    latitude: Array,
    mean_irradiance: Numeric = TOTAL_SOLAR_IRRADIANCE,
    variation: Numeric = SOLAR_IRRADIANCE_VARIATION,
) -> jnp.ndarray:
  """Returns TOA incident radiation flux, normalized to between -1 and 1."""
  scale = mean_irradiance + variation
  return get_radiation_flux(
      orbital_time,
      longitude,
      latitude,
      mean_irradiance=mean_irradiance / scale,
      variation=variation / scale,
  )


class SolarRadiation:
  """Top of atmosphere incident solar radiation (TISR)."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      reference_datetime: datetime.datetime | np.datetime64,
  ):
    """Initialize SolarRadiation.

    Args:
      coords: horizontal and vertical descritization.
      physics_specs: object holding physical constants and definition of custom
        units to use for initialization of the state.
      reference_datetime: datetime corresponding to nondimensionalized time = 0.
    """
    if isinstance(reference_datetime, np.datetime64):
      reference_datetime = datetime64_to_datetime(reference_datetime)
    self.reference_datetime = reference_datetime
    self.reference_orbital_time = datetime_to_orbital_time(reference_datetime)

    self.coords = coords
    self.lon, sin_lat = self.coords.horizontal.nodal_mesh
    self.lat = np.arcsin(sin_lat)
    self.orbital_rate = jax.tree_util.tree_map(
        physics_specs.nondimensionalize,
        OrbitalTime(2 * jnp.pi / units.year, 2 * jnp.pi / units.day),  # pyrefly: ignore[bad-argument-count, unsupported-operation]
    )

    self.total_solar_irradiance = physics_specs.nondimensionalize(
        TOTAL_SOLAR_IRRADIANCE
    )
    self.solar_irradiance_variation = physics_specs.nondimensionalize(
        SOLAR_IRRADIANCE_VARIATION
    )
    self.physics_specs = physics_specs

  def datetime_to_time(self, when: datetime.datetime | np.datetime64) -> float:
    """Returns nondimensional time corresponding to the specified datetime."""
    return datetime_to_time(when, self.physics_specs, self.reference_datetime)

  def time_to_orbital_time(self, time: Numeric) -> OrbitalTime:
    """Returns the OribtalTime corresponding to the specified nondim time."""
    orbital_time = self.reference_orbital_time + self.orbital_rate * time
    # Reduce the magnitude of the result to avoid loss of precision errors
    # downstream. Avoid jnp.fmod, which is not very precise on float32.
    orbital_time -= orbital_time // (2 * jnp.pi) * (2 * jnp.pi)
    return orbital_time

  def solar_hour_angle(self, time: Numeric) -> jnp.ndarray:
    """Returns solar hour angle in radians."""
    now = self.time_to_orbital_time(time)
    return get_hour_angle(
        orbital_phase=now.orbital_phase,
        synodic_phase=now.synodic_phase,
        longitude=self.lon,
    )

  def radiation_flux(self, time: Numeric) -> jnp.ndarray:
    """Returns non-dimensionalized TOA incident solar radiation flux."""
    now = self.time_to_orbital_time(time)
    return get_radiation_flux(
        now,
        self.lon,
        self.lat,
        mean_irradiance=self.total_solar_irradiance,
        variation=self.solar_irradiance_variation,
    )

  @classmethod
  def normalized(
      cls,
      coords: coordinate_systems.CoordinateSystem,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      reference_datetime: datetime.datetime | np.datetime64,
  ) -> SolarRadiation:
    """Initialize SolarRadiation for normalized solar radiation."""
    this = cls(
        coords=coords,
        physics_specs=physics_specs,
        reference_datetime=reference_datetime,
    )
    scale = this.total_solar_irradiance + this.solar_irradiance_variation
    this.total_solar_irradiance /= scale
    this.solar_irradiance_variation /= scale
    return this
