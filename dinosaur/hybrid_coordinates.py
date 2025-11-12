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

"""A vertical coordinate system that interpolates between sigma and pressure."""

from __future__ import annotations

import dataclasses
import functools
import importlib

import dinosaur
from dinosaur import sigma_coordinates
from dinosaur import typing
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


Array = typing.Array
einsum = functools.partial(jnp.einsum, precision=lax.Precision.HIGHEST)
# For consistency with commonly accepted notation, we use Greek letters within
# some of the functions below.
# pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class HybridCoordinates:
  """Specifies the vertical coordinate with hybrid sigma-pressure levels.

  This allows for matching the vertical coordinate system used by ECMWF and most
  other operational forecasting systems.

  The pressure corresponding to a level is given by the formula `a + b * sp`
  where `sp` is surface pressure.

  Attributes:
    a_boundaries: offset coefficient for the boundaries of each level, starting
      at the level closest to the top of the atmosphere.
    b_boundaries: slope coefficient for the boundaries of each level, starting
      at the level closest to the top of the atmosphere.
    layers: number of vertical layers.
  """

  a_boundaries: np.ndarray
  b_boundaries: np.ndarray

  def __post_init__(self):
    if len(self.a_boundaries) != len(self.b_boundaries):
      raise ValueError(
          'Expected `a_boundaries` and `b_boundaries` to have the same length, '
          f'got {len(self.a_boundaries)} and {len(self.b_boundaries)}.'
      )

  @classmethod
  def from_coefficients(
      cls,
      a_coeffs: list[float] | np.ndarray,
      b_coeffs: list[float] | np.ndarray,
      p0: float = 1000.0,
  ) -> HybridCoordinates:
    """Creates coordinates from A, B coefficients and reference pressure.

    Implements the formulation: P(k) = A(k) * p0 + B(k) * Ps

    This is common in NCAR/CAM literature.

    Args:
        a_coeffs: Dimensionless A coefficients.
        b_coeffs: Dimensionless B coefficients (sigma component).
        p0: Reference pressure in hPa (default 1000 hPa).
    """
    a_boundaries = np.array(a_coeffs) * p0
    return cls(a_boundaries=a_boundaries, b_boundaries=np.array(b_coeffs))

  @classmethod
  def analytic_levels(
      cls,
      n_levels: int,
      p_top: float = 0.0,
      p0: float = 1000.0,
      sigma_exponent: float = 3.0,
      stretch_exponent: float = 2.0,
  ) -> HybridCoordinates:
    """Generates analytically smooth hybrid coordinates.

    This uses a power-law strategy to blend from pressure to sigma.

    The vertical domain is defined by a master coordinate `eta` in [0, 1].
    The distribution of `eta` is stretched to concentrate levels near the
    surface.

    The pressure is then split into A and B coefficients such that:
        P(eta) = A(eta) * p_ref + B(eta) * p_s

    When p_s == p_ref, the total pressure is exactly P(eta).

    Args:
        n_levels: Number of vertical layers (resulting in n_levels + 1
          interfaces).
        p_top: The pressure at the top of the model (in Pascals).
        p0: Reference surface pressure (P0) for defining A coefficients.
        sigma_exponent: Controls the "hybridization". 1.0 = Pure Sigma (terrain
          following everywhere). >1.0 = Hybrid (becomes more isobaric aloft).
          Typical values are 3.0 to 5.0 for Earth-like atmospheres.
        stretch_exponent: Controls vertical resolution spacing. 1.0 = Linear
          spacing. >1.0 = Concentrates levels near the surface (PBL).
    """
    # We use a power law to concentrate resolution at the surface.
    k = np.linspace(0, 1, n_levels + 1)
    eta = k**stretch_exponent

    # This connects p_top (at eta=0) to p_ref (at eta=1)
    p_profile = p_top + eta * (p0 - p_top)

    # B must be 0 at top and 1 at surface.
    # We use a power law: B = eta^r.
    # Higher `sigma_exponent` means B decays faster, making the grid
    # "pure pressure" (flat) more quickly as you go up.
    b_boundaries = eta**sigma_exponent

    # derived from the constraint: P_profile = A * p_ref + B * p_ref
    # Therefore: A = (P_profile / p_ref) - B
    a_boundaries = (p_profile / p0) - b_boundaries

    # Enforce machine precision consistency at boundaries
    a_boundaries[0] = p_top / p0  # Top is pure pressure (if B[0]=0)
    b_boundaries[0] = 0.0
    a_boundaries[-1] = 0.0  # Surface is pure sigma
    b_boundaries[-1] = 1.0
    return cls.from_coefficients(a_boundaries, b_boundaries, p0)

  def get_eta(self, p_surface: float) -> np.ndarray:
    """Returns the eta values for a given pressure."""
    etas = self.a_centers / p_surface + self.b_centers
    return etas

  @property
  def pressure_thickness(self) -> np.ndarray:
    """Returns thickness of pressure part of hybrid coordinates."""
    return np.diff(self.a_boundaries)

  @property
  def sigma_thickness(self) -> np.ndarray:
    """Returns thickness of sigma part of hybrid coordinates."""
    return np.diff(self.b_boundaries)

  @property
  def a_centers(self) -> np.ndarray:
    """Returns center values for pressure part of hybrid coordinates."""
    return (self.a_boundaries[1:] + self.a_boundaries[:-1]) / 2

  @property
  def b_centers(self) -> np.ndarray:
    """Returns center values for sigma part of hybrid coordinates."""
    return (self.b_boundaries[1:] + self.b_boundaries[:-1]) / 2

  @property
  def center_to_center(self) -> np.ndarray:
    """Returns center-to-center distance in sigma part of coordinates."""
    return np.diff(self.b_centers)

  def pressure_boundaries(self, surface_pressure: typing.Numeric) -> Array:
    """Returns boundaries of each layer in pressure units."""
    return (
        self.a_boundaries[:, np.newaxis, np.newaxis]
        + self.b_boundaries[:, np.newaxis, np.newaxis] * surface_pressure
    )

  def pressure_centers(self, surface_pressure: typing.Numeric) -> Array:
    """Returns centers of each layer in pressure units."""
    boundaries = self.pressure_boundaries(surface_pressure)
    return (boundaries[1:] + boundaries[:-1]) / 2

  @classmethod
  def from_sigma_levels(
      cls, sigma_levels: sigma_coordinates.SigmaCoordinates
  ) -> HybridCoordinates:
    """Creates hybrid coordinates that effectively are sigma coordinates."""
    b_boundaries = np.array(sigma_levels.boundaries)
    a_boundaries = np.zeros_like(b_boundaries)
    return cls(a_boundaries=a_boundaries, b_boundaries=b_boundaries)

  @classmethod
  def _from_resource_csv(cls, path: str) -> HybridCoordinates:
    levels_csv = importlib.resources.files(dinosaur).joinpath(path)
    with levels_csv.open() as f:
      a_in_pa, b = np.loadtxt(f, skiprows=1, usecols=(1, 2), delimiter='\t').T
    a = a_in_pa / 100  # convert from Pa to hPa
    # any reasonable hybrid coordinate system falls in this range (certainly
    # including UFS and ECMWF)
    assert 100 < a.max() < 1000
    return cls(a_boundaries=a, b_boundaries=b)

  @classmethod
  def ECMWF137(cls) -> HybridCoordinates:  # pylint: disable=invalid-name
    """Returns the 137 model levels used by ECMWF's IFS (e.g., in ERA5).

    Pressure is returned in units of hPa.

    For details, see the ECMWF wiki:
    https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
    """
    return cls._from_resource_csv('data/ecmwf137_hybrid_levels.csv')

  @classmethod
  def UFS127(cls) -> HybridCoordinates:  # pylint: disable=invalid-name
    """Returns the 127 model levels used by NOAA's UFS (GFS v16).

    Pressure is returned in units of hPa.

    For details, see the documentation for UFS Replay:
    https://ufs2arco.readthedocs.io/en/latest/example_pressure_interpolation.html

    Source data:
    https://github.com/NOAA-PSL/ufs2arco/blob/v0.1.2/ufs2arco/replay_vertical_levels.yaml
    """
    return cls._from_resource_csv('data/ufs127_hybrid_levels.csv')

  @property
  def layers(self) -> int:
    return len(self.a_boundaries) - 1

  def __hash__(self):
    return hash(
        (tuple(self.a_boundaries.tolist()), tuple(self.b_boundaries.tolist()))
    )

  def __eq__(self, other):
    return (
        isinstance(other, HybridCoordinates)
        and np.array_equal(self.a_boundaries, other.a_boundaries)
        and np.array_equal(self.b_boundaries, other.b_boundaries)
    )

  def get_sigma_boundaries(
      self, surface_pressure: typing.Numeric
  ) -> typing.Array:
    """Returns centers of sigma levels for a given surface pressure.

    Args:
      surface_pressure: scalar surface pressure, in the same units as
        `a_boundaries`.

    Returns:
      Array with shape `(layers + 1,)`.
    """
    return self.a_boundaries / surface_pressure + self.b_boundaries

  def get_sigma_centers(self, surface_pressure: typing.Numeric) -> typing.Array:
    """Returns centers of sigma levels for a given surface pressure.

    Args:
      surface_pressure: scalar surface pressure, in the same units as
        `a_boundaries`.

    Returns:
      Array with shape `(layers,)`.
    """
    boundaries = self.get_sigma_boundaries(surface_pressure)
    return (boundaries[1:] + boundaries[:-1]) / 2

  def to_approx_sigma_coords(
      self, layers: int, surface_pressure: float = 1013.25
  ) -> sigma_coordinates.SigmaCoordinates:
    """Interpolate these hybrid coordinates to approximate sigma levels.

    The resulting coordinates will typically not be equidistant.

    Args:
      layers: number of sigma layers to return.
      surface_pressure: reference surface pressure to use for interpolation. The
        default value is 1013.25, which is one standard atmosphere in hPa.

    Returns:
      New SigmaCoordinates object wih the requested number of layers.
    """
    original_bounds = self.get_sigma_boundaries(surface_pressure)
    interpolated_bounds = jax.vmap(jnp.interp, (0, None, None))(
        jnp.linspace(0, 1, num=layers + 1),
        jnp.linspace(0, 1, num=self.layers + 1),
        original_bounds,
    )
    interpolated_bounds = np.array(interpolated_bounds)
    # Some hybrid coordinates (e.g., from UFS) start at a non-zero pressure
    # level. It is not clear that this makes sense for Dinosaur, so to be safe,
    # we set the first level to 0 (zero pressure) and the last level to 1
    # (surface pressure).
    interpolated_bounds[0] = 0.0
    interpolated_bounds[-1] = 1.0
    return sigma_coordinates.SigmaCoordinates(interpolated_bounds)
