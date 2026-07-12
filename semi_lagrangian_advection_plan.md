# Semi-Lagrangian advection in Dinosaur: implementation plan (initial version)

*Plan for [issue #55](https://github.com/neuralgcm/dinosaur/issues/55). Drafted 2026-07-11.*

## 1. Goals and non-goals

Semi-Lagrangian (SL) advection is the key ingredient that lets spectral
transform models like ECMWF's IFS take ~6x longer time steps than Eulerian
advection permits (Diamantakis 2014). Since the dynamical core dominates
NeuralGCM cost, SL advection is a path to substantially faster training and
inference. A second, independent motivation raised on issue #55: grid-point SL
transport is local, so it can carry sharp tracer fields (aerosol plumes,
chemistry) that the current spectral transport cannot represent without Gibbs
ringing and negative concentrations.

**Goals of this initial version**

1. A semi-Lagrangian treatment of advection for the primitive equations on
   sigma coordinates that is **compatible with Dinosaur's implicit-explicit
   (IMEX) solver structure** (`explicit_terms` / `implicit_terms` /
   `implicit_inverse`), reusing the existing implicit solve unchanged.
2. A **one-step (Runge-Kutta-style) time discretization** in the spirit of
   Dinosaur's preferred `imex_rk_sil3` / `crank_nicolson_rk2` solvers — *not* a
   two-time-level extrapolating scheme (SETTLS) or three-time-level leapfrog
   like those used operationally at ECMWF, which require multistep memory and
   have documented extrapolation instabilities.
3. Straightforward, correct interpolation on the **existing full Gaussian
   grids**: gather-based linear and cubic-Lagrange interpolation with proper
   longitude periodicity and cross-pole halos.
4. **Tracer transport with an opt-in quasi-monotone limiter** (per-tracer), the
   feature explicitly requested by downstream users for aerosol/chemistry
   positivity.
5. Everything **differentiable** (fixed iteration counts, gather-based
   interpolation, no data-dependent control flow), unit-tested, and validated
   against the Eulerian core on standard test cases.

**Non-goals (deferred; see §11)**

- No grid changes: no reduced/octahedral Gaussian grids, no HEALPix. Departure
  points near the poles may span many longitude cells; on the full Gaussian
  grid we simply gather at whatever indices result.
- No TPU-efficiency work: no static-stencil "matmul gather" formulation, no
  quasi-cubic (32-point) cost optimization, no data layout changes. Plain
  `jnp.take`-style gathers are acceptable even if slow on TPU.
- No inherently conservative SL (SLICE/CSLAM), no mass fixers beyond an
  optional trivial global rescaling.
- No hybrid-coordinate support in the first version (sigma only; the hybrid
  class is a straightforward follow-up once sigma works).
- No changes to the default Eulerian code path; SL is a parallel opt-in
  equation class + steppers.

## 2. Where advection lives in Dinosaur today

State (`primitive_equations.State`) holds modal (spherical-harmonic)
`vorticity`, `divergence`, `temperature_variation` (T′ = T − T_ref(σ)),
`log_surface_pressure`, and modal `tracers`. Each call to
`PrimitiveEquationsSigma.explicit_terms`:

- transforms to nodal space (`compute_diagnostic_state_sigma`), computing
  nodal winds `cos_lat_u` via `spherical_harmonic.get_cos_lat_vector`, the
  sigma-velocity `sigma_dot_{explicit,full}` at layer interfaces, and
  `u_dot_grad_log_sp`;
- computes **advection terms pseudo-spectrally**: for scalars the flux-form
  identity `v·∇ψ = ∇·(vψ) − ψ∇·v` (`horizontal_scalar_advection`) plus
  centered vertical advection (`sigma_coordinates.centered_vertical_advection`);
  for momentum the vector-invariant form, i.e. rotational term `(ζ+f) k×v`
  plus the kinetic-energy gradient `∇(|v|²/2)` (`curl_and_div_tendencies`,
  `kinetic_energy_tendency`);
- computes non-advective terms: pressure-gradient/geopotential terms, adiabatic
  heating `κT_v ω/p`, orography, moisture corrections, and the
  `log_surface_pressure` tendency.

`implicit_terms` is the standard linearization about a resting isothermal-ish
reference state (T_ref(σ), constant lnpₛ): gravity-wave coupling of
(δ, T′, lnpₛ) via the geopotential matrix **G**, the thermodynamic matrix
**H**, and the layer-thickness row; `implicit_inverse` solves the per-total-
wavenumber block system with precomputed NumPy inverses. Vorticity and tracers
have zero implicit terms.

Time steppers (`time_integration.py`) are one-step IMEX methods:
`crank_nicolson_rk2` (Heun + trapezoidal), the general `imex_runge_kutta`
driven by an `ImExButcherTableau`, and the default `imex_rk_sil3`
(Whitaker & Kar 2013). Modal filters (hyperdiffusion / exponential) are
applied per-step via `step_with_filters`.

Also relevant precedent: `primitive_equations.semi_lagrangian_vertical_advection_step_sigma`
already implements a first-order, vertical-only SL step via 1-D interpolation —
this plan generalizes that idea to full 3-D transport inside the IMEX loop.

## 3. Time discretization: a one-step semi-Lagrangian IMEX-RK scheme

### 3.1 Setup and notation

Split the primitive equations, written along fluid trajectories, as

$$\frac{DX}{Dt} = N(X) + L(X), \qquad \frac{d\mathbf r}{dt} = \mathbf V(X),$$

where `L` is exactly Dinosaur's existing linear implicit operator, `N` is the
remaining **non-advective** explicit forcing (§4 gives the term-by-term
mapping), and all advection — horizontal and vertical, previously part of
`explicit_terms` — is absorbed into the material derivative `D/Dt`.

The SL discretization integrates along the trajectory arriving at each grid
point. Notation: for an arrival grid point, `T_D[·]` denotes interpolation of
a (nodal) field at the departure point of a trajectory family `D`; fields
evaluated at the arrival point need no interpolation. The classic trapezoidal
semi-implicit SL (SLSI) discretization (Diamantakis 2014, Eq. 9; Temperton,
Hortal & Simmons 2001) is

$$X^{n+1} = T_D\big[X^n + \tfrac{\Delta t}{2}(N^n + L X^n)\big]
          + \tfrac{\Delta t}{2}\,N^{n+1/2\,{\rm(extrap)}}
          + \tfrac{\Delta t}{2} L X^{n+1},$$

which is second order and unconditionally stable for advection — but ECMWF
obtains `N^{n+1/2}` and the trajectory winds by **time extrapolation** from
`{t^n, t^{n−1}}` (SETTLS), making the scheme multistep and introducing the
extrapolation instabilities (stratospheric noise) that SETTLS and its limiters
exist to manage.

### 3.2 Key idea: replace extrapolation with an RK predictor

ECMWF's own "iterative centred implicit" (ICI) scheme (Diamantakis 2014,
§3.1) replaces extrapolation with iteration: compute a predictor `X⁽⁰⁾ ≈
X^{n+1}`, then redo the SLSI step using time-*interpolation*
`N^{n+1/2} ≈ ½(N⁽⁰⁾ + N^n)`. One iteration suffices to eliminate the noise;
ECMWF avoids it operationally only because it doubles cost. **ICI with one
iteration is precisely a two-stage, one-step IMEX Runge-Kutta method** — no
multistep memory, self-starting, and it slots naturally into Dinosaur's RK
infrastructure. This is the recommended scheme.

One-step SL time stepping has further independent precedent: Tumolo &
Bonaventura (2015) built an adaptive discontinuous-Galerkin NWP core around a
TR-BDF2 (one-step, two-implicit-stage) semi-Lagrangian discretization, and
semi-Lagrangian Runge-Kutta / exponential integrators are an established line
in the numerical-analysis literature (Celledoni & Kometa 2009; Celledoni,
Kometa & Verdier 2016; Peixoto & Schreiber 2019 for the rotating
shallow-water equations).

### 3.3 The workhorse scheme: `semi_lagrangian_crank_nicolson_rk2`

It is exactly the semi-Lagrangian lift of Dinosaur's existing
`crank_nicolson_rk2` (Heun for explicit terms + Crank-Nicolson for implicit
terms). Let `G_inv(·, η) = (1 − η L)⁻¹` (the existing `implicit_inverse`).
One step from `X^n`:

**Stage 1 (predictor).**
Departure points `D₁` from the midpoint iteration (§5) using winds
`V^n` (no time extrapolation; first-order-in-time trajectories):

$$X^* = G_{\rm inv}\Big(\,T_{D_1}\big[X^n + \tfrac{\Delta t}{2} L X^n
        + \Delta t\, N(X^n)\big],\ \tfrac{\Delta t}{2}\Big).$$

**Stage 2 (corrector).**
Recompute departure points `D₂` with time-centered winds
`V^{n+1/2} = ½(V^n + V^*)` (both fields known; interpolated at the trajectory
midpoint, or averaged along the trajectory à la Temperton et al. 2001), then

$$X^{n+1} = G_{\rm inv}\Big(\,T_{D_2}\big[X^n + \tfrac{\Delta t}{2}(L X^n
          + N(X^n))\big] + \tfrac{\Delta t}{2} N(X^*),\ \tfrac{\Delta t}{2}\Big).$$

Properties:

- **Second order** in time for all terms (trapezoidal centering along the
  trajectory: old-time-level quantities ride from the departure point, new
  ones sit at the arrival point).
- **Reduces exactly to `crank_nicolson_rk2` when transport is the identity**
  (zero winds ⇒ `T_D = Id` recovers, term for term, the existing stepper:
  `g = u0 + dt/2·G(u0)`, `u1 = G_inv(g + dt·h1, dt/2)`,
  `u2 = G_inv(g + dt·h2, dt/2)` with `h2 = ½(h1 + F(u1))`). This gives a sharp
  unit test and makes the scheme's relationship to the existing code obvious.
- **No CFL restriction from advection**; stability is limited by the explicit
  treatment of `N` (Coriolis and non-linear residuals — slow modes), the
  trajectory-convergence (Lipschitz) condition `Δt · max‖∇V‖ < 1`, and
  accuracy.
- Cost per step: 2 evaluations of `N`, 2 departure-point solves, 2 transport
  applications, 2 implicit solves. Roughly 2× an Eulerian
  `crank_nicolson_rk2` step; the payoff is the 3-6× larger `Δt`.

An optional **off-centering (decentering) parameter** ε shifts the implicit
weights to `(½+ε)Δt` at arrival and `(½−ε)Δt` at departure — the standard
remedy for orographic resonance in SLSI models (small accuracy cost, first
order in the ε-term). Plumb it through from the start; default ε = 0.

### 3.4 Generalization: `semi_lagrangian_imex_runge_kutta(tableau, ...)`

The same construction generalizes any `ImExButcherTableau`: for stage `i` with
abscissa `c_i`, build the bracket
`B_i = X^n + Δt Σ_{j<i} (ã_ij N_j + a_ij L_j)`, transport it along stage
trajectories spanning `c_i Δt`, and solve
`Y_i = G_inv(T_{D_i}[B_i], a_ii Δt)`; mirror the same pattern for the final
combination (`b_ex`, `b_im`). Two caveats to document in code:

1. **Formal order.** Interpolating *all* previous-stage tendencies at the
   full-stage departure point treats them as attached to the parcel at `t^n`.
   For stage tendencies with `0 < c_j < c_i` this is an `O(Δt²)` misplacement,
   so the SL lift of a 3rd-order tableau (e.g. SIL3's explicit part) is
   formally 2nd order unless stage-consistent interpolation (separate
   trajectory-segment interpolation per `(i, j)` pair, as in the SL
   exponential-RK integrators of Celledoni & Kometa 2009 and Celledoni,
   Kometa & Verdier 2016) is used. Given that
   SIL3's implicit part is 2nd order anyway and spatial interpolation error
   dominates in practice, this is acceptable; a stage-consistent variant is
   listed as future work.
2. **Negative tableau weights** (SIL3 has them) combine tendencies sampled at
   slightly inconsistent positions; empirical noise checks against
   `semi_lagrangian_crank_nicolson_rk2` are part of validation.

Deliverable-wise, the 2-stage scheme in §3.3 is the required outcome; the
general tableau version is a stretch goal implemented behind the same
interface so `imex_rk_sil3`-style tableaus can be evaluated experimentally.

### 3.5 New equation interface

```python
# time_integration.py
class SemiLagrangianImplicitExplicitODE(ImplicitExplicitODE):
  """ODE of the form DX/Dt = N(X) + L(X) along trajectories dr/dt = V(X).

  explicit_terms(x)   -> N(x): non-advective explicit tendencies (modal).
  implicit_terms(x)   -> L(x): unchanged from the Eulerian equations.
  implicit_inverse(x, eta): unchanged.
  """

  def nodal_velocities(self, state) -> typing.Pytree:
    """Nodal (u, v) per level, vertical velocity sigma-dot at interfaces,
    and the vertically averaged wind for the continuity equation."""
    raise NotImplementedError

  def semi_lagrangian_transport(self, bracket, departure) -> PyTreeState:
    """Applies T_D[bracket]: remaps the advected representation of a modal
    pytree (state or state-like linear combination) to arrival points."""
    raise NotImplementedError
```

The steppers (`semi_lagrangian_crank_nicolson_rk2`, and the tableau-general
version) consume this interface, so shallow water and the primitive equations
share the same time-integration code, mirroring how `ImplicitExplicitODE` is
shared today. `sim_time` passes through transport untouched and keeps its
existing convention (`N` contributes rate 1.0).

## 4. Semi-Lagrangian form of the primitive equations (sigma coordinates)

Rather than transporting (ζ, δ) — whose advective form contains stretching
terms with no clean transport interpretation — the SL equations transport
grid-point **velocity components**, following IFS practice (Ritchie et al.
1995; Temperton et al. 2001), and only convert to (ζ, δ) modally at arrival
via the existing `spherical_harmonic.uv_nodal_to_vor_div_modal`. Because
`get_cos_lat_vector` (modal (ζ,δ) → nodal winds) and `uv_nodal_to_vor_div_modal`
are linear, transporting a *bracket* that includes (ζ, δ)-space terms is
well-defined: convert the bracket's vorticity/divergence components to bracket
winds, transport, convert back.

Term-by-term mapping (dry case shown; moist corrections follow the same
pattern using the existing helper functions):

| Equation | Eulerian term (current code) | SL fate |
|---|---|---|
| momentum | `(ζ+f) k×v` rotational term | advection part → **trajectories**; Coriolis `−f k×v` → **N** (nodal, at stage points) |
| momentum | `∇(|v|²/2)` (`kinetic_energy_tendency`) | **dropped** — artifact of the vector-invariant form, absorbed by directly advecting `v` |
| momentum | `σ̇ ∂v/∂σ` | **trajectories** (3-D departure points) |
| momentum | `−R T′_v ∇ln pₛ` (explicit PGF part) | **N** |
| momentum | `−∇Φₛ` (orography; currently `−g∇²(orography)` in the δ equation) | **N** (as a nodal gradient) |
| momentum | `−∇(G T′ + R T_ref ln pₛ)` (implicit PGF; currently `−∇²(...)` in the δ equation) | **L** — unchanged; its old-time-level value is transported inside the bracket as a nodal vector field |
| thermodynamic | `v·∇T′` + `σ̇_full ∂T′/∂σ` | **trajectories** |
| thermodynamic | `σ̇_explicit ∂T_ref/∂σ` | **N** (T_ref is a function of σ, not a transported field; zero for constant T_ref) |
| thermodynamic | `κ T_v ω/p` explicit part (`nodal_temperature_adiabatic_tendency`) | **N** (reused as-is) |
| thermodynamic | `H δ` (implicit) | **L** — unchanged |
| continuity | `−∫ v·∇ln pₛ dσ` | **trajectories** (2-D, vertically averaged wind; see below) |
| continuity | `−∫ δ dσ` (implicit) | **L** — unchanged; `N_{lnpₛ} = 0` |
| tracers | flux-form horizontal + vertical advection | **trajectories** (pure transport; `N = L = 0`) |

Notes:

- **Vorticity keeps a zero implicit term.** The momentum-space implicit
  forcing is a gradient, `L_v = −∇(G T′ + R T_ref ln pₛ)`, whose curl
  vanishes — consistent with the existing `implicit_terms`
  (`vorticity_implicit = 0`), which is why **`implicit_inverse` is reused with
  no modification**. After building the provisional arrival-point winds
  (transported bracket + `Δt`-weighted arrival `N` terms), we transform
  `(u*, v*) → (ζ*, δ*)` and apply the existing per-wavenumber solve on
  (δ, T′, lnpₛ).
- **Continuity is exact with 2-D trajectories.** Since `ln pₛ` is independent
  of σ, `∫₀¹ v·∇ln pₛ dσ = v̄·∇ln pₛ` with `v̄ = ∫₀¹ v dσ`, so
  `D̄(ln pₛ)/D̄t = −∫₀¹ δ dσ` *exactly*, where `D̄/D̄t` follows the
  vertically averaged wind. The right-hand side is exactly the current
  implicit term, so the SL continuity equation needs one extra (cheap,
  single-level, horizontal-only) departure-point family and **no** explicit
  tendency. With orography, `ln pₛ` carries a rough orographic signature;
  interpolating a smoothed variable (e.g. `ln pₛ + Φₛ/(R T̄)`, à la
  Ritchie & Tanguay 1996) is noted as a follow-up.
- **T_ref handling:** the transported thermodynamic variable is `T′`
  (unchanged state format). Because `T′ = T − T_ref(σ)` and trajectories move
  in σ, the term `−σ̇ ∂T_ref/∂σ` appears as a source: its explicit part stays
  in `N` (reusing the existing `sigma_dot_explicit` machinery) and its
  implicit part is already inside **H**. For the common constant-`T_ref`
  configuration this term vanishes identically.
- **Moisture:** virtual-temperature adjustments enter only through `N`
  (explicit PGF residuals and adiabatic terms), reusing
  `divergence/vorticity_tendency_due_to_humidity` logic re-expressed in
  (u, v) space where needed. The humidity tracer itself is transported like
  any other tracer.

## 5. Departure points on the sphere

Work in 3-D Cartesian coordinates on the unit sphere (pole-singularity-free):

1. Precompute arrival unit vectors `r_a` from `grid.nodal_mesh` and the local
   tangent basis `(ê_λ, ê_φ)`.
2. Convert stage winds to Cartesian: `V = u ê_λ + v ê_φ` (three smooth scalar
   fields — no polar discontinuity, unlike (u, v)).
3. Fixed-point midpoint iteration (Robert-style; Diamantakis 2014 Eq. 5), a
   **fixed number of iterations (default 2)** for reverse-mode AD friendliness:

   ```
   r_d ← r_a                                  # initial guess uses V(r_a)
   repeat 2x:
     r_m  = normalize((r_a + r_d) / 2)
     r_d  = normalize(r_a − Δt · V_mid(r_m))  # displacement in tangent space,
                                              # renormalized to the sphere
   ```

   where `V_mid` is the time-appropriate wind for the stage (predictor: `V^n`;
   corrector: `½(V^n + V^*)`), interpolated **bilinearly** (linear wind
   interpolation in the trajectory solve is standard and sufficient:
   Staniforth & Côté 1991). The normalized-displacement update is 2nd-order
   equivalent to the exact great-circle solution.
4. Convert `r_d → (λ_d, φ_d)` via `atan2` / `arcsin`.
5. Vertical: `σ_d = clip(σ_a − Δt·σ̇_mid, σ_min, σ_max)`, with `σ̇` interpolated
   linearly (from interface values, as in `compute_vertical_velocity_sigma`)
   and clipping to the valid interpolation range (consistent with the
   zero-gradient boundary conditions of the existing vertical advection and
   the existing first-order SL vertical step). The 3-D iteration updates
   horizontal and vertical positions together.

Convergence of the iteration requires `Δt < 1/max‖∂V/∂x‖` (trajectories do
not cross; Pudykiewicz et al. 1985; Smolarkiewicz & Pudykiewicz 1992) —
in practice hours, far beyond target time steps; two iterations give
second-order departure points.

## 6. Interpolation on the full Gaussian grid

New module `dinosaur/semi_lagrangian.py`, all shapes static, all ops gathers +
einsums:

- **Longitude**: uniform spacing ⇒ fractional index `(λ_d − λ₀)/Δλ`; stencil
  indices wrap modulo `longitude_nodes`.
- **Latitude**: Gaussian nodes are non-uniform ⇒ locate the bracketing ring
  with `jnp.searchsorted` on `sin φ` (monotone), then use **non-uniform
  Lagrange weights** on the actual node latitudes (exact for the polynomial
  order regardless of spacing).
- **Cross-pole halo**: extend the field with 2 rows beyond each pole, taking
  values from longitude `λ + π` (a `jnp.roll` by `longitude_nodes // 2`; exact
  for the even node counts of the standard `Grid.T*`/`Grid.TL*` constructors —
  the odd counts produced by `Grid.with_wavenumbers` would need a half-cell
  shift, so the initial version asserts an even number of longitude nodes).
  Scalars copy directly;
  for wind fields interpolated as Cartesian components no sign flip is needed
  (this is a key reason to interpolate Cartesian components; (u, v) halos
  would need sign flips and are only piecewise-smooth at the pole).
- **Schemes**: `order='linear'` (2×2 horizontal) and `order='cubic'`
  (4×4 horizontal, full tensor-product Lagrange — the "naive" 64-point/
  quasi-cubic distinction is an efficiency concern deferred with the rest of
  TPU work). Vertical: linear between layer centers initially (matching
  `vertical_interpolation.interp` semantics with constant extrapolation);
  cubic-in-σ later.
- **Quasi-monotone limiter** (Bermejo & Staniforth 1992): clip the
  interpolated value to the [min, max] of the surrounding linear-stencil cell
  (2×2×2 corners), applied after the full 3-D interpolation (the variant
  ECMWF found preferable for the tropopause cold bias, Diamantakis 2014 §4).
  Exposed as an option on the interpolator; **enabled per-tracer** via the
  equation class (`monotone_tracers: set[str]`), off for dynamical variables
  by default.
- Linear interpolation is far too diffusive as the default for transported
  fields (well documented, e.g. Diamantakis 2014 Fig. 10); **cubic is the
  default** for all transported fields, linear the default for winds inside
  the trajectory solve.

**Vector transport.** Momentum (and momentum-bracket) fields are transported
as the three Cartesian components of the horizontal wind, interpolated
componentwise, then **rotated from the departure tangent plane to the arrival
tangent plane** with the closed-form great-circle (Rodrigues) rotation
`R(r_d → r_a)` and projected onto `(ê_λ, ê_φ)` at arrival. This is exact
parallel transport along the great circle; skipping the rotation is an
`O(|Δr|²)` directional error, so the rotation stays on by default.

API sketch:

```python
# dinosaur/semi_lagrangian.py
@dataclasses.dataclass(frozen=True)
class GridInterpolator:
  grid: spherical_harmonic.Grid
  order: str = 'cubic'            # 'linear' | 'cubic'
  monotone: bool = False          # Bermejo-Staniforth clip
  def __call__(self, field, lon_d, sin_lat_d): ...   # horizontal, any batch dims

def departure_points(u, v, sigma_dot, coords, dt, *, iterations=2): ...
def transport_scalar(field, departure, interpolator): ...        # 3-D
def transport_scalar_2d(field, departure, interpolator): ...     # for ln ps
def transport_wind(u, v, departure, interpolator): ...           # Cartesian + rotation
```

`SemiLagrangianPrimitiveEquations` (subclassing `PrimitiveEquationsSigma`)
implements `nodal_velocities`, `semi_lagrangian_transport`, and overrides
`explicit_terms` to return only `N` (reusing the existing per-term helpers;
`implicit_terms`/`implicit_inverse` inherited untouched). Modal↔nodal flow per
stage: bracket (modal) → nodal advected representation (winds via
`get_cos_lat_vector`, scalars via `to_nodal`) → gather-interpolate → arrival
nodal fields → `to_modal` (+ `clip_wavenumbers`, matching `explicit_terms`
conventions) → add arrival-side terms → `implicit_inverse`.

## 7. Data-representation decisions

- **State stays modal** (no format change): SL fields make a nodal round-trip
  inside each step. The `to_modal` at arrival acts as the spectral fit IFS
  also performs on dynamical variables; existing modal filters
  (hyperdiffusion/exponential) apply unchanged.
- **Caveat to document prominently:** for the sharp-tracer positivity use
  case, a modal round-trip re-introduces Gibbs ringing each step no matter how
  good the transport is. Fixing that requires (opt-in, per-tracer) **nodal
  tracer storage**, exactly as IFS keeps humidity/tracers in grid-point space.
  This is scoped as milestone M5b: tracers listed in `nodal_tracers` are
  carried in `State.tracers` as nodal arrays, skipped by modal filters, and
  never transformed (SL transport + physics both operate nodally). It is a
  deliberate, visible format extension — not required for the dynamics
  milestones, but required to fully deliver the aerosol/chemistry ask.
- Grid note: with SL transport the quadratic-nonlinearity dealiasing argument
  no longer applies to advection, which is what let ECMWF run linear "TL"
  truncation grids. Dinosaur already has `Grid.TL*` constructors; evaluating
  TL grids with SL is cheap once the core lands (listed under future work).

## 8. Differentiability & JAX constraints

- Fixed trajectory-iteration count; no `while_loop` tolerances.
- `searchsorted` + `take`-based gathers: piecewise-differentiable, standard
  for SL-in-ML; gradients flow through interpolation weights (and through the
  departure points into the weights — the fully differentiable path).
- The quasi-monotone clip uses `jnp.clip` (subgradients at the clip boundary,
  same tradeoff as relu).
- `implicit_inverse` requires concrete `step_size` — unchanged, since stage
  `a_ii Δt` values are Python floats exactly as today.
- Everything `vmap`s/`scan`s exactly like the current steppers
  (`trajectory_from_step`, `nested_checkpoint_scan` compose unchanged).
- Unit test: `jax.grad` of a scalar loss through 1-2 SL steps vs finite
  differences.

## 9. Verification and validation

**Unit tests (`semi_lagrangian_test.py`)**

1. Interpolator: exact for constants and for polynomials up to the stencil
   order; longitude wrap; cross-pole halo (advect a blob over the pole with a
   solid-body flow through the poles and compare against the analytic
   rotation); limiter produces no new extrema and preserves positivity;
   Cartesian-wind transport of a solid-body flow returns the same flow.
2. Departure points: solid-body rotation has analytic great-circle departure
   points — verify 2nd-order convergence in Δt and iteration behavior.
3. Steppers: with zero velocities, `semi_lagrangian_crank_nicolson_rk2`
   reproduces `crank_nicolson_rk2` to machine precision on a toy
   `SemiLagrangianImplicitExplicitODE`; time-order convergence (2nd) on a toy
   advection-relaxation problem with an analytic solution.

**Passive 2-D transport (prescribed winds)**

4. Williamson et al. (1992) case 1 (cosine bell, solid-body rotation at
   several orientations incl. over the poles): error norms vs resolution and
   vs the spectral-advection baseline; Nair & Lauritzen (2010) deformational
   flow; Gaussian-hills positivity comparison (mirrors the jax-gcm aerosol
   failure mode: sharp source on near-zero background — spectral transport
   rings, SL+limiter stays non-negative).

**Shallow water (cheap full-loop de-risk, optional milestone)**

5. `ShallowWaterEquations` in SL form (transport nodal (u, v) and height
   perturbation): `testSteadyStateGeostrophicFlow` analog (Williamson case 2)
   and barotropic-instability case vs the Eulerian core; large-Δt stability
   scan.

**Primitive equations**

6. **Consistency:** Jablonowski & Williamson steady state (`steady_state_jw`)
   must remain steady (a sensitive test of the vector transport + PGF
   residual split); JW baroclinic wave (`baroclinic_perturbation_jw`) —
   SL vs Eulerian solutions converge to each other at 2nd order as Δt → 0 at
   fixed resolution.
7. **Climate:** Held-Suarez long run at T42/T85 — zonal-mean statistics
   against the Eulerian core within sampling variability.
8. **The point of it all:** time-step extension experiments — max stable Δt
   for SL vs Eulerian cores at T42/T85/T170 with matched filters (target ≥3×;
   ECMWF experience suggests up to 6×), and wall-clock cost per simulated day
   on CPU/GPU (TPU numbers recorded but explicitly not optimized).
9. Tracer-in-dynamics test: `gaussian_scalar` tracer in Held-Suarez flow;
   positivity with/without limiter; mass-conservation drift tracked
   (`grid.integrate`), with the optional global proportional fixer evaluated.
10. Gradient check (§8) and a small `primitive_equations_integration_test.py`
    -style regression pinning key numbers.

## 10. Milestones

- **M0 — RHS split refactor (no behavior change).** Expose `N`-only tendencies
  on `PrimitiveEquationsSigma` (factor the existing helpers so
  `explicit_terms ≡ N + advection` is testable by reconstruction);
  add `nodal_velocities`. Regression: reconstructed explicit terms match
  current `explicit_terms` to machine precision.
- **M1 — `semi_lagrangian.py` core.** Geometry (Cartesian arrival points,
  tangent bases, Rodrigues rotation), departure-point solver, linear/cubic
  interpolators with wrap + pole halos, limiter, scalar & wind transport.
  Unit tests (§9.1-2).
- **M2 — Passive transport validation.** Prescribed-wind advection tests
  (§9.4), including the positivity stress case.
- **M3 — SL time steppers.** `SemiLagrangianImplicitExplicitODE`,
  `semi_lagrangian_crank_nicolson_rk2` (+ off-centering), toy-problem tests
  (§9.3). Stretch: tableau-general `semi_lagrangian_imex_runge_kutta`.
- **M3b (optional) — Shallow-water SL** for a cheap end-to-end shakeout (§9.5).
- **M4 — `SemiLagrangianPrimitiveEquations` (dry, sigma).** Momentum/thermo/
  continuity transport per §4, JW steady-state + baroclinic-wave consistency
  (§9.6).
- **M5 — Moist terms + tracers + validation.** Moisture in `N`, per-tracer
  limiter, Held-Suarez climate, Δt-extension study, gradient tests, notebook
  + docs. **M5b:** opt-in nodal tracer storage for the sharp-tracer use case.
- **M6 — Cleanups/follow-ups** spun out per §11.

Each milestone lands as a separate PR with tests; nothing touches the default
Eulerian path.

## 11. Risks and open questions

- **Order reduction in the general-tableau lift** (§3.4): accepted at 2nd
  order; stage-consistent interpolation is the known fix if ever needed.
- **Polar noise from naive lat-lon interpolation:** departure points near the
  poles span many longitude cells; the interpolation is still well-defined
  (full Gaussian grid, wrap + halo) but accuracy is anisotropic. Monitored via
  the pole-crossing tests; ECMWF-grade fixes (reduced grids) are exactly the
  deferred issue-#55 follow-up.
- **Orographic resonance** at large Δt (classic SLSI failure): mitigated by
  the off-centering option; Held-Suarez is flat, so add an orography-enabled
  spot check (isothermal-rest-with-orography state exists in
  `primitive_equations_states`).
- **Conservation:** SL conserves neither mass nor tracer mass exactly.
  Tracked in tests; trivial proportional fixer optional; proper fixers
  (Bermejo & Conde 2002; Priestley 1993) and conservative remap (SLICE/CSLAM)
  deferred.
- **Coriolis treatment:** initial version keeps Coriolis explicit in `N`
  (matches current explicit treatment; `fΔt ≈ 0.26` at Δt = 30 min is
  comfortably within RK2 stability). If accuracy at very long steps
  disappoints, the standard alternative is advecting planetary momentum
  (`v + Ω×r` treatment, as in IFS options; Temperton et al. 2001).
- **Cost accounting:** 2 transforms + 2 transports per step must beat the
  Eulerian step at ≥3× Δt. On CPU/GPU this is very likely; on TPU the gathers
  will be slow until the deferred efficiency work — measured and reported,
  not optimized, in this phase.
- **Interaction with modal filters:** filters currently target the tail
  spectrum produced by pseudo-spectral products; SL changes the noise
  spectrum. Filter settings may need retuning for large Δt (tracked in the
  Δt-extension study).

## 12. Deferred efficiency roadmap (context from issue #55)

For completeness, the follow-on path once this version is correct: quasi-cubic
(32-point) interpolation; static-stencil gathers expressed as banded matmuls
(TPU-friendly, per the einsum formulation sketched in issue #55); reduced
Gaussian / octahedral / HEALPix-like grids so polar stencils become static;
TL-truncation grids; conservative/monotone upgrades (SLICE-3D, CSLAM);
hybrid-coordinate support; stage-consistent high-order SL-RK.

## 13. References

- Bermejo, R. & Staniforth, A. (1992). The conversion of semi-Lagrangian
  advection schemes to quasi-monotone schemes. *Mon. Wea. Rev.*, 120,
  2622-2631.
- Bermejo, R. & Conde, J. (2002). A conservative quasi-monotone
  semi-Lagrangian scheme. *Mon. Wea. Rev.*, 130, 423-430.
- Celledoni, E. & Kometa, B. K. (2009). Semi-Lagrangian Runge-Kutta
  exponential integrators for convection dominated problems. *J. Sci.
  Comput.*, 41, 139-164. doi:10.1007/s10915-009-9291-3
- Celledoni, E., Kometa, B. K. & Verdier, O. (2016). High order
  semi-Lagrangian methods for the incompressible Navier-Stokes equations.
  *J. Sci. Comput.*, 66, 91-115. doi:10.1007/s10915-015-0015-6
- Diamantakis, M. (2014). The semi-Lagrangian technique in atmospheric
  modelling: current status and future challenges. *ECMWF Seminar on Numerical
  Methods for Atmosphere and Ocean Modelling*, 183-200.
- Hortal, M. (2002). The development and testing of a new two-time-level
  semi-Lagrangian scheme (SETTLS) in the ECMWF forecast model. *Q. J. R.
  Meteorol. Soc.*, 128, 1671-1687.
- Jablonowski, C. & Williamson, D. L. (2006). A baroclinic instability test
  case for atmospheric model dynamical cores. *Q. J. R. Meteorol. Soc.*, 132,
  2943-2975.
- Lauritzen, P. H., Nair, R. D. & Ullrich, P. A. (2010). A conservative
  semi-Lagrangian multi-tracer transport scheme (CSLAM) on the cubed-sphere
  grid. *J. Comput. Phys.*, 229, 1401-1424.
- Nair, R. D. & Lauritzen, P. H. (2010). A class of deformational flow test
  cases for linear transport problems on the sphere. *J. Comput. Phys.*, 229,
  8868-8887.
- Peixoto, P. S. & Schreiber, M. (2019). Semi-Lagrangian exponential
  integration with application to the rotating shallow water equations.
  *SIAM J. Sci. Comput.*, 41(5). doi:10.1137/18M1206497
- Priestley, A. (1993). A quasi-conservative version of the semi-Lagrangian
  advection scheme. *Mon. Wea. Rev.*, 121, 621-629.
- Ritchie, H., Temperton, C., Simmons, A., Hortal, M., Davies, T., Dent, D. &
  Hamrud, M. (1995). Implementation of the semi-Lagrangian method in a
  high-resolution version of the ECMWF forecast model. *Mon. Wea. Rev.*, 123,
  489-514.
- Ritchie, H. & Tanguay, M. (1996). A comparison of spatially averaged
  Eulerian and semi-Lagrangian treatments of mountains. *Mon. Wea. Rev.*, 124,
  167-181.
- Simmons, A. J. & Burridge, D. M. (1981). An energy and angular-momentum
  conserving vertical finite-difference scheme and hybrid vertical
  coordinates. *Mon. Wea. Rev.*, 109, 758-766.
- Staniforth, A. & Côté, J. (1991). Semi-Lagrangian integration schemes for
  atmospheric models — a review. *Mon. Wea. Rev.*, 119, 2206-2223.
- Temperton, C., Hortal, M. & Simmons, A. (2001). A two-time-level
  semi-Lagrangian global spectral model. *Q. J. R. Meteorol. Soc.*, 127,
  111-127.
- Tumolo, G. & Bonaventura, L. (2015). A semi-implicit, semi-Lagrangian
  discontinuous Galerkin framework for adaptive numerical weather prediction.
  *Q. J. R. Meteorol. Soc.*, 141, 2582-2601. doi:10.1002/qj.2544
- Whitaker, J. S. & Kar, S. K. (2013). Implicit-explicit Runge-Kutta methods
  for fast-slow wave problems. *Mon. Wea. Rev.*, 141, 3426-3434.
- Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R. & Swarztrauber,
  P. N. (1992). A standard test set for numerical approximations to the
  shallow water equations in spherical geometry. *J. Comput. Phys.*, 102,
  211-224.
- Zerroukat, M. & Allen, T. (2012). A three-dimensional monotone and
  conservative semi-Lagrangian scheme (SLICE-3D) for transport problems.
  *Q. J. R. Meteorol. Soc.*, 138, 1640-1651.
