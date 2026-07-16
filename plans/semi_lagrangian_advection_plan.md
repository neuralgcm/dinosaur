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
2. A **one-step (Runge-Kutta-style) time discretization** as the primary
   scheme, in the spirit of Dinosaur's preferred `imex_rk_sil3` /
   `crank_nicolson_rk2` solvers: self-starting, no multistep memory, and none
   of the documented extrapolation instabilities of two-time-level SETTLS or
   three-time-level leapfrog schemes. A SETTLS stepper is planned as an
   opt-in *option* (§3.6) — halving per-step cost, matching operational IFS
   practice, and mirroring how Dinosaur already ships
   `semi_implicit_leapfrog` alongside the RK solvers.
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
  class, built on the Simmons & Burridge (1981) vertical discretization, is a
  straightforward follow-up once sigma works).
- No changes to the default Eulerian code path; SL is a parallel opt-in
  equation class + steppers.

## 2. Where advection lives in Dinosaur today

State (`primitive_equations.State`) holds modal (spherical-harmonic)
`vorticity`, `divergence`, `temperature_variation` (T′ = T − T_ref(σ)),
`log_surface_pressure`, and modal `tracers`. Each call to
`PrimitiveEquationsSigma.explicit_terms`:

- transforms to nodal space (`compute_diagnostic_state_sigma`), computing
  nodal cosθ-scaled winds `cos_lat_u` (modal cosθ·v from
  `spherical_harmonic.get_cos_lat_vector`, then `to_nodal`), the
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

$$
\frac{DX}{Dt} = N(X) + L(X), \qquad \frac{d\mathbf r}{dt} = \mathbf V(X),
$$

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

$$
X^{n+1} = T_D\big[X^n + \tfrac{\Delta t}{2}\big(L X^n + N^{n+1/2}\big)\big]
  + \tfrac{\Delta t}{2}\,N^{n+1/2}
  + \tfrac{\Delta t}{2} L X^{n+1},
$$

where the midpoint-time non-linear term `N^{n+1/2}` appears **twice**:
interpolated at the departure point (inside `T_D`) and evaluated at the
arrival point, centering the non-linear forcing at `t^{n+1/2}` along the
trajectory. (Placing `N^n` at the departure point instead would center it at
`t^{n+1/4}` and drop the scheme to first order.) This discretization is second
order and unconditionally stable for advection — but ECMWF obtains `N^{n+1/2}`
and the trajectory winds by **time extrapolation** from `{t^n, t^{n−1}}` — the
standard `(3/2)N^n − (1/2)N^{n−1}` or the more stable SETTLS variant (§3.6) —
making the scheme multistep and introducing the extrapolation instabilities
(stratospheric noise) that SETTLS and its limiters exist to manage.

### 3.2 Key idea: replace extrapolation with an RK predictor

ECMWF's own "iterative centred implicit" (ICI) scheme (Diamantakis 2014,
§3.1) replaces extrapolation with iteration: compute a predictor `X⁽⁰⁾ ≈
X^{n+1}`, then redo the SLSI step using time-*interpolation*
`N^{n+1/2} ≈ ½(N⁽⁰⁾ + N^n)`. One iteration suffices to eliminate the noise;
ECMWF avoids it operationally only because it doubles the per-step cost of
the dynamics *relative to SETTLS*, whose one-evaluation-per-step ledger is
bought with multistep memory (stored `N^{t−Δt}`, `V^{t−Δt}`) — a discount
Dinosaur's one-step solvers already forgo by design: `imex_rk_sil3` makes 3
explicit-tendency evaluations per step and `crank_nicolson_rk2` makes 2, in
exchange for self-starting steps and clean `scan`/autodiff semantics.
Measured against Dinosaur's own RK baseline, ICI-style stepping therefore
adds no doubling (see the cost bullet in §3.3).

Two honest caveats on the identification. As documented, ICI is not quite
one-step — its first guess `X⁽⁰⁾` still uses time extrapolation — and its
corrector centers the non-linear term as `½(N⁽⁰⁾ + N^n)` at *both* trajectory
endpoints, where §3.3 uses the endpoint trapezoid `½(N^n_d + N(X^*)_a)`.
**Replacing the extrapolated first guess with an RK predictor turns
ICI-with-one-iteration into a genuine two-stage, one-step IMEX Runge-Kutta
method** — no multistep memory, self-starting, slotting naturally into
Dinosaur's RK infrastructure — and that is the recommended scheme. Both
centerings are second order, but ECMWF's "one iteration eliminates the noise"
is evidence from a close sibling, not from this exact scheme; the noise claim
is verified, not assumed, in the Δt study (§9.8).

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

$$
X^* = G_{\rm inv}\Big(\,T_{D_1}\big[X^n + \tfrac{\Delta t}{2} L X^n
  + \Delta t\, N(X^n)\big],\ \tfrac{\Delta t}{2}\Big).
$$

**Stage 2 (corrector).**
Recompute departure points `D₂` with time-centered winds
`V^{n+1/2} = ½(V^n + V^*)` (both fields known; interpolated at the trajectory
midpoint, or averaged along the trajectory à la Temperton et al. 2001), then

$$
X^{n+1} = G_{\rm inv}\Big(\,T_{D_2}\big[X^n + \tfrac{\Delta t}{2}(L X^n
  + N(X^n))\big] + \tfrac{\Delta t}{2} N(X^*),\ \tfrac{\Delta t}{2}\Big).
$$

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
  treatment of `N` — where Coriolis is the binding term at large Δt, because
  Heun has no imaginary-axis stability (see the Coriolis note in §4 and §11)
  — by the trajectory-convergence (Lipschitz) condition `Δt · max‖∇V‖ < 1`
  (§5), and by accuracy.
- Cost per step: 2 evaluations of `N`, 2 departure-point solves, 2 transport
  applications, 2 implicit solves. Roughly 2× an Eulerian
  `crank_nicolson_rk2` step; the payoff is the 3-6× larger `Δt`. (The SETTLS
  option of §3.6 halves this per-step cost at the price of multistep memory.)

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

1. **Formal order — the naive lift is only first order.** Interpolating a
   stage-`j` tendency (an Eulerian snapshot at `t^n + c_jΔt`) at the
   full-segment departure point samples it where the parcel was at `t^n` — a
   positional error `c_jΔt·|V|`, i.e. `O(Δt)` in the sampled value. The
   leading step error is `−Δt²(Σ_j b_j c_j)(V·∇)N = −(Δt²/2)(V·∇)N`
   (`Σ_j b_j c_j = ½` for any 2nd-order-consistent tableau), so the lift is
   **globally first order** for any tableau with interior abscissae — worse
   than §3.3. Constant-wind counterexample: for `DX/Dt = N(x)` the naive
   lift yields `X^n_d + Δt·N_d`, missing the exact update's
   `(Δt²/2)V·∇N` term. §3.3 escapes because its abscissae are `{0, 1}`:
   `N^n` rides from the departure point and `N(X^*)` is added at the arrival
   point — both at their correct parcel positions. Consequence: the general
   stepper is only worth building with **stage-consistent transport**
   (interpolate stage-`j` quantities at the parcel position at `t^n + c_jΔt`
   along the stage-`i` trajectory — one trajectory-segment family per
   `(i, j)` pair, as in the SL exponential-RK integrators of Celledoni &
   Kometa 2009 and Celledoni, Kometa & Verdier 2016), at the cost of extra
   interpolation passes.
2. **Negative tableau weights** (SIL3 has them) combine tendencies sampled at
   slightly inconsistent positions; empirical noise checks against
   `semi_lagrangian_crank_nicolson_rk2` are part of validation.

Deliverable-wise, the 2-stage scheme in §3.3 is the required outcome; the
general tableau version is a stretch goal implemented behind the same
interface, **with stage-consistent transport as a requirement** (a naive
first-order lift would be strictly worse than the workhorse), so
`imex_rk_sil3`-style tableaus can be evaluated experimentally.

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

### 3.6 Optional: a SETTLS two-time-level stepper

One-step is a preference, not a hard constraint: Dinosaur already ships a
three-time-level `semi_implicit_leapfrog` whose step state is a
`(previous, current)` tuple with matching filters (`leapfrog_utils.py`). A
SETTLS stepper is the analogous two-time-level *option* for the SL scheme,
worth having for two reasons: it is the operationally proven configuration
(IFS since 1998), and it halves per-step cost relative to §3.3 — one `N`
evaluation, one departure-point solve, one transport pass, one implicit
solve — by reusing the previous step's tendencies instead of recomputing
them with a predictor.

Formulation (Hortal 2002; Diamantakis 2014, Eqs. 10-12): carry `(N^{n−1},
V^{n−1})` and replace the RK predictor with the stable two-term extrapolation
averaged along the trajectory,

$$
X^{n+1} = G_{\rm inv}\Big(\,T_D\big[X^n + \tfrac{\Delta t}{2}\big(L X^n
  + 2N^n - N^{n-1}\big)\big] + \tfrac{\Delta t}{2}\,N^n,\ \tfrac{\Delta t}{2}\Big),
$$

with departure points iterated using the same extrapolation for the winds:
`r_d ← r − (Δt/2)(V^n(r) + [2V^n − V^{n−1}](r_d))`.

Design notes, mirroring the leapfrog pattern:

- **Step state:** a tuple `(X^n, aux)` with `aux = (N^{n−1} modal, nodal
  (u, v, σ̇)^{n−1})`. Tendencies are carried, not recomputed — that is where
  the saving comes from. It is just a larger `scan` carry; checkpointing and
  `trajectory_from_step` compose unchanged, as they do for leapfrog.
- **Bootstrap:** take the first step with
  `semi_lagrangian_crank_nicolson_rk2` (self-starting) and record
  `(N^0, V^0)` — no accuracy-degraded startup step. Filter wrappers
  analogous to `leapfrog_step_filter` adapt the modal filters to the tuple
  state.
- **Trade-offs to document:** the residual extrapolation instability that
  SETTLS manages but does not eliminate (stratospheric noise; the SETTLS
  trajectory limiter of Diamantakis 2014 §3.3 is the known remedy, deferred);
  no digital-filter initialization (`TimeReversedImExODE` assumes a plain
  state); and for training, one bootstrap step per rollout window plus `aux`
  in every checkpoint.

Because it consumes the same §3.5 interface — transport, departure points,
`N`, `L`, `G_inv` shared verbatim — this is a small stepper wrapper plus
filter adapters, scheduled after the RK scheme lands (M5c in §10) and folded
into the Δt-extension study (§9.8) as a three-way comparison: Eulerian RK vs
SL-RK2 vs SL-SETTLS on per-step cost, max stable Δt, and noise.

## 4. Semi-Lagrangian form of the primitive equations (sigma coordinates)

Rather than transporting (ζ, δ) — whose advective form contains stretching
terms with no clean transport interpretation — the SL equations transport
grid-point **velocity components**, following IFS practice (Ritchie et al.
1995; Temperton et al. 2001), and only convert to (ζ, δ) modally at arrival
via the existing `spherical_harmonic.uv_nodal_to_vor_div_modal`. Because
`get_cos_lat_vector` (modal (ζ, δ) → modal cosθ·v, made nodal via `to_nodal`)
and `uv_nodal_to_vor_div_modal`
are linear, transporting a *bracket* that includes (ζ, δ)-space terms is
well-defined: convert the bracket's vorticity/divergence components to bracket
winds, transport, convert back.

Term-by-term mapping (dry case shown; moist corrections follow the same
pattern using the existing helper functions):

| Equation | Eulerian term (current code) | SL fate |
|---|---|---|
| momentum | `(ζ+f) k×v` rotational term | advection part → **trajectories**; Coriolis `−f k×v` → **N** in explicit-`f` mode (small Δt only), or removed entirely via planetary-momentum transport (large Δt; see Notes) |
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
- **Coriolis and large Δt.** Heun's explicit part has no imaginary-axis
  stability, so explicit `−f k×v` in `N` amplifies inertial modes by
  `≈ (fΔt)⁴/8` per step: ~6·10⁻⁴/step at Δt = 30 min (`fΔt ≈ 0.26` at the
  poles; e-folding ≈ 35 simulated days — fine for short tests, marginal for
  climate runs) and disqualifying at the targeted 3-6× extension
  (`fΔt ≈ 0.8-1.6` → 5-59% growth per step). Three Coriolis modes, selected
  by an equation-level option (`coriolis_mode`): **explicit** `f` in `N`
  (small-Δt consistency testing, §9.6, where the growth is negligible over
  test horizons); **advected planetary momentum** — the default large-Δt
  configuration (IFS's LADVF option; Temperton et al. 2001): transport
  `v + 2Ω×r`, with the analytic `2Ω×r` field added at the departure point
  and subtracted at the arrival point so only `v` is ever interpolated —
  since `D(v + 2Ω×r)/Dt` has no Coriolis term, `f` drops out of `N`
  entirely, the §6 rotation/projection machinery applies unchanged, and the
  implicit solve is untouched; and **implicit** Coriolis (IFS's LIMPF
  option; Temperton 1997), a designed follow-up sketched in §11. Note the
  implicit mode is a *scheme* change, not a solver-algorithm change: unlike
  `implicit_inverse_method` (same operator, same results, different
  algorithm), it moves Coriolis from `N` into `L`, so `implicit_terms` and
  `implicit_inverse` must change together and results differ — which is why
  it is an equation-level mode (precedent: `humidity_key` also changes the
  equations) rather than a new `implicit_inverse_method` value.
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
not cross; Pudykiewicz et al. 1985; Smolarkiewicz & Pudykiewicz 1992). That
margin is real, not academic: resolved jet-region shear/vorticity of
~1·10⁻⁴ s⁻¹ puts the bound at ≈ 2.8 h — comparable to the top of the §9.8
Δt scan (6× a 30-min baseline = 3 h) — so the fixed 2-iteration solve loses
convergence exactly where the study pushes hardest. The Δt study therefore
tracks the iteration increment `‖r_d⁽²⁾ − r_d⁽¹⁾‖` as a diagnostic, and the
faster-converging departure-point algorithm of Diamantakis & Váňa (2022) is
the known upgrade if that limit binds. Within the convergent regime, two
iterations give second-order departure points.

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
  `Grid.with_wavenumbers` yields `order·m + 1` longitude nodes, odd whenever
  `order·m` is even, and odd counts would need a half-cell shift — so the
  initial version asserts an even number of longitude nodes).
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
`O(|Δr|²)` directional error, so the rotation stays on by default. One
convention trap: dinosaur's wind diagnostics are cosθ-scaled
(`cos_lat_u = cosθ·v`; `vor_div_to_uv_nodal` divides it back out), so the
Cartesian conversion must consume true `(u, v)` with the cosθ factor
stripped — an easy silent bug, pinned by the solid-body unit test.

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
   SL vs Eulerian differences shrink at 2nd order **over a bounded Δt window**
   (e.g. Δt_E to 6·Δt_E at fixed resolution). Not "as Δt → 0": SL error
   behaves like `O(Δt²) + O(E_interp/Δt)` — every step commits an
   interpolation remap error and the step count grows as `1/Δt` — so below an
   interpolation-error floor the difference *grows* again. The test pins the
   window and documents the measured floor (or compares both cores against a
   Δt- and resolution-converged reference).
7. **Climate:** Held-Suarez long run at T42/T85 — zonal-mean statistics
   against the Eulerian core within sampling variability.
8. **The point of it all:** time-step extension experiments — max stable Δt
   for SL vs Eulerian cores at T42/T85/T170 with matched filters (target ≥3×;
   ECMWF experience suggests up to 6×), and wall-clock cost per simulated day
   on CPU/GPU (TPU numbers recorded but explicitly not optimized). With M5c:
   a three-way Eulerian vs SL-RK2 vs SL-SETTLS comparison (per-step cost, max
   stable Δt, stratospheric noise). Diagnostics tracked across the Δt scan:
   polar inertial-mode amplification (explicit-`f` vs planetary-momentum
   Coriolis modes, §4) and the departure-iteration increment
   `‖r_d⁽²⁾ − r_d⁽¹⁾‖` (§5).
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
  (§9.3). Stretch: tableau-general `semi_lagrangian_imex_runge_kutta`
  (stage-consistent transport required, per §3.4).
- **M3b (optional) — Shallow-water SL** for a cheap end-to-end shakeout (§9.5).
- **M4 — `SemiLagrangianPrimitiveEquations` (dry, sigma).** Momentum/thermo/
  continuity transport per §4 with the explicit-`f` and
  advected-planetary-momentum Coriolis modes (the implicit mode stays a §12
  follow-up), JW steady-state + baroclinic-wave consistency (§9.6).
- **M5 — Moist terms + tracers + validation.** Moisture in `N`, per-tracer
  limiter, Held-Suarez climate, Δt-extension study, gradient tests, notebook
  + docs. **M5b:** opt-in nodal tracer storage for the sharp-tracer use case.
- **M5c (optional) — SETTLS stepper** (§3.6): tuple-state stepper + filter
  wrappers + RK2 bootstrap; extends the Δt study to the three-way comparison.
- **M6 — Cleanups/follow-ups** spun out per §11.

Each milestone lands as a separate PR with tests; nothing touches the default
Eulerian path.

## 11. Risks and open questions

- **Order of the general-tableau lift** (§3.4): the naive lift is first
  order, so the stretch-goal stepper ships with stage-consistent transport
  or not at all; the 2-stage workhorse is unaffected.
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
- **Coriolis treatment:** explicit `f` under Heun is *not* the benign choice
  it is under SIL3. Heun's stability function has `|R(iy)|² = 1 + y⁴/4 > 1`
  for all `y ≠ 0` (no imaginary-axis stability), while SIL3's third-order
  explicit part is stable to `|fΔt| ≤ √3` — which is why the Eulerian core
  never sees this. Growth `≈ (fΔt)⁴/8` per step: ~6·10⁻⁴ at Δt = 30 min
  (e-folding ≈ 35 simulated days; whether the modal filters mask it in the
  Held-Suarez runs is checked, not assumed) and 5-59% per step at the 3-6×
  extension. Mitigation per §4: advected planetary momentum (`v + 2Ω×r`,
  IFS LADVF; Temperton et al. 2001) is the large-Δt configuration, with
  explicit `f` only for small-Δt phases. Off-centering the `N` trapezoid
  cannot substitute: even the maximally decentered corrector is stable only
  to `|fΔt| ≤ 1`, short of the polar 6× regime — so the two real remedies
  are planetary momentum and implicit `f`. The Δt study tracks an
  inertial-mode growth diagnostic (§9.8).
- **Implicit Coriolis (`coriolis_mode='implicit'`, follow-up option):** the
  LIMPF-style alternative (Temperton 1997) keeps the transported quantity a
  plain wind and instead puts `−f k×v` into `L`. Structure: `f = 2Ωμ`, and
  both μ-multiplication and the `(1−μ²)∂_μ` operators couple total
  wavenumbers `l±1` within each zonal wavenumber `m`, with the same
  ε-recurrence weights dinosaur already implements (`cos_lat_d_dlat`,
  `sec_lat_d_dlat_cos2`) — so *adding* `L_f` to `implicit_terms` is easy,
  and ζ simply joins the implicit state. The work is the inverse: it is no
  longer one small matrix per `l` but, per `m`, a block-tridiagonal solve
  in `l` coupling (ζ, δ) — with (T′, lnpₛ) eliminated per-`l` by Schur
  complement against the existing small inverses — i.e. order
  `3·(2K)²·L²/2` precomputed floats (~0.1 GB f32 at T85/L24, ~0.7 GB at
  T170/L32), forfeiting the m-independence that makes the current solve
  cheap. Two requirements if built: the reference implementation must be a
  *direct* (exact) banded solve — an under-converged iterative inverse
  would change the effective scheme, not just the algorithm — and
  `implicit_inverse(x, η) ≡ (1 − η·implicit_terms)⁻¹x` must hold exactly
  for the extended operator, since the steppers assume it. Two attractions:
  with Coriolis in `L`, the SL bracket automatically centers the old-time
  Coriolis along the trajectory (the `½Δt·L X^n` term rides from the
  departure point) — precisely the LIMPF treatment — and it equally
  stabilizes the SETTLS option, which has the same explicit-`f` exposure.
  Deferred to §12 unless planetary momentum disappoints in the Δt study.
- **Cost accounting:** 2 transforms + 2 transports per step must beat the
  Eulerian step at ≥3× Δt. On CPU/GPU this is very likely; on TPU the gathers
  will be slow until the deferred efficiency work — measured and reported,
  not optimized, in this phase. If the two-pass RK structure ever binds,
  SETTLS (§3.6) halves it.
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
hybrid-coordinate support; stage-consistent high-order SL-RK; the
implicit-Coriolis `coriolis_mode` (LIMPF-style per-`m` block-tridiagonal
implicit solve; §11).

**SLHD — flow-adaptive damping through the interpolator (design sketch).**
The principled form of tracer-noise control, compared with a uniform
grid-space (Shapiro-type) filter, is the Semi-Lagrangian Horizontal
Diffusion of Váňa et al. (2008): transport interpolates with a pointwise
blend

    f(x_d) = (1 − κ)·f_cubic(x_d) + κ·f_linear(x_d),

with κ ∈ [0, κ_max] diagnosed from the local resolved flow deformation
(total strain |D| from the stretching and shearing components, computable
from the modal state's spectral derivatives), e.g.
κ = κ_max·(1 − exp(−(Δt·|D|/d₀)^p)). This is a Smagorinsky-style closure of
the advective cascade: dissipation lands exactly where deformation
generates grid-scale variance and at a rate tied to the local cascade
rate, while coherently translating sharp features keep full cubic
accuracy. Implementation shape in this codebase: `GridInterpolator` gains
the blend (one extra 2✕2(✕2) gather on transported fields, ~+1–3% step
time), the equation classes gain the κ diagnostic evaluated at arrival
nodes, and `semi_lagrangian_transport` threads κ into the tracer
interpolators; the blend is a convex combination of the limited cubic and
the automatically-monotone linear interpolant, so quasi-monotone
positivity guarantees survive unchanged. Known blind spot (the reason the
operational ALADIN/ALARO configuration pairs SLHD with weak uniform
supporting diffusion): SLHD dissipates only through the act of
interpolation, whose damping vanishes as the local displacement
approaches zero or an integer number of cells — stationary noise in
weakly-deforming regions escapes it. Calibration surface is κ_max, d₀ and
p, which warrant an ERA5 A/B like the one in §13's noise diagnosis. Note
that diagnosis before building: the observed ERA5 wave trains are forced
by the resolved dynamics, which SLHD cannot remove either.

## 13. Implementation updates

Issues encountered while implementing this plan, by milestone.

**M0 — RHS split refactor.**

- The reconstruction identity `explicit_terms ≈ explicit_advective_terms +
  explicit_nonadvective_terms` holds only up to float32 re-association error,
  not bit-exactly: splitting `(ζ + f)` products and applying `to_modal` /
  `clip_wavenumbers` to each part separately re-associates linear operations.
  Observed max absolute error ~5e-8 on T21 (nondimensional units, tendencies
  O(0.1)); the regression test uses `rtol=1e-4, atol=1e-6`. The *outputs* of
  `explicit_terms` with default flags are bit-for-bit identical to the
  pre-refactor code (verified empirically against the parent commit, eager
  and jitted); the traced expression tree gains two `+0` ops that XLA folds.
- `nodal_velocities` returns σ̇ at all `layers + 1` boundaries (zeros appended
  at σ = 0, 1) rather than only interior boundaries, since the trajectory
  solver needs the full interpolation range.
- Adversarial review (fresh sub-agent) found the initial tests pinned the
  *completeness* of the split but not the *classification* (swapping Coriolis
  and ζ-advection between N and A would have passed). Added algebraic
  classification tests: N must be affine in (ζ, δ) at fixed (T′, ln pₛ,
  tracers); momentum advection must be homogeneous quadratic and scalar
  advection homogeneous linear, all vanishing at zero winds. The remaining
  algebraically-indistinguishable swap (adiabatic heating ↔ T′ advection,
  both linear in winds) is pinned later by the M4 steady-state test.
- Pre-existing test failures (not caused by this work): the two sharding
  subtests of `primitive_equations_integration_test.py::IntegrationTest::
  test_distributed_simulation_consistency` fail on a single-device CPU host,
  verified identical at the base commit.

**M1 — `semi_lagrangian.py` core.**

- Latitude interpolation uses latitude φ (radians) as the coordinate, not
  sin φ as sketched in §6: with cross-pole halo rows the extended axis must
  stay strictly monotone, and sin φ folds at the poles. Halo-ring
  coordinates are mirrored about the pole (`±π − φ`), keeping the extended
  axis increasing; `searchsorted` operates on it directly.
- "Cartesian-wind transport of a solid-body flow returns the same flow"
  (§9.1) is true only up to O(Δt): parallel transport differs from the flow
  map of solid-body rotation by an in-plane twist of angle ≈ ωΔt·cosθ (θ =
  angle from the rotation axis) — exactly the turning that Coriolis/metric
  terms supply in the momentum equations (f = 2Ω·cosθ_axis). The unit test
  instead verifies wind transport against an independent per-point Rodrigues
  parallel-transport implementation (tight tolerance), plus the O(ω²Δt)
  flow-return bound as documentation of the physics.

**M3 — SL time steppers.**

- The `SemiLagrangianImplicitExplicitODE` interface grew a third method
  beyond the §3.5 sketch: `departure_points(velocities, dt)` sits between
  `nodal_velocities` and `semi_lagrangian_transport`, so steppers can form
  time-centered winds by tree-averaging two velocity pytrees (stage 2 needs
  `½(V^n + V^*)`) without knowing the equation-specific velocity layout.
- The off-centering parameter ε decenters both the implicit terms and the
  corrector's non-linear trapezoid (`α = (½+ε)Δt` at arrival, `β = (½−ε)Δt`
  at departure); the predictor keeps the full `Δt·N^n` weight. With ε = 0
  and zero velocities the stepper reproduces `crank_nicolson_rk2` to
  float64 round-off, and a ring advection-relaxation toy with exact
  (spectral-shift) transport confirms clean second-order convergence
  dropping toward first order with ε > 0.
- The tableau-general `semi_lagrangian_imex_runge_kutta` stretch goal is
  deliberately not built, following §3.4's own analysis: without
  stage-consistent transport the lift is first order (strictly worse than
  the workhorse), and stage-consistent transport (one trajectory family per
  (i, j) stage pair) is a research-grade extension deferred with the §12
  roadmap.

**Final review pass (M5/M5b/M5c) and wrap-up.**

- The final adversarial review verified the SETTLS formula, bootstrap
  history, sim_time bookkeeping, the coefficient-1 bracket invariant, and —
  by auditing every method in the stepping path — that nodal tracers never
  touch a spectral transform. Its main finding was a test blind spot: both
  ring toys had state-independent forcing, so nothing pinned SETTLS's
  defining `2N^n − N^{n−1}` extrapolation (mutants deleting or
  time-swapping it passed everything). A state-dependent-forcing toy now
  discriminates: correct scheme converges at order 2.0, both mutants drop
  to ~0.9.
- The gradient test was widened per review: both steppers (RK2 and
  SETTLS), two directional derivatives (vorticity scale through
  departure-point/wind-transport paths; tracer scale through scalar
  transport), run in float64 so central differences resolve the gradient —
  tolerance tightened from 1e-2 to 1e-4 (the original float32 test hid
  ~1e-2 cancellation noise).
- Documented limitations surfaced by review: `compose_equations` requires
  structurally matching tendency pytrees (Held-Suarez forcing fails loudly
  on states carrying tracers — pre-existing Eulerian limitation); the
  SETTLS bootstrap step is unfiltered (documented workaround).
- Full-suite gotcha worth knowing: several existing test modules enable
  jax x64 globally at import time, so full-suite runs execute later tests
  in float64. The instability-onset and threshold-calibrated SL test
  classes now pin float32 in setUp.
- Test organization: JAX only handles the x64 flag reliably when it is set
  once at startup, so per-test toggling was removed. All float64 tests
  (convergence-order measurements, departure/transport exactness against
  analytic rotations, and the gradient/finite-difference comparisons) live
  in a dedicated `semi_lagrangian_x64_test.py` that enables x64 at import
  time, following the existing `time_integration_test.py` convention;
  `semi_lagrangian_test.py` keeps the dtype-agnostic unit tests. Every
  validation claim in this section is backed by a committed test; the only
  command-line-only runs were mutation checks (verifying that committed
  tests discriminate against seeded bugs) and threshold-calibration runs
  whose measured values are recorded in test comments.
- Post-PR review follow-up: the §3.5 interface originally reused the name
  `explicit_terms` for the non-advective forcing N, leaving semi-Lagrangian
  equations silently accepted by Eulerian steppers (which would integrate
  advection-free dynamics). The prohibition is now encoded in the data
  model: the interface exposes N as `nonadvective_terms`, and
  `explicit_terms` on semi-Lagrangian equations raises TypeError with
  guidance (tested for both equation classes), and the interface was
  subsequently made a standalone class — `ImplicitExplicitODE` is
  deliberately not a base, since a semi-Lagrangian equation is not usable
  where an Eulerian one is expected. The raising `explicit_terms` remains
  necessary regardless: Eulerian steppers duck-type, and the concrete
  classes stay isinstance-compatible through their Eulerian equation
  parents. The advective/non-advective
  split methods on the Eulerian class remain distinct from `explicit_terms`
  by design: the fused Eulerian path sums nodal terms before a single
  spectral transform per equation, which is cheaper than adding the two
  split tendencies and keeps the default path bit-for-bit unchanged; the
  reconstruction identity is pinned by a regression test instead. Also from
  review: `GridInterpolator` no longer supports arbitrary leading batch
  dimensions via silent reshapes — fields are [lon, lat] or
  [levels, lon, lat] with an explicit `jax.vmap` over the levels axis,
  matching dinosaur's array conventions. The limiter option became a string
  (`limiter='quasi_monotone' | None` instead of `monotone: bool`, validated
  with a clear error) so future shape-preservation variants — e.g. an
  IFS-style positive-definite clip — extend the same option rather than
  breaking it; conservative fixers (Priestley/Bermejo–Conde) remain
  deliberately outside this interface, as global post-transport corrections
  at the equation level.
- First GPU numbers (Modal, float32, `benchmarks/
  semi_lagrangian_gather_benchmark.py`): the linearized 1-D take and
  multi-array advanced indexing are equivalent on A100-80GB and H100
  (within a few percent at T85/L32-T340/L64; e.g. H100 T170/L48: 1.73 vs
  1.74 ms), while the linearized form is 1.3-2✕ faster on CPU — so the
  implementation keeps it. Library-level on H100: `transport_scalar`
  0.41 / 2.55 / 37 ms and `departure_points_3d` 1.0 / 6.3 / 121 ms at
  T85/L32, T170/L48, T340/L64 — the departure solve, not the gather, is
  the dominant SL cost on GPU. Replacing the per-component
  stack([...]) loops with `jax.vmap` over the three Cartesian wind
  components (guaranteeing one stencil computation and one batched
  gather) halved the H100 departure solve at T340/L64 (121 → 64 ms)
  while remaining neutral at smaller sizes and on CPU. Synthetic gather payload bandwidth reaches
  ~560 GB/s (H100) at T85 and drops to ~160 GB/s at T340 with fully random
  indices (a pessimistic bound; real departure points are spatially
  coherent).
- Digital filter initialization now supports semi-Lagrangian equations
  (removing the §3.6-era limitation): `TimeReversedSemiLagrangianODE`
  negates the non-advective and implicit tendencies and the trajectory
  velocities while reusing transport unchanged — valid because in reversed
  time the material derivative of any transported quantity flips sign, so
  planetary-momentum transport needs no special handling.
  `digital_filter_initialization` dispatches on the equation type. Tests:
  reversed dynamics match the Eulerian reversal at zero velocities; a
  forward-backward round trip on the uniform-velocity ring toy is exact to
  machine precision; the JW steady state is exactly as steady backward as
  forward (drifts agree to 0.5%); SL DFI matches Eulerian DFI on a ±3 h
  window to l2 < 2e-3.
- The §9.8 Δt-extension study now has real-data GPU numbers (Modal
  A100-80GB, float32): 2-day T170/L32 forecasts from ERA5 (1990-05-01),
  raw initialization, hyperdiffusion filter as in the ERA5 notebook.
  Eulerian `imex_rk_sil3` at its notebook dt = 5 min costs 7.2 s per
  simulated day; the SL core is stable through dt = 60 min with day-2
  surface-pressure agreement of 0.2% (10-20 min), 0.3% (30 min), 0.5%
  (45 min) and 0.7% (60 min). Quality degrades before stability does: the
  temperature maximum develops a localized hot anomaly beyond 30 min
  (+9 K at 45 min, +34 K at 60 min — the classic large-Δt symptom that
  off-centering exists to treat), so dt = 30 min (6✕ the Eulerian
  notebook, 3.9✕ wall-clock speedup at 1.8 s per simulated day) is the
  tuned operating point used in the semi-Lagrangian ERA5 notebook.
- **Notebook study (Modal A100-80GB).** All three repo notebooks were
  re-executed headless on A100 (portability fixes: `np.cast` removal,
  `HybridCoordinates` module move, `get_geopotential`/
  `compute_vertical_velocity` renames — the originals no longer ran against
  main) and semi-Lagrangian copies were added alongside them. Measured
  wall-clock for the simulation cells (same GPU, compiled runs):
  ERA5 T170/L32 2-day forecast 14.3 s Eulerian (dt = 5 min) vs 3.9 s SL
  (dt = 30 min, ~3.6-3.9✕); Held-Suarez T42/L24 1200-day climate run
  2 min 16 s Eulerian (dt = 10 min) vs 1 min 32 s SL (dt = 60 min, 1.5✕);
  baroclinic T42/L24 two-week wave at wall-clock parity (SL per-step gather
  overhead offsets the 12✕ step reduction at this small size — the SL
  advantage grows with resolution). The ERA5 SL notebook stores all three
  moisture tracers nodally with the quasi-monotone limiter and runs DFI
  with the semi-Lagrangian core; its executed verification cell shows
  min cloud liquid water = 0.0 exactly (both DFI and unfiltered runs, and
  strictly positive humidity), versus −1.1e-4 with negative humidity in
  the Eulerian notebook — closing the negative-cloud-water caveat written
  into the original notebook.
- **Off-centering does not fix the large-Δt terrain artifact.** An ε-scan
  on the 2-day ERA5 forecast (ε ∈ {0, 0.05, 0.1, 0.2} at dt = 45/60 min)
  left the hot anomaly essentially unchanged (340 → 335/334/348 K at
  60 min) while degrading the cold extreme (183 → 148 K at ε = 0.2), and
  the anomaly localizes to the steepest terrain (Andes coast, Himalayan
  foothills) rather than ridge tops. This rules out classical stationary
  orographic resonance (which decentering damps) and implicates the
  explicit terrain terms (`RT'∇ln pₛ` residual) at large Δt — the
  Ritchie & Tanguay (1996) smoothed interpolated variable noted in §4 is
  the indicated follow-up, and dt = 30 min remains the tuned operating
  point.
- One integration fix from the notebook runs: diagnostics that feed a full
  state into `compute_diagnostic_state_sigma` (e.g. the notebooks'
  vertical-velocity output) must strip nodal tracers first, since that
  helper spectrally transforms `state.tracers`. The SL ERA5 notebook drops
  tracers from the diagnostic call; the equation classes themselves already
  split nodal tracers internally.
- PR curation: notebooks are grouped as `notebooks/eulerian/` and
  `notebooks/semi_lagrangian/` (same file names, README links updated);
  the portability fixes for NumPy 2.0 and current-main APIs apply to both
  groups. The one-off gather microbenchmark script was dropped from the
  PR — its measurements are recorded above and it remains available in
  the branch history.
- Compile-time audit: the per-field interpolation entry points
  (`interpolate_3d` and the horizontal single-field path) are now wrapped
  in `jax.jit` with static (grid, order, limiter), so repeated call sites
  with the same signature — the two RK2 stages, several tracers with the
  same limiter — share one lowered computation instead of inlining copies.
  On the T85/L32 3-tracer step this removes 30% of lowered ops (6442 →
  4518; gathers 64 → 30) with runtime unchanged — but an A100 A/B shows
  GPU compile time is *unchanged* (55.0 → 55.2 s for the T170/L32 step),
  so this is a tracing/lowering cleanup, not a compile-time fix. The GPU
  compile cost is dominated by the ~58 MB of spectral-basis constants
  embedded once per `to_modal`/`to_nodal` call site in shared
  `spherical_harmonic` code (the SL step lowers to 59 MB of module text
  vs 32 MB Eulerian); deduplicating those transforms is the identified
  follow-up, outside the semi-Lagrangian scope.
- Hybrid-coordinate support (originally deferred in the non-goals) was
  added after all: the hybrid *level* coordinate has fixed per-level nodes
  (`s = (A + B·pₛ_ref)/pₛ_ref`), so the transport machinery applies
  unchanged via a small `VerticalNodes` generalization; ṡ is diagnosed from
  the interface mass fluxes, and the ΔB-weighted mean wind makes the 2-D
  ln pₛ trajectory replacement exact (ΣΔB = 1). The hybrid class gains the
  same advective/non-advective split (with two extra non-advective
  linearization-residual terms) and `SemiLagrangianPrimitiveEquationsHybrid`
  mirrors the sigma class. Validation is deliberately minimal: the A = 0
  configuration is pinned to the sigma semi-Lagrangian class (nodal
  velocities to 2e-6; 12-step runs to 4e-3, the accumulated
  Simmons–Burridge-vs-sigma vertical discretization difference also carried
  by the Eulerian equivalence tests), plus reconstruction, a one-day
  baroclinic consistency check against the Eulerian hybrid core on genuinely
  hybrid levels (l2 = 1.0e-3), and stepper-rejection coverage. Genuinely
  hybrid configurations inherit the Eulerian hybrid class's
  "not thoroughly verified" caveat.
- **Warm-started departure iterations (IFS 48r1 follow-up).** After reading
  Diamantakis & Váňa (2022; ECMWF Newsletter 173) — whose geocentric-
  Cartesian trajectory solve matches the formulation here — their warm-start
  refinement was implemented and measured: `initial_guess` on both
  departure-point solvers, threaded through the equation interface and all
  three SL equation classes (which also gain a `departure_iterations`
  field), with two stepper hooks: `warm_start_corrector` on the RK2 stepper
  (corrector seeded from the predictor's departure points, no multistep
  memory) and `warm_start_departures` on SETTLS (departure points carried
  in the step state, the direct IFS analogue). Warm-starting from a
  k-iteration solve and iterating j more is *exactly* a (k+j)-iteration
  solve (tested); on a spun-up T85L8 baroclinic wave at dt = 60 min the
  previous step's solution is as good a guess as one cold iteration
  (max departure-point residual 1.3e-4 vs 1.8e-4, displacement scale
  2.0e-2), so one warm iteration ≈ two cold (2.6e-6 vs 3.2e-6). End-to-end
  (12 h vs an 8-iteration reference, f64): cold-1 is unusable (T′ error
  3.6e-3 at dt = 30 min, comparable to the discretization error), warm-1
  is fine (1.1e-4 RK2, 1.5e-5 SETTLS), and SETTLS warm-2 compounds the
  carried refinement to 1.2e-7 vs 3.9e-6 cold. A100 timings at T170/L32
  f32: the 3-D solve alone is 4.75/8.99/4.80 ms for cold-1/cold-2/warm-1
  (the warm start itself is free); full steps are RK2 35.2 (cold-2) /
  37.9 (warm-2) / 28.8 (warm-1) ms and SETTLS 19.1 / 19.3 / 14.9 ms.
  Defaults chosen from these numbers: SETTLS warm start on (cost-neutral,
  residual 20-100✕ better); RK2 corrector warm start **off** at equal
  iterations — it is slightly slower because a cold corrector's first
  iteration interpolates at the constant arrival mesh, which XLA
  constant-folds (visible on CPU and GPU alike) — and documented as the
  enabler for `departure_iterations=1` (−18% RK2 / −22% SETTLS step time).
  Yardstick for the single-iteration accuracy: the converged
  (8-iteration) scheme's own sensitivity to time-step refinement
  (30 → 7.5 min) in the same 12 h T85 experiment is 2.3e-4 on T′, so
  warm-1 trajectory truncation sits ~2✕ below it for RK2 (1.1e-4) and
  ~15✕ below for SETTLS (1.5e-5). Note the refinement comparison is not a
  clean time-order measurement (observed rate ~0.6, not 2): in SL cores
  smaller steps mean *more* interpolation applications, so interpolation
  diffusion partially offsets the smaller truncation terms — a
  well-known SL trade-off that also argues for the long-step operating
  points used here.
- **Warm-started single iterations adopted as the default.** An A100
  notebook study at the ERA5 operating point (T170/L32, dt = 30 min,
  2-day forecast, notebook-faithful nodal + quasi-monotone tracers)
  compared the warm-1 configuration against the then-default cold-2:
  32.9 vs 39.5 ms/step (−17%; simulation cell 3.21 s vs 3.93 s), no
  terrain anomaly (max T 305.8 K at 0.28 km elevation, versus the
  +9/+34 K spikes of the dt = 45/60 min pathology), min cloud liquid
  water exactly 0.0, and the same distance from the Eulerian dt = 5 min
  reference (T′ rel-l2 9.4e-3 vs 9.1e-3; the two SL runs are 4✕ closer
  to each other than either is to the reference). The day-2
  warm-vs-cold differences (T 2.1e-3, pₛ 9.2e-4, humidity 9.9e-2) are
  *smaller than the sensitivity of the converged scheme to a benign
  30 → 20 min time-step change* (T 3.2e-3, pₛ 2.6e-3, humidity 1.26e-1
  — sharp filamentary moisture fields diverge chaotically between any
  two configurations at this horizon), so the warm start perturbs the
  forecast less than a time-step tweak does. Defaults flipped
  accordingly: `departure_iterations = 1` on the equation classes and
  `warm_start_corrector = True` on the RK2 stepper; all three
  semi-Lagrangian notebooks re-executed in this configuration
  (Held–Suarez 1200-day cell 1 m 31 s → 1 m 14 s; baroclinic two-week
  cell 9.33 → 5.93 s, now clearly ahead of the Eulerian core at T42
  rather than at parity; ERA5 DFI cell 1 m 24 s → 1 m 12 s). The
  extreme-Δt stress tests request `departure_iterations = 2` explicitly:
  near the convergence margin dt·max‖∇V‖ < 1 (3 h steps, 8✕-Eulerian
  shallow-water steps) a single warm-started iteration drifts ~5✕ more,
  a caveat now documented on the equation classes.
- **Nodal-tracer noise: diagnosis and a negative A/B result.** The
  executed ERA5 SL notebook shows 2–6-gridpoint oscillations in humidity
  near the Maritime Continent and the Amazon at day 1–2 (coherent
  stationary wave trains plus fine stipple near sharp maxima); the same
  regions in the Eulerian notebook are smooth. Spectral Gibbs in the
  tracer is impossible by construction (nodal storage; min cloud water is
  exactly 0.0), so the candidate mechanisms were (a) grid-scale variance
  accumulating because nodal tracers receive no dissipation at all (no
  spectral hyperdiffusion by design, no moist physics in this demo),
  (b) cubic-interpolation overshoot terraced by the quasi-monotone clip,
  and (c) noise imprinted by the resolved spectral dynamics near sharp
  terrain. `semi_lagrangian.nodal_diffusion_filter` +
  `primitive_equations.step_filter_for_nodal_tracers` were added to test
  (a): a separable index-space smoother (periodic longitude, cross-pole
  halo latitude; `mu = 1 − exp(−dt/tau)` at the 2Δ scale; order 2 is a
  Shapiro-type δ⁴ kernel with the modal ∇⁴'s selectivity, clipped to the
  local 3✕3 range so positivity and no-new-extrema survive exactly). The
  A100 A/B at the dynamics' own tau left the wave trains essentially
  unchanged (forecast cell 3.29 s vs 3.27 s — the filter is free — and
  positivity intact): a filter that damps passive 2Δ content to ~1e-12
  over the run barely touched them, so they are continuously forced,
  stationary, ~4–6Δ patterns from the resolved dynamics — mechanism (c),
  which no tracer-side scheme (including SLHD, §12) can remove. The
  Eulerian notebook is smooth there only because its spectral tracer
  representation truncates (~3Δ at T170 on the quadratic grid) and
  hyperdiffuses those scales, at the cost of Gibbs negatives. Having
  served its diagnostic purpose without fixing anything observable, the
  filter was reverted rather than carried as speculative API (its design
  — a separable index-space smoother with cross-pole halos, e-folding
  parametrization matching the spectral filter, and a Shapiro-δ⁴ order
  with a local-range clip — lives in the branch history should a
  physics-free configuration need the closure); the indicated follow-ups
  for the wave trains are dynamics-side — stronger or steeper tail
  hyperdiffusion, smoother orography, or the Ritchie–Tanguay
  smoothed-terrain variable already flagged by the large-Δt study.
- **Ritchie & Tanguay smoothed-terrain transport, adopted as default.**
  `terrain_smoothed_log_sp` on both SL primitive-equation classes
  transports `ψ = ln pₛ + Φₛ/(R·T̄)` (terrain-locked parts cancel before
  interpolation; the static correction is removed exactly at arrival
  nodes) and adds the compensating `v̄·∇C` to the explicit forcing,
  evaluated spectrally with the trajectory wind's vertical weights (Δσ or
  ΔB — the continuity exactness identity survives). Premise measured
  sharply (a 4 km mountain rough at T42 truncation makes `ln pₛ` 33✕
  rougher than ψ), exact no-op over flat terrain, consistent on steep
  terrain (T′ l2 3.2e-3 over 2 days at dt = 60 min). ERA5 T170/L32 A100
  probes: exactly neutral at dt = 30 min (rel-l2 7e-4, identical extremes,
  unmeasurable cost) and part of the large-Δt terrain fix — at dt = 60 min
  with converged trajectories the hot anomaly goes 327.2 K (raw) → 322.4 K
  (smoothed) → 318.1 K (GEM-style ≥6Δx orography filtering alone) →
  **306.9 K, the healthy value, with both** — i.e. the smoothed variable
  and orography smoothness are complementary halves, matching GEM's
  operational ≥6Δx orography rule (Husain et al. 2020). Two more probe
  findings: the catastrophic cold extremes previously seen at dt = 60 min
  were the single-warm-iteration default (gone at
  `departure_iterations=2`, per the documented convergence-margin caveat),
  and the earlier probe run that conflated the two effects is superseded.
- **Production-model details audit (agent survey, 2026-07-15).** dinosaur's
  SL core was audited against IFS Cy48r1 Part III (read in full), ARPEGE/
  ALADIN, GEM, UM/ENDGame, JMA GSM, CAM SLD and the SL-era GFS. Coverage
  verdict: the departure solve, warm starts, LADVF Coriolis,
  parallel-transport wind rotation, SETTLS bracket, ln pₛ continuity,
  limiter variant and nodal tracer storage match or supersede operational
  practice. High-priority genuinely-new items, in order: (1) cubic/quintic
  *vertical* interpolation (every production model is ≥cubic vertically;
  IFS quintic+WENO for T, q since cy47r1) — the cubic option is now
  implemented, see below; (2) IFS SETTLS vertical-extrapolation
  safeguards (first-order limiter on the η̇ extrapolation with a
  documented smooth tanh differentiable variant, and the cy48r1
  warm-start sign-change fallback); (3) GEM-style ≥6Δx orography
  filtering (validated by the probe above); (4) quasi-monotone
  interpolation on dynamical variables (IFS default, we never limit
  dynamics — cheap A/B). Medium tier: arrival-only physics-tendency slot
  (Wedi/SLAVEPP: vertical diffusion must not be trapezoided along the
  trajectory), COMAD weights, single-precision dry-mass fixer (IFS needed
  one at cy47r2 — relevant to float32 climate runs). The suggested
  per-step stencil/weight reuse across advected fields was measured and
  closed without code: XLA's common-subexpression elimination already
  deduplicates the stencil arithmetic across per-field interpolation
  calls — six fields at shared departure points cost 8.05 ms on A100
  (166 ms CPU) vs 8.12 ms (167 ms) with explicit single-stencil sharing
  and 19.8 ms (391 ms) if unshared, ratio ~1.0 even with the real
  mixed-limiter call pattern. The *contraction form*, however, was worth
  changing: production codes evaluate the tensor-product interpolation as
  a cascade of 1-D interpolations so the gathered stencil tensor never
  exists in memory, and the XLA analogue is gather→multiply→reduce
  fusion, which the einsum contraction forecloses by lowering to
  dot_general (materialized operands). `_contract_stencil` now picks per
  backend (einsum on CPU, where it is 1.8✕ faster; the fused form on
  accelerators): 26.3 vs 32.0 ms/step with linear vertical and 33.3 vs
  58.4 ms/step with cubic on the ERA5 benchmark — the in-context
  materialization penalty far exceeds the isolated-call 1.15–1.20✕.
  Cumulative software gains at the ERA5 operating point: 39.5 (original)
  → 32.9 (warm-started single iteration) → 26.3 ms/step (fused
  contraction), 1.5✕ total. Also flagged: IFS runs
  warm+2 departure iterations even after the 48r1 warm start (Diamantakis
  & Magnusson 2016) — consistent with our stress tests; warm+1 stands
  validated at T170/dt=30 only.
- **Cubic vertical interpolation option.** `vertical_order='cubic'` in
  `interpolate_3d` (4-point Lagrange on the non-uniform nodes, degraded
  to linear in the first and last cells per operational practice, exact
  no-extrapolation preserved, quasi-monotone limiter still clipping
  against the two bracketing levels only), exposed as
  `vertical_interpolation_order` on both SL equation classes. Exact for
  cubic σ-profiles away from boundaries; equals linear in boundary cells
  bit-for-bit. A100 A/B at the ERA5 operating point: behaviorally sound
  (cloud water still exactly 0.0; T changes rel-l2 6.2e-3 with a 19 K
  colder global minimum, consistent with removing vertical interpolation
  diffusion near the tropopause — the sharpening IFS pairs with its SLVF
  vertical smoother) but initially +82% step time (32.0 → 58.4 ms: the
  gathered tensor doubles to 64 points per field). The fused stencil
  contraction (below) collapses that penalty: 33.3 vs 26.3 ms/step, +27%
  — cubic vertical now costs about what linear did before the fusion
  change. Default stays 'linear' on behavioral grounds (the sharper
  vertical structure deserves validation, and IFS pairs high-order
  vertical with its SLVF smoother); quasi-cubic (§12) remains the further
  cost reduction if wanted.
- **Moist wave trains: mechanism confirmed as gravity waves; off-centering
  is the effective knob.** Closing the diagnosis chain: the dt = 30 min
  humidity wave trains survive the smoothed-terrain transport (R–T A/B)
  *and* GEM-style ≥6Δx orography filtering unchanged — eliminating the
  remaining terrain hypotheses — and they sit over the Bay of Bengal and
  equatorial ocean, not over mountains. The surviving explanation, now
  tested: gravity waves radiated by geostrophic adjustment of ERA5's
  convectively driven divergence (which the dry core cannot sustain),
  neutrally propagated by the centered semi-implicit trapezoid; the
  undiffused nodal tracer records them faithfully while the Eulerian
  notebook's spectrally truncated + hyperdiffused humidity hides the same
  dynamics. Off-centering — whose operational purpose is exactly
  first-order selective damping of these fast modes — shows the predicted
  dose response on the executed notebooks: ε = 0.05 (the UM operational
  value) substantially fades the wave trains, ε = 0.1 essentially
  eliminates them, with resolved filaments and plumes visually intact,
  cloud liquid water still exactly 0.0, and humidity extrema unchanged at
  the 1e-8 level. The notebooks remain centered (ε = 0) by default —
  formal second order is the better demo default, the waves are cosmetic
  in a physics-free configuration, and any full model's moist physics
  provides the missing humidity-variance sink — with ε documented as the
  remedy when a clean tracer field matters.
- Submitted as https://github.com/neuralgcm/dinosaur/pull/135 (the plan
  moved into plans/ in the same PR).

**M5c — SETTLS stepper.**

- Implemented per §3.6 with one documented deviation: trajectories reuse
  the midpoint departure-point iteration with winds extrapolated to
  `t + Δt/2` (`(3V^n − V^{n−1})/2`, the Temperton et al. 2001 form) rather
  than Hortal's endpoint-form iteration, so the equation interface is
  consumed verbatim (one wind pytree per departure solve). The RHS follows
  SETTLS exactly (`2N^n − N^{n−1}` riding from departure, `N^n` at
  arrival).
- Step state is a `(x, (N_prev, V_prev))` tuple with an RK2 bootstrap
  (`semi_lagrangian_settls_init`) and a `settls_step_filter` adapter, as
  planned. Second-order convergence verified on both ring toys (including
  the state-dependent-velocity one, which exercises the wind
  extrapolation); on the one-day T21 baroclinic wave SETTLS tracks the RK2
  stepper to T′ l2 < 2e-3 at half the per-step cost.
- Scope notes: the notebook deliverable mentioned in §10 was not produced
  (docstrings + this section carry the documentation); the wall-clock and
  full 3-6✕ Δt study remain deferred to GPU/TPU hardware as discussed
  under M4.

**M5 / M5b — Moist terms, tracers, gradients, nodal storage.**

- Moisture needed no new physics code: the M0 split already classifies the
  humidity corrections as non-advective, so moist SL is `humidity_key` +
  the humidity tracer transported like any other. Verified by a q=0
  reduction test and a moist baroclinic consistency run vs the Eulerian
  core.
- The plan §7 caveat is now quantified: a barely resolved tracer at T21
  undershoots to ~−6% of peak in *modal* storage with or without the
  quasi-monotone limiter (the per-step modal round trip reintroduces the
  ringing the limiter removed). With `nodal_tracers` storage (M5b) the same
  configuration is exactly non-negative over a two-day run, with no new
  maxima and small tracer-mass drift. Nodal tracers may not participate in
  the dynamics (`humidity_key`/`cloud_keys` rejected) and are excluded from
  modal filters via `step_filter_excluding_nodal_tracers`.
- `compose_equations` now preserves the semi-Lagrangian interface when its
  ImplicitExplicitODE member is semi-Lagrangian (extra explicit equations
  become additional non-advective forcing), letting Held-Suarez forcing
  compose unchanged: a 3-day forced T21 run stays stable with physical
  temperatures.
- `jax.grad` through two full SL primitive-equation steps matches central
  finite differences to 1e-2 relative (float32 end-to-end).
- Adversarial review of M3b/M4 verified the LADVF factor-2/sign/projection
  derivation independently and mutation-tested it (sign or factor errors
  exceed test thresholds by >100✕). Review-driven fixes: documented the
  coefficient-1-on-state bracket invariant that planetary-momentum
  transport imposes on steppers (transport is affine, not linear), warned
  in both SL equation docstrings that Eulerian steppers would silently
  integrate advection-free dynamics, covered the previously untested
  multi-layer pressure coupling and orography paths in the shallow-water
  tests, aligned the SW nodal-conversion clipping convention, and
  corrected the int32 gather-limit comment.

**M4 — `SemiLagrangianPrimitiveEquations` (dry, sigma).**

- Worked essentially on first assembly thanks to the earlier de-risking: on
  the JW steady state at T21/L8 with Δt = 30 min, both Coriolis modes hold
  the jet with T′ drift 1.1e-3 over one day — comparable to the Eulerian
  core at Δt = 10 min (1.3e-3, dominated by spatial truncation of the
  initial balance in both cases). The baroclinic wave tracks the Eulerian
  core to T′ l2 ≈ 1.2e-3 after one day.
- Δt extension measured at T42/L8 on the steady state (CPU-affordable
  scope): the Eulerian `imex_rk_sil3` core hits its spectral advective
  stability limit almost exactly where `u·k_max·Δt = √3` predicts
  (stable at 2.5 h, NaN at 3 h over two days), while SL at 3 h stays
  bounded with T′ drift 6.7% — degraded accuracy, consistent with the §5
  trajectory-convergence margin (`Δt·max‖∇V‖` ≈ 0.5-0.8 there), and
  unaffected by modal filtering (the error is large-scale, not spectral
  noise). The gentle JW jet at coarse CPU resolutions cannot showcase the
  3-6✕ headline (stronger jets/finer grids are where Eulerian Δt collapses
  and SL does not); that study remains with the deferred GPU/TPU work
  (§9.8).
- `explicit_nonadvective_terms` gained an `include_coriolis` flag so the
  planetary-momentum mode can drop Coriolis from N without duplicating the
  method.

**M3b — Shallow-water SL equations.**

- Implemented as `SemiLagrangianShallowWaterEquations` subclassing the
  Eulerian class (implicit terms/inverse inherited) — and used to de-risk
  the advected-planetary-momentum Coriolis treatment ahead of M4:
  `transport_wind`/`transport_wind_2d` gained a `planetary_rotation_rate`
  option that adds the analytic `2Ω✕R` at the departure point and removes
  it at arrival, so only the relative wind is interpolated. Derivation
  pinned in the code: the horizontal projection of `2Ω✕v` is exactly
  `f k✕v`, so the covariant (parallel-transport) advection of `v + 2Ω✕R`
  has no Coriolis force.
- The plan's promised de-risk delivered: on the steady geostrophic flow at
  T42, the Eulerian `imex_rk_sil3` core goes non-finite at Δt = 0.4
  (nondimensional; ~8✕ its stable step), while the SL core remains steady
  to l2 ≈ 3e-3 over 50 such steps. Both Coriolis modes hold the steady
  state to l2 < 1e-4 at moderate Δt.
- SL-vs-Eulerian differences on the barotropic-instability flow plateau at
  the interpolation-error floor (~5e-4 potential l2) rather than shrinking
  with Δt — the §9.6 bounded-window behavior, so the consistency test pins
  closeness at fixed Δt instead of asserting convergence.

**M2 — Passive transport validation.**

- The dominant error in long passive-advection runs is accumulated per-remap
  interpolation error, so at fixed final time the error *decreases* with
  larger Δt (measured Williamson case 1 at T42: l2 = 0.014 at 64 steps,
  0.07 at 100, 0.16 at 250, 0.20 at 512 — identical in float32 and float64).
  Step counts commensurate with the grid are misleading (128 steps = exactly
  one cell per step at T42's 128 longitudes → near-exact); tests use
  non-commensurate counts and thresholds calibrated to measured values.
- Nair & Lauritzen deformational-flow return error is resolution-limited at
  T42 (l2 = 0.24; the mid-time filaments are thinner than the grid), falling
  to 0.021 at T85 with the same step count — consistent with published
  behavior, so the T42 threshold is set accordingly rather than chasing the
  headline numbers from finer grids.
- The positivity stress case behaves as issue #55 hopes, with an important
  correction found by adversarial review: the first version's spectral
  baseline used a tilted flow for which the *unfiltered* flux-form
  pseudo-spectral operator is pole-unstable, so its negative values were a
  blow-up artifact, not Gibbs ringing. The test now advects along the
  equator (where the same baseline is verified stable and accurate) with a
  ~1.5-cell hill: genuine Gibbs ringing measures −5.8% of peak, unlimited
  cubic SL undershoots, and quasi-monotone SL stays exactly non-negative.
  Notably the spectral core remains *more accurate in l2* on this barely
  resolved field — the SL win is positivity, not accuracy.
- Review also showed the full-revolution Williamson tests were individually
  vacuous against a no-op transport (exact solution = initial condition);
  they now compare a partial revolution against the analytically rotated
  field. A state-dependent-velocity ring toy was added after review noted
  constant-velocity toys cannot detect a broken stage-2 wind average
  `½(V^n + V^*)`; mutation-tested (correct scheme: order 2.00; a
  no-averaging mutant: 1.02).

## 14. References

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
- Diamantakis, M. & Váňa, F. (2022). A fast converging and concise algorithm
  for computing the departure points in semi-Lagrangian weather and climate
  models. *Q. J. R. Meteorol. Soc.*, 148, 670-684. doi:10.1002/qj.4224
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
- Pudykiewicz, J., Benoit, R. & Staniforth, A. (1985). Preliminary results
  from a partial LRTAP model based on an existing meteorological forecast
  model. *Atmos.-Ocean*, 23, 267-303.
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
- Smolarkiewicz, P. K. & Pudykiewicz, J. A. (1992). A class of
  semi-Lagrangian approximations for fluids. *J. Atmos. Sci.*, 49, 2082-2096.
- Staniforth, A. & Côté, J. (1991). Semi-Lagrangian integration schemes for
  atmospheric models — a review. *Mon. Wea. Rev.*, 119, 2206-2223.
- Temperton, C. (1997). Treatment of the Coriolis terms in semi-Lagrangian
  spectral models. *Atmos.-Ocean*, 35(sup1), 293-302.
  doi:10.1080/07055900.1997.9687353
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
