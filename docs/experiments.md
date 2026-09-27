# Experiment log

The project's memory: every campaign, including the ones that led nowhere. All numbers
come from actual runs (`results/`).

**Code provenance.** The code on the main line was reduced to the retained method once
the exploration was over. The abandoned variants (lazy integration of plastic elements,
catch-up, neighbour propagation, post-step and predictor-only checks, global reversal
wake-up, modified Newton), the learned policies (Phase 3) and the failure-case study are
preserved at git tag **`exploration-complete`** (commit `b062a19`). Their raw results are
kept in `results/` so that every table below remains traceable.

**Conventions.** "save" = fraction of the **element-level cost** (strains, law, K^e,
f^e, assembly, extrapolation, checks, decisions) saved against the **Sysala reference**
(same code, orchestrator off, elastic tangent reused on elements without a plastic
point), under the `scalar` cost model unless stated; "u" = maximum relative displacement
error over the history; "a" = relative error on the accumulated plastic strain at the
final step; "it" = Newton iterations (reference in brackets). Unit costs are re-measured
by micro-benchmark at every campaign, so "save" values for one configuration fluctuate
between campaigns; the operation counters do not. Re-running the κ sweep on another
machine after the code clean-up gave bit-identical counters and displacements, but
savings shifted by −5 to +9 points (κ = 1: notched plate 0.517 → 0.527, cantilever
0.486 → 0.575, cyclic plate 0.569 → 0.576, cyclic cantilever 0.456 → 0.410; see
`results/phase2/kappa_sweep.jsonl`). The *ranking* of configurations is stable; absolute
percentages should be read as ±10 points.

## 0. Solver and validation (Phase 1)

Solver: plane-strain Q4 with B-bar, J2 plasticity with isotropic (linear + Voce) and
linear kinematic hardening, vectorised radial return + scalar reference version,
consistent tangent, Newton-Raphson with a tangent predictor, displacement control,
per-phase counters and timers. Tests (`tests/`): consistent tangent against finite
differences (< 1e-5 relative, isotropic and kinematic), vectorised = scalar (1e-8), patch
test on a distorted mesh (1e-12), simple shear against the closed-form J2 solution
(1e-10), quadratic Newton convergence (e.g. 3.1e-1 → 5.6e-2 → 3.5e-4 → 8.0e-8 → 6.2e-14),
Bauschinger effect, **orchestrator "all active" = reference bit for bit**, and **exact
orchestrator = reference**.

Notable incident: the first Newton diverged on every plastic case under displacement
control, because the first iteration evaluated the tangent on a non-equilibrated iterate
(elements on the loaded edge at 10 % strain). Fixed with the standard tangent predictor
(linearisation with the previous converged tangent before any integration). The fix is
shared by the reference and the orchestrated runs.

Cases: notched plate (3096 elements, 6450 dofs, 60 steps, u_max 0.06 mm), overloaded
plate (u_max 0.12, until the ligament yields), cantilever (2000 elements, 50 steps),
cyclic plate and cantilever (126 and 105 steps, λ: 0 → 1 → −0.6 → 1) with isotropic
hardening (`cyclic_*`) and combined isotropic + kinematic hardening (`cyclic_*_kin`,
H_k = 4 GPa), parameterised families (notch radius, height; cantilever height and
length). Steel: E = 200 GPa, ν = 0.3, σ_y0 = 250 MPa, H = 1000 MPa, σ_∞ = 400 MPa, δ = 20.

## 1. Phase 0 — oracle potential study

Scripts: `experiments/phase0_oracle.py`, `phase0_stepsize.py`, `phase0_baselines.py`;
raw results `results/phase0/*.json`, figures `results/phase0/*.png`.

### 1.1 Method

For every element and every step, a greedy policy on the **true trajectory**: from the
last step at which the element was integrated (k0), evaluate the error of the stress
extrapolation at step k; if it is below tolerance, the element is "extrapolated" at step
k (k0 unchanged), otherwise it is "integrated" (k0 := k). Three extrapolators:

- **freeze**: σ̂_k = σ_k0;
- **linear in time**: σ̂_k = σ_k0 + (k−k0)(σ_k0 − σ_k0−1);
- **tangent**: σ̂_k = σ_k0 + D_k0 (ε_k − ε_k0), with D_k0 the stored consistent tangent.
  This is the extrapolator that corresponds to *lazy integration*: no law evaluation,
  K^e block reused, internal variables frozen until wake-up, where the law is integrated
  over the total increment.

Error = norm of the in-plane components of (σ̂ − σ_true) / σ_y0, max over the 4 Gauss
points. Tolerances swept from 0.1 % to 10 %. For the tangent extrapolator we also check
the **error of the deferred integration** at wake-up (σ and α obtained by integrating the
law from the state k0 with the true strain ε_k, against the true state).

Cost weighting: costs measured on this code by micro-benchmark, under two models:
`vector` (vectorised NumPy implementation) and `scalar` (the same algorithms written point
by point in pure Python, whose plastic/elastic *ratio* is a proxy for a compiled
per-element code). Cost of one element-iteration, mean over the five cases:

| model | active elastic element | active plastic element | quiet (extrapolated) element | plastic/elastic ratio | quiet/elastic ratio |
|---|---|---|---|---|---|
| scalar | 54–56 µs | 160–165 µs | 25–26 µs | 2.9–3.0 | 0.46–0.48 |
| vector | 3.9–4.1 µs | 5.0–5.1 µs | 0.8 µs | 1.25–1.28 | 0.19–0.20 |

(The quiet cost here is the Gauss-point path; the element-level path adopted in Phase 2
brings it down to 2.6 µs, see § 2.1.)

### 1.2 Raw potential (fraction of element-steps that can be extrapolated), tangent extrapolator

| case | plastic fraction max / mean | tol 0.2 % | tol 1 % | tol 5 % |
|---|---|---|---|---|
| notched plate | 0.36 / 0.080 | 0.976 | 0.986 | 0.993 |
| overloaded plate | 0.53 / 0.269 | 0.936 | 0.964 | 0.982 |
| cantilever | 0.23 / 0.125 | 0.969 | 0.981 | 0.990 |
| cyclic plate | 0.36 / 0.034 | 0.978 | 0.983 | 0.989 |
| cyclic cantilever | 0.28 / 0.117 | 0.948 | 0.964 | 0.977 |

Extrapolators compared at tol 1 % (plate / overloaded / cantilever / cyclic plate /
cyclic cantilever): freeze 0.18 / 0.39 / 0.48 / 0.04 / 0.26; linear 0.88 / 0.83 / 0.85 /
0.85 / 0.78; tangent 0.99 / 0.96 / 0.98 / 0.98 / 0.96. **Freezing is unusable** (the
stress of elastic elements evolves with the load); **the tangent dominates**.

### 1.3 An honest reading of the raw potential: two components

1. **Elastic elements** (72 to 97 % of the elements depending on case and time) are
   extrapolated *exactly* by the elastic tangent. This component is trivial, and it
   dominates the raw number.
2. **Plastic elements** can also be extrapolated much of the time: fraction of plastic
   element-steps extrapolated at tol 1 % = 0.83 / 0.86 / 0.85 / 0.64 / 0.74. But this
   fraction **depends on the load step** (`phase0_stepsize.py`, notched plate, tol 1 %):

   | number of steps | 10 | 20 | 40 | 60 | 120 |
   |---|---|---|---|---|---|
   | plastic element-steps extrapolated | 0.39 | 0.62 | 0.75 | 0.83 | 0.90 |
   | raw potential | 0.94 | 0.97 | 0.98 | 0.99 | 0.99 |
   | weighted potential (scalar, vs naive) | 0.47 | 0.54 | 0.54 | 0.59 | 0.59 |

   Part of the extrapolability of plastic elements is therefore "the steps are small for
   them", which global load-step control would also capture. What remains spatially
   heterogeneous, and global step control does not capture, is the distribution of
   integration frequencies: at tol 1 % and 60 steps, the median per-element integration
   frequency is 0, the 90th percentile 0.05, the maximum 0.13 (plate) and 0.20
   (cantilever). No element needs integrating at every step.
3. **Deferred integration** is practically exact on these trajectories: error on α at
   wake-up, 99th percentile ≤ 1.4e-6 at tol 1 %, ≤ 4.4e-5 at tol 5 % (radial return is
   exact on proportional paths). The internal state stays admissible by construction.

### 1.4 Cost-weighted potential (tangent extrapolator)

Three references: naive solver (everything re-integrated and re-assembled), "Sysala"
baseline (K^e block reused for elements without a plastic point, the state of the art),
and the oracle applied **on top of** Sysala. `scalar` cost model:

| case | Sysala vs naive | oracle vs naive (0.2 % / 1 % / 5 %) | **oracle vs Sysala** (0.2 % / 1 % / 5 %) |
|---|---|---|---|
| notched plate | 0.24 | 0.55 / 0.57 / 0.59 | **0.40 / 0.44 / 0.46** |
| overloaded plate | 0.15 | 0.59 / 0.64 / 0.67 | **0.51 / 0.57 / 0.61** |
| cantilever | 0.23 | 0.55 / 0.58 / 0.60 | **0.42 / 0.46 / 0.48** |
| cyclic plate | 0.29 | 0.51 / 0.53 / 0.55 | **0.31 / 0.34 / 0.36** |
| cyclic cantilever | 0.23 | 0.52 / 0.55 / 0.58 | **0.38 / 0.42 / 0.46** |

With the `vector` model (much cheaper quiet element), oracle vs naive 0.76–0.79 at 1 %;
oracle vs Sysala 0.31–0.55. With the abstract model (elastic = 1, plastic = 3, quiet = ρ):
ρ = 0 gives 0.92–0.95, ρ = 0.25 gives 0.72–0.77, ρ = 0.5 gives 0.50–0.61 (at 1 %). **The
potential is driven, to first order, by the residual cost of the quiet element** rather
than by the tolerance: between 0.2 % and 5 % it moves by only a few points.

### 1.5 What the potential does not cover: the linear solver

Share of total time spent at element level (strains, law, K^e, f^e, assembly) in this
code: 0.27–0.28 (plates), 0.42 (cantilevers). The rest is essentially LU factorisation
(SuperLU). A 50 % element-level gain is therefore worth 14 to 21 % of total time here.
This share is specific to a 2D Python code with a simple J2 law: for expensive laws
(crystal plasticity, damage with a local Newton on many variables, heavy UMATs) the
element share dominates and the potential transfers almost entirely; for a large 3D
model with a direct solver, the solver share is often the majority.

### 1.6 Space-time maps

`results/phase0/*_spacetime.png` and `*_snapshots.png`. On the notched plate, the
elements that need integrating concentrate (i) on the **plastic front** as it passes
from elastic to plastic (tangent change), then (ii) sporadically, every few steps, inside
the plastic zone when hardening drift pushes the tangent out of tolerance. On the cyclic
plate, the active region on reloading is a band (the plastic "wings") covering 13 % of
the elements at the last step; on unloading everything becomes extrapolable at once (0 %
integrated during the whole intermediate elastic phase), and wake-up happens when the
reversed stress reaches the hardened surface. With purely isotropic hardening, wake-up
in compression is late and limited; kinematic hardening (Bauschinger effect) was added
to make the cyclic case more discriminating on wake-up.

### 1.7 Decision: GO, with two explicit caveats

The exit criterion (weighted potential ≥ 25–30 %) was met on all five cases, including
**against the strong baseline**: 31 to 61 % of element-level cost (scalar), 31 to 55 %
(vector). Caveats: (1) **it is an upper bound** — the oracle sees the true trajectory;
online, the extrapolation changes the equilibrium and the error propagates; (2) **gain
on total time = element-level potential × element share**, 0.27 to 0.43 here.

Revisions made to the brief after Phase 0: the reference is the Sysala solver, not the
naive one; the "small and moving active region" intuition is refined (the plastic zone
grows monotonically under monotonic load; the region that *needs re-integration* is the
plastic front plus a rotating fraction of the plastic zone; only the cyclic case truly
moves); the load step is a confounding factor and is always reported.

### 1.8 Upper bounds recomputed with the final cost model

With the final cost model (quiet path f_0 + K_0 Δu, 2.6 µs vs 55 µs for an active elastic
element) — `experiments/oracle_vs_online.py`, `results/phase2/bounds_vs_online.json`:

| case | mean plastic fraction (cost share) | (a) ideal exact policy | (b) tangent oracle 1 % (0.2 %) | online exact κ=1 |
|---|---|---|---|---|
| notched plate | 0.080 (0.35) | 0.605 | 0.891 (0.851) | 0.517 |
| cantilever | 0.125 (0.40) | 0.559 | 0.896 (0.857) | 0.486 |
| cyclic plate, kin. | 0.036 (0.21) | 0.737 | 0.871 (0.838) | 0.569 |
| cyclic cantilever, kin. | 0.116 (0.39) | 0.571 | 0.852 (0.798) | 0.456 |

(a): an element is integrated if and only if it is plastic at the current step, perfect
knowledge, free check — the ceiling of any **exact** policy. (b): the Phase 0 bound
(extrapolation of plastic elements included). The gap (b)−(a), 13 to 34 points, is the
"extrapolation of plastic elements" share; the gap (a)−(online), 7 to 17 points, is the
cost of online checking.

## 2. Phase 2 — heuristic orchestrator

Archived campaign runner (tag `exploration-complete`): `experiments/phase2_campaign.py`;
results `results/phase2/results.jsonl` (244 rows, 8 of them failed runs), tables `results/phase2/tables.md`,
Pareto plots `results/phase2/pareto_*.png`. On the main line, the retained configuration
is re-run by `experiments/phase2_kappa_sweep.py` (`results/phase2/kappa_sweep.jsonl`) and
summarised in `docs/figures/exploration_map.png`.

### 2.1 Design choices fixed before the campaigns (and why)

1. **Element-level extrapolation**, f^e = f^e_0 + K^e_0 (u_e − u_e0), rather than per
   Gauss point (σ̂ then B^T σ̂). Mathematically identical; the former costs 64
   multiplications against ≈230 flops. Measured: quiet element 2.6 µs (scalar) vs 25 µs
   for the Gauss-point path and 55 µs for an active elastic element. Without this choice
   the online gain was **zero** on the first attempt (−0.8 %).
2. **Exact check of elastic elements** by the yield function on the extrapolated stress
   (the elastic predictor of the radial return, without the correction). For an element
   anchored elastic, σ̂ is the exact elastic stress: the check is exact. Measured cost:
   8.1 µs per Gauss point (scalar), i.e. 32 µs per checked element, **more than half an
   active elastic element**. Hence the importance of not checking everything.
3. **Check-skipping bound**: an element whose margin at its last check exceeds
   κ × (largest stress change per step observed) × (age + 1) is not checked. κ is the
   safety factor. Bug fixed along the way: elements never checked (infinite margin) were
   skipped forever.
4. **Modified Newton** (reuse the predictor's factorisation when the active fraction is
   small): **diverges** (contraction rate ≈ 0.7, > 25 iterations) on the first attempt
   with threshold 0.3. Abandoned: the previous converged tangent is too far off on the
   plastic elements. Gains on the solver phase remain out of reach of this approach.

### 2.2 Campaign A — when and how much to check (`A_monitor`, 5 modes × 5 κ × 4 cases)

Plastic elements always active. Excerpts (save scalar / u):

| mode | κ | plate | cantilever | cyclic plate kin. | cyclic cantilever kin. |
|---|---|---|---|---|---|
| iteration (every iteration) | 0 (check all) | 0.049 / 1e-13 | 0.067 / 5e-13 | 0.069 / 9e-13 | 0.056 / 4e-12 |
| iteration | 0.5 | 0.560 / 4.6e-4 | 0.515 / 2.3e-5 | 0.648 / 2.0e-4 | 0.511 / 3.6e-6 |
| **iteration** | **1** | **0.517 / 9e-14** | **0.486 / 5e-13** | **0.569 / 9e-13** | **0.456 / 4e-12** |
| iteration | 2 | 0.436 / 9e-14 | 0.427 / 5e-13 | 0.417 / 9e-13 | 0.346 / 4e-12 |
| predictor only (it. 0) | 1 | 0.552 / 2.0e-5 | 0.496 / 6.5e-6 | 0.616 / 7.6e-5 | 0.477 / 7.9e-6 |
| predictor + convergence | 1 | 0.411 / 1e-10, it 127 (102) | 0.476 / 1e-11, it 107 (97) | 0.487 / 2e-9, it 252 (205) | 0.449 / 5e-10, it 231 (212) |
| convergence only | 1 | 0.285 / 2e-10, it 184 | 0.259 / 8e-11, it 176 | 0.405 / 7e-10, it 374 | 0.208 / 1e-9, it 399 |
| post-step (wake at next step) | 1 | 0.537 / 7.6e-4 | 0.499 / 4.7e-4 | 0.638 / 1.3e-2 | 0.458 / 6.6e-3 |

Reading:
- **Checking everything at every iteration cancels the gain** (5–7 %). Checking is the
  dominant cost item of the method.
- **κ = 1 is the smallest exact factor** on all four cases; κ = 0.5 misses yield events
  (errors 1e-5 to 5e-4). κ = 2 costs 6 to 15 points for nothing.
- Checking only after convergence is **dominated**: every wake-up forces re-convergence
  (+80 % iterations). Checking during the iteration adds no iteration: wake-ups are
  absorbed by the current Newton loop (iterations identical to the reference on all four
  cases).
- Checking at the predictor only gains 2 to 5 points for a 1e-5 error (yielding that
  happens during the corrections is detected at the next step).
- Post-step mode gains 2 to 7 points under monotonic load, but costs 1e-2 error on cyclic
  cases (a step solved as elastic on elements that yield on reloading).

Wall time (Python, factorisation dominated): 1 to 16 % faster for the exact κ = 1
configuration; the element share of the time is 27–43 % (Phase 0), which caps the wall
gain; it is positive despite the Python overhead of the orchestrator
(`docs/figures/walltime_breakdown.png`).

### 2.3 Campaign B — lazy integration of plastic elements (`B_plastic`)

"Drift" policy: an element anchored plastic stays quiet as long as the linearised plastic
strain since the anchor stays below tol_α and it does not unload (n : Δε ≥ 0); prediction
at the start of the step from the previous increment, in-iteration check as a safety net.
Result: **negative across the board**.

| tol_α | plate: save / u / a / it | cantilever | cyclic plate kin. | cyclic cantilever kin. |
|---|---|---|---|---|
| always active | 0.450 / 1e-13 / 1e-13 / 102 | 0.424 / 5e-13 / 2e-13 / 97 | 0.447 / 9e-13 / 1e-13 / 205 | 0.341 / 4e-12 / 4e-12 / 212 |
| 3e-5 | 0.394 / 4.4e-5 / 5.9e-5 / 112 | 0.436 / 2.1e-5 / 4.6e-5 / 101 | 0.437 / 8.9e-4 / 2.1e-3 / 208 | 0.342 / 1.2e-3 / 1.9e-3 / 215 |
| 1e-4 | 0.455 / 9.9e-4 / 1.9e-3 / 122 | 0.458 / 1.2e-4 / 3.0e-4 / 114 | 0.409 / 1.2e-2 / 1.1e-2 / 234 | 0.335 / 7.4e-3 / 7.9e-3 / 238 |
| 3e-4 | 0.418 / 4.2e-3 / 1.1e-2 / 124 | 0.493 / 1.8e-3 / 2.1e-3 / 121 | 0.393 / 7.8e-2 / 4.7e-2 / 259 | 0.326 / 3.3e-2 / 3.8e-2 / 284 |
| 1e-3 | 0.426 / 1.8e-2 / 1.5e-1 / 127 | 0.299 / 8.4e-3 / 7.6e-2 / 138 | 0.351 / 3.5e-1 / 2.6e-1 / 275 | 0.129 / 1.1e-1 / 1.6e-1 / 336 |
| 1e-3 without unloading wake-up | 0.487 / 1.7e-2 / 1.2e-1 | 0.319 / 8.4e-3 / 7.9e-2 | 0.593 / **6.0** / 0.69 / 82 | — |

(This campaign ran with κ = 2, hence the lower "always active" baseline.)

Why it fails although the oracle said otherwise: (i) the oracle follows the true
trajectory; online, the extrapolation error of a plastic element changes the equilibrium,
hence the strain of neighbouring elements, and the error spreads through the plastic
zone — precisely the zone that drives the response; (ii) waking a plastic element causes
a stress jump (deferred state vs. extrapolated state) that costs Newton iterations (+10
to +60 %), which eats the gain; (iii) on cyclic cases, deferred integration over an
increment that contains a load reversal is **invalid** (non-proportional path,
path-dependent plasticity) — without unloading wake-up the solution is destroyed (600 %
error). The plastic elements' share of the cost (21–40 %) is also too small to be worth
it. **Conclusion: plastic elements stay active. The online active region is the plastic
zone, no more and no less.**

### 2.4 Campaign D — catch-up (`D_catchup`)

With the "drift" policy at tol_α = 1e-3: a full catch-up every 5 steps brings the error
down from 1.8e-2 to 2.7e-3 (plate) and from 8.4e-3 to 2.6e-3 (cantilever) at the cost of
22 % active elements; on cyclic cases no catch-up recovers the error (2e-1 to 4e-1).
Max-skip 3: errors 2e-3 monotonic, 1e-1 cyclic. Catch-up does not rescue the lazy
policy; with the exact policy there is nothing to catch up.

### 2.5 Campaign C — neighbourhood and precautionary margin (`C_spatial`, κ = 2)

Forcing the neighbours of woken elements active costs 10 to 27 points (plate: 0.449 →
0.339 → 0.275 for 0, 1, 2 layers; cyclic plate: 0.424 → 0.170) with no accuracy gain
(already exact). Precautionary margin on the yield function (wake if f̂ > −m): 0.02 →
−0.5 pt, 0.1 → −4 pts, no gain. In post-step mode, a margin of 0.02 to 0.2 does not
reduce the error (it stays at 3e-4 to 8e-3): the error comes from elements that cross the
limit *during* the step, not from those close to it at the start. **Conclusion: spatial
propagation and precaution are useless with an exact check; the right granularity is the
element.**

### 2.6 Campaign E — wake-up and load reversal (`E_reversal`, cyclic cases)

Global wake-up on a sign change of Δλ: no effect on accuracy (already exact), −0.5 pt of
cost. The exact local check handles reversal without global information: on unloading
every element becomes elastic and quiet (active fraction 0 over 45 steps); on reloading
the plastic band wakes up element by element at the exact moment its extrapolated stress
reaches the hardened surface (maps `results/phase2/*_online_spacetime.png`). The price is
in the checking: on the cyclic plate, 41 % of the elements are checked per step on
average (vs 23 % under monotonic load), because the κ bound uses the mesh-wide maximum
rate, which is high during unloading.

### 2.7 Documented failure case (`results/failure/`)

Cyclic plate with kinematic hardening: exact 9e-13; post-step 1.3e-2 (error appearing at
the first plastic step and persisting); lazy tol_α = 1e-3: 0.35; lazy without unloading
wake-up: 6.0 (the active fraction drops to 0.02: the solver "converges" on a structure
that has become almost elastic, in 82 iterations instead of 205). The spatial map of the
final error on α shows the error concentrated in the plastic band, where the D_ep
extrapolation went through the reversal.

### 2.8 Phase 2 summary

Retained configuration: exact check in every iteration, κ = 1, plastic elements active,
no neighbourhood, no margin, no catch-up (useless), final full step. Element-level
saving against Sysala: **0.46 to 0.57** (scalar), 0.47 to 0.61 (vector), at error
≤ 1e-12 on the four cases; Newton iterations identical to the reference. Wall time of
this prototype: 1 to 16 % faster.

## 3. Phase 3 — learning

Archived scripts (tag `exploration-complete`): `experiments/phase3_learned.py`,
`experiments/phase3_analysis.py`, `adaptfem/learning.py`. Results kept:
`results/phase3/results_all.jsonl`, `tables.md`, `best.txt`, `phase3_pareto.png`.

### 3.1 What we ask of learning, and why only that

Phase 2 leaves a single loss-free lever: **which elastic elements to check, and when**.
The heuristic decides through the κ bound (margin / rate × age). The learned policy
predicts a **horizon** (number of steps before yielding) from features available at
check time: margin f, its last two changes, α, ‖β‖, plastic fraction and max margin of
the neighbourhood, sign and ratio of the load increments, global rate r, heuristic
horizon −f/r, steps since the last plastic event. Deployment is identical for every
policy: an element checked at step k with horizon h is not checked again before
k + ⌊s(h−1)⌋ + 1 (s = safety factor), and stays checked within the iterations of the
current step. Labels: steps until first yielding, computed a posteriori on the reference
histories (the solver is its own teacher; no external dataset). Training on five members
of a family of notched plates (R ∈ {2.5, 3, 4, 5}, H ∈ {16, 18, 20, 24}; 801,348 samples,
10 % within 16 steps of the limit); test on two unseen members (R = 4, H = 20; R = 3.5,
H = 22), then on the cantilever and both cyclic cases (H3). Models: linear regression,
gradient boosting on log(1+h) with absolute loss (median) and with quantiles 10 %, 5 %,
2 % (conservative variants, the learning-side analogue of κ), 2–3 seeds. Bound: **oracle
horizon** (true number of steps until yielding, from the reference). Inference time is
counted in the cost.

### 3.2 Prediction quality (offline, samples within 16 steps of the limit)

| case | MAE gbm median / q10 / q05 / q02 (steps) | unsafe fraction (predicted h > true h) | heuristic MAE | heuristic unsafe |
|---|---|---|---|---|
| plate R4 H20 (family, unseen) | 1.51 / 1.41 / 1.64 / 2.08 | 0.65 / 0.17 / 0.13 / 0.09 | 6.24 | 0.00 |
| plate R3.5 H22 (family, unseen) | 1.51 / 1.48 / 1.70 / 2.12 | 0.65 / 0.20 / 0.14 / 0.10 | 6.26 | 0.00 |
| cantilever (other family) | 2.30 / 2.33 / 2.34 / 2.71 | 0.57 / 0.24 / 0.26 / 0.19 | 5.62 | 0.00 |
| cyclic plate kin. | 1.80 / 3.02 / 2.76 / 3.29 | 0.27 / 0.08 / 0.09 / 0.08 | 6.05 | 0.00 |
| cyclic cantilever kin. | 2.96 / 3.15 / 2.86 / 3.12 | 0.56 / 0.37 / 0.38 / 0.24 | 5.98 | 0.00 |

The learned model predicts the horizon 2 to 4 times more accurately than the heuristic,
and better within the family (1.4–2.1) than outside (2.3–3.5): family specialisation
(H3) is real **at the prediction level**. But it is unsafe on 6 to 38 % of the samples
near the limit, whereas the heuristic never is, by construction (it always
underestimates).

### 3.3 Online result (save scalar vs Sysala, in-iteration check, plastic elements active; mean ± std over seeds)

| case | heuristic κ=1 (exact) | oracle horizon s=1 (exact) | best learned **exact** | best learned at u < 1e-3 | learned gbm median s=0.5 |
|---|---|---|---|---|---|
| plate R4 H20 | 0.518±0.003 | 0.560±0.002 | none | q02 s=0.5: 0.483 (u 3.6e-4) | 0.520 (u 1.5e-2) |
| plate R3.5 H22 | 0.525±0.001 | 0.575±0.002 | none | q02 s=0.5: 0.490 (u 2.4e-4) | 0.557 (u 1.7e-2) |
| cantilever | 0.490±0.002 | 0.520±0.002 | none | q02 s=0.5: 0.432 (u 7.4e-4) | 0.453 (u 4.2e-3) |
| cyclic plate kin. | 0.573±0.006 | 0.693±0.003 | none | none (min 8.5e-3) | 0.651 (u 2.1e-1) |
| cyclic cantilever kin. | 0.457±0.003 | 0.530±0.003 | none | none (min 5.2e-3) | 0.454 (u 1.1e-1) |

Reading:
- **No learned policy reaches the exact regime**, whatever the model, quantile or safety
  factor (24 configurations × 5 cases). When it saves more than the heuristic, it does so
  by missing yield events (error 2e-3 to 4e-1, and 5 to 15 % more Newton iterations,
  because late wake-ups come with an accumulated increment).
- **The bound on anticipation is narrow**: the oracle, which anticipates perfectly, gains
  3 to 12 points over the heuristic, the maximum being on the cyclic plate (0.573 →
  0.693), where checking is most expensive (41 % of the elements).
- Why the learned model fails to close that gap although it predicts better: (i)
  deployment turns a prediction error into *no check for h steps*, with no intermediate
  re-check — the heuristic re-checks as soon as the margin can have been consumed at the
  observed rate, which is a physical guarantee, not a statistical one; (ii) training
  features are measured on converged states, deployment features on iterates; (iii) the
  6–38 % unsafe predictions fall precisely on the elements that yield, i.e. those whose
  omission costs the most; (iv) lowering the quantile reduces the error but also the
  gain, down to below the heuristic (q02 s=0.5: −3 to −6 points).
- **H3**: the family-specialisation advantage shows in the MAE, not in the deployed
  gain, which is capped by safety.

### 3.4 H2 (decision under a global budget, reinforcement learning) — not tested, and why

In the exact regime there is no error budget to distribute along the trajectory: the
only decision is local (check this element at this step or not), and its cost and
consequence are local and immediate. A trajectory problem only appears if some error is
accepted (post-step or lazy modes), and Phase 2 shows these modes bring at best 2 to 7
points for 1e-3 to 1e-2 error. The room a reinforcement-learning policy could exploit is
therefore capped by the heuristic → oracle gap (3 to 12 points) plus those few points; it
does not justify the cost of training a trajectory policy. We did not run this
experiment; it is an argued choice, not a result.

### 3.5 Phase 3 summary

**Learning does not beat the tuned heuristic on any configuration at equal accuracy**;
it predicts better (MAE ÷ 2 to 4, better still within its family) but does not decide
more safely; the maximum theoretical gain of perfect anticipation is 3 to 12 points of
element-level cost, concentrated on the cyclic cases.
