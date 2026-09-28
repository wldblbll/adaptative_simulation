# Adaptive allocation of computational effort in nonlinear finite elements: lazy integration on a fixed mesh, oracle bound, exact heuristic and measured value of learning

*Working draft. All numbers come from the `experiments/` scripts and the `results/`
files; scripts of abandoned variants are archived at git tag `exploration-complete`.*

## Abstract

In an incremental elastoplastic finite element computation, every Newton iteration
re-integrates the constitutive law and re-assembles the contribution of every element,
although plasticity only occupies a fraction of the domain. We study an **online,
fixed-mesh** orchestration that integrates the law on an active set of elements only,
and replaces the contribution of the others by their exact linearisation about their
last integrated state, f^e = f^e_0 + K^e_0 (u_e − u_e0). An a posteriori oracle study
bounds the gain: 94–99 % of element-steps can be extrapolated to within 1 % of σ_y by
tangent extrapolation, i.e. 85–90 % of the element-level cost. This bound does not
transfer online: extrapolating plastic elements degrades equilibrium, costs iterations
and becomes invalid at load reversal. What does transfer, and fully, is the elastic
share: a deterministic orchestrator that keeps plastic elements active and checks
elastic elements with the elastic predictor, with a skipping bound on the margin to
yield, **reproduces the reference solution to 1e-12** while saving **46 to 57 % of the
element-level cost** against a reference that already reuses the elastic tangent, with
an identical number of Newton iterations, on four cases including two cyclic ones. The
remaining cost item is the check itself. A learned policy (horizon before yielding,
gradient-boosting regression trained on the simulations of a family of geometries) is
compared with the tuned heuristic and with an oracle horizon: no learned policy reaches
the exact regime; when it saves more than the heuristic it does so by missing yield
events, and the oracle caps the gain of perfect anticipation at 3–12 points. The gain on
total time is capped by the element share of the time (27–43 % in our prototype), which
the method does not affect.

## 1. Introduction

The starting observation is simple: in an incremental nonlinear computation, the
computational effort is uniform over the mesh at every step, whereas the active physics
(plasticity, damage, contact) only occupies a fraction of the domain at any time, and
that fraction moves. The dominant approach to accelerating these computations with
learning is to learn a shortcut of the physics (surrogate models, neural operators).
Such models are less accurate than finite elements and only valid within their training
domain, which makes them ill-suited to an R&D context where new technologies have no
dataset.

We follow another hypothesis: learning does not replace physics, it allocates effort.
The physics stays exact wherever it matters, the solver remains the guarantor of the
solution, and the policy trains on the simulations themselves. The research question is
twofold: can effort be dynamically concentrated on the active regions without degrading
the solution beyond a tolerance? and does learning bring a measurable gain over a
well-tuned deterministic criterion?

Scope: the mesh is an input and is never modified. This constraint is deliberate: meshing
rules are specific to each organisation, and a method that takes the mesh as is can be
grafted onto an existing code without touching the pre-processor. A corollary not to be
distorted: if the user over-refined an inactive region, its degrees of freedom stay in
the system; the method reduces the cost of useless over-refinement, it does not remove
it.

Contributions: (i) an oracle study that quantifies the upper bound of the gain, weighted
by the measured cost, as a function of tolerance, load step and observed variable; (ii)
an implementation with neither reduced model nor surrogate (element linearisation,
deferred law integration, elastic-predictor check) that reproduces the reference
exactly; (iii) a measured explanation of why the oracle bound does not transfer; (iv) a
comparison at equal information between a tuned heuristic, a learned policy and an
oracle, on cyclic cases designed to test the anticipation of wake-up.

## 2. Related work and positioning

*Caveat: the full texts of the references could not be accessed from the working
environment; only abstracts were used (`docs/literature.md`). Statements about the
switching criteria of the closest relatives must be checked before submission.*

**Learned adaptive refinement.** Yang et al. (AISTATS 2023) cast AMR as a Markov decision
process trained directly from the simulation; Freymuth et al. (NeurIPS 2023) and Foucart
et al. (JCP 2023) follow. We keep the argument (deciding on an instantaneous estimate
does not optimise the trajectory) and the constraint (no dataset), but we do not touch
the mesh.

**Hybrid solvers through domain decomposition.** Wang, Hakimzadeh, Ruan and Goswami
(CMAME 2025) delegate the nonlinear subdomains to a DeepONet and the rest to FE through a
Schwarz method, with an ML region that can grow; a non-overlapping framework
(arXiv:2606.08796) notes that "the current subdomain assignment is prescribed a priori".
Two structural differences: the direction of delegation (we keep the exact solver on the
hard region and lighten the easy one) and the dynamics (shrinking and wake-up, not only
growth).

**Multi-time-step, subcycling, asynchronous integrators.** Belytschko & Mullen (1977),
Smolinski (1993, 1997), Lew, Marsden, Ortiz & West (2003). Closest numerical parent: one
time step per element. But in explicit dynamics, where the constraint is stability; in
implicit quasi-statics the question becomes "must the element be re-integrated and
re-assembled?".

**Modified Newton and tangent reuse.** Čermák, Sysala & Valdman (2019) pre-assemble the
elastic operator and update the tangent only at plastic points; Yusa et al. (2018)
exploit the concentration of the nonlinearity in a preconditioner. Methodological
consequence: **our reference is not the naive solver but this "Sysala" solver**; every
gain is measured against it.

**Close relatives found while checking the niche.** Radermacher & Reese (Comput. Mech.
2014): selective POD on approximately elastic subdomains, full finite elements elsewhere,
*adaptive* sub-structuring. Kerfriden, Goury, Rabczuk & Bordas (CMAME 2013): partitioned
ROM concentrating the effort around the damage zone without a priori knowledge. Gendre,
Allix, Gosselet (2009): non-intrusive global/local for local plasticity, region fixed a
priori. Ryckelynck (2005, 2009): hyper-reduction with a reduced integration domain.
Palchoudhary, Kerfriden et al. (2024): local Neuber-type plastic correction accelerated by
learning.

**Honest positioning.** The idea of adaptively partitioning the domain into plastic and
quasi-elastic regions on a fixed mesh is not new. What we found nowhere: a measurement of
the upper bound of the gain; an implementation without reduced basis, whose guarantee is
the solver's (nothing to train); and the question of what a learned criterion adds
against a tuned heuristic, with wake-up as the discriminating case.

## 3. Potential study (oracle)

### 3.1 Method

Reference solver: plane-strain Q4 with B-bar, J2 plasticity, linear + Voce isotropic
hardening (and linear kinematic hardening for the cyclic "kin" cases), radial return,
consistent tangent, Newton-Raphson with a tangent predictor, displacement control.
Validation: consistent tangent against finite differences (< 1e-5), vectorised radial
return identical to a scalar version, patch test, simple shear against the closed-form
solution (1e-10), observed quadratic convergence (3e-1, 6e-2, 4e-4, 8e-8, 6e-14). Cases:
notched plate (3096 elements), plate overloaded until the ligament yields, cantilever
(2000 elements), and cyclic versions (0 → 1 → −0.6 → 1). Steel: E = 200 GPa, ν = 0.3,
σ_y0 = 250 MPa.

For every element and every step, a greedy policy on the true trajectory decides whether
the element could have been extrapolated from its last integrated step k_0 at a given
tolerance on stress (norm of the in-plane components, normalised by σ_y0, max over the
Gauss points). Three extrapolators: freeze (σ_0), linear in time, **tangent** (σ_0 + D_0
Δε, D_0 the stored consistent tangent). The error of the deferred integration at wake-up
(law integrated from the anchor over the total increment) is also measured. Cost is
weighted by micro-benchmarks on this code, under two models: `vector` (vectorised NumPy)
and `scalar` (the same algorithms point by point, whose plastic/elastic ratio, 2.9–3.0,
is a proxy for a compiled code).

### 3.2 Results

Freezing is unusable (4–48 % of element-steps at 1 %) because the stress of elastic
elements evolves with the load; linear extrapolation in time gives 78–88 %; **tangent
extrapolation gives 94–99 %**. That number has two components. Elastic elements (72 to
97 % of the elements depending on case and time) are extrapolated exactly. Plastic
elements can be extrapolated for 64 to 86 % of their steps at 1 %, but that fraction
**depends on the load step**: on the plate, 39 % at 10 steps, 62 % at 20, 83 % at 60,
90 % at 120. Part of the plastic extrapolability is therefore "the steps are small for
these elements". Deferred integration is practically exact (error on α at wake-up
≤ 1.4e-6 at the 99th percentile at 1 %): radial return is exact on a proportional path.

Weighted by the final cost (quiet element = f_0 + K_0 Δu, 2.6 µs vs 55 µs for an active
elastic element and 160 µs for a plastic one, scalar model), the oracle bound at 1 % is
**85 to 90 %** of the element-level cost against the Sysala reference. A more useful
bound is that of the **ideal exact policy** (integrate an element if and only if it is
plastic, perfect knowledge, free check): 56 to 74 %. The gap between the two, 13 to 34
points, is the "extrapolation of plastic elements" share (figures
`results/phase0/*_potential.png`, `*_spacetime.png`).

Caveat: this is an upper bound. The oracle looks at the true trajectory; online,
extrapolation changes the equilibrium and the error propagates. Second caveat: the share
of time spent at element level is 27–43 % in this 2D prototype (LU factorisation
dominates); the gain on total time is that share multiplied by the element-level gain.
For expensive laws (crystal plasticity, damage with many variables) the element share
dominates; for large 3D models with a direct solver it is often a minority.

## 4. Method

### 4.1 Element extrapolation and deferred integration

At every Newton iteration, an *active* element is integrated normally (strain, law,
consistent tangent, K^e, f^e). A *quiet* element returns f^e = f^e_0 + K^e_0 (u_e −
u_e0), where index 0 denotes its anchor (last integrated state), and K^e = K^e_0. This is
exactly ∫ B^T (σ_0 + D_0 B Δu) — the tangent extrapolation of Phase 0 — but computed in 64
multiplications instead of a loop over Gauss points (≈230 flops): quiet element 2.6 µs vs
25 µs through the Gauss path. Internal variables stay at the anchor. At wake-up, the law
is integrated **once over the total increment** from the anchor: the state is the output
of a radial return, hence admissible; the path is approximated by a segment, which is
only legitimate if there is no reversal in the interval (§ 5.3). For an element anchored
elastic that stayed elastic, the extrapolation is exact.

### 4.2 Checking elastic elements

An element anchored elastic is checked with the elastic predictor of the radial return:
σ̂ = σ_0 + D_0 Δε at the Gauss points, then f(σ̂, α, β) = ‖dev σ̂ − β‖ − √(2/3) σ_y(α).
Since D_0 = C, σ̂ is the exact elastic stress for the current strain: **if f ≤ 0, the
element is exactly elastic and its extrapolated contribution is exact; if f > 0, it is
woken up and integrated within the current iteration.** No error is therefore possible
on a checked element. The check costs 8 µs per Gauss point (scalar), 32 µs per element,
more than half an active elastic element: checking everything at every iteration cancels
the gain (§ 5.1).

**Skipping bound.** After a check, the margin m = −max_g f is stored. Let r be the
mesh-wide maximum of the (normalised) stress change over the previous step. The element
is not checked while m > κ r (age + 1), age being the number of steps since its last
check. κ is a safety factor (κ = 1: the margin may be consumed at the maximum observed
rate). It is a composite criterion (margin to yield × rate of change), deterministic,
with no tuning other than κ.

### 4.3 Policies tested

Check timing: every iteration; predictor only (iteration 0); predictor + verification at
convergence; convergence only (with re-convergence on wake-up); post-step (wake-up at the
next step, non-intrusive). Plastic elements: always active, or lazy integration with a
tolerance on the linearised plastic strain since the anchor
(Δα̂ = √(2/3) · 2μ n:Δε / (2μ + (2/3)(H_k + H'))) and wake-up on unloading (n:Δε < 0),
predicted at the start of the step from the previous increment. Full catch-up every N
steps, maximum number of skipped steps, neighbour propagation, precautionary margin,
global wake-up on a sign change of Δλ. Mandatory final full step.

### 4.4 Protocol

Every comparison uses the same mesh, load steps and integrator; the reference is the same
code with the orchestrator disabled, in Sysala mode. Costs are reported as counted
operations (Gauss-point integrations, plastic points, K^e computed, extrapolated
elements, checked points, decisions, factorisations) weighted by measured unit costs,
**and** as wall time. The orchestrator's overhead (checks, decisions, inference) is always
included. Errors: relative displacement (max over the history), RMS and energy stress
errors, accumulated plastic strain at the final step, reaction force. Fixed seeds,
configurations serialised in `results/*/results*.jsonl`.

## 5. Results of the heuristic orchestrator

Full tables: `docs/experiments.md`, `results/phase2/tables.md`; overview:
`docs/figures/exploration_map.png`.

### 5.1 Exact configuration

Check at every iteration, κ = 1, plastic elements active:

| case | mean active fraction | checked / step (Gauss points) | element-level saving scalar (vector) | wall time saved | u_max | Newton it. (ref.) |
|---|---|---|---|---|---|---|
| notched plate | 0.108 | 4101 (of 12384) | 0.517 (0.514) | 14 % | 9e-14 | 102 (102) |
| cantilever | 0.160 | 2291 (of 8000) | 0.486 (0.471) | 14 % | 5e-13 | 97 (97) |
| cyclic plate kin. | 0.055 | 6980 | 0.569 (0.606) | 16 % | 9e-13 | 205 (205) |
| cyclic cantilever kin. | 0.137 | 4306 | 0.456 (0.467) | 1 % | 4e-12 | 212 (212) |

The number of Newton iterations is identical to the reference: in-iteration wake-ups are
absorbed. κ = 0.5 misses yield events (errors 4e-6 to 5e-4); κ = 2 costs 6 to 15 points
for nothing; checking everything brings the gain down to 5–7 %. The maps
`results/phase2/*_online_spacetime.png` show the integrated set coinciding with the
plastic zone, and the checked set (23 % of elements under monotonic load, 41 % cyclic)
around it. **Why this configuration wins**: for an elastic element, the check criterion
and the guarantee are the same object (the elastic predictor), and check skipping is
bounded by a physical quantity (margin / rate); there is no approximation to catch up.

### 5.2 Non-exact variants

Checking at the predictor only gains 2 to 5 points for 1e-5 error. Post-step mode
(non-intrusive) gains 2 to 7 points under monotonic load for 5e-4, but 1e-2 on cyclic
cases. Checking at convergence only is dominated (+80 % iterations). Neighbour
propagation costs 10 to 27 points with no gain; the precautionary margin costs with no
gain; global wake-up on reversal adds nothing to the exact local check.

### 5.3 Lazy integration of plastic elements fails online

With tol_α from 3e-5 to 3e-3, the displacement error ranges from 2e-5 to 5e-2 under
monotonic load and from 9e-4 to 8e-1 on cyclic cases, Newton iterations grow by 10 to
60 %, and the gain is never better than the exact configuration (except by destroying the
solution). Three measured mechanisms (`results/failure/`): (i) the extrapolation error of
a plastic element changes the equilibrium and spreads through the region that drives the
response; (ii) wake-up produces a stress jump that costs iterations; (iii) at load
reversal, deferred integration over a non-proportional increment is invalid
(path-dependent plasticity) — without unloading wake-up, the error is 600 %, and
"convergence" happens on a structure that has become almost elastic. The "plastic
elements" share of the oracle bound (13 to 34 points) **does not transfer**. The online
active region is the plastic zone, no more and no less; what is at stake is the cost of
checking elastic elements.

### 5.4 What remains: the check

Between the ideal exact policy (free check: 56–74 %) and the exact heuristic (46–57 %),
the 7 to 17 point gap is the cost of checking under the κ bound. It is the only place
where a better decision can still gain without losing anything: deciding *which*
elements to check and *when*. That is the question put to learning.

## 6. What learning adds

### 6.1 Protocol

The only loss-free lever left by Phase 2 is the choice of which elastic elements to check
and when. The heuristic decides through the κ bound; the learned policy predicts a
horizon h (steps before yielding) from twelve features available at check time (margin,
its changes, hardening, neighbourhood, load direction, global rate, heuristic horizon,
steps since the last plastic event). Deployment is identical for all: no re-check before
k + ⌊s(h−1)⌋ + 1. Labels are computed a posteriori on the reference histories: the solver
is its own teacher, with no external dataset. Training on five members of a family of
notched plates (801,348 samples), test on two unseen members then on the cantilever and
cyclic cases (H3). Models: linear, gradient boosting (median, quantiles 10, 5, 2 %), 2–3
seeds. Bound: oracle horizon. Inference time counted in the cost.

### 6.2 Results

Offline, the learned model predicts the horizon 2 to 4 times better than the heuristic
(MAE 1.4–2.1 steps within the family, 2.3–3.5 outside, vs 5.6–6.3), but it is unsafe
(predicted h > true h) on 6 to 38 % of the samples near the limit, whereas the heuristic
never is. Online (`docs/figures/learning_vs_heuristic.png`):

| case | heuristic κ=1 (exact) | oracle horizon (exact) | best learned exact | best learned at u < 1e-3 |
|---|---|---|---|---|
| plate R4 H20 (family) | 0.518 | 0.560 | none | 0.483 (u 3.6e-4) |
| plate R3.5 H22 (family) | 0.525 | 0.575 | none | 0.490 (u 2.4e-4) |
| cantilever | 0.490 | 0.520 | none | 0.432 (u 7.4e-4) |
| cyclic plate kin. | 0.573 | 0.693 | none | none |
| cyclic cantilever kin. | 0.457 | 0.530 | none | none |

**H1 (anticipate instead of observe).** Perfect anticipation is worth 3 to 12 points of
element-level cost, the maximum on the cyclic case where checking is most expensive. No
learned policy captures them: over 24 configurations × 5 cases, when it saves more than
the heuristic it does so by missing yield events (error 2e-3 to 4e-1 and +5 to +15 %
iterations, late wake-ups arriving with an accumulated increment). The mechanism is
structural: a prediction error becomes an absence of check for h steps, whereas the κ
bound re-checks as soon as the margin can have been consumed at the observed rate — a
physical guarantee, not a statistical one. Lowering the quantile reduces the error but
also the gain, down to below the heuristic.

**H3 (family specialisation).** Real on prediction quality (lower MAE within the family),
with no effect on the deployed gain, which is capped by safety.

**H2 (global budget, reinforcement learning).** Not tested, by argued choice: in the
exact regime there is no error budget to spread along the trajectory; the modes that
accept an error bring 2 to 7 points (§ 5.2); the room for a trajectory policy is capped by
the heuristic → oracle gap. It does not justify the training.

### 6.3 Answer to the question

On this problem, learning brings no measurable gain over the tuned heuristic. The reason
is not that the model predicts poorly; it is that the useful decision is a *safety*
decision, and the physical criterion (elastic predictor + margin/rate) is already safe
and nearly optimal: the gap to perfect is 3 to 12 points. Where learning keeps a role:
reducing the cost of checking in cases where it is massive (cyclic: 41 % of the
elements), provided a safety guarantee that regression alone does not provide — for
instance a learned policy that only *delays* the check below the κ bound, never beyond.

## 7. Limitations

- Oracle bound and real gain are two different things: the bound includes an
  extrapolation of plastic elements that does not survive the equilibrium feedback loop.
- The gain is on element-level cost; the share of that cost in total time (27–43 % here)
  caps the wall-clock gain. A modified Newton to save factorisations was tried and
  diverges.
- 2D, plane strain, J2, simple laws: the plastic/elastic ratio (3) and the cost of the
  check (0.6 elastic element) are those of this law; a more expensive law increases the
  gain, a simpler one reduces it.
- Small strains only. The exactness argument relies on elastic elements responding
  linearly; with geometric nonlinearity (large displacements) or laws without an elastic
  domain (hyperelasticity, viscoelasticity) the cached linearisation is no longer exact
  and the method needs a different guarantee.
- The load step is a confounding factor for Phase 0, not for Phase 2 (plastic elements are
  active there).
- Proportional loading; the exact check does not depend on this assumption, but the
  deferred integration of an element that stayed elastic and is then woken does, over a
  segment: exact as long as the element was elastic.
- Cost model: unit costs are measured in Python (two models); ratios, not absolute
  values, are transferable, and absolute savings move by several points between machines
  and campaigns.
- Full texts of the closest related work not read.

## 8. Integration into an existing code

See `docs/industrial_integration.md`. In short: the layer needs (A) an element loop
filterable per iteration, (B) a per-element cache (f^e_0, K^e_0, u_e0), (C) an elastic
predictor callable without correction, (D) an entry point inside the Newton iteration to
grow the active set. (A)–(C) amount to overriding the integration routine; (D) is the
intrusive point, avoidable with the "convergence" mode (exact, +80 % iterations) or the
"post-step" mode (non-intrusive, error 1e-3 to 1e-2). Guarantees: the default
configuration reproduces the reference to 1e-12; a learned policy would only decide when
to check. Favourable families: expensive laws, localised and stable nonlinearity, cycles
with long elastic phases, repeated families of parts. Unfavourable: diffuse plasticity,
large models where the linear solver dominates, structures uniformly close to yield
(everything must be checked).

## 9. Conclusion and outlook

To the first question the answer is yes, with a precision: integration and assembly
effort can be concentrated on the plastic zone **with no degradation at all** (1e-12),
saving 46 to 57 % of the element-level cost against a reference that already reuses the
elastic tangent, with an identical number of Newton iterations, including under cyclic
loading — but the online active region is the plastic zone, not less: extrapolating
plastic elements, which the oracle credited with 13 to 34 extra points, survives neither
equilibrium feedback nor load reversal. To the second, the answer is no: on this problem,
a learned policy does not beat the tuned heuristic at equal accuracy, because the useful
decision is a safety decision that the physical criterion already provides, and because
the gain of perfect anticipation is capped at 3–12 points.

The gain on total time remains capped by the element share of the time. Outlook, in
order of expected value: (1) expensive constitutive laws (crystal plasticity, damage),
where that share dominates and the active/quiet cost ratio is well above 3; (2) 3D and
elements with more degrees of freedom, where the K^e_0 cache costs memory but the quiet
path remains a matrix-vector product; (3) extend the gain to the solver phase: keeping
the factorisation when the active set is small requires a modified Newton which, in its
simple form, diverges; a low-rank update of the active blocks is the natural route; (4) a
learned policy **under guarantee** (delay the check below the κ bound, never beyond) for
cyclic cases where checking costs the most; (5) check the positioning against Radermacher
& Reese (2014) and Kerfriden et al. (2013) on the full texts.
