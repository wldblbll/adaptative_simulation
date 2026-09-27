# Verdict

One page, no detours. Numbers from `results/` (details in `docs/experiments.md`).

## Is it a good idea?

**Yes, in a narrower form than originally planned.** Concentrating law integration and
assembly on the plastic zone — replacing every other element by its exact linearisation
f^e_0 + K^e_0 Δu and guarding it with the elastic predictor — works **with no
degradation at all** of the solution (1e-12) and no extra Newton iteration. The idea as
first written — also extrapolating "calm but plastic" regions — does not work online: the
oracle credited it, equilibrium feedback and load reversal destroy it.

## How much, on which problems, at what error?

- **46 to 57 % of the element-level cost** (scalar cost model; 47 to 61 % vectorised)
  against a reference that already reuses the elastic tangent, at error ≤ 1e-12, on a
  notched plate, a cantilever, and their cyclic versions with kinematic hardening. +2 to
  +7 more points for an error of 1e-5 to 1e-3 (check at the predictor only, or after the
  step).
- On **total time** in this 2D Python prototype: 1 to 16 % faster, because the element
  share of the time is only 27 to 43 % (factorisation dominates). The wall-clock gain is
  the element gain times that share: real for expensive constitutive laws (the J2
  plastic/elastic ratio is only 3), small for large models dominated by the linear
  solver.
- Oracle upper bound (Phase 0): 85–90 % of element-level cost; bound for an ideal exact
  policy (free checks): 56–74 %. The gap between the two (13–34 points) is the plastic
  share, not transferable; the gap between ideal and achieved (7–17 points) is the cost
  of checking.

## Limits, counter-productive cases?

- Diffuse plasticity: the active set tends to the whole domain, checking adds to the
  cost. Structures uniformly close to yield: everything must be checked, the gain drops
  to 5–7 %.
- Every non-exact mode (lazy integration of plastic elements) is counter-productive:
  errors of 1e-3 to 6.0, +10 to +60 % Newton iterations, and under cyclic loading the
  solution is destroyed (non-proportional path inside the deferred integration).
- Neighbour propagation and precautionary margins cost 10 to 27 points and bring nothing.
  Modified Newton to save factorisations diverges.
- Scope: 2D, J2, small strains, proportional loading, Python prototype; cost ratios
  measured under two models (absolute savings shift by several points from one machine
  to another, the ranking does not); full texts of the closest related work not read.

## Does AI add anything, where, how much?

**No, not in this setting.** A learned policy (gradient boosting, 800k samples drawn from
the simulations of one family of parts, no external dataset) predicts the horizon before
yielding 2 to 4 times better than the heuristic, and better within its family than
outside (H3 holds at the prediction level). But none of its 24 configurations reaches the
exact regime on any of the 5 cases: when it saves more than the tuned heuristic, it does
so by missing yield events (error 2e-3 to 4e-1). The bound on what perfect anticipation
could bring (H1) is **3 to 12 points** of element-level cost (oracle 52–69 % vs heuristic
46–57 %), concentrated on the cyclic cases where checking costs the most. H2
(reinforcement learning) was not tested, by argued choice: there is no error budget to
distribute in the exact regime. The underlying reason: the useful decision is a *safety*
decision, and the physical criterion (elastic predictor + margin/rate) is safe by
construction and nearly optimal.

## What should come next?

1. Port the method to an expensive constitutive law (crystal plasticity, damage) and to
   3D: that is where the element-level gain weighs on total time.
2. Attack the solver phase: low-rank update of the factorisation on the active blocks
   only, rather than a modified Newton.
3. If learning at all: a policy that only *delays* the check below the κ bound (never
   beyond), evaluated on cyclic cases, the only place where 3–12 points remain.
4. Read Radermacher & Reese (2014) and Kerfriden et al. (2013) in full before any
   submission, and reposition if needed.
