# Integration into an existing FE code

What would it take to graft the orchestrator onto an existing industrial FE code? This
note builds on the prototype (`adaptfem/`), which was designed so that the orchestrator
is a **layer outside the Newton loop**, with no change to the element kernel or to the
constitutive law.

## 1. What the orchestrator does, and what it does not do

At every Newton iteration, the solver asks each element for its internal force f^e(u)
and, at iterations where the tangent is re-assembled, for its block K^e(u). The
orchestrator intercepts that request and, for an element classified as *quiet*, returns

    f^e = f^e_0 + K^e_0 (u_e − u_e,0),     K^e = K^e_0

where the index 0 denotes the last state in which the element was actually integrated
(its "anchor"). No Gauss point is visited, no internal variable is touched. When the
element is woken up, the law is integrated **once over the total increment** since the
anchor (deferred integration): the resulting state is the output of a radial return,
hence admissible; the strain path between anchor and wake-up is approximated by a
straight segment. In the retained (exact) configuration only elastic elements are ever
quiet, and for them this segment is exact.

It does nothing else: no change to the mesh, the load step, the Newton tolerance or the
linear solver. The mesh remains the user's; the degrees of freedom of the quiet regions
stay in the linear system. *The method reduces the cost of useless over-refinement, it
does not remove it.*

## 2. Hooks needed in the solver

| need | minimal form | available in common codes? |
|---|---|---|
| A. Filterable element loop | iterate over a subset of elements for law integration and K^e, f^e computation | Yes in most codes (loops over groups, deactivated elements); a per-iteration filter is rarely exposed |
| B. Per-element anchor storage | f^e_0 (n_dof,e), K^e_0 (dense), u_e,0 | K^e is often recomputed rather than stored: the most intrusive point memory-wise (64 reals per Q4, ~600 per HEX8 with 24 dofs, ~4700 for a HEX20) |
| C. Read access to the anchor's internal variables | stress, strain, stored consistent tangent, hardening variables | Codes store the state; the per-Gauss-point consistent tangent is rarely stored (it is recomputed). Without it, the tangent extrapolation must be replaced by K^e_0 alone, which is enough for the equilibrium equation; the per-Gauss-point tangent only serves the **check** (computing the extrapolated stress for the yield test) |
| D. An exposed "elastic predictor" | trial stress σ̂ = σ_0 + D_0 Δε and yield function f(σ̂) without return mapping | This is exactly the first half of any radial-return algorithm; it must be callable on its own |
| E. An entry point inside the Newton loop | after the residual computation, before the convergence decision: ability to add elements to the active set and recompute their contribution | Rarely exposed; this is the algorithmic intrusiveness point |
| F. Cost counters | number of law calls, assembled elements, factorisations | Needed to verify the gain; often missing |

Deferred integration (wake-up) assumes the law tolerates a "long" increment without
sub-stepping; industrial laws with automatic sub-stepping handle it.

## 3. Level of intrusiveness

- **External, non-intrusive layer**: impossible with the usual APIs, because the
  interception happens *inside* a Newton iteration (point E). A user-element approach
  (UEL/UMAT) allows a degraded form: a UMAT can decide to return the extrapolated tangent
  and stress without radial return (points C, D) — but it is still called at every Gauss
  point, so the cost of the element loop and of B^T σ is still paid. The gain is then
  limited to the radial return and the tangent, i.e. the "law" share of the cost. For
  heavy laws that is most of it; for J2 it is marginal (see the cost measurements in
  `docs/experiments.md`).
- **Override of the integration routine + loop filter**: the realistic level. Change the
  assembly loop to skip quiet elements and take their contribution from the cache
  (points A, B), and expose the yield test (D).
- **Kernel modification**: needed for wake-up inside the iteration (E). Less intrusive
  alternative: wake-up *after convergence* of the step, then re-iterate the step with the
  enlarged active set ("converged" mode of the prototype) — also exact, at the price of
  extra iterations (measured: +40 to +80 % Newton iterations on our cases). The
  "post-step" mode (wake-up at the next step) is non-intrusive but introduces an error
  (one step solved with an elastic response on an element that yields); it is acceptable
  only with a catch-up.

## 4. Minimal interface to expose

```
orchestrator.begin_step(k, load_increment)          -> active element set
orchestrator.monitor(iterate)                         -> elements to wake (uses trial stress only)
element.integrate(el_subset, u)                       -> stress, tangent, K^e, f^e (existing kernel)
element.linearised_force(el_subset, u)                -> f^e_0 + K^e_0 (u_e - u_e0)  (cache)
element.trial_yield(el_subset, u)                     -> f(sigma_hat) per Gauss point   (no return mapping)
solver.newton_hook_after_residual(callback)           -> allows growing the active set mid-iteration
cost.counters                                          -> law calls, K^e computed, monitored points, factorisations
```

## 5. Robustness guarantees a software vendor would require

1. **Exact reproduction of the reference** in the default configuration. The prototype
   does it: with the exact yield check at every iteration and plastic elements kept
   active, the solution matches the reference to 1e-13 on every case
   (`results/phase2/tables.md`). This is the configuration to ship; the modes that
   introduce an error (lagged check, lazy integration of plastic elements) must stay
   experimental.
2. **No dependence on learned data for correctness**: a learned policy would only decide
   *when to check* an element; a missed check turns into a detection delayed by one step,
   never into an inadmissible state.
3. **Mandatory final full step**, and a full catch-up on demand (every N steps, or on a
   drift criterion) for any non-exact configuration.
4. **Controlled degradation**: if the active set exceeds a threshold (for example 60 % of
   the elements), the orchestrator switches itself off: the monitoring overhead is no
   longer justified.
5. **Traceability**: per-step log of the active fraction, wake-ups and monitoring cost;
   counters comparable to those of the reference run.

## 6. Realistic path

An opt-in option ("lazy element integration"), off by default, with two levels: *exact*
(the option's default) and *experimental* (tolerance parameters exposed). User tuning
limited to the safety factor of the check (κ); everything else parameter-free. A
mandatory first step in any code: measure the share of time spent at the element level,
which caps the gain (see § 7).

## 7. Which families of computations benefit, and which do not

**Favourable**: constitutive laws that are expensive per Gauss point (crystal plasticity,
viscoplasticity with a local Newton, damage with many variables, user laws), localised
and stable nonlinear regions (notches, fillets, bolted joints, local contact), cyclic
loading with long elastic phases (fatigue), repeated computations on a family of parts.
There, the element share dominates the total time and the active fraction is small.

**Unfavourable**: diffuse plasticity (forming, crash), where the active set tends to the
whole domain; large 3D models with a direct solver, where factorisation dominates (the
element-level gain is capped by the element share, measured at 27–43 % in our 2D
prototype); elastic or piecewise-linear elastic laws, where integration is already
trivial; explicit dynamics (no Newton loop, cost driven by the critical time step — the
territory of classic subcycling). Also out of scope as is: laws with no elastic domain
(hyperelasticity, viscoelasticity) and large-deformation analyses, where an "elastic"
element is no longer linear and the cached linearisation stops being exact. The exact
check costs one evaluation of the yield function on the monitored elements: if almost
every element is close to the limit (a structure uniformly loaded near yield), the check
costs as much as the computation and the gain disappears.
