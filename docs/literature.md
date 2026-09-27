# Literature review and positioning

*Methodological note.* This review was put together from an environment with
restricted network access: publisher sites (Elsevier, Springer, arXiv, PubMed,
ResearchGate) could not be read directly. What follows comes from abstracts and excerpts
reachable through search results, not from reading the full texts. Statements about the
detailed content of the papers (switching criteria, reported gains) must therefore be
**re-checked against the full texts** before any submission. No reference below is
invented: each one matches a title and a source found online.

## 1. The four research streams identified at the start of the project

### 1.1 Learned adaptive mesh refinement (out of scope, but a methodological frame)

- Yang, Dzanic, Petersen, Kudo, Mittal, Tomov, Camier, Zhao, Zha, Kolev, Anderson, Faissol,
  *Reinforcement Learning for Adaptive Mesh Refinement*, AISTATS 2023 (arXiv:2103.01342).
  AMR cast as a Markov decision process, policy trained directly from the simulation,
  model size independent of the mesh size.
- Freymuth et al., *Swarm Reinforcement Learning for Adaptive Mesh Refinement*, NeurIPS 2023
  (arXiv:2304.00818). Each element is an agent; spatial reward; message passing.
- Foucart, Charous, Lermusiaux, *Deep reinforcement learning for adaptive mesh refinement*,
  J. Comput. Phys. 2023 (arXiv:2209.12351). AMR as a local POMDP.
- Multi-Agent RL for AMR (arXiv:2211.00801); *Multi-Objective AMR using RL* (LLNL report, OSTI 1989992).

What we take from them: the argument "deciding on an instantaneous estimate does not
optimise the whole trajectory" (our hypothesis H2), and the constraint "train from the
simulation, no dataset". What we do not do: touch the mesh.

### 1.2 Hybrid FE / neural-operator solvers through domain decomposition

- Wang, Hakimzadeh, Ruan, Goswami, *Time-marching neural operator–FE coupling*, CMAME 2025
  (arXiv:2504.11383). A pre-trained DeepONet solves the "expensive" subdomains (stress
  concentrations), FE the rest, alternating Schwarz coupling. **Important for us**: the
  abstract mentions "an adaptive subdomain evolution strategy enables the ML-resolved
  region to expand dynamically". The decomposition is therefore not entirely fixed: the
  ML region can **grow** during the computation.
- *A Non-Overlapping Schwarz Hybrid FE–Neural Operator Framework for Solid Mechanics on
  Irregular Domains* (arXiv:2606.08796, June 2026). From its outlook: "The current
  subdomain assignment is prescribed a priori; an adaptive decomposition strategy, driven by
  local error indicators or nonlinearity measures, would eliminate this assumption and make
  the framework applicable to problems where critical regions are not known in advance."
- *Hybrid coupling with operator inference and the overlapping Schwarz alternating method*
  (Sandia, arXiv:2511.20687).

Two structural differences with our approach:
1. **The delegation goes the other way.** These works put the learned surrogate on the
   *hard* (nonlinear) region and keep FE on the easy one. We do the opposite: the exact
   solver stays on the active region, and the *quiet* region is the one treated cheaply.
   Our hypothesis is that this is the only way to keep a guarantee on the solution without
   a prior dataset.
2. **Dynamics of the partition.** In Wang et al. the ML region only grows. Our question
   includes shrinking (a region calms down) and wake-up (unloading/reloading), which is
   exactly the cyclic case.

### 1.3 Multi-time-step methods, subcycling, multirate (closest numerical relative)

- Belytschko & Mullen (1977) and follow-ups; Smolinski, *Explicit multi-time step
  integration for first and second order FE semidiscretizations*, CMAME 1993; *Stability
  of explicit subcycling with linear interpolation*, CMAME 1997.
- Asynchronous variational integrators (AVI): each element has its own time step,
  element-by-element constitutive update (Lew, Marsden, Ortiz, West 2003; extension to
  elastoplastic contact, Acta Mech. Solida Sinica 2023).
- *Multi-temporal decomposition for elastoplastic ratcheting solids* (arXiv:2308.11821).

Difference: these methods are designed for explicit dynamics, where the constraint is
stability (local critical time step). In implicit quasi-statics there is no critical time
step; the question becomes "must the law be re-integrated and the element re-assembled?".
AVI is the closest conceptual parent (a time step *per element*), but in an explicit
setting.

### 1.4 Modified Newton, quasi-Newton, reuse of tangent and assembly

- Čermák, Sysala, Valdman, *Efficient and flexible MATLAB implementation of 2D and 3D
  elastoplastic problems*, Appl. Math. Comput. 2019 (arXiv:1805.04155). **Key point**: the
  tangent matrix is split into three sparse matrices (elastic operator,
  strain-displacement, stress-strain derivative); the first two are assembled once and
  for all; tangent assembly time is proportional to the **number of plastic integration
  points**.
- Yusa, Okada, Yamada, Yoshimura, *Scalable parallel elastic–plastic FE analysis using a
  quasi-Newton method with a balancing domain decomposition preconditioner* (2018):
  exploits the local concentration of the nonlinearity.

Consequence for our baseline: **re-assembling only the plastic blocks is already the
state of the art.** Our reference solver must therefore be compared in two forms:
(a) naive (everything re-integrated and re-assembled at every iteration) and (b) "Sysala"
(pre-assembled elastic tangent, updated only at plastic points). Any gain we claim must
be measured **against (b)**, otherwise it is artificial. What (b) does not remove: the
strain computation, the elastic predictor, the yield check at every Gauss point, and the
assembly of the residual over all elements, at every Newton iteration. That is where the
additional saving of the extrapolation lies.

## 2. Close relatives found while checking the niche

The brief asked for an explicit search on selective element activation, freezing of
elastic regions, selective assembly and lazy integration. Results:

- **Radermacher & Reese, *Model reduction in elastoplasticity: proper orthogonal
  decomposition combined with adaptive sub-structuring*, Comput. Mech. 54 (2014) 677–687.**
  The closest relative found. "Selective" POD applied only to the approximately elastic
  subdomains, full finite elements in the plastic subdomains, **adaptive**
  sub-structuring. The title itself contains "adaptive sub-structuring". The switching
  criterion and its dynamics (wake-up, shrinking) could not be checked on the full text.
  **Must be read.**
- **Kerfriden, Goury, Rabczuk, Bordas, *A partitioned model order reduction approach to
  rationalise computational expenses in nonlinear fracture mechanics*, CMAME 256 (2013)
  169–188.** Domain decomposition + projection-based ROM; effort is concentrated around the
  damage zone; "no a priori knowledge of the damage pattern is required".
- Gendre, Allix, Gosselet, Comte, *Non-intrusive and exact global/local techniques for
  structural problems with local plasticity*, Comput. Mech. 2009; Gosselet et al. 2018
  (global/local seen as Schwarz). The local plastic region is handled by a nonlinear
  sub-model iteratively coupled to a linear global model. The local region is **defined a
  priori**.
- Ryckelynck, *A priori hyperreduction method: an adaptive approach* (2005); *Hyper-reduction
  of mechanical models involving internal variables*, IJNME 2009. The "reduced integration
  domain" (RID): the constitutive law is integrated on a few elements only, the rest is
  extrapolated through the reduced basis. A form of **selective integration of the law**,
  but the extrapolation goes through a global reduced basis, not a local state.
- *Model order reduction of nonlinear THM systems by means of elastic and plastic domain
  sub-structuring*, Finite Elem. Anal. Des. 2024: "the computationally expensive nonlinear
  iterative procedure is confined in the zone where plasticity is assumed to be
  restricted" — the plastic zone is assumed known.
- Palchoudhary, Peter, Maurel, Ovalle, Kerfriden, *A plastic correction algorithm for
  full-field elasto-plastic FE simulations*, Comput. Mech. 2024 (arXiv:2402.06313): local
  Neuber-type plastic correction around notches, accelerated by a meta-model and a
  learned correction layer. Approximate, not guaranteed by the solver.
- ML approaches "inside Newton" without changing the spatial allocation:
  *Neural-Initialized Newton* (arXiv:2511.06802), *ML-accelerated time integration of
  plasticity models* (arXiv:2606.14548), COMMET (arXiv:2510.00884), RL for time-step
  selection (Mach. Learn. Comput. Sci. Eng. 2025).

## 3. Verdict on the niche

**The niche as stated in the original brief ("online, moving allocation on a fixed mesh")
is not empty.** Radermacher & Reese (2014) and Kerfriden et al. (2013) already perform an
adaptive elastic/plastic partition on a fixed mesh, and Wang et al. (2025) dynamically
grow the ML region. What we found nowhere, and what makes the repositioned contribution:

1. **An oracle potential study** that quantifies, a posteriori and weighted by the real
   cost, the fraction of element-steps that could be extrapolated, and its dependence on
   the tolerance and on the observed variable (stress vs. accumulated plastic strain). We
   found no published upper bound of this kind.
2. **A local extrapolation from the internal state** (last converged state + local
   tangent, deferred integration of the law) with neither reduced basis nor surrogate: the
   guarantee is the solver's, there is nothing to train offline. The close relatives all
   go through a ROM (POD, RID) or a surrogate.
3. **Explicit treatment of wake-up** (cyclic unloading/reloading) as a decision problem,
   and the question of what a learned criterion adds **against a tuned heuristic**, with
   the comparison made against the "Sysala" baseline (plastic-only re-assembly) rather than
   a naive solver. We found no comparison of this kind.

The honest positioning is therefore: *we do not claim the idea of partitioning the
domain into active and quiet regions, which is known; we claim the measurement of its
real potential, an implementation without reduced model, and an answer to the question of
the added value of learning for the decision.*

To check on the full texts before any submission: switching criterion and wake-up
handling in Radermacher & Reese; algebraic criterion in Kerfriden et al.; growth
mechanism in Wang et al.
