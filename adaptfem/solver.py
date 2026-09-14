"""Incremental Newton-Raphson solver with full cost instrumentation and history recording.

Design rule: the orchestrator (Phase 2) only changes *which elements are integrated* at a
given iteration. Everything else (mesh, element, material, Newton loop, convergence test)
is shared with the reference run, which is this same solver with `orchestrator=None`.
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse.linalg import splu

from .element import NGP

_PS3 = np.array([0, 1, 3])


class Cost:
    """Operation counters and phase timers. Counters are transferable; timers are
    implementation-dependent and reported separately."""

    COUNTERS = ("gp_integrations", "gp_plastic", "local_newton_iters", "element_K",
                "element_fint", "element_extrapolated", "assemblies_K", "assemblies_f",
                "factorizations", "solves", "newton_iters", "steps", "full_steps",
                "orchestrator_decisions")
    TIMERS = ("strain", "constitutive", "element_K", "element_fint", "assembly",
              "factorization", "solve", "orchestrator", "extrapolation", "total")

    def __init__(self):
        self.c = {k: 0 for k in self.COUNTERS}
        self.t = {k: 0.0 for k in self.TIMERS}

    @contextmanager
    def timer(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] += time.perf_counter() - t0

    def add(self, name, n=1):
        self.c[name] += n

    def as_dict(self):
        d = {f"n_{k}": v for k, v in self.c.items()}
        d.update({f"t_{k}": v for k, v in self.t.items()})
        return d

    def __repr__(self):
        return "Cost(" + ", ".join(f"{k}={v}" for k, v in self.as_dict().items()) + ")"


@dataclass
class State:
    """Converged material state at every Gauss point (flattened: index = el*NGP + g)."""
    stress: np.ndarray      # (ngp,4)
    eps_p: np.ndarray       # (ngp,4)
    alpha: np.ndarray       # (ngp,)
    plastic: np.ndarray     # (ngp,) bool  (plastic at the last integration)
    D: np.ndarray           # (ngp,3,3) consistent tangent at the last integration
    eps: np.ndarray         # (ngp,3) strain at the last integration
    Ke: np.ndarray          # (nel,8,8) element tangent at the last integration
    fe: np.ndarray          # (nel,8) element internal force at the last integration

    @classmethod
    def zeros(cls, nel, C3):
        ngp = nel * NGP
        return cls(np.zeros((ngp, 4)), np.zeros((ngp, 4)), np.zeros(ngp),
                   np.zeros(ngp, bool), np.broadcast_to(C3, (ngp, 3, 3)).copy(),
                   np.zeros((ngp, 3)), np.zeros((nel, 8, 8)), np.zeros((nel, 8)))

    def copy(self):
        return State(*(getattr(self, f).copy() for f in
                       ("stress", "eps_p", "alpha", "plastic", "D", "eps", "Ke", "fe")))


@dataclass
class StepRecord:
    step: int
    lam: float
    n_iter: int
    residuals: list
    reaction: float
    active_fraction: float = 1.0


class FEModel:
    def __init__(self, elements, material, fixed_dofs, ubar, load_history, fext=None,
                 tol=1e-8, maxit=25, record_history=False, record_tangent=True):
        self.el = elements
        self.mat = material
        self.ndof = elements.ndof
        self.nel = elements.nel
        self.fixed = np.asarray(fixed_dofs, dtype=int)
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.ubar = ubar                       # prescribed displacement at lambda = 1
        self.lam_hist = np.asarray(load_history, float)
        self.fext_unit = np.zeros(self.ndof) if fext is None else fext
        self.tol, self.maxit = tol, maxit
        self.record_history = record_history
        self.record_tangent = record_tangent
        self.cost = Cost()
        self.history = {"eps": [], "stress": [], "eps_p": [], "alpha": [], "plastic": [],
                        "D": [], "u": [], "n_iter": [], "lam": [], "reaction": [],
                        "gp_plastic": [], "active": [], "residuals": []}
        self.records = []
        self.reaction_dofs = None

    # ---------------------------------------------------------------------
    def integrate_elements(self, u, state_prev, el, state_work):
        """Integrate the constitutive law on elements `el` (index array) from the
        converged state `state_prev`; write results into `state_work`. Returns nothing."""
        c = self.cost
        gp = (el[:, None] * NGP + np.arange(NGP)[None]).ravel()
        with c.timer("strain"):
            eps = self.el.strains(u, el).reshape(-1, 3)
        with c.timer("constitutive"):
            sig, ep, al, pl, D, nit = self.mat.integrate(
                eps, state_prev.eps_p[gp], state_prev.alpha[gp], need_tangent=True)
        c.add("gp_integrations", gp.size)
        c.add("gp_plastic", int(pl.sum()))
        c.add("local_newton_iters", nit)
        state_work.stress[gp] = sig
        state_work.eps_p[gp] = ep
        state_work.alpha[gp] = al
        state_work.plastic[gp] = pl
        state_work.D[gp] = D
        state_work.eps[gp] = eps
        with c.timer("element_fint"):
            state_work.fe[el] = self.el.element_fint(sig[:, _PS3].reshape(-1, NGP, 3), el)
        c.add("element_fint", el.size)
        with c.timer("element_K"):
            state_work.Ke[el] = self.el.element_K(D.reshape(-1, NGP, 3, 3), el)
        c.add("element_K", el.size)

    def extrapolate_elements(self, u, state_prev, el, state_work):
        """Lazy path: no constitutive call. Stress is linearised about the last integrated
        state with the stored tangent; Ke is reused; fe = fe_last + Ke (u - u_last) computed
        from the strain increment (same flops as an elastic element residual)."""
        if el.size == 0:
            return
        c = self.cost
        gp = (el[:, None] * NGP + np.arange(NGP)[None]).ravel()
        with c.timer("extrapolation"):
            eps = self.el.strains(u, el).reshape(-1, 3)
            deps = eps - state_prev.eps[gp]
            dsig3 = np.einsum("nij,nj->ni", state_prev.D[gp], deps)
            sig = state_prev.stress[gp].copy()
            sig[:, _PS3] += dsig3
            # zz component: plane-strain elastic-like update (not used in equilibrium)
            state_work.stress[gp] = sig
            state_work.eps[gp] = state_prev.eps[gp]       # keep the anchor strain
            state_work.eps_p[gp] = state_prev.eps_p[gp]
            state_work.alpha[gp] = state_prev.alpha[gp]
            state_work.plastic[gp] = state_prev.plastic[gp]
            state_work.D[gp] = state_prev.D[gp]
            state_work.Ke[el] = state_prev.Ke[el]
            state_work.fe[el] = self.el.element_fint(sig[:, _PS3].reshape(-1, NGP, 3), el)
        c.add("element_extrapolated", el.size)

    # ---------------------------------------------------------------------
    def solve_step(self, k, u_prev, state_prev, active=None, orchestrator=None):
        """One load step. `active`: element index array integrated exactly this step
        (None = all). Returns (u, state, record)."""
        c = self.cost
        lam = self.lam_hist[k]
        u = u_prev.copy()
        u[self.fixed] = lam * self.ubar[self.fixed]
        fext = lam * self.fext_unit
        all_el = np.arange(self.nel)
        if active is None:
            active = all_el
        quiet = np.setdiff1d(all_el, active)
        state = state_prev.copy()
        residuals = []
        converged = False
        # Tangent predictor (standard in displacement control): linearise the internal
        # force about the previous converged state with the previous converged tangent,
        # solve for the free DOFs, and only then start integrating. Same code path for
        # the reference and the orchestrated runs.
        lam_prev = self.lam_hist[k - 1] if k > 0 else 0.0
        du_p = u - u_prev
        with c.timer("assembly"):
            K = self.el.csr(self.el.assemble_K_data(state_prev.Ke))
            Kff = K[self.free][:, self.free].tocsc()
        c.add("assemblies_K")
        R_lin = (lam - lam_prev) * self.fext_unit - K @ du_p
        with c.timer("factorization"):
            lu = splu(Kff)
        c.add("factorizations")
        with c.timer("solve"):
            u[self.free] += lu.solve(R_lin[self.free])
        c.add("solves")
        for it in range(self.maxit + 1):
            self.integrate_elements(u, state_prev, active, state)
            self.extrapolate_elements(u, state_prev, quiet, state)
            with c.timer("assembly"):
                fint = self.el.assemble_fint(state.fe)
            c.add("assemblies_f")
            R = fext - fint
            Rf = R[self.free]
            ref = max(np.linalg.norm(fint), np.linalg.norm(fext), 1e-12)
            rn = np.linalg.norm(Rf) / ref
            residuals.append(rn)
            if rn < self.tol:
                converged = True
                break
            if it == self.maxit:
                break
            with c.timer("assembly"):
                Kdata = self.el.assemble_K_data(state.Ke)
                K = self.el.csr(Kdata)
                Kff = K[self.free][:, self.free].tocsc()
            c.add("assemblies_K")
            with c.timer("factorization"):
                lu = splu(Kff)
            c.add("factorizations")
            with c.timer("solve"):
                du = lu.solve(Rf)
            c.add("solves")
            u[self.free] += du
            c.add("newton_iters")
        if not converged:
            raise RuntimeError(f"Newton did not converge at step {k} (lam={lam:.4f}); "
                               f"residuals={residuals}")
        c.add("steps")
        if active.size == self.nel:
            c.add("full_steps")
        reaction = float(fint[self.reaction_dofs].sum()) if self.reaction_dofs is not None else 0.0
        rec = StepRecord(k, lam, len(residuals) - 1, residuals, reaction, active.size / self.nel)
        # quiet elements keep their anchor (last integrated) state: only the extrapolated
        # stress is provisional. For history we store the *provisional* stress on quiet
        # elements, and flag them.
        if self.record_history:
            h = self.history
            h["eps"].append(self.el.strains(u).reshape(-1, 3).astype(np.float64))
            h["stress"].append(state.stress.copy())
            h["eps_p"].append(state.eps_p.copy())
            h["alpha"].append(state.alpha.copy())
            h["plastic"].append(state.plastic.copy())
            if self.record_tangent:
                h["D"].append(state.D.astype(np.float32))
            h["u"].append(u.copy())
            h["n_iter"].append(rec.n_iter)
            h["lam"].append(lam)
            h["reaction"].append(reaction)
            h["gp_plastic"].append(state.plastic.reshape(self.nel, NGP).sum(1).astype(np.int8))
            act = np.zeros(self.nel, bool); act[active] = True
            h["active"].append(act)
            h["residuals"].append(np.array(residuals))
        return u, state, rec

    def run(self, orchestrator=None, verbose=False):
        """Run the full load history. `orchestrator` (Phase 2) has
        `select(k, u_prev, state_prev, model) -> active element index array or None`
        and `after_step(k, u, state, rec, model)`."""
        c = self.cost
        with c.timer("total"):
            u = np.zeros(self.ndof)
            state = State.zeros(self.nel, self.mat.C3)
            state.Ke = self.el.element_K(np.broadcast_to(self.mat.C3, (self.nel, NGP, 3, 3)))
            for k in range(len(self.lam_hist)):
                active = None
                if orchestrator is not None:
                    with c.timer("orchestrator"):
                        active = orchestrator.select(k, u, state, self)
                    c.add("orchestrator_decisions", self.nel)
                u, state, rec = self.solve_step(k, u, state, active, orchestrator)
                self.records.append(rec)
                if orchestrator is not None:
                    with c.timer("orchestrator"):
                        orchestrator.after_step(k, u, state, rec, self)
                if verbose:
                    print(f"step {k:4d} lam={rec.lam:+.4f} it={rec.n_iter} "
                          f"plastic_gp={int(state.plastic.sum())} act={rec.active_fraction:.3f} "
                          f"R={rec.reaction:+.4e}")
        return u, state

    def stacked_history(self):
        h = self.history
        out = {}
        for k, v in h.items():
            if k == "residuals":
                out[k] = np.array(v, dtype=object)
            elif len(v):
                out[k] = np.stack(v)
        return out
