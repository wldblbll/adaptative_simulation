"""Phase 2: deterministic heuristic orchestrator.

Each load step, elements are either *active* (constitutive law integrated, tangent
recomputed) or *quiet* (stress linearised about the last integrated state with the
stored tangent, element stiffness reused, internal variables frozen until wake-up, at
which point the law is integrated over the accumulated increment).

Decision rules (all deterministic, all cheap):
  * elastic-anchored quiet elements are monitored with the exact trial yield check on
    the extrapolated stress (this is exactly the elastic predictor a standard code would
    compute), either inside every Newton iteration (wake immediately) or after the step;
  * plastic-anchored quiet elements are woken on local unloading (flow direction against
    the strain increment) and when the linearised accumulated plastic strain since the
    anchor exceeds `tol_alpha` (tangent-drift proxy);
  * structural rules: catch-up (full step) every N steps, maximum consecutive skipped
    steps, neighbour propagation, global load-reversal wake-up, mandatory final full step.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .element import NGP
from .material import SQ23, dev_norm
from .mesh import element_adjacency


@dataclass
class OrchestratorConfig:
    name: str = "heuristic"
    monitor: str = "yield"            # "yield" | "none"
    monitor_when: str = "iteration"   # "iteration" | "predictor" (it 0 + at convergence) | "predictor_only" | "converged" | "step"
    yield_margin: float = 0.0         # wake elastic-anchored element if f_hat > -margin
    plastic_policy: str = "always_active"   # "always_active" | "drift"
    tol_alpha: float = 1e-3           # linearised accumulated plastic strain allowed since anchor
    unload_wake: bool = True
    catchup_every: int = 0            # 0 = never
    max_skip: int = 0                 # 0 = unlimited
    neighbor_layers: int = 0
    global_reversal_wake: bool = False
    warmup_full_steps: int = 1        # first steps always full
    plastic_predict: bool = True      # wake plastic-anchored elements at step start from the predicted increment
    plastic_check_in_iteration: bool = True   # safety net: also check them inside iterations
    monitor_policy: str = "kappa"     # "kappa" | "learned" | "oracle" | "all"
    horizon_safety: float = 0.5       # learned/oracle: re-monitor after floor(safety * predicted horizon) steps
    monitor_skip_kappa: float = 2.0   # >0: skip monitoring of elements whose yield margin exceeds kappa x (max stress change per step)

    def to_dict(self):
        return asdict(self)


class HeuristicOrchestrator:
    def __init__(self, cfg: OrchestratorConfig, model, horizon_model=None):
        self.cfg = cfg
        self.nel = model.nel
        self.horizon_model = horizon_model       # object with .predict(X) (learned) or .horizon(k, el) (oracle)
        self.next_monitor = np.zeros(self.nel, dtype=int)   # step at which an element must be monitored again
        self.f_last = np.full(self.nel, -1.0)
        self.f_hist = np.full((self.nel, 2), -1.0)
        self.age_hist = np.ones(self.nel, dtype=int)
        self.last_plastic_step = np.full(self.nel, -10 ** 6)
        self.t_policy = 0.0
        self.anchor = np.full(self.nel, -1, dtype=int)
        self.skip = np.zeros(self.nel, dtype=int)
        self.pending_wake = np.zeros(self.nel, bool)
        self.last_woken = np.zeros(self.nel, bool)
        if cfg.neighbor_layers > 0 or cfg.monitor_policy == "learned":
            self.adj_ptr, self.adj_idx = element_adjacency(model.el.elems)
        self.log = []
        ngp = self.nel * NGP
        self.eps_end = np.zeros((ngp, 3))        # strain at the end of the last step
        self.deps_last = np.zeros((ngp, 3))      # last step's strain increment
        self.margin = np.full(self.nel, np.inf)  # -max f_hat at last monitoring
        self.margin_age = np.zeros(self.nel, int)
        self.rate = 0.0                          # max normalised stress change per step observed

    # -- helpers -----------------------------------------------------------
    def _neighbors(self, mask, layers):
        out = mask.copy()
        for _ in range(layers):
            idx = np.nonzero(out)[0]
            nb = np.concatenate([self.adj_idx[self.adj_ptr[e]:self.adj_ptr[e + 1]] for e in idx]) if idx.size else np.array([], int)
            out[nb] = True
        return out

    def _plastic_checks(self, model, state_prev, gp, deps):
        """For plastic-anchored GPs: unloading indicator and linearised plastic strain."""
        mat = model.mat
        _, d = dev_norm(state_prev.stress[gp])
        xi = d - state_prev.beta[gp]
        xn = np.sqrt(xi[:, 0] ** 2 + xi[:, 1] ** 2 + xi[:, 2] ** 2 + 2 * xi[:, 3] ** 2)
        n = xi / np.maximum(xn, 1e-300)[:, None]
        n_deps = n[:, 0] * deps[:, 0] + n[:, 1] * deps[:, 1] + n[:, 3] * deps[:, 2]   # n:deps (gxy engineering)
        Hp = mat.dsig_y(state_prev.alpha[gp])
        dgam = 2 * mat.mu * n_deps / (2 * mat.mu + 2 / 3 * (mat.Hk + Hp))
        return n_deps, SQ23 * dgam

    def _evaluate(self, model, state_prev, quiet, u, in_iteration=True):
        """Return boolean wake mask over `quiet` elements."""
        cfg = self.cfg
        wake = np.zeros(quiet.size, bool)
        pl_anchor = state_prev.plastic.reshape(self.nel, NGP).any(1)[quiet]
        if cfg.monitor == "yield":
            # elements whose margin at the last monitoring exceeds what the stress can have
            # moved since then are provably still elastic: skip them
            el_mon = quiet[~pl_anchor]
            if cfg.monitor_policy == "kappa":
                if cfg.monitor_skip_kappa > 0 and self.rate > 0:
                    bound = cfg.monitor_skip_kappa * self.rate * (self.margin_age[el_mon] + 1)
                    mg = self.margin[el_mon]
                    el_mon = el_mon[(mg <= bound) | ~np.isfinite(mg)]
            elif cfg.monitor_policy in ("learned", "oracle"):
                # horizon applies to later steps; within the current step an element that
                # was monitored once stays monitored (its converged state must be verified)
                el_mon = el_mon[(self.next_monitor[el_mon] <= self._k) | (self.margin_age[el_mon] == 0)]
            if el_mon.size:
                sig_hat, _, gpm = model.stress_hat(u, state_prev, el_mon)
                model.cost.add("gp_monitored", gpm.size)
                f = model.mat.yield_function(sig_hat, state_prev.alpha[gpm],
                                             state_prev.beta[gpm]).reshape(-1, NGP).max(1)
                if cfg.monitor_policy in ("learned", "oracle") and in_iteration:
                    import time as _t
                    t0 = _t.perf_counter()
                    if cfg.monitor_policy == "oracle":
                        h = self.horizon_model.horizon(self._k, el_mon)
                    else:
                        from .learning import online_features
                        X = online_features(self, model, state_prev, el_mon, f, self._k)
                        h = self.horizon_model.predict(X)
                    self.next_monitor[el_mon] = self._k + np.maximum(np.floor(cfg.horizon_safety * (h - 1)), 0).astype(int) + 1
                    self.t_policy += _t.perf_counter() - t0
                    model.cost.add("orchestrator_decisions", el_mon.size)
                self.f_hist[el_mon, 1] = self.f_hist[el_mon, 0]
                self.age_hist[el_mon] = np.maximum(self.margin_age[el_mon], 1)
                self.f_hist[el_mon, 0] = self.f_last[el_mon]
                self.f_last[el_mon] = f
                self.margin[el_mon] = -f
                self.margin_age[el_mon] = 0
                w = np.zeros(quiet.size, bool)
                w[np.searchsorted(quiet, el_mon)] = f > -cfg.yield_margin
                wake |= w
        if pl_anchor.any() and (cfg.plastic_check_in_iteration or not in_iteration):
            if cfg.plastic_policy == "always_active":
                wake |= pl_anchor
            elif cfg.plastic_policy == "drift":
                el_pl = quiet[pl_anchor]
                _, eps, gp = model.stress_hat(u, state_prev, el_pl)
                model.cost.add("gp_monitored", gp.size)
                deps = eps - state_prev.eps[gp]
                n_deps, dalpha = self._plastic_checks(model, state_prev, gp, deps)
                pl_gp = state_prev.plastic[gp]
                unload = ((n_deps < 0) & pl_gp).reshape(-1, NGP).any(1) if cfg.unload_wake else False
                drift = ((dalpha > cfg.tol_alpha) & pl_gp).reshape(-1, NGP).any(1)
                w = np.zeros(quiet.size, bool)
                w[pl_anchor] = unload | drift
                wake |= w
        return wake

    # -- interface used by FEModel -----------------------------------------
    def select(self, k, u_prev, state_prev, model):
        cfg = self.cfg
        self._k = k
        T = len(model.lam_hist)
        if k < cfg.warmup_full_steps or k == T - 1:
            return np.arange(self.nel)
        if cfg.catchup_every and (k % cfg.catchup_every == 0):
            return np.arange(self.nel)
        if cfg.global_reversal_wake and k >= 2:
            d1 = model.lam_hist[k] - model.lam_hist[k - 1]
            d0 = model.lam_hist[k - 1] - model.lam_hist[k - 2]
            if d1 * d0 < 0:
                return np.arange(self.nel)
        act = self.pending_wake.copy()
        if cfg.max_skip:
            act |= self.skip >= cfg.max_skip
        if cfg.plastic_policy == "drift" and cfg.plastic_predict and k >= 2:
            pl_el = state_prev.plastic.reshape(self.nel, NGP).any(1)
            cand = np.nonzero(pl_el & ~act)[0]
            if cand.size:
                gp = (cand[:, None] * NGP + np.arange(NGP)[None]).ravel()
                deps = self.eps_end[gp] + self.deps_last[gp] - state_prev.eps[gp]
                n_deps, dalpha = self._plastic_checks(model, state_prev, gp, deps)
                pl_gp = state_prev.plastic[gp]
                unload = ((n_deps < 0) & pl_gp).reshape(-1, NGP).any(1) if cfg.unload_wake else False
                drift = ((dalpha > cfg.tol_alpha) & pl_gp).reshape(-1, NGP).any(1)
                act[cand[unload | drift]] = True
        if cfg.plastic_policy == "always_active":
            act |= state_prev.plastic.reshape(self.nel, NGP).any(1)
        if cfg.neighbor_layers > 0:
            act |= self._neighbors(self.last_woken, cfg.neighbor_layers)
        return np.nonzero(act)[0]

    def monitor(self, k, it, u, state_prev, state, quiet, model):
        if self.cfg.monitor_when not in ("iteration", "converged", "predictor", "predictor_only") or quiet.size == 0:
            return None
        wake = self._evaluate(model, state_prev, quiet, u)
        return quiet[wake]

    def after_step(self, k, u, state, rec, model):
        cfg = self.cfg
        active = rec.active
        self.pending_wake[:] = False
        self.deps_last = model.eps_cur - self.eps_end
        self.eps_end = model.eps_cur.copy()
        self.margin_age += 1
        dsig = state.stress_hat - self._sig_prev if hasattr(self, "_sig_prev") else np.zeros_like(state.stress_hat)
        self._sig_prev = state.stress_hat.copy()
        self.rate = float(np.sqrt((dsig[:, [0, 1, 3]] ** 2 * np.array([1, 1, 2])).sum(1)).max() / model.mat.sig_y0)
        if cfg.monitor_when == "step":
            quiet = np.nonzero(~active)[0]
            if quiet.size:
                # state.* of quiet elements is still the anchor; stress_hat is the equilibrium stress
                wake = self._evaluate(model, state, quiet, u, in_iteration=False)
                self.pending_wake[quiet[wake]] = True
        pl_now = state.plastic.reshape(self.nel, NGP).any(1)
        self.last_plastic_step[pl_now] = k
        self.f_last[active] = model.mat.yield_function(state.stress[(np.nonzero(active)[0][:, None] * NGP + np.arange(NGP)[None]).ravel()],
                                                       state.alpha[(np.nonzero(active)[0][:, None] * NGP + np.arange(NGP)[None]).ravel()],
                                                       state.beta[(np.nonzero(active)[0][:, None] * NGP + np.arange(NGP)[None]).ravel()]).reshape(-1, NGP).max(1)
        newly = active & (self.anchor < k - 1)          # woken this step (were quiet before)
        self.last_woken = newly | (active & (self.anchor == -1))
        self.anchor[active] = k
        self.skip[active] = 0
        self.skip[~active] += 1
        self.log.append(dict(step=k, active=float(active.mean()), woken=int(model.cost.c["elements_woken_in_iteration"])))
