"""Exact selective element integration ("lazy element integration").

Each load step, elements are either *active* (constitutive law integrated, tangent
recomputed) or *quiet* (element force replaced by its exact linearisation about the last
integrated state, f_e = f_e0 + K_e0 (u_e - u_e0), no Gauss-point work at all).

Rules (all deterministic, all cheap):
  * every element with a plastic Gauss point stays active (plasticity is path dependent:
    extrapolating it is not safe -- see docs/experiments.md);
  * an elastic element is quiet, and its linearisation is then *exact*; it is woken as soon
    as the trial (elastic predictor) stress violates the yield criterion. The check runs
    inside every Newton iteration, so a woken element is absorbed by the current Newton
    loop and no extra iteration is needed;
  * the check itself is skipped for elements that provably cannot have reached the yield
    surface: margin at the last check > kappa x (largest normalised stress change per step
    observed on the mesh) x (steps since that check + 1). kappa is the only parameter;
    kappa = 1 is the smallest value that stayed exact on every test case;
  * the first step and the last step are always fully integrated.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .element import NGP


@dataclass
class OrchestratorConfig:
    name: str = "exact"
    monitor_skip_kappa: float = 1.0   # safety factor of the monitoring-skip bound (0 = check every quiet element)
    warmup_full_steps: int = 1        # first steps always fully integrated

    def to_dict(self):
        return asdict(self)


class HeuristicOrchestrator:
    def __init__(self, cfg: OrchestratorConfig, model):
        self.cfg = cfg
        self.nel = model.nel
        self.margin = np.full(self.nel, np.inf)   # -max f_hat at the last check (inf = never checked)
        self.margin_age = np.zeros(self.nel, int)  # steps since the last check
        self.rate = 0.0                            # max normalised stress change per step observed
        self._sig_prev = None

    # -- interface used by FEModel -----------------------------------------
    def select(self, k, u_prev, state_prev, model):
        """Active set at the start of step k: every element with a plastic Gauss point."""
        if k < self.cfg.warmup_full_steps or k == len(model.lam_hist) - 1:
            return np.arange(self.nel)
        return np.nonzero(state_prev.plastic.reshape(self.nel, NGP).any(1))[0]

    def monitor(self, k, it, u, state_prev, state, quiet, model):
        """Called inside every Newton iteration: return the quiet elements to wake."""
        el_mon = quiet
        kappa = self.cfg.monitor_skip_kappa
        if kappa > 0 and self.rate > 0:
            bound = kappa * self.rate * (self.margin_age[el_mon] + 1)
            mg = self.margin[el_mon]
            el_mon = el_mon[(mg <= bound) | ~np.isfinite(mg)]
        if el_mon.size == 0:
            return None
        # elastic predictor: trial stress from the stored (elastic) tangent, no return mapping
        sig_hat, _, gp = model.stress_hat(u, state_prev, el_mon)
        model.cost.add("gp_monitored", gp.size)
        f = model.mat.yield_function(sig_hat, state_prev.alpha[gp],
                                     state_prev.beta[gp]).reshape(-1, NGP).max(1)
        self.margin[el_mon] = -f
        self.margin_age[el_mon] = 0
        return el_mon[f > 0]

    def after_step(self, k, u, state, rec, model):
        self.margin_age += 1
        sig = state.stress_hat
        if self._sig_prev is not None:
            dsig = sig - self._sig_prev
            self.rate = float(np.sqrt((dsig[:, [0, 1, 3]] ** 2 * np.array([1, 1, 2])).sum(1)).max()
                              / model.mat.sig_y0)
        self._sig_prev = sig.copy()
