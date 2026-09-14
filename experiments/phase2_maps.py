"""Space-time maps of the online (exact) orchestrator: integrated vs monitored vs skipped
elements at every step, compared with the plastic zone of the reference."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptfem import cases
from adaptfem.orchestrator import OrchestratorConfig, HeuristicOrchestrator
from adaptfem.element import NGP

out = "results/phase2"; os.makedirs(out, exist_ok=True)


class Recorder(HeuristicOrchestrator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.mon_steps = []
    def select(self, k, u_prev, state_prev, model):
        self._mon = np.zeros(self.nel, bool)
        return super().select(k, u_prev, state_prev, model)
    def _evaluate(self, model, state_prev, quiet, u, in_iteration=True):
        age0 = self.margin_age.copy()
        w = super()._evaluate(model, state_prev, quiet, u, in_iteration)
        self._mon |= (self.margin_age == 0) & (age0 != 0) | ((self.margin_age == 0) & (self.margin != np.inf) & self._mon)
        self._mon[quiet[self.margin_age[quiet] == 0]] = True
        return w
    def after_step(self, k, u, state, rec, model):
        self.mon_steps.append(self._mon.copy())
        super().after_step(k, u, state, rec, model)


for name in ("notched_plate", "cantilever", "cyclic_notched_kin", "cyclic_cantilever_kin"):
    c = cases.CASES[name](h=0.5)
    m = c.model(record_history=True, reuse_elastic_K=True)
    o = Recorder(OrchestratorConfig(monitor_skip_kappa=1.0), m)
    m.run(orchestrator=o)
    h = m.stacked_history()
    act = h["active"]; mon = np.array(o.mon_steps); pl = h["gp_plastic"] > 0
    cen = c.nodes[c.elems].mean(1)
    if "notch" in c.meta:
        nx, ny = c.meta["notch"]; order = np.argsort(np.hypot(cen[:, 0] - nx, cen[:, 1] - ny)); lab = "distance to notch"
    else:
        order = np.argsort(cen[:, 0]); lab = "x"
    code = np.zeros(act.shape); code[mon & ~act] = 1; code[act] = 2
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    im = axs[0].imshow(code[:, order].T, aspect="auto", interpolation="nearest", cmap=matplotlib.colors.ListedColormap(["white", "gold", "black"]), vmin=0, vmax=2)
    axs[0].set_title(f"{name}: online orchestrator (exact, kappa=1) — black: integrated, gold: monitored only, white: skipped")
    axs[0].set_ylabel(f"elements sorted by {lab}")
    axs[1].imshow(pl[:, order].T, aspect="auto", interpolation="nearest", cmap="Reds"); axs[1].set_title("reference plastic zone"); axs[1].set_xlabel("load step")
    ax2 = axs[1].twinx(); ax2.plot(c.lam_hist, "k:", lw=1); ax2.set_ylabel("load factor")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"{name}_online_spacetime.png"), dpi=130); plt.close(fig)
    print(f"{name}: mean active {act.mean():.3f}, mean monitored-only {(mon & ~act).mean():.3f}, mean plastic {pl.mean():.3f}, max active {act.mean(1).max():.3f}")
