"""Failure case: configurations where the orchestrator degrades the solution, with the
spatial and temporal signature of the error. Two mechanisms are shown:
 (1) lagged detection (post-step monitoring): an element yields during a step while
     treated as elastic, the equilibrium of that step is wrong, the error persists;
 (2) lazy integration of plastic elements with a loose drift tolerance: the error
     accumulates in the plastic zone and is only partially recovered at the final step.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptfem import cases
from adaptfem.orchestrator import OrchestratorConfig, HeuristicOrchestrator
from adaptfem.metrics import errors
from adaptfem.element import NGP

out = "results/failure"; os.makedirs(out, exist_ok=True)
c = cases.cyclic_notched_kin(h=0.5)
ref = c.model(record_history=True, reuse_elastic_K=True); ref.run(); H = ref.stacked_history()
cfgs = {"exact (iteration monitoring, plastic active)": dict(),
        "lagged (post-step monitoring)": dict(monitor_when="step"),
        "lazy plastic, tol_alpha=1e-3": dict(plastic_policy="drift", tol_alpha=1e-3),
        "lazy plastic, tol_alpha=1e-3, no unload wake": dict(plastic_policy="drift", tol_alpha=1e-3, unload_wake=False)}
res = {}
for name, kw in cfgs.items():
    m = c.model(record_history=True, reuse_elastic_K=True)
    o = HeuristicOrchestrator(OrchestratorConfig(**kw), m)
    m.run(orchestrator=o)
    h = m.stacked_history(); e = errors(h, H, m, c.material.sig_y0)
    res[name] = dict(hist=h, err=e, act=[r.active_fraction for r in m.records])
    print(f"{name}: u_max={e['u_max']:.1e} sig_rms_max={e['stress_rms_max']:.1e} alpha_rel_final={e['alpha_rel_final']:.1e} R_max={e['reaction_max']:.1e}")
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for name, r in res.items():
    axs[0].semilogy(np.maximum(r["err"]["per_step"]["u"], 1e-16), label=name)
    axs[1].semilogy(np.maximum(r["err"]["per_step"]["alpha_max"], 1e-16), label=name)
    axs[2].plot(r["act"], label=name)
axs[0].set_ylabel("relative displacement error"); axs[1].set_ylabel("max |alpha - alpha_ref|"); axs[2].set_ylabel("active fraction")
ax2 = axs[2].twinx(); ax2.plot(c.lam_hist, "k:", lw=1); ax2.set_ylabel("load factor")
axs[0].legend(fontsize=8); axs[2].set_xlabel("load step")
for a in axs: a.grid(alpha=.3)
fig.suptitle("Failure modes on the cyclic notched plate (kinematic hardening)")
fig.tight_layout(); fig.savefig(os.path.join(out, "failure_time.png"), dpi=130); plt.close(fig)
# spatial map of the final plastic-strain error for the lazy configuration
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
verts = c.nodes[c.elems]
for ax, (name, r) in zip(axs, list(res.items())[1:]):
    da = np.abs(r["hist"]["alpha"][-1] - H["alpha"][-1]).reshape(-1, NGP).max(1)
    pc = PolyCollection(verts, array=da, cmap="magma", edgecolors="none"); ax.add_collection(pc); ax.autoscale(); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{name}\nfinal |alpha - alpha_ref| (max {da.max():.1e})", fontsize=8); plt.colorbar(pc, ax=ax, fraction=0.03)
fig.tight_layout(); fig.savefig(os.path.join(out, "failure_space.png"), dpi=130); plt.close(fig)
json.dump({k: {kk: vv for kk, vv in v["err"].items() if kk != "per_step"} for k, v in res.items()}, open(os.path.join(out, "failure.json"), "w"), indent=1)
