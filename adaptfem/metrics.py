"""Error metrics between an orchestrated run and the reference run (same mesh, steps,
integrator), and cost aggregation with measured unit costs."""
import numpy as np
from .element import NGP


def errors(hist, ref, model, sig_y0):
    """Per-step and summary errors. hist/ref: stacked histories."""
    w = np.repeat(model.el.wdetJ.reshape(-1), 1)                # (ngp,)
    W = w.sum()
    Cinv = np.linalg.inv(model.mat.C3)
    out = {}
    du = np.linalg.norm(hist["u"] - ref["u"], axis=1) / np.maximum(np.linalg.norm(ref["u"], axis=1), 1e-30)
    ds = (hist["stress"] - ref["stress"])[:, :, [0, 1, 3]]
    rms_sig = np.sqrt((w * (ds ** 2).sum(2)).sum(1) / W) / sig_y0
    max_sig = np.sqrt((ds ** 2).sum(2)).max(1) / sig_y0
    en = np.einsum("tni,ij,tnj->tn", ds, Cinv, ds)
    en_ref = np.einsum("tni,ij,tnj->tn", ref["stress"][:, :, [0, 1, 3]], Cinv, ref["stress"][:, :, [0, 1, 3]])
    e_energy = np.sqrt((w * en).sum(1) / np.maximum((w * en_ref).sum(1), 1e-30))
    da = hist["alpha"] - ref["alpha"]
    rms_alpha = np.sqrt((w * da ** 2).sum(1) / W)
    max_alpha = np.abs(da).max(1)
    rel_alpha = np.abs(da).max(1) / np.maximum(np.abs(ref["alpha"]).max(1), 1e-30)
    dR = np.abs(hist["reaction"] - ref["reaction"]) / np.maximum(np.abs(ref["reaction"]).max(), 1e-30)
    out["per_step"] = dict(u=du, stress_rms=rms_sig, stress_max=max_sig, energy=e_energy,
                           alpha_rms=rms_alpha, alpha_max=max_alpha, reaction=dR)
    for k, v in out["per_step"].items():
        out[f"{k}_final"] = float(v[-1]); out[f"{k}_max"] = float(v.max())
    out["alpha_rel_final"] = float(rel_alpha[-1]); out["alpha_rel_max"] = float(rel_alpha.max())
    return out


def element_cost(cost, unit, decision_time=0.0):
    """Aggregate the run's counters with a unit-cost dict (from benchmark.measure_costs).
    decision_time: measured wall time of the policy inference (added as is: it is a real
    cost of the same implementation; a compiled inference would be cheaper, so would the
    rest)."""
    c = cost.c
    return (decision_time + c["gp_integrations"] * (unit["strain_gp"] + unit["trial_gp"])
            + c["gp_plastic"] * unit["return_gp"]
            + c["element_K"] * (unit["K_el"] + unit["asmK_el"])
            + c["element_fint"] * (unit["fint_el"] + unit["asmf_el"])
            + c["element_extrapolated"] * unit["quiet_el"]
            + c["gp_monitored"] * unit["monitor_gp"])
