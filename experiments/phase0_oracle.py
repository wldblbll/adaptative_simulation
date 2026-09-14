"""Phase 0: oracle study. Runs the reference solver on every case, stores the full
history, measures per-element costs, and evaluates the a posteriori extrapolation
potential as a function of tolerance. Produces figures and a JSON summary.

Usage: python experiments/phase0_oracle.py [--h 0.5] [--cases notched_plate,cantilever,...]
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptfem import cases
from adaptfem.oracle import oracle_greedy, cost_weighted_potential, unit_cost_model, EXTRAPOLATORS
from adaptfem.benchmark import measure_costs
from adaptfem.element import NGP

TOLS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
RHOS = [0.0, 0.1, 0.25, 0.5]


def elem_order(case):
    """Ordering of elements for space-time maps: by distance to the notch centre
    (notched) or by x (cantilever)."""
    cen = case.nodes[case.elems].mean(1)
    if "notch" in case.meta:
        nx, ny = case.meta["notch"]
        return np.argsort(np.hypot(cen[:, 0] - nx, cen[:, 1] - ny)), "distance to notch [mm]", np.hypot(cen[:, 0] - nx, cen[:, 1] - ny)
    return np.argsort(cen[:, 0]), "x [mm]", cen[:, 0]


def plot_mesh_field(ax, case, values, cmap="viridis", vmin=None, vmax=None, title=""):
    verts = case.nodes[case.elems]
    pc = PolyCollection(verts, array=values, cmap=cmap, edgecolors="none")
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return pc


def run_case(name, h, outdir, tols, rhos):
    case = cases.CASES[name](h=h)
    model = case.model(record_history=True)
    t0 = time.perf_counter()
    model.run()
    wall = time.perf_counter() - t0
    hist = model.stacked_history()
    T = len(case.lam_hist)
    np.savez_compressed(os.path.join(outdir, f"{name}_history.npz"),
                        **{k: v for k, v in hist.items() if k != "residuals"},
                        nodes=case.nodes, elems=case.elems)
    costs = measure_costs(model.el, case.material)
    cst = model.cost
    t_elem = sum(cst.t[k] for k in ("strain", "constitutive", "element_K", "element_fint", "assembly"))
    summary = dict(case=name, nel=int(model.nel), ndof=int(model.ndof), steps=T,
                   wall_time=wall, newton_iters=int(cst.c["newton_iters"]),
                   t_element_level=t_elem, t_factorization=cst.t["factorization"], t_solve=cst.t["solve"],
                   element_level_share=t_elem / cst.t["total"],
                   first_yield_step=int(np.argmax((hist["gp_plastic"] > 0).any(1))),
                   final_plastic_el_fraction=float((hist["gp_plastic"][-1] > 0).mean()),
                   max_plastic_el_fraction=float((hist["gp_plastic"] > 0).mean(1).max()),
                   mean_plastic_el_fraction=float((hist["gp_plastic"] > 0).mean()),
                   costs=costs, counters=cst.as_dict(), oracle={})
    pl_el = hist["gp_plastic"] > 0
    order, xlabel, dist = elem_order(case)

    # --- tolerance sweep -------------------------------------------------
    res = {}
    for ex in EXTRAPOLATORS:
        res[ex] = {}
        for tol in tols:
            o = oracle_greedy(hist, case.material, model.nel, tol, ex, check_lazy=(ex == "tangent"))
            integ = o["integrated"]
            d = dict(frac_extrapolated=float(1 - integ.mean()),
                     frac_extrapolated_after_yield=float(1 - integ[summary["first_yield_step"]:].mean()),
                     frac_plastic_elements_extrapolated=float((~integ & pl_el).sum() / max(pl_el.sum(), 1)),
                     active_fraction_per_step=integ.mean(1).tolist())
            for cm in ("vector", "scalar"):
                d[f"cost_potential_{cm}"] = float(cost_weighted_potential(integ, hist, costs[cm])[0])
            for rho in rhos:
                d[f"cost_potential_rho{rho}"] = float(cost_weighted_potential(
                    integ, hist, unit_cost_model(rho, costs["scalar"]["ratio_plastic_elastic"]))[0])
            if ex == "tangent":
                m = ~integ
                d["lazy_sig_err_p50"] = float(np.median(o["lazy_sig_err"][m])) if m.any() else 0.0
                d["lazy_sig_err_max"] = float(o["lazy_sig_err"][m].max()) if m.any() else 0.0
                d["lazy_alpha_err_max"] = float(o["lazy_alpha_err"][m].max()) if m.any() else 0.0
                d["lazy_alpha_err_p99"] = float(np.quantile(o["lazy_alpha_err"][m], 0.99)) if m.any() else 0.0
            res[ex][str(tol)] = d
    summary["oracle"] = res
    # also the "one-step skip" upper bound (always extrapolate from k-1) for tangent
    one = {}
    for tol in tols:
        o = oracle_greedy(hist, case.material, model.nel, tol, "tangent", max_skip=1)
        one[str(tol)] = float(1 - o["integrated"].mean())
    summary["oracle_one_step_tangent"] = one

    # --- figures ---------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ex in EXTRAPOLATORS:
        axs[0].plot(tols, [res[ex][str(t)]["frac_extrapolated"] for t in tols], "o-", label=ex)
    axs[0].set_xscale("log"); axs[0].set_xlabel("tolerance (stress error / sig_y0)")
    axs[0].set_ylabel("fraction of element-steps extrapolated"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[0].set_title(f"{name}: raw potential (upper bound)")
    for rho in rhos:
        axs[1].plot(tols, [res["tangent"][str(t)][f"cost_potential_rho{rho}"] for t in tols], "o-", label=f"quiet cost = {rho}")
    axs[1].plot(tols, [res["tangent"][str(t)]["cost_potential_vector"] for t in tols], "s--", label="measured (NumPy)")
    axs[1].plot(tols, [res["tangent"][str(t)]["cost_potential_scalar"] for t in tols], "^--", label="measured (scalar)")
    axs[1].set_xscale("log"); axs[1].set_xlabel("tolerance"); axs[1].set_ylabel("element-level cost saved")
    axs[1].set_title("cost-weighted potential, tangent extrapolation"); axs[1].legend(fontsize=7); axs[1].grid(alpha=.3)
    for tol in (0.002, 0.01, 0.05):
        axs[2].plot(res["tangent"][str(tol)]["active_fraction_per_step"], label=f"tol={tol}")
    axs[2].plot(pl_el.mean(1), "k:", label="plastic element fraction")
    axs[2].set_xlabel("step"); axs[2].set_ylabel("fraction of elements integrated"); axs[2].legend(); axs[2].grid(alpha=.3)
    axs[2].set_title("active fraction vs step (tangent)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{name}_potential.png"), dpi=130); plt.close(fig)

    tol_map = 0.01
    o = oracle_greedy(hist, case.material, model.nel, tol_map, "tangent")
    integ = o["integrated"]
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axs[0].imshow(integ[:, order].T, aspect="auto", cmap="Greys", interpolation="nearest")
    axs[0].set_ylabel(f"elements sorted by {xlabel}"); axs[0].set_title(f"{name}: elements that must be integrated (black), oracle tangent tol={tol_map}")
    axs[1].imshow(pl_el[:, order].T, aspect="auto", cmap="Reds", interpolation="nearest")
    axs[1].set_ylabel(f"elements sorted by {xlabel}"); axs[1].set_xlabel("load step"); axs[1].set_title("plastic elements (red)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{name}_spacetime.png"), dpi=130); plt.close(fig)

    snaps = np.unique(np.linspace(max(summary["first_yield_step"], 1), T - 1, 4).astype(int))
    fig, axs = plt.subplots(2, len(snaps), figsize=(4 * len(snaps), 5))
    for j, k in enumerate(snaps):
        plot_mesh_field(axs[0, j], case, integ[k].astype(float), cmap="Greys", vmin=0, vmax=1,
                        title=f"step {k} (lam={case.lam_hist[k]:+.2f}): integrated {integ[k].mean():.0%}")
        plot_mesh_field(axs[1, j], case, hist["alpha"][k].reshape(-1, NGP).max(1), cmap="magma",
                        title="accumulated plastic strain")
    fig.suptitle(f"{name}: oracle active set (tangent, tol={tol_map}) — black = must integrate")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{name}_snapshots.png"), dpi=130); plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--cases", default=",".join(cases.CASES))
    ap.add_argument("--out", default="results/phase0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    all_summ = {}
    for name in args.cases.split(","):
        t0 = time.perf_counter()
        s = run_case(name, args.h, args.out, TOLS, RHOS)
        all_summ[name] = s
        print(f"[{name}] nel={s['nel']} steps={s['steps']} wall={s['wall_time']:.1f}s "
              f"elem-level share={s['element_level_share']:.2f} plastic frac max={s['max_plastic_el_fraction']:.2f} "
              f"(analysis {time.perf_counter() - t0:.0f}s)")
        for tol in ("0.002", "0.01", "0.05"):
            r = s["oracle"]["tangent"][tol]
            print(f"   tol={tol}: raw={r['frac_extrapolated']:.3f} after-yield={r['frac_extrapolated_after_yield']:.3f} "
                  f"cost(vec)={r['cost_potential_vector']:.3f} cost(scalar)={r['cost_potential_scalar']:.3f} "
                  f"rho0.25={r['cost_potential_rho0.25']:.3f} lazy_alpha_p99={r['lazy_alpha_err_p99']:.2e}")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(all_summ, f, indent=1)


if __name__ == "__main__":
    main()
