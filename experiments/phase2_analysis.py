"""Analysis of Phase 2 campaigns: Pareto cost/error plots and summary tables."""
import json, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r.get("ok")]


def pareto_front(x, y):
    """x: cost saving (higher better), y: error (lower better)."""
    idx = np.argsort(-np.asarray(x))
    front = []; best = np.inf
    for i in idx:
        if y[i] < best:
            front.append(i); best = y[i]
    return sorted(front, key=lambda i: x[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/phase2/results.jsonl")
    ap.add_argument("--out", default="results/phase2")
    ap.add_argument("--err", default="u_max")
    args = ap.parse_args()
    rows = load(args.inp)
    cases = sorted({r["case"] for r in rows})
    camps = sorted({r["campaign"] for r in rows})
    # --- Pareto per case, colour by campaign -----------------------------
    fig, axs = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4.5), squeeze=False)
    for j, cs in enumerate(cases):
        ax = axs[0, j]
        for cp in camps:
            rr = [r for r in rows if r["case"] == cs and r["campaign"] == cp]
            if not rr:
                continue
            x = [r["elem_saving_scalar_vs_sysala"] for r in rr]
            y = [max(r["errors"][args.err], 1e-16) for r in rr]
            ax.scatter(x, y, label=cp, s=25, alpha=.8)
        rr = [r for r in rows if r["case"] == cs]
        x = np.array([r["elem_saving_scalar_vs_sysala"] for r in rr]); y = np.array([max(r["errors"][args.err], 1e-16) for r in rr])
        f = pareto_front(x, y)
        ax.plot(x[f], y[f], "k--", lw=1, label="Pareto front")
        ax.set_yscale("log"); ax.set_xlabel("element-level cost saved vs Sysala (scalar model)")
        ax.set_ylabel(f"error: {args.err}"); ax.set_title(cs); ax.grid(alpha=.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"pareto_{args.err}.png"), dpi=130); plt.close(fig)

    # --- tables ---------------------------------------------------------------
    lines = ["| campaign | config | case | active | save scalar vs Sysala | save vector | wall vs Sysala | Newton it (ref) | monitored GP/step | u_max | stress_rms_max | alpha_rel_final | reaction_max |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r["errors"]; c = r["counters"]; cr = r["counters_ref_sysala"]
        T = len(r["active_per_step"])
        lines.append(f"| {r['campaign']} | {r['config']['name']} | {r['case']} | {r['active_fraction_mean']:.3f} | {r['elem_saving_scalar_vs_sysala']:.3f} | "
                     f"{r['elem_saving_vector_vs_sysala']:.3f} | {r['wall_saving_vs_sysala']:+.2f} | {c['n_newton_iters']} ({cr['n_newton_iters']}) | "
                     f"{c['n_gp_monitored'] / T:.0f} | {e['u_max']:.1e} | {e['stress_rms_max']:.1e} | {e['alpha_rel_final']:.1e} | {e['reaction_max']:.1e} |")
    open(os.path.join(args.out, "tables.md"), "w").write("\n".join(lines) + "\n")
    # --- best exact configs (error < 1e-10) and best at error < 1e-3 --------
    summ = []
    for cs in cases:
        rr = [r for r in rows if r["case"] == cs]
        for thr, lab in ((1e-10, "exact"), (1e-3, "err<1e-3"), (1e-2, "err<1e-2")):
            ok = [r for r in rr if r["errors"][args.err] < thr]
            if ok:
                b = max(ok, key=lambda r: r["elem_saving_scalar_vs_sysala"])
                summ.append(f"{cs:24s} {lab:9s} best={b['config']['name']:24s} save={b['elem_saving_scalar_vs_sysala']:.3f} (vector {b['elem_saving_vector_vs_sysala']:.3f}) wall={b['wall_saving_vs_sysala']:+.2f} err={b['errors'][args.err]:.1e}")
    print("\n".join(summ))
    open(os.path.join(args.out, "best.txt"), "w").write("\n".join(summ) + "\n")


if __name__ == "__main__":
    main()
