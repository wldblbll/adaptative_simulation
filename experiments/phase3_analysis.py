"""Aggregate Phase 3 results: per case, saving vs error for heuristic / learned / oracle
policies (mean and std over GBM seeds), prediction quality tables."""
import json, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/phase3/results.jsonl")
    ap.add_argument("--out", default="results/phase3")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp)]
    rows = [r for r in rows if r.get("ok")]
    cases = []
    for r in rows:
        if r["case"] not in cases:
            cases.append(r["case"])
    lines = ["| case | group | policy | setting | save scalar (mean±std) | save w/o decision cost | monitored GP/step | Newton it (ref) | u_max | alpha_rel_final |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    best = []
    fig, axs = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.2), squeeze=False)
    for j, cs in enumerate(cases):
        ax = axs[0, j]
        rr = [r for r in rows if r["case"] == cs]
        groups = {}
        for r in rr:
            key = (r["policy"], json.dumps({k: v for k, v in r["extra"].items()}, sort_keys=True))
            groups.setdefault(key, []).append(r)
        for (pol, ex), g in groups.items():
            sv = np.array([x["save_scalar"] for x in g]); sv0 = np.array([x["save_scalar_nodecision"] for x in g])
            eu = np.array([x["errors"]["u_max"] for x in g]); ea = np.array([x["errors"]["alpha_rel_final"] for x in g])
            mon = np.mean([x["monitored_gp_per_step"] for x in g]); it = np.mean([x["newton_iters"] for x in g])
            lines.append(f"| {cs} | {g[0]['group']} | {pol} | {ex} | {sv.mean():.3f}±{sv.std():.3f} | {sv0.mean():.3f} | {mon:.0f} | {it:.0f} ({g[0]['newton_iters_ref']}) | {eu.max():.1e} | {ea.max():.1e} |")
            mk = {"kappa": "o", "learned": "s", "oracle": "^"}[pol]
            col = {"kappa": "C0", "oracle": "C3"}.get(pol, "C2" if "gbm" in ex else "C1")
            ax.errorbar(sv.mean(), max(eu.max(), 1e-15), xerr=sv.std(), fmt=mk, color=col, ms=6, alpha=.85)
        ax.set_yscale("log"); ax.set_xlabel("element cost saved vs Sysala (scalar, incl. decision cost)")
        ax.set_ylabel("max relative displacement error"); ax.set_title(cs, fontsize=9); ax.grid(alpha=.3)
        # best exact per policy
        for pol in ("kappa", "learned", "oracle"):
            ok = [(k, g) for k, g in groups.items() if k[0] == pol and max(x["errors"]["u_max"] for x in g) < 1e-9]
            if ok:
                k, g = max(ok, key=lambda kg: np.mean([x["save_scalar"] for x in kg[1]]))
                best.append(f"{cs:26s} {pol:8s} exact best {k[1]:40s} save={np.mean([x['save_scalar'] for x in g]):.3f}±{np.std([x['save_scalar'] for x in g]):.3f}")
            ok = [(k, g) for k, g in groups.items() if k[0] == pol and max(x["errors"]["u_max"] for x in g) < 1e-3]
            if ok:
                k, g = max(ok, key=lambda kg: np.mean([x["save_scalar"] for x in kg[1]]))
                best.append(f"{cs:26s} {pol:8s} err<1e-3 best {k[1]:40s} save={np.mean([x['save_scalar'] for x in g]):.3f}±{np.std([x['save_scalar'] for x in g]):.3f} u={max(x['errors']['u_max'] for x in g):.1e}")
    from matplotlib.lines import Line2D
    axs[0, 0].legend(handles=[Line2D([], [], marker="o", color="C0", ls="", label="heuristic (kappa sweep)"),
                              Line2D([], [], marker="s", color="C2", ls="", label="learned GBM (safety sweep)"),
                              Line2D([], [], marker="s", color="C1", ls="", label="learned linear"),
                              Line2D([], [], marker="^", color="C3", ls="", label="oracle horizon (upper bound)")], fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "phase3_pareto.png"), dpi=130); plt.close(fig)
    open(os.path.join(args.out, "tables.md"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(args.out, "best.txt"), "w").write("\n".join(best) + "\n")
    print("\n".join(best))
    # prediction quality
    pq = {}
    for r in rows:
        pq[r["case"]] = r["prediction_quality"]
    for cs, q in pq.items():
        print(cs, {k: (round(v["mae"], 2), round(v["unsafe"], 3)) for k, v in q.items()})


if __name__ == "__main__":
    main()
