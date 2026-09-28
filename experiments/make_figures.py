"""Figures of the README, built only from the stored results (no simulation is re-run).

Inputs : results/phase2/bounds_vs_online.json, results/phase2/results.jsonl,
         results/phase3/results_all.jsonl
Outputs: docs/figures/*.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

# validated categorical slots (light surface) + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
GREY_DARK, GREY_LIGHT = "#6f6e69", "#c9c8c2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GREY_LIGHT, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})

CASES = ["notched_plate", "cantilever", "cyclic_notched_kin", "cyclic_cantilever_kin"]
LABEL = {"notched_plate": "Notched plate", "cantilever": "Cantilever",
         "cyclic_notched_kin": "Notched plate, cyclic", "cyclic_cantilever_kin": "Cantilever, cyclic",
         "notched_R4_H20_L40": "Notched plate R4 H20\n(unseen family member)",
         "notched_R3.5_H22_L40": "Notched plate R3.5 H22\n(unseen family member)"}
ERR_FLOOR = 1e-14


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def rows(path):
    return [json.loads(l) for l in open(path)]


# 1. Headline: achieved vs ceilings ---------------------------------------------------
def fig_headline():
    b = json.load(open("results/phase2/bounds_vs_online.json"))
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(CASES))[::-1]
    for yi, cs in zip(y, CASES):
        v = b[cs]
        a, i, o = 100 * v["online_exact_kappa1"], 100 * v["ideal_exact"], 100 * v["oracle_tangent_1pct"]
        ax.barh(yi, a, height=0.5, color=BLUE, zorder=2)
        ax.barh(yi, i - a, left=a, height=0.5, color=BLUE, alpha=0.22, zorder=2)
        ax.plot([i, o], [yi, yi], color=GREY_DARK, lw=1.5, ls=(0, (2, 2)), zorder=1)
        ax.scatter([o], [yi], s=60, marker="o", facecolor=SURFACE, edgecolor=GREY_DARK, linewidths=1.8, zorder=3)
        ax.text(a - 1.5, yi, f"{a:.0f} %", ha="right", va="center", color="white", fontsize=9.5, fontweight="bold", zorder=4)
        ax.text(i + 1, yi + 0.33, f"{i:.0f}", ha="left", va="center", color=INK2, fontsize=8.5)
        ax.text(o, yi + 0.33, f"{o:.0f}", ha="center", va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(y, [LABEL[c] for c in CASES])
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(CASES) - 0.4)
    ax.set_xlabel("element-level cost saved vs. an already-optimised reference (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="achieved online, exact to 1e-12"),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.22, label="ceiling of any exact policy (perfect knowledge, free checks)"),
               Line2D([], [], color=GREY_DARK, marker="o", mfc=SURFACE, lw=0, label="oracle that also extrapolates plastic zones (not reachable online)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=9)
    ax.set_title("Skipping elastic elements saves about half of the element work, with no loss of accuracy", loc="left")
    save(fig, "headline_savings.png")


# 2. Wall-time decomposition (Amdahl) -------------------------------------------------
def fig_walltime():
    r2 = [r for r in rows("results/phase2/results.jsonl") if r.get("ok") and r["config"]["name"] == "mon_iteration_k1.0"]
    by = {r["case"]: r for r in r2}
    parts = [("element-level work", ("strain", "constitutive", "element_K", "element_fint", "assembly"), BLUE),
             ("selection + checks", ("orchestrator", "extrapolation"), ORANGE),
             ("linear solver", ("factorization", "solve"), GREY_DARK),
             ("other", None, GREY_LIGHT)]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ylab, ypos = [], []
    for j, cs in enumerate(CASES):
        r = by[cs]
        for k, (lab, ctr) in enumerate((("reference", r["counters_ref_sysala"]), ("exact method", r["counters"]))):
            tot_ref = r["counters_ref_sysala"]["t_total"]
            yv = -(j * 2.7 + k)
            left = 0.0
            known = 0.0
            for name, keys, col in parts:
                if keys is None:
                    w = ctr["t_total"] - known
                else:
                    w = sum(ctr[f"t_{x}"] for x in keys); known += w
                w = 100 * w / tot_ref
                ax.barh(yv, w, left=left, height=0.8, color=col, edgecolor=SURFACE, linewidth=2)
                if name == "element-level work":
                    ax.text(left + w / 2, yv, f"{w:.0f}", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
                left += w
            ax.text(left + 1, yv, f"{left:.0f} %", va="center", color=INK2, fontsize=8.5)
            ylab.append(f"{LABEL[cs]} — {lab}" if k == 0 else "exact method"); ypos.append(yv)
    ax.set_yticks(ypos, ylab, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 112)
    ax.set_xlabel("wall-clock time, % of the reference run (Python prototype)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts],
              labels=[p[0] for p in parts], loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=4, fontsize=9)
    ax.set_title("…but total time barely moves: the linear solver, untouched, dominates", loc="left")
    save(fig, "walltime_breakdown.png")


# 3. Map of everything explored -------------------------------------------------------
def family(cfg):
    if cfg.get("plastic_policy") == "drift":
        return "lazy"
    k = cfg.get("monitor_skip_kappa", 2.0)
    if cfg.get("monitor_when") in ("step", "predictor_only") or 0 < k < 1:
        return "relaxed"
    return "exact"


FAM = {"exact": ("exact check of elastic elements, every Newton iteration", BLUE, "o"),
       "relaxed": ("relaxed check (post-step, predictor only, or κ < 1)", ORANGE, "^"),
       "lazy": ("plastic elements also extrapolated (lazy integration)", AQUA, "s")}


def fig_exploration():
    r2 = [r for r in rows("results/phase2/results.jsonl") if r.get("ok")]
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True, sharey=True)
    for ax, cs in zip(axs.ravel(), CASES):
        ax.axhspan(ERR_FLOOR / 3, 1e-10, color=BLUE, alpha=0.06, lw=0)
        ax.text(0.02, 2e-13, "exact", color=BLUE, fontsize=8.5, transform=ax.get_yaxis_transform())
        for f, (lab, col, mk) in FAM.items():
            rr = [r for r in r2 if r["case"] == cs and family(r["config"]) == f]
            x = [100 * r["elem_saving_scalar_vs_sysala"] for r in rr]
            y = [max(r["errors"]["u_max"], ERR_FLOOR) for r in rr]
            ax.scatter(x, y, s=34, marker=mk, color=col, edgecolor=SURFACE, linewidths=0.8, alpha=0.9, zorder=3)
        best = [r for r in r2 if r["case"] == cs and r["config"]["name"] == "mon_iteration_k1.0"][0]
        bx, byv = 100 * best["elem_saving_scalar_vs_sysala"], max(best["errors"]["u_max"], ERR_FLOOR)
        ax.scatter([bx], [byv], s=150, marker="o", facecolor="none", edgecolor=INK, linewidths=1.6, zorder=4)
        ax.annotate("retained (κ = 1)", (bx, byv), xytext=(bx - 34, 1e-8), fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_yscale("log"); ax.set_ylim(ERR_FLOOR / 3, 30); ax.set_xlim(-5, 80)
        ax.set_title(LABEL[cs], loc="left", fontsize=10)
    for ax in axs[1]:
        ax.set_xlabel("element-level cost saved (%)")
    for ax in axs[:, 0]:
        ax.set_ylabel("max. relative displacement error")
    fig.legend(handles=[Line2D([], [], marker=m, color=c, lw=0, markersize=7, label=l) for l, c, m in FAM.values()],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=1, fontsize=9)
    fig.suptitle("~240 configurations explored: only the exact check is both accurate and cheap;\n"
                 "anything that extrapolates plasticity buys little and costs 1e-3 to 600 % error",
                 x=0.02, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, "exploration_map.png")


# 4. Learned policy vs heuristic vs oracle --------------------------------------------
def fig_learning():
    r3 = [r for r in rows("results/phase3/results_all.jsonl") if r.get("ok") and r.get("policy")]
    cases = ["notched_R4_H20_L40", "notched_R3.5_H22_L40", "cantilever", "cyclic_notched_kin", "cyclic_cantilever_kin"]
    fig, axs = plt.subplots(1, 5, figsize=(13.5, 3.8), sharey=True)
    for ax, cs in zip(axs, cases):
        ax.axhspan(ERR_FLOOR / 3, 1e-10, color=BLUE, alpha=0.06, lw=0)
        L = [r for r in r3 if r["case"] == cs and r["policy"] == "learned"]
        ax.scatter([100 * r["save_scalar"] for r in L], [max(r["errors"]["u_max"], ERR_FLOOR) for r in L],
                   s=26, color=ORANGE, marker="^", edgecolor=SURFACE, linewidths=0.6, zorder=3)
        K = [r for r in r3 if r["case"] == cs and r["policy"] == "kappa" and r["extra"].get("monitor_skip_kappa") == 1.0]
        O = [r for r in r3 if r["case"] == cs and r["policy"] == "oracle" and r["extra"].get("horizon_safety") == 1.0]
        for R, col, mk, sz in ((K, BLUE, "o", 70), (O, AQUA, "D", 55)):
            ax.scatter([100 * np.mean([r["save_scalar"] for r in R])], [max(np.mean([r["errors"]["u_max"] for r in R]), ERR_FLOOR)],
                       s=sz, color=col, marker=mk, edgecolor=SURFACE, linewidths=1.2, zorder=4)
        ax.set_yscale("log"); ax.set_ylim(ERR_FLOOR / 3, 3)
        ax.set_xlim(35, 78)
        ax.set_title(LABEL[cs], loc="left", fontsize=9.5)
        ax.set_xlabel("cost saved (%)")
    axs[0].set_ylabel("max. relative displacement error")
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, lw=0, markersize=8, label="physics heuristic (κ = 1)"),
                        Line2D([], [], marker="D", color=AQUA, lw=0, markersize=7, label="perfect anticipation (oracle horizon)"),
                        Line2D([], [], marker="^", color=ORANGE, lw=0, markersize=7, label="learned policies (gradient boosting, 24 configs)")],
               loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    fig.suptitle("Machine learning predicts yielding better, but never stays exact; perfect anticipation would add only 3–12 points",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "learning_vs_heuristic.png")


if __name__ == "__main__":
    fig_headline()
    fig_walltime()
    fig_exploration()
    fig_learning()
