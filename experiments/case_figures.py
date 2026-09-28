"""Figures of the simulated parts: geometry, mesh, boundary conditions, load histories,
and simulation results on the mesh (von Mises stress, plastic strain, force-displacement
response, element status of the exact method). Runs the reference and the orchestrated
solver on the cyclic notched plate and the cantilever (~2 min).

Outputs: docs/figures/test_cases.png, docs/figures/notched_plate_results.png,
         docs/figures/cantilever_results.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptfem import cases
from adaptfem.element import NGP
from adaptfem.orchestrator import OrchestratorConfig, HeuristicOrchestrator

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, SURFACE, EDGE, GREY_LIGHT = "#0b0b0b", "#52514e", "#fcfcfb", "#3a3935", "#dddcd6"
# single-hue sequential ramps (light -> dark)
STRESS_CMAP = LinearSegmentedColormap.from_list(
    "stress", ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
PLASTIC_CMAP = LinearSegmentedColormap.from_list(
    "plastic", ["#fdf1ea", "#fbd3bf", "#f6a47e", "#eb6834", "#c24d1c", "#8f3511", "#5e2108"])
STATUS_CMAP = ListedColormap(["#f1f0ec", "#f6b596", BLUE])   # skipped, checked only, integrated

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GREY_LIGHT,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def von_mises(stress):
    """stress (..., 4) = (sxx, syy, szz, sxy) -> von Mises equivalent stress."""
    sx, sy, sz, txy = (stress[..., i] for i in range(4))
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * txy ** 2)


def per_element(gp_field, nel):
    return gp_field.reshape(nel, NGP).mean(1)


def mesh_plot(ax, case, values=None, cmap=None, vmin=None, vmax=None, lw=0.12, edge=EDGE, face="none"):
    verts = case.nodes[case.elems]
    if values is None:
        pc = PolyCollection(verts, facecolors=face, edgecolors=edge, linewidths=lw)
    else:
        pc = PolyCollection(verts, array=values, cmap=cmap, edgecolors="face" if lw == 0 else edge, linewidths=lw)
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_aspect("equal"); ax.autoscale_view()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return pc


def clamp(ax, x, y0, y1, side="left", n=9, d=1.2):
    ax.plot([x, x], [y0, y1], color=INK, lw=2, solid_capstyle="butt", zorder=5)
    s = -1 if side == "left" else 1
    for y in np.linspace(y0, y1, n):
        ax.plot([x, x + s * d], [y, y - d], color=INK, lw=0.9, zorder=5)


def arrows(ax, x, ys, dx, dy, label, lx, ly, **kw):
    for y in ys:
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6, mutation_scale=11), zorder=6)
    ax.text(lx, ly, label, color=INK, fontsize=9, **kw)


def run(case, orchestrated=False):
    m = case.model(record_history=True, reuse_elastic_K=True)
    rec = None
    if orchestrated:
        rec = StatusRecorder(OrchestratorConfig(monitor_skip_kappa=1.0), m)
        m.run(orchestrator=rec)
    else:
        m.run()
    return m, m.stacked_history(), rec


class StatusRecorder(HeuristicOrchestrator):
    """Exact orchestrator that also records which quiet elements were checked each step."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.checked = []

    def select(self, k, u_prev, state_prev, model):
        self._chk = np.zeros(self.nel, bool)
        return super().select(k, u_prev, state_prev, model)

    def monitor(self, k, it, u, state_prev, state, quiet, model):
        w = super().monitor(k, it, u, state_prev, state, quiet, model)
        self._chk[quiet[self.margin_age[quiet] == 0]] = True
        return w

    def after_step(self, k, u, state, rec, model):
        self.checked.append(self._chk.copy())
        super().after_step(k, u, state, rec, model)


# ---------------------------------------------------------------------------------
def fig_test_cases(plate, beam):
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.1, 1], wspace=0.22, hspace=0.3)

    # notched plate
    ax = fig.add_subplot(gs[0, 0])
    mesh_plot(ax, plate, lw=0.15)
    L, H, R = plate.meta["L"], plate.meta["H"], plate.meta["R"]
    clamp(ax, 0, 0, H, "left")
    ax.plot([L, L], [0, H], color=INK, lw=2)
    arrows(ax, L + 0.6, np.linspace(2, H - 2, 5), 3.5, 0, "imposed\nu = λ · 0.06 mm", L + 0.8, H + 0.8)
    ax.text(-0.5, H + 0.8, "clamped", fontsize=9, color=INK)
    ax.set_xlim(-3, L + 9); ax.set_ylim(-1, H + 4.5)
    ax.set_title(f"Notched plate in tension — 40 × 20 mm, notch R = {R:g} mm\n"
                 f"{len(plate.elems)} Q4 elements, {2 * len(plate.nodes)} degrees of freedom", loc="left")
    # zoom on the notch
    axz = fig.add_subplot(gs[0, 1])
    mesh_plot(axz, plate, lw=0.35)
    axz.set_xlim(L / 2 - 8, L / 2 + 8); axz.set_ylim(-0.3, 8)
    axz.set_title("Zoom on the notch: element size 0.5 mm", loc="left")
    ax.add_patch(plt.Rectangle((L / 2 - 8, 0), 16, 8, fill=False, ec=ORANGE, lw=1.2, ls="--", zorder=6))

    # cantilever
    ax = fig.add_subplot(gs[1, 0])
    mesh_plot(ax, beam, lw=0.12)
    Lb, Hb = beam.meta["L"], beam.meta["H"]
    clamp(ax, 0, 0, Hb, "left", n=6)
    arrows(ax, Lb + 1.2, [Hb / 2 - 2], 0, 4, "imposed\nv = λ · 1 mm", Lb + 2.2, Hb / 2 - 1.5)
    ax.set_xlim(-3, Lb + 16); ax.set_ylim(-1.5, Hb + 2)
    ax.set_title(f"Cantilever in bending — 50 × 10 mm, clamped left\n"
                 f"{len(beam.elems)} Q4 elements, {2 * len(beam.nodes)} degrees of freedom", loc="left")

    # load histories
    ax = fig.add_subplot(gs[1, 1])
    mono = np.concatenate([[0], cases.notched_plate().lam_hist])
    cyc = np.concatenate([[0], plate.lam_hist])
    ax.plot(np.arange(len(mono)), mono, color=INK2, lw=2, label="monotonic cases")
    ax.plot(np.arange(len(cyc)), cyc, color=ORANGE, lw=2, label="cyclic cases")
    ax.axhline(0, color=GREY_LIGHT, lw=1)
    ax.set_xlabel("load step"); ax.set_ylabel("load factor λ")
    ax.set_title("Load histories (displacement control)", loc="left")
    ax.grid(alpha=0.4, color=GREY_LIGHT)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.annotate("load", (5, 0.62), fontsize=8.5, color=INK2)
    ax.annotate("unload, then\nreverse to −0.6", (58, 0.2), fontsize=8.5, color=INK2)
    ax.annotate("reload", (106, 0.1), fontsize=8.5, color=INK2)

    fig.suptitle("Test parts — 2D plane strain, steel (E = 200 GPa, ν = 0.3, σ_y = 250 MPa), J2 plasticity with hardening",
                 x=0.02, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=0.995)
    save(fig, "test_cases.png")


def fig_results(case, Href, Hon, rec, k_peak, k_show, disp_label, disp_scale, name, title, zoom=None):
    nel = len(case.elems)
    wide = np.ptp(case.nodes[:, 0]) > 3 * np.ptp(case.nodes[:, 1])
    if wide:
        fig = plt.figure(figsize=(12, 9.0))
        gs0 = fig.add_gridspec(3, 2, height_ratios=[0.62, 0.62, 1.2], wspace=0.15, hspace=0.45)
        pos = (gs0[0, :], gs0[1, :], gs0[2, 0], gs0[2, 1])
    else:
        fig = plt.figure(figsize=(12, 7.4))
        gs0 = fig.add_gridspec(2, 2, wspace=0.12, hspace=0.32)
        pos = (gs0[0, 0], gs0[0, 1], gs0[1, 0], gs0[1, 1])

    def field_ax(pos, vals, cmap, vmin, vmax, ttl, cblabel):
        ax = fig.add_subplot(pos)
        pc = mesh_plot(ax, case, vals, cmap, vmin, vmax, lw=0)
        if zoom:
            ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
        ax.set_title(ttl, loc="left")
        cb = fig.colorbar(pc, ax=ax, orientation="vertical" if wide else "horizontal", fraction=0.05, pad=0.02 if wide else 0.04,
                          aspect=12 if wide else 40, shrink=0.9)
        cb.set_label(cblabel, color=INK2); cb.outline.set_visible(False)
        return ax

    vm = per_element(von_mises(Href["stress"][k_peak]), nel)
    field_ax(pos[0], vm, STRESS_CMAP, 0, np.quantile(vm, 0.995),
             f"von Mises stress at peak load (step {k_peak + 1})", "σ_vM [MPa]")
    alpha = per_element(Href["alpha"][-1], nel)
    field_ax(pos[1], alpha, PLASTIC_CMAP, 0, np.quantile(alpha, 0.995),
             "accumulated plastic strain, end of history", "ε_p accumulated [–]")

    # force-displacement
    ax = fig.add_subplot(pos[2])
    lam = np.concatenate([[0], Href["lam"]])
    R = np.concatenate([[0], Href["reaction"]])
    Ron = np.concatenate([[0], Hon["reaction"]])
    ax.plot(lam * disp_scale, R, color=INK2, lw=2.2, label="reference (all elements integrated)")
    ax.plot(lam * disp_scale, Ron, color=BLUE, lw=0, marker="o", markersize=3.2, markevery=2,
            label="exact method (same curve to 1e-12)")
    ax.scatter([lam[k_show + 1] * disp_scale], [R[k_show + 1]], s=70, facecolor="none", edgecolor=ORANGE, lw=2, zorder=5)
    ax.annotate(f"step {k_show + 1}\n(status map →)", (lam[k_show + 1] * disp_scale, R[k_show + 1]),
                xytext=(-10, -80), textcoords="offset points", ha="right",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), fontsize=8.5, color=INK2)
    ax.axhline(0, color=GREY_LIGHT, lw=1); ax.axvline(0, color=GREY_LIGHT, lw=1)
    ax.set_xlabel(disp_label); ax.set_ylabel("reaction force [N per mm of thickness]")
    ax.set_title("Force–displacement response (hysteresis = plasticity)", loc="left")
    ax.grid(alpha=0.4, color=GREY_LIGHT)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # element status of the exact method at step k_show
    act = Hon["active"][k_show]
    chk = rec.checked[k_show] & ~act
    status = np.where(act, 2, np.where(chk, 1, 0))
    ax = fig.add_subplot(pos[3])
    mesh_plot(ax, case, status, STATUS_CMAP, -0.5, 2.5, lw=0)
    if zoom:
        ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
    n_act, n_chk = int(act.sum()), int(chk.sum())
    ax.set_title(f"What the exact method computes at step {k_show + 1}", loc="left")
    ax.legend(handles=[Patch(color=BLUE, label=f"integrated (plastic): {n_act} el. ({100 * n_act / nel:.0f} %)"),
                       Patch(color="#f6b596", label=f"elastic, checked only: {n_chk} ({100 * n_chk / nel:.0f} %)"),
                       Patch(color="#f1f0ec", label=f"elastic, skipped: {nel - n_act - n_chk} ({100 * (nel - n_act - n_chk) / nel:.0f} %)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, fontsize=8.5)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11.5, fontweight="bold", color=INK, y=0.98)
    save(fig, name)


if __name__ == "__main__":
    plate = cases.cyclic_notched_kin(h=0.5)
    beam = cases.cyclic_cantilever_kin(h=0.5)
    fig_test_cases(plate, beam)

    for case, disp_label, scale, name, title in (
            (plate, "imposed displacement of the right edge [mm]", plate.meta["u_max"], "notched_plate_results.png",
             "Cyclic notched plate: plasticity starts at the notch root and spreads in bands to the clamped corners"),
            (beam, "imposed tip deflection [mm]", beam.meta["v_max"], "cantilever_results.png",
             "Cyclic cantilever: plasticity stays in the outer fibres near the clamp, the rest of the beam stays elastic")):
        m_ref, Href, _ = run(case)
        m_on, Hon, rec = run(case, orchestrated=True)
        du = np.abs(Hon["u"] - Href["u"]).max() / np.abs(Href["u"]).max()
        lam = Href["lam"]
        k_peak = int(np.argmax(lam))
        # a reloading step where the plastic zone is waking up again
        rel = np.nonzero((np.diff(lam, prepend=lam[0]) > 0) & (np.arange(len(lam)) > np.argmin(lam)))[0]
        pl_frac = Hon["active"].mean(1)
        cand = [k for k in rel if 0.02 < pl_frac[k] < 0.5]
        k_show = cand[len(cand) // 2] if cand else rel[len(rel) // 2]
        print(f"{case.name}: max rel. displacement diff = {du:.1e}, peak step {k_peak + 1}, shown step {k_show + 1}")
        zoom = ((case.meta["L"] / 2 - 12, case.meta["L"] / 2 + 12), (-0.5, case.meta["H"])) if "notch" in case.meta else None
        fig_results(case, Href, Hon, rec, k_peak, k_show, disp_label, scale, name, title, zoom=None)
