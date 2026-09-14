"""Micro-benchmarks of the per-element / per-Gauss-point costs used to weight the oracle
potential. Two cost models are measured:

* `vector`: this NumPy implementation (batched over all points);
* `scalar`: the same algorithms written point by point in pure Python. Per-operation
  overhead is roughly uniform there, so the elastic/plastic cost *ratio* is a proxy for a
  compiled per-element code (Fortran/C++), where the return mapping with a local Newton
  loop costs several times an elastic trial.

Both are reported; neither is claimed to be the industrial truth.
"""
import time
import numpy as np
from .element import NGP

_PS3 = np.array([0, 1, 3])


def _bench(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def measure_costs(el, mat, n_el=None, seed=0):
    rng = np.random.default_rng(seed)
    n_el = el.nel if n_el is None else min(n_el, el.nel)
    idx = np.arange(n_el)
    ngp = n_el * NGP
    u = rng.normal(0, 1e-4, el.ndof)
    eps_e = rng.normal(0, 3e-4, (ngp, 3))          # elastic strains
    eps_p_ = rng.normal(0, 1, (ngp, 3))            # plastic strains: fixed magnitude, random direction
    eps_p_ *= 6e-3 / np.linalg.norm(eps_p_, axis=1)[:, None]
    ep0 = np.zeros((ngp, 4)); a0 = np.zeros(ngp)
    sig, ep, al, be, pl, D, _ = mat.integrate(eps_p_, ep0, a0)
    assert pl.all(), "benchmark: plastic batch is not fully plastic"
    assert not mat.integrate(eps_e, ep0, a0)[4].any(), "benchmark: elastic batch is plastic"
    out = {}
    # vectorised -----------------------------------------------------------
    t_strain = _bench(lambda: el.strains(u, idx))
    t_el = _bench(lambda: mat.integrate(eps_e, ep0, a0))
    t_pl = _bench(lambda: mat.integrate(eps_p_, ep0, a0))
    Dg = D.reshape(n_el, NGP, 3, 3)
    t_K = _bench(lambda: el.element_K(Dg, idx))
    sg = sig[:, _PS3].reshape(n_el, NGP, 3)
    t_f = _bench(lambda: el.element_fint(sg, idx))
    Ke = el.element_K(Dg, idx); fe = el.element_fint(sg, idx)
    t_asmK = _bench(lambda: el.assemble_K_data(Ke, idx))
    t_asmf = _bench(lambda: el.assemble_fint(fe, idx))
    deps = rng.normal(0, 1e-4, (ngp, 3))
    ue0 = rng.normal(0, 1e-4, (n_el, 8))
    def extrap():
        due = u[el.dofs[idx]] - ue0
        fe + np.einsum("nij,nj->ni", Ke, due)
    t_x = _bench(extrap)
    b0 = np.zeros((ngp, 4))
    def monitor():
        e = el.strains(u, idx).reshape(-1, 3)
        ds = np.einsum("nij,nj->ni", D, e - deps)
        s = sig.copy(); s[:, _PS3] += ds
        mat.yield_function(s, a0, b0)
    t_mon = _bench(monitor)
    out["vector"] = dict(monitor_gp=t_mon / ngp,
        strain_gp=t_strain / ngp, trial_gp=t_el / ngp, return_gp=max(t_pl - t_el, 0) / ngp,
        K_el=t_K / n_el, fint_el=t_f / n_el, asmK_el=t_asmK / n_el, asmf_el=t_asmf / n_el,
        extrap_el=t_x / n_el)
    # scalar ---------------------------------------------------------------
    m = min(400, ngp)
    def scal_el():
        for i in range(m):
            mat.integrate_scalar(eps_e[i], ep0[i], a0[i])
    def scal_pl():
        for i in range(m):
            mat.integrate_scalar(eps_p_[i], ep0[i], a0[i])
    t_sel = _bench(scal_el, 3) / m
    t_spl = _bench(scal_pl, 3) / m
    me = min(200, n_el)
    B = el.B; w = el.wdetJ
    def scal_K():
        for e in range(me):
            K = np.zeros((8, 8))
            for g in range(NGP):
                K += w[e, g] * B[e, g].T @ (Dg[e, g] @ B[e, g])
    def scal_f():
        for e in range(me):
            f = np.zeros(8)
            for g in range(NGP):
                f += w[e, g] * (B[e, g].T @ sg[e, g])
    def scal_strain():
        for e in range(me):
            ue = u[el.dofs[e]]
            for g in range(NGP):
                B[e, g] @ ue
    def scal_x():
        for e in range(me):
            fe[e] + Ke[e] @ (u[el.dofs[e]] - ue0[e])
    def scal_mon():
        for e in range(me):
            ue = u[el.dofs[e]]
            for g in range(NGP):
                i = e * NGP + g
                e_ = B[e, g] @ ue
                s3 = sig[i, _PS3] + Dg[e, g] @ (e_ - deps[i])
                mat.yield_function_scalar([s3[0], s3[1], sig[i, 2], s3[2]], a0[i], b0[i])
    out["scalar"] = dict(monitor_gp=_bench(scal_mon, 3) / (me * NGP),
        strain_gp=_bench(scal_strain, 3) / (me * NGP), trial_gp=t_sel, return_gp=max(t_spl - t_sel, 0),
        K_el=_bench(scal_K, 3) / me, fint_el=_bench(scal_f, 3) / me,
        asmK_el=out["vector"]["asmK_el"], asmf_el=out["vector"]["asmf_el"],
        extrap_el=_bench(scal_x, 3) / me)
    for k, d in out.items():
        d["active_el_elastic"] = NGP * (d["strain_gp"] + d["trial_gp"]) + d["K_el"] + d["fint_el"] + d["asmK_el"] + d["asmf_el"]
        d["active_el_plastic"] = d["active_el_elastic"] + NGP * d["return_gp"]
        d["quiet_el"] = d["extrap_el"] + d["asmf_el"]
        d["monitor_el"] = NGP * d["monitor_gp"]
        d["ratio_plastic_elastic"] = d["active_el_plastic"] / d["active_el_elastic"]
        d["ratio_quiet_elastic"] = d["quiet_el"] / d["active_el_elastic"]
    return out
