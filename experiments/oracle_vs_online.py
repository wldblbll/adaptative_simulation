"""Upper bounds vs achieved online gains, same cost model (scalar unit costs measured on
the final implementation). Bounds: (a) ideal exact policy: an element is integrated iff
it is plastic at that step, perfect knowledge, no monitoring cost; (b) Phase 0 oracle
(tangent extrapolation, greedy, tol 1%), plastic elements included, no monitoring cost.
Achieved: exact heuristic (kappa=1) and oracle-horizon monitoring, from Phase 2/3 rows."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptfem import cases
from adaptfem.oracle import oracle_greedy
from adaptfem.benchmark import measure_costs
from adaptfem.element import NGP

rows2 = [json.loads(l) for l in open("results/phase2/results.jsonl")]
out = {}
for name in ("notched_plate", "cantilever", "cyclic_notched_kin", "cyclic_cantilever_kin"):
    c = cases.CASES[name](h=0.5)
    m = c.model(record_history=True, reuse_elastic_K=True); m.run(); H = m.stacked_history()
    u = measure_costs(m.el, c.material)["scalar"]
    n_iter = H["n_iter"] + 1
    pl = H["gp_plastic"] > 0
    gp_pl = H["gp_plastic"].astype(float)
    c_el = NGP * (u["strain_gp"] + u["trial_gp"]) + u["fint_el"] + u["asmf_el"]          # Sysala elastic element
    c_pl = c_el + u["K_el"] + u["asmK_el"] + gp_pl * u["return_gp"]                      # plastic element
    c_sys = np.where(pl, c_pl, c_el)
    C_sys = (n_iter[:, None] * c_sys).sum()
    C_ideal = (n_iter[:, None] * np.where(pl, c_pl, u["quiet_el"])).sum()
    integ = oracle_greedy(H, c.material, m.nel, 0.01, "tangent")["integrated"]
    C_or = (n_iter[:, None] * np.where(integ, c_sys, u["quiet_el"])).sum()
    integ2 = oracle_greedy(H, c.material, m.nel, 0.002, "tangent")["integrated"]
    C_or2 = (n_iter[:, None] * np.where(integ2, c_sys, u["quiet_el"])).sum()
    r_exact = [r for r in rows2 if r["case"] == name and r["config"]["name"] == "mon_iteration_k1.0"][0]
    out[name] = dict(ideal_exact=1 - C_ideal / C_sys, oracle_tangent_1pct=1 - C_or / C_sys, oracle_tangent_0p2pct=1 - C_or2 / C_sys,
                     online_exact_kappa1=r_exact["elem_saving_scalar_vs_sysala"], plastic_frac_mean=float(pl.mean()),
                     plastic_cost_share=float((n_iter[:, None] * np.where(pl, c_pl, 0)).sum() / C_sys))
    print(f"{name:24s} plastic frac {pl.mean():.3f} (cost share {out[name]['plastic_cost_share']:.2f}) | ideal exact {out[name]['ideal_exact']:.3f} | "
          f"oracle tangent 1% {out[name]['oracle_tangent_1pct']:.3f} (0.2%: {out[name]['oracle_tangent_0p2pct']:.3f}) | online exact kappa=1 {out[name]['online_exact_kappa1']:.3f}")
json.dump(out, open("results/phase2/bounds_vs_online.json", "w"), indent=1)
