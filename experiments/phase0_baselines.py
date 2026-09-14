"""Compare the oracle potential against the known baseline that already reuses the
elastic stiffness of elastic elements (Cermak/Sysala/Valdman): how much is *additional*?"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptfem import cases
from adaptfem.oracle import oracle_greedy
from adaptfem.element import NGP

summ = json.load(open("results/phase0/summary.json"))
rows = []
for name, s in summ.items():
    h = dict(np.load(f"results/phase0/{name}_history.npz"))
    c = cases.CASES[name](h=0.5)
    nel = s["nel"]
    n_iter = h["n_iter"] + 1
    pl = h["gp_plastic"].astype(float) / NGP           # fraction of plastic GPs per element
    out = dict(case=name)
    for cm in ("scalar", "vector"):
        k = s["costs"][cm]
        gp = NGP * (k["strain_gp"] + k["trial_gp"])
        c_naive = gp + k["K_el"] + k["fint_el"] + k["asmK_el"] + k["asmf_el"] + pl * NGP * k["return_gp"]
        # Sysala baseline: elastic elements (no plastic GP) skip K and its assembly
        el_only = (pl == 0)
        c_sys = np.where(el_only, gp + k["fint_el"] + k["asmf_el"], c_naive)
        C_naive = (n_iter[:, None] * c_naive).sum(); C_sys = (n_iter[:, None] * c_sys).sum()
        out[f"{cm}_sysala_vs_naive"] = 1 - C_sys / C_naive
        for tol in (0.002, 0.01, 0.05):
            integ = oracle_greedy(h, c.material, nel, tol, "tangent")["integrated"]
            c_orc = np.where(integ, c_sys, k["quiet_el"])   # orchestrated on top of Sysala
            C_orc = (n_iter[:, None] * c_orc).sum()
            out[f"{cm}_oracle_vs_naive_{tol}"] = 1 - C_orc / C_naive
            out[f"{cm}_oracle_vs_sysala_{tol}"] = 1 - C_orc / C_sys
    out["elem_share"] = s["element_level_share"]
    rows.append(out)
    print(f"{name:18s} scalar: sysala/naive={out['scalar_sysala_vs_naive']:.3f} | oracle vs naive "
          + " ".join(f"{t}:{out[f'scalar_oracle_vs_naive_{t}']:.3f}" for t in (0.002, 0.01, 0.05))
          + " | vs sysala " + " ".join(f"{t}:{out[f'scalar_oracle_vs_sysala_{t}']:.3f}" for t in (0.002, 0.01, 0.05))
          + f" || vector: sysala/naive={out['vector_sysala_vs_naive']:.3f} oracle vs sysala 0.01: {out['vector_oracle_vs_sysala_0.01']:.3f}"
          + f" | elem share {out['elem_share']:.2f}")
json.dump(rows, open("results/phase0/baselines.json", "w"), indent=1)
