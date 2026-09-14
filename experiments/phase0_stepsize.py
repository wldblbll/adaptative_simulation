"""Step-size sensitivity of the oracle potential: is the potential a spatial effect
(heterogeneous activity) or just 'the steps are small'?"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptfem import cases
from adaptfem.oracle import oracle_greedy, cost_weighted_potential
from adaptfem.benchmark import measure_costs

out = {}
for name in ("notched_plate", "cantilever"):
    out[name] = {}
    for n in (10, 20, 40, 60, 120):
        c = cases.CASES[name](h=0.5, n_steps=n)
        m = c.model(record_history=True)
        m.run()
        h = m.stacked_history()
        costs = measure_costs(m.el, c.material)["scalar"]
        row = dict(iters_per_step=m.cost.c["newton_iters"] / n, reaction_final=float(h["reaction"][-1]))
        for tol in (0.002, 0.01, 0.05):
            o = oracle_greedy(h, c.material, m.nel, tol, "tangent")
            integ = o["integrated"]
            freq = integ.mean(0)                      # per-element integration frequency
            pl = h["gp_plastic"] > 0
            row[str(tol)] = dict(raw=float(1 - integ.mean()),
                                 plastic_el_steps_extrap=float((~integ & pl).sum() / max(pl.sum(), 1)),
                                 cost_scalar=float(cost_weighted_potential(integ, h, costs)[0]),
                                 freq_p50=float(np.median(freq)), freq_p90=float(np.quantile(freq, .9)),
                                 freq_max=float(freq.max()), n_el_always=int((freq > 0.99).sum()))
        out[name][str(n)] = row
        r = row["0.01"]
        print(f"{name} n_steps={n:4d} it/step={row['iters_per_step']:.2f} R={row['reaction_final']:.1f} | tol=0.01: raw={r['raw']:.3f} "
              f"plastic-extrap={r['plastic_el_steps_extrap']:.3f} cost={r['cost_scalar']:.3f} freq p50/p90/max={r['freq_p50']:.2f}/{r['freq_p90']:.2f}/{r['freq_max']:.2f} always={r['n_el_always']}")
json.dump(out, open("results/phase0/stepsize.json", "w"), indent=1)
