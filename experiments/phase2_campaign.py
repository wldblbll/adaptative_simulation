"""Phase 2 campaign runner: heuristic orchestrator configurations x cases.

Each row of the output JSONL is fully reproducible: it stores the case, the orchestrator
config, the counters, the timers, the unit-cost models and the error metrics against the
reference run (same mesh, steps, integrator; orchestrator off).
"""
import argparse, json, os, sys, time, itertools, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptfem import cases
from adaptfem.orchestrator import OrchestratorConfig, HeuristicOrchestrator
from adaptfem.metrics import errors, element_cost
from adaptfem.benchmark import measure_costs

_REF = {}


def reference(name, h):
    key = (name, h)
    if key not in _REF:
        c = cases.CASES[name](h=h)
        res = {}
        for mode in ("naive", "sysala"):
            m = c.model(record_history=True, reuse_elastic_K=(mode == "sysala"))
            t0 = time.perf_counter(); m.run(); t = time.perf_counter() - t0
            res[mode] = dict(model=m, hist=m.stacked_history(), wall=t, cost=m.cost)
        res["unit"] = measure_costs(res["naive"]["model"].el, c.material)
        res["case"] = c
        _REF[key] = res
    return _REF[key]


def run_config(name, h, cfg: OrchestratorConfig, ref=None):
    ref = ref or reference(name, h)
    c = ref["case"]
    m = c.model(record_history=True, reuse_elastic_K=True)
    orch = HeuristicOrchestrator(cfg, m)
    t0 = time.perf_counter()
    try:
        m.run(orchestrator=orch)
        ok = True; err_msg = ""
    except RuntimeError as e:
        ok = False; err_msg = str(e)[:200]
    wall = time.perf_counter() - t0
    row = dict(case=name, h=h, config=cfg.to_dict(), ok=ok, error=err_msg, wall=wall,
               wall_ref_naive=ref["naive"]["wall"], wall_ref_sysala=ref["sysala"]["wall"],
               counters=m.cost.as_dict(), counters_ref_naive=ref["naive"]["cost"].as_dict(),
               counters_ref_sysala=ref["sysala"]["cost"].as_dict())
    if ok:
        hist = m.stacked_history()
        e = errors(hist, ref["sysala"]["hist"], m, c.material.sig_y0)
        row["errors"] = {k: v for k, v in e.items() if k != "per_step"}
        row["active_fraction_mean"] = float(np.mean([r.active_fraction for r in m.records]))
        row["active_per_step"] = [float(r.active_fraction) for r in m.records]
        for cm in ("scalar", "vector"):
            u = ref["unit"][cm]
            co = element_cost(m.cost, u)
            row[f"elem_cost_{cm}"] = co
            row[f"elem_saving_{cm}_vs_naive"] = 1 - co / element_cost(ref["naive"]["cost"], u)
            row[f"elem_saving_{cm}_vs_sysala"] = 1 - co / element_cost(ref["sysala"]["cost"], u)
        row["wall_saving_vs_sysala"] = 1 - wall / ref["sysala"]["wall"]
    return row


def grid(**kw):
    keys = list(kw)
    for vals in itertools.product(*kw.values()):
        yield dict(zip(keys, vals))


CAMPAIGNS = {
    "smoke": lambda: [dict(name="default")],
    # A/E: monitoring strategy for elastic-anchored elements (where, how often, how safe)
    "A_monitor": lambda: [dict(name=f"mon_{w}_k{k}", monitor_when=w, monitor_skip_kappa=k)
                          for w in ("iteration", "predictor", "predictor_only", "converged", "step")
                          for k in (0.0, 0.5, 1.0, 2.0, 5.0)],
    # B: plastic-anchored elements: keep active vs lazy integration with drift tolerance
    "B_plastic": lambda: [dict(name="pl_always", plastic_policy="always_active")]
                       + [dict(name=f"pl_drift_{t}", plastic_policy="drift", tol_alpha=t) for t in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)]
                       + [dict(name=f"pl_drift_{t}_nopred", plastic_policy="drift", tol_alpha=t, plastic_predict=False) for t in (1e-4, 1e-3)]
                       + [dict(name=f"pl_drift_{t}_nounload", plastic_policy="drift", tol_alpha=t, unload_wake=False) for t in (1e-4, 1e-3)],
    # D: catch-up policies (only matter with lazy plastic integration or lagged monitoring)
    "D_catchup": lambda: [dict(name=f"D_step_catch{n}", monitor_when="step", catchup_every=n) for n in (0, 5, 10, 20)]
                       + [dict(name=f"D_drift_catch{n}", plastic_policy="drift", tol_alpha=1e-3, catchup_every=n) for n in (0, 5, 10, 20)]
                       + [dict(name=f"D_drift_skip{n}", plastic_policy="drift", tol_alpha=1e-3, max_skip=n) for n in (3, 10)],
    # C: spatial granularity / neighbour propagation, precautionary margin
    "C_spatial": lambda: [dict(name=f"C_nb{l}", neighbor_layers=l) for l in (0, 1, 2)]
                       + [dict(name=f"C_margin{m}", yield_margin=m) for m in (0.02, 0.1)]
                       + [dict(name=f"C_step_margin{m}", monitor_when="step", yield_margin=m) for m in (0.02, 0.05, 0.1, 0.2)],
    # E: wake-up on reversal (cyclic cases)
    "E_reversal": lambda: [dict(name="E_base"),
                           dict(name="E_global_reversal", global_reversal_wake=True),
                           dict(name="E_step", monitor_when="step"),
                           dict(name="E_step_global", monitor_when="step", global_reversal_wake=True),
                           dict(name="E_drift_1e-3", plastic_policy="drift", tol_alpha=1e-3),
                           dict(name="E_drift_1e-3_noun", plastic_policy="drift", tol_alpha=1e-3, unload_wake=False),
                           dict(name="E_drift_1e-3_global", plastic_policy="drift", tol_alpha=1e-3, global_reversal_wake=True)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="smoke")
    ap.add_argument("--cases", default="notched_plate,cantilever,cyclic_notched_kin,cyclic_cantilever_kin")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--out", default="results/phase2/results.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cfgs = [OrchestratorConfig(**d) for d in CAMPAIGNS[args.campaign]()]
    with open(args.out, "a") as f:
        for name in args.cases.split(","):
            ref = reference(name, args.h)
            print(f"[{name}] ref naive {ref['naive']['wall']:.1f}s, sysala {ref['sysala']['wall']:.1f}s")
            for cfg in cfgs:
                row = run_config(name, args.h, cfg, ref)
                row["campaign"] = args.campaign
                f.write(json.dumps(row) + "\n"); f.flush()
                if row["ok"]:
                    e = row["errors"]
                    print(f"  {cfg.name:22s} act={row['active_fraction_mean']:.3f} "
                          f"save(scalar) vs sysala={row['elem_saving_scalar_vs_sysala']:.3f} vs naive={row['elem_saving_scalar_vs_naive']:.3f} "
                          f"wall={row['wall_saving_vs_sysala']:+.2f} it={row['counters']['n_newton_iters']} "
                          f"| err u={e['u_max']:.1e} sig_rms={e['stress_rms_max']:.1e} alpha_rel_final={e['alpha_rel_final']:.1e} R={e['reaction_max']:.1e}")
                else:
                    print(f"  {cfg.name:22s} FAILED: {row['error']}")


if __name__ == "__main__":
    main()
