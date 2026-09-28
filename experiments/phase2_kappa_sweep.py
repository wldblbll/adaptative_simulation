"""Exact orchestrator vs reference: sweep of the monitoring safety factor kappa.

Each row of the output JSONL is fully reproducible: it stores the case, the orchestrator
config, the counters, the timers, the unit-cost models and the error metrics against the
reference run (same mesh, steps, integrator; orchestrator off).

The historical campaigns of the exploration (lazy plastic integration, catch-up,
neighbour propagation, post-step monitoring, reversal wake-up) are archived at the git
tag `exploration-complete`; their rows are kept in results/phase2/results.jsonl.
"""
import argparse, json, os, sys, time
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


KAPPAS = (0.0, 0.5, 1.0, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="notched_plate,cantilever,cyclic_notched_kin,cyclic_cantilever_kin")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--out", default="results/phase2/kappa_sweep.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cfgs = [OrchestratorConfig(name=f"exact_k{k}", monitor_skip_kappa=k) for k in KAPPAS]
    with open(args.out, "a") as f:
        for name in args.cases.split(","):
            ref = reference(name, args.h)
            print(f"[{name}] ref naive {ref['naive']['wall']:.1f}s, sysala {ref['sysala']['wall']:.1f}s")
            for cfg in cfgs:
                row = run_config(name, args.h, cfg, ref)
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
