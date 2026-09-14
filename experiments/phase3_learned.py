"""Phase 3: learned monitoring policy vs tuned heuristic vs oracle horizon.

H1 (anticipation): the learned horizon predictor anticipates yielding from features
available online; the heuristic uses margin / rate. Both deployed identically.
H3 (family specialisation): train on a parametrised family of notched plates, evaluate
on unseen members of the family, then on other geometries (cantilever, cyclic).
Everything is trained from reference simulations only (the solver is its own teacher).
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptfem import cases
from adaptfem.learning import extract_dataset, HorizonModel, OracleHorizon, FEATURES
from adaptfem.orchestrator import OrchestratorConfig, HeuristicOrchestrator
from adaptfem.metrics import errors, element_cost
from adaptfem.benchmark import measure_costs


def run_ref(c):
    m = c.model(record_history=True, reuse_elastic_K=True)
    t0 = time.perf_counter(); m.run(); wall = time.perf_counter() - t0
    return m, m.stacked_history(), wall


def evaluate(c, ref, H, unit, policy, hm, extra, seed=None):
    m = c.model(record_history=True, reuse_elastic_K=True)
    o = HeuristicOrchestrator(OrchestratorConfig(monitor_policy=policy, **extra), m, horizon_model=hm)
    t0 = time.perf_counter()
    try:
        m.run(orchestrator=o); ok = True
    except RuntimeError as ex:
        return dict(ok=False, error=str(ex)[:100])
    wall = time.perf_counter() - t0
    e = errors(m.stacked_history(), H, m, c.material.sig_y0)
    T = len(c.lam_hist)
    row = dict(ok=True, case=c.name, policy=policy, extra=extra, seed=seed, wall=wall, wall_ref=ref["wall"],
               t_policy=o.t_policy, newton_iters=m.cost.c["newton_iters"], newton_iters_ref=ref["model"].cost.c["newton_iters"],
               monitored_gp_per_step=m.cost.c["gp_monitored"] / T, decisions=m.cost.c["orchestrator_decisions"],
               active_mean=float(np.mean([r.active_fraction for r in m.records])),
               errors={k: v for k, v in e.items() if k != "per_step"})
    for cm in ("scalar", "vector"):
        row[f"save_{cm}"] = 1 - element_cost(m.cost, unit[cm], o.t_policy) / element_cost(ref["model"].cost, unit[cm])
        row[f"save_{cm}_nodecision"] = 1 - element_cost(m.cost, unit[cm]) / element_cost(ref["model"].cost, unit[cm])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--out", default="results/phase3/results.jsonl")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    h = args.h
    # --- training family: notched plates, unseen test members --------------
    train_cases = [cases.notched_family(R=R, H=Hh, h=h) for R, Hh in ((3.0, 20.0), (5.0, 20.0), (4.0, 16.0), (4.0, 24.0), (2.5, 18.0))]
    test_family = [cases.notched_family(R=4.0, H=20.0, h=h), cases.notched_family(R=3.5, H=22.0, h=h)]
    other = [cases.cantilever(h=h), cases.cyclic_notched_kin(h=h), cases.cyclic_cantilever_kin(h=h)]
    Xs, ys = [], []
    t0 = time.perf_counter()
    for c in train_cases:
        m, Hh, _ = run_ref(c)
        X, y, _ = extract_dataset(Hh, c, c.elems, c.material); Xs.append(X); ys.append(y)
        print(f"train case {c.name}: {X.shape[0]} samples, {(y < 16).mean():.3f} near-yield")
    X = np.vstack(Xs); y = np.concatenate(ys)
    print(f"training set {X.shape} ({time.perf_counter() - t0:.0f}s)")
    models = {}
    for seed in range(args.seeds):
        t0 = time.perf_counter(); models[("gbm", seed)] = HorizonModel("gbm", seed).fit(X, y)
        print(f"gbm seed {seed} fit {time.perf_counter() - t0:.1f}s")
    models[("linear", 0)] = HorizonModel("linear").fit(X, y)
    # feature importance proxy: permutation on a held-out family member (reported later)
    with open(args.out, "a") as f:
        for group, cs in (("family_unseen", test_family), ("other", other)):
            for c in cs:
                mref, H, wall = run_ref(c)
                ref = dict(model=mref, wall=wall)
                unit = measure_costs(mref.el, c.material)
                Xt, yt, _ = extract_dataset(H, c, c.elems, c.material)
                near = yt < 16
                mae = {}
                for key, mod in models.items():
                    p = mod.predict(Xt); mae[f"{key[0]}_{key[1]}"] = dict(mae=float(np.abs(p[near] - yt[near]).mean()), unsafe=float((p[near] > yt[near]).mean()))
                hh = np.minimum(Xt[:, FEATURES.index("h_heur")], 16)
                mae["heuristic"] = dict(mae=float(np.abs(hh[near] - yt[near]).mean()), unsafe=float((hh[near] > yt[near]).mean()))
                print(f"[{group}] {c.name}: prediction MAE/unsafe on near-yield samples: {mae}")
                runs = [("kappa", None, dict(monitor_skip_kappa=k), None) for k in (0.0, 1.0, 1.5, 2.0, 3.0)]
                orc = OracleHorizon(H, c.elems, c.material)
                runs += [("oracle", orc, dict(horizon_safety=s), None) for s in (1.0, 0.5)]
                for (kind, seed), mod in models.items():
                    for s in (0.75, 0.5, 0.25):
                        runs.append(("learned", mod, dict(horizon_safety=s, name=kind), seed))
                for policy, hm, extra, seed in runs:
                    row = evaluate(c, ref, H, unit, policy, hm, extra, seed)
                    row.update(group=group, case=c.name, prediction_quality=mae)
                    f.write(json.dumps(row) + "\n"); f.flush()
                    if row["ok"]:
                        e = row["errors"]
                        print(f"   {policy:8s} {str(extra):45s} seed={seed} save={row['save_scalar']:.3f} (no-decision {row['save_scalar_nodecision']:.3f}) "
                              f"mon/step={row['monitored_gp_per_step']:.0f} it={row['newton_iters']}/{row['newton_iters_ref']} | u={e['u_max']:.1e} a={e['alpha_rel_final']:.1e}")
                    else:
                        print("   FAILED", policy, extra, row["error"])


if __name__ == "__main__":
    main()
