# Results

Raw outputs of every campaign. Operation counters and error metrics are deterministic;
the cost-weighted savings depend on unit costs micro-benchmarked on the machine that ran
the campaign (see `docs/experiments.md`, "Conventions"). Full load histories (`.npz`) are
not versioned.

| path | produced by | content |
|---|---|---|
| `phase0/` | `experiments/phase0_oracle.py`, `phase0_stepsize.py`, `phase0_baselines.py` | oracle potential per case (`summary.json`), load-step sensitivity, comparison with the Sysala baseline; potential curves, space-time maps and snapshots |
| `phase2/results.jsonl` | `phase2_campaign.py` at tag `exploration-complete` | the 244 rows of the heuristic campaigns A–E (monitoring modes, lazy plastic integration, catch-up, neighbourhood, reversal wake-up) |
| `phase2/tables.md`, `best.txt`, `pareto_*.png`, `campaign.log` | `phase2_analysis.py` at tag `exploration-complete` | tables and Pareto plots of those campaigns |
| `phase2/kappa_sweep.jsonl` | `experiments/phase2_kappa_sweep.py` | re-run of the retained method after the code clean-up (counters and errors identical to the archived rows) |
| `phase2/bounds_vs_online.json` | `experiments/oracle_vs_online.py` | oracle and ideal-exact ceilings vs. achieved savings |
| `phase2/*_online_spacetime.png` | `experiments/phase2_maps.py` | integrated / checked / skipped elements per step vs. reference plastic zone |
| `phase3/` | `phase3_learned.py`, `phase3_analysis.py` at tag `exploration-complete` | learned policies vs. heuristic vs. oracle horizon (`results_all.jsonl`, tables, Pareto plot) |
| `failure/` | `failure_case.py` at tag `exploration-complete` | documented failure of lazy plastic integration on the cyclic plate |

Each JSONL row stores the serialised configuration, counters, timers and errors, and is
reproducible on its own.
