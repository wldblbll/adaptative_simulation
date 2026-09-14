# adaptfem — allocation adaptative de l'effort de calcul en EF non linéaire

Code de recherche accompagnant le projet « Peut-on concentrer dynamiquement l'effort de
calcul sur les zones physiquement actives d'un calcul éléments finis non linéaire, et
l'apprentissage apporte-t-il un gain mesurable par rapport à une heuristique bien
conçue ? ». Le maillage est une donnée d'entrée et n'est jamais modifié ; seule la phase
de résolution incrémentale (intégration de la loi, assemblage, tangente, Newton) est
orchestrée. La référence de toute comparaison est **ce même code avec l'orchestrateur
désactivé**.

## Installation

```
pip install -r requirements.txt
python -m pytest -q
```

## Organisation

| chemin | contenu |
|---|---|
| `adaptfem/material.py` | plasticité J2 déformation plane, écrouissage isotrope linéaire + Voce, retour radial, tangente cohérente (vectorisé + version scalaire de référence) |
| `adaptfem/element.py` | Q4, 2×2 Gauss, B-bar, assemblage à motif précalculé |
| `adaptfem/mesh.py` | maillages structurés, plaque entaillée |
| `adaptfem/solver.py` | Newton-Raphson incrémental, prédicteur tangent, compteurs et chronomètres, enregistrement d'historique, chemin « extrapolé » pour l'orchestrateur |
| `adaptfem/cases.py` | cas tests : plaque entaillée, plaque surchargée, console, versions cycliques, bloc en cisaillement (validation) |
| `adaptfem/oracle.py` | étude a posteriori de Phase 0 |
| `adaptfem/benchmark.py` | micro-benchmark des coûts par élément / point de Gauss |
| `adaptfem/orchestrator.py` | orchestrateur heuristique (Phase 2) : contrôle exact par prédicteur élastique, borne de saut κ, politiques plastiques, rattrapage, politiques de contrôle (κ / apprise / oracle) |
| `adaptfem/metrics.py` | erreurs vs référence, agrégation des coûts |
| `adaptfem/learning.py` | Phase 3 : extraction des étiquettes depuis les simulations, modèles d'horizon, horizon oracle |
| `experiments/` | scripts de campagne |
| `docs/` | bibliographie, rapports, article |
| `results/` | sorties (JSON, figures ; les historiques `.npz` ne sont pas versionnés) |

## Reproduire la Phase 0

```
python experiments/phase0_oracle.py       # référence + oracle sur les 5 cas (~3 min)
python experiments/phase0_stepsize.py     # sensibilité au pas de charge
python experiments/phase0_baselines.py    # comparaison à la ligne de base « tangente élastique réutilisée »
```

Voir `docs/rapport_intermediaire.md` pour les résultats et la décision go/no-go.

## Reproduire les Phases 2 et 3

```
# Phase 2 : campagnes heuristiques (≈ 45 min), analyse, cartes, cas d'échec, bornes
for c in A_monitor B_plastic D_catchup C_spatial; do
  python experiments/phase2_campaign.py --campaign $c --cases notched_plate,cantilever,cyclic_notched_kin,cyclic_cantilever_kin
done
python experiments/phase2_campaign.py --campaign E_reversal --cases cyclic_notched_kin,cyclic_cantilever_kin,cyclic_notched,cyclic_cantilever
python experiments/phase2_analysis.py --err u_max
python experiments/phase2_maps.py
python experiments/failure_case.py
python experiments/oracle_vs_online.py

# Phase 3 : politiques apprises vs heuristique vs oracle (≈ 2 × 40 min)
python experiments/phase3_learned.py --h 0.5 --seeds 3
python experiments/phase3_learned.py --h 0.5 --seeds 2 --models gbm_q10,gbm_q05,gbm_q02 --safeties 1.0,0.5 --out results/phase3/results_quantile.jsonl
cat results/phase3/results.jsonl results/phase3/results_quantile.jsonl > results/phase3/results_all.jsonl
python experiments/phase3_analysis.py --in results/phase3/results_all.jsonl
```

Les fichiers `results/**/results*.jsonl` contiennent, par ligne, la configuration
sérialisée, les compteurs, les chronomètres et les erreurs : chaque ligne est
reproductible seule.

## Documents

| document | contenu |
|---|---|
| `docs/biblio.md` | revue bibliographique et positionnement |
| `docs/rapport_intermediaire.md` | Phase 0 : créneau, potentiel oracle, décision GO |
| `docs/rapport_experiences.md` | mémoire de toutes les campagnes, y compris négatives |
| `docs/article.md` | article (version de travail) |
| `docs/integration_industrielle.md` | note d'intégration dans un code existant |
| `docs/verdict.md` | synthèse en une page |

## Résultat en une phrase

L'orchestrateur exact (éléments plastiques actifs, éléments élastiques linéarisés et
contrôlés par le prédicteur élastique avec borne de saut κ = 1) reproduit la référence
à 1e-12 près en économisant 46 à 57 % du coût élémentaire ; l'extrapolation des
éléments plastiques ne survit pas en ligne ; l'apprentissage ne bat pas l'heuristique
réglée à exactitude égale, l'anticipation parfaite ne valant que 3 à 12 points.

## Licence

MIT.
