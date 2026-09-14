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

## Licence

MIT.
