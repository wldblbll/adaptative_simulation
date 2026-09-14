# Rapport intermédiaire — revue bibliographique et étude oracle (Phase 0)

Date : 2026-09-14. Branche : `claude/adaptive-effort-fem-fp3i4n`.
Toutes les valeurs ci-dessous proviennent d'exécutions réelles du code du dépôt
(`experiments/phase0_oracle.py`, `experiments/phase0_stepsize.py`,
`experiments/phase0_baselines.py`), résultats bruts dans `results/phase0/*.json`.

## 1. Le créneau est-il libre ? — Partiellement. Repositionnement nécessaire.

Détail dans `docs/biblio.md`. En résumé :

- L'idée de partitionner le domaine en zone plastique (calcul complet) et zone
  quasi-élastique (traitement réduit), **de façon adaptative et à maillage fixé**, existe
  déjà : Radermacher & Reese (Comput. Mech. 2014, « POD combined with *adaptive*
  sub-structuring ») ; Kerfriden et al. (CMAME 2013, ROM partitionnée « sans connaissance a
  priori du motif d'endommagement »). Wang et al. (CMAME 2025) font croître dynamiquement
  la zone confiée à l'opérateur neuronal dans un couplage EF–DeepONet.
- Le réassemblage de la tangente **limité aux points plastiques** (matrice élastique
  préassemblée) est de l'état de l'art courant (Čermák, Sysala, Valdman 2019). Toute
  comparaison contre un solveur « naïf » qui réassemble tout serait donc biaisée.
- Ce que nous n'avons trouvé nulle part : (i) une **mesure de la borne haute** du gain,
  pondérée par le coût, en fonction de la tolérance ; (ii) une extrapolation locale par
  **état interne et tangente locale** (intégration différée de la loi) sans base réduite ni
  substitut ; (iii) la question **heuristique optimisée contre critère appris**, avec le
  réveil (décharge/recharge) comme cas discriminant.

Repositionnement proposé pour l'article : *nous ne revendiquons pas l'idée de la
partition active/calme, connue ; nous revendiquons sa quantification, une réalisation
sans modèle réduit garantie par le solveur, et la réponse à la question de l'apport de
l'apprentissage pour la décision.* Réserve : les textes intégraux des relatifs proches
n'ont pas pu être lus (accès éditeurs bloqué) ; les résumés seuls ont été exploités.

## 2. Ce qui a été construit pour répondre à la Phase 0

L'étude oracle exige un historique complet ; le cœur du solveur de Phase 1 a donc été
écrit d'abord (`adaptfem/`, ~700 lignes) :

- Q4 déformation plane avec B-bar, plasticité J2, écrouissage isotrope linéaire + Voce
  (retour radial avec Newton local, tangente cohérente), Newton-Raphson incrémental avec
  prédicteur tangent, pilotage en déplacement.
- Instrumentation : compteurs (intégrations par point de Gauss, points plastiques,
  itérations locales, matrices élémentaires, forces internes, assemblages,
  factorisations, résolutions, itérations de Newton) et chronomètres par phase.
- Validation (`tests/`, 7 tests) : tangente cohérente vérifiée par différences finies
  (erreur relative < 1e-5), retour radial vectorisé identique à une version scalaire,
  patch test sur maillage distordu (exact à 1e-12), cisaillement simple contre la solution
  fermée J2 à écrouissage linéaire (rtol 1e-10), convergence quadratique de Newton
  observée (ex. 3.1e-1 → 5.6e-2 → 3.5e-4 → 8.0e-8 → 6.2e-14).
- Cas tests : plaque entaillée (3096 éléments, u_max = 0.06 mm, 60 pas), variante
  surchargée jusqu'à la plastification du ligament (u_max = 0.12), poutre console
  (2000 éléments, 50 pas), et versions cycliques des deux (charge → décharge et
  inversion à −0.6 → recharge, 126 et 105 pas). Matériau acier : E = 200 GPa, ν = 0.3,
  σ_y0 = 250 MPa, H = 1000 MPa, σ_∞ = 400 MPa, δ = 20.

## 3. Méthode oracle

Pour chaque élément et chaque pas, politique gloutonne sur la **trajectoire vraie** : à
partir du dernier pas où l'élément a été intégré (k0), on évalue l'erreur de
l'extrapolation de contrainte au pas k ; si elle est sous la tolérance, l'élément est
« extrapolé » au pas k (k0 inchangé), sinon il est « intégré » (k0 := k). Trois
extrapolateurs :

- **gel** : σ̂_k = σ_k0 ;
- **linéaire en temps** : σ̂_k = σ_k0 + (k−k0)(σ_k0 − σ_k0−1) ;
- **tangente** : σ̂_k = σ_k0 + D_k0 (ε_k − ε_k0), D_k0 tangente cohérente stockée.
  C'est l'extrapolateur qui correspond à l'*intégration paresseuse* : aucune évaluation
  de la loi, bloc K^e réutilisé, variables internes figées jusqu'au réveil, où la loi est
  intégrée sur l'incrément total.

Erreur = norme des composantes planes de (σ̂ − σ_vrai) / σ_y0, max sur les 4 points de
Gauss. Tolérances balayées : 0.1 % à 10 %. Pour l'extrapolateur tangente, on vérifie
aussi l'**erreur de l'intégration différée** au réveil (σ et α obtenus en intégrant la
loi depuis l'état k0 avec la déformation vraie ε_k, contre l'état vrai).

Pondération par le coût : coûts mesurés sur ce code par micro-benchmark, sous deux
modèles : `vector` (implémentation NumPy vectorisée) et `scalar` (mêmes algorithmes
écrits point par point en Python pur, dont le *ratio* plastique/élastique est un proxy
d'un code compilé par élément). Coût d'un élément-itération, moyenne des cinq cas :

| modèle | élément actif élastique | élément actif plastique | élément calme (extrapolé) | ratio plast./élast. | ratio calme/élast. |
|---|---|---|---|---|---|
| scalar | 54–56 µs | 160–165 µs | 25–26 µs | 2.9–3.0 | 0.46–0.48 |
| vector | 3.9–4.1 µs | 5.0–5.1 µs | 0.8 µs | 1.25–1.28 | 0.19–0.20 |

Le coût « calme » n'est pas nul : il inclut le calcul de la déformation, le produit
D·Δε et la force interne (mêmes flops qu'un résidu élastique). Les timings varient
d'environ ±5 % d'une exécution à l'autre.

## 4. Résultats

### 4.1 Potentiel brut (fraction d'éléments-pas extrapolables), extrapolateur tangente

| cas | frac. plastique max / moy. | tol 0.2 % | tol 1 % | tol 5 % |
|---|---|---|---|---|
| plaque entaillée | 0.36 / 0.080 | 0.976 | 0.986 | 0.993 |
| plaque surchargée | 0.53 / 0.269 | 0.936 | 0.964 | 0.982 |
| console | 0.23 / 0.125 | 0.969 | 0.981 | 0.990 |
| plaque cyclique | 0.36 / 0.034 | 0.978 | 0.983 | 0.989 |
| console cyclique | 0.28 / 0.117 | 0.948 | 0.964 | 0.977 |

Comparaison des extrapolateurs à tol 1 % (plaque / surchargée / console / plaque cycl. /
console cycl.) : gel 0.18 / 0.39 / 0.48 / 0.04 / 0.26 ; linéaire 0.88 / 0.83 / 0.85 /
0.85 / 0.78 ; tangente 0.99 / 0.96 / 0.98 / 0.98 / 0.96. **Le gel est inutilisable** (la
contrainte des éléments élastiques évolue avec la charge) ; **la tangente domine**, et
c'est bien l'objet « critère = gain » anticipé dans le cahier des charges.

### 4.2 Lecture honnête du potentiel brut : deux composantes

1. **Les éléments élastiques** (72 à 97 % des éléments selon le cas et l'instant) sont
   extrapolés *exactement* par la tangente élastique. Cette composante est triviale et
   c'est elle qui domine le chiffre brut.
2. **Les éléments plastiques** sont eux aussi extrapolables une bonne partie du temps :
   fraction des éléments-pas plastiques extrapolés à tol 1 % = 0.83 / 0.86 / 0.85 / 0.64
   / 0.74. Mais cette fraction **dépend du pas de charge** (`phase0_stepsize.py`, plaque
   entaillée, tol 1 %) :

   | nombre de pas | 10 | 20 | 40 | 60 | 120 |
   |---|---|---|---|---|---|
   | éléments-pas plastiques extrapolés | 0.39 | 0.62 | 0.75 | 0.83 | 0.90 |
   | potentiel brut | 0.94 | 0.97 | 0.98 | 0.99 | 0.99 |
   | potentiel pondéré (scalar, vs naïf) | 0.47 | 0.54 | 0.54 | 0.59 | 0.59 |

   Une partie de l'extrapolabilité des éléments plastiques est donc « les pas sont
   petits pour eux », ce qu'un pilotage global du pas de charge capterait aussi. Ce qui
   reste spatialement hétérogène, et que le pas global ne capte pas, c'est la
   distribution des fréquences d'intégration : à tol 1 % et 60 pas, la médiane des
   fréquences d'intégration par élément est 0, le 90e centile 0.05, le maximum 0.13 (plaque)
   et 0.20 (console). Aucun élément n'a besoin d'être intégré à chaque pas.
3. L'**intégration différée** est pratiquement exacte sur ces trajectoires : erreur sur
   α au réveil, 99e centile ≤ 1.4e-6 à tol 1 %, ≤ 4.4e-5 à tol 5 % (retour radial exact
   pour les trajets proportionnels). L'état interne reste admissible par construction
   (il est toujours la sortie d'un retour radial).

### 4.3 Potentiel pondéré par le coût (extrapolateur tangente)

Trois références : solveur naïf (tout réintégré et réassemblé), ligne de base « Sysala »
(bloc K^e réutilisé pour les éléments sans point plastique, état de l'art), et oracle
appliqué **par-dessus** Sysala. Modèle de coût `scalar` :

| cas | Sysala vs naïf | oracle vs naïf (0.2 % / 1 % / 5 %) | **oracle vs Sysala** (0.2 % / 1 % / 5 %) |
|---|---|---|---|
| plaque entaillée | 0.24 | 0.55 / 0.57 / 0.59 | **0.40 / 0.44 / 0.46** |
| plaque surchargée | 0.15 | 0.59 / 0.64 / 0.67 | **0.51 / 0.57 / 0.61** |
| console | 0.23 | 0.55 / 0.58 / 0.60 | **0.42 / 0.46 / 0.48** |
| plaque cyclique | 0.29 | 0.51 / 0.53 / 0.55 | **0.31 / 0.34 / 0.36** |
| console cyclique | 0.23 | 0.52 / 0.55 / 0.58 | **0.38 / 0.42 / 0.46** |

Avec le modèle `vector` (calme beaucoup moins cher), oracle vs naïf 0.76–0.79 à 1 % ; oracle vs
Sysala 0.31–0.55. Avec le modèle abstrait (élastique = 1, plastique = 3, calme = ρ) :
ρ = 0 donne 0.92–0.95, ρ = 0.25 donne 0.72–0.77, ρ = 0.5 donne 0.50–0.61 (à 1 %). **Le
potentiel est piloté au premier ordre par le coût résiduel de l'élément calme**, plus
que par la tolérance : entre 0.2 % et 5 % la variation n'est que de quelques points.

### 4.4 Ce que le potentiel ne couvre pas : la part du solveur linéaire

Part du temps total passée au niveau élémentaire (déformations, loi, K^e, f^e,
assemblage) dans ce code : 0.27–0.28 (plaques), 0.42 (consoles). Le reste est essentiellement la factorisation LU (SuperLU). Un gain de 50 %
au niveau élémentaire vaut donc 14 à 21 % de temps total ici. Deux remarques :

- Cette part est propre à un code 2D Python avec loi J2 simple. Pour des lois coûteuses
  (plasticité cristalline, endommagement avec Newton local sur de nombreuses variables,
  UMAT lourdes), la part élémentaire domine et le potentiel se transfère presque
  entièrement. Pour un grand modèle 3D à solveur direct, la part du solveur est souvent
  majoritaire et le gain élémentaire seul est plafonné.
- Quand la zone active est petite, les blocs tangents des éléments calmes ne changent
  pas ; la factorisation précédente peut être conservée (Newton modifié / quasi-Newton
  sur les seuls blocs actifs). Cette extension est dans le périmètre (réutilisation de
  tangente) et sera évaluée en Phase 2 par comptage des factorisations.

### 4.5 Cartes spatio-temporelles

`results/phase0/*_spacetime.png` et `*_snapshots.png`. Sur la plaque entaillée, les
éléments à intégrer se concentrent (i) sur le **front plastique** au moment du passage
élastique → plastique (changement de tangente), puis (ii) sporadiquement, tous les
quelques pas, dans la zone plastique quand la dérive de l'écrouissage fait sortir la
tangente de la tolérance. Sur la plaque cyclique, la zone active à la recharge est une
bande (les « ailes » plastiques) couvrant 13 % des éléments au dernier pas ; à la
décharge, tout redevient extrapolable d'un coup (0 % intégré pendant toute la phase
élastique intermédiaire), et le réveil se produit lorsque la contrainte inverse atteint
la surface écrouie. Avec écrouissage purement isotrope, le réveil en compression est
tardif et limité ; un écrouissage cinématique (effet Bauschinger) sera ajouté pour
rendre le cas cyclique plus discriminant sur la question du réveil (H1).

## 5. Décision : GO, avec deux réserves explicites

Le critère de sortie (potentiel pondéré ≥ 25–30 %) est atteint sur les cinq cas, y compris
**contre la ligne de base forte** (réutilisation de la tangente élastique) : 31 à 61 % du
coût élémentaire (modèle scalar), 31 à 55 % (modèle vector). Réserves :

1. **C'est une borne haute.** L'oracle voit la trajectoire vraie ; en ligne,
   l'extrapolation modifie l'équilibre et l'erreur se propage. L'extrapolation tangente
   étant consistante au premier ordre, la dérive attendue est O(Δε²) par pas sauté, mais
   cela reste à mesurer (Phase 2).
2. **Gain sur le temps total = potentiel élémentaire × part élémentaire.** Cette part est
   de 0.27 à 0.43 ici. La Phase 2 devra aussi mesurer ce qu'apporte la réutilisation de
   factorisation quand la zone active est petite.

## 6. Révisions proposées au cahier des charges

- **La ligne de base de référence est le solveur « Sysala »**, pas le solveur naïf. Le
  solveur de référence sera exécuté dans les deux modes et toutes les Pareto seront
  tracées contre le mode Sysala.
- **L'intuition « zone active petite et mobile » se précise** : la zone plastique croît
  de façon monotone en chargement monotone (elle ne « se déplace » pas), mais la zone
  qui a *besoin d'être réintégrée* est plus petite qu'elle : le front plastique plus une
  fraction tournante de la zone plastique. Le cas cyclique est le seul où la zone active
  se déplace vraiment.
- **Le pas de charge est un facteur confondant** : les rapports de Phase 2 indiqueront
  toujours le nombre de pas, et une comparaison à un pilotage adaptatif global du pas
  sera ajoutée comme troisième ligne de base si le temps le permet.
- **Le critère « variation de tangente » est déjà validé comme extrapolateur** ; reste à
  savoir s'il est aussi un bon *critère de décision en ligne* (il ne peut être évalué
  sur un élément calme, dont la tangente n'est plus calculée — c'est la question E).

## 7. Reproduction

```
pip install -r requirements.txt
python -m pytest -q
python experiments/phase0_oracle.py          # ~3 min, figures + summary.json
python experiments/phase0_stepsize.py        # ~3 min
python experiments/phase0_baselines.py       # ~1 min, nécessite summary.json
```
