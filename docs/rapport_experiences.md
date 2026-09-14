# Rapport d'expériences

Mémoire du projet : toutes les campagnes, y compris celles qui n'ont rien donné. Tous les
chiffres proviennent d'exécutions réelles (`results/`), reproductibles par les scripts
indiqués. Conventions : « save » = fraction du **coût élémentaire** (déformations, loi,
K^e, f^e, assemblage, extrapolation, contrôle, décision) économisée par rapport à la
**référence Sysala** (même code, orchestrateur désactivé, tangente élastique réutilisée
sur les éléments sans point plastique), sous le modèle de coût `scalar` sauf mention ;
« u » = erreur relative maximale en déplacement sur l'historique ; « a » = erreur
relative sur la déformation plastique cumulée au pas final ; « it » = itérations de
Newton (référence entre parenthèses). Les coûts unitaires sont re-mesurés par
micro-benchmark à chaque campagne : les valeurs de « save » fluctuent de ±2 points
d'une campagne à l'autre pour une même configuration.

## 0. Infrastructure et validation (Phase 1)

Solveur : Q4 déformation plane B-bar, J2 avec écrouissage isotrope (linéaire + Voce) et
cinématique linéaire, retour radial vectorisé + version scalaire de référence, tangente
cohérente, Newton-Raphson avec prédicteur tangent, pilotage en déplacement, compteurs et
chronomètres par phase. Tests (`tests/`, 8) : tangente cohérente par différences finies
(< 1e-5 relatif, isotrope et cinématique), vectorisé = scalaire (1e-8), patch test
(1e-12), cisaillement simple contre solution fermée (1e-10), convergence quadratique de
Newton, effet Bauschinger, et **orchestrateur « tout actif » = référence bit à bit**.

Incident notable : le premier Newton divergeait sur tout cas plastique en pilotage en
déplacement, parce que la première itération évaluait la tangente sur un itéré non
équilibré (éléments du bord chargé à 10 % de déformation). Corrigé par le prédicteur
tangent standard (linéarisation avec la tangente convergée précédente avant toute
intégration). Cette correction est partagée par la référence et l'orchestré.

Cas : plaque entaillée (3096 éléments, 6450 ddl, 60 pas, u_max 0.06 mm), plaque
surchargée (u_max 0.12), console (2000 éléments, 50 pas), plaque et console cycliques
(126 et 105 pas, λ : 0 → 1 → −0.6 → 1) en écrouissage isotrope (`cyclic_*`) et
combiné isotrope + cinématique (`cyclic_*_kin`, H_k = 4 GPa), familles paramétrées
(rayon d'entaille, hauteur ; hauteur et longueur de console).

## 1. Phase 0 — oracle

Détails et tableaux dans `docs/rapport_intermediaire.md`. Résumé :

- Extrapolateur **tangente** (σ̂ = σ_0 + D_0 Δε) : 94–99 % des éléments-pas
  extrapolables à 1 % de σ_y0 ; gel : 4–48 % ; linéaire en temps : 78–88 %.
- Deux composantes : éléments élastiques (extrapolation exacte, 72–97 % des éléments) et
  éléments plastiques (64–86 % de leurs pas extrapolables à 1 %, mais **cette fraction
  dépend du pas de charge** : 39 % à 10 pas, 90 % à 120 pas).
- Pondéré par le coût (modèle `scalar`, ancien chemin « calme » par points de Gauss) :
  31 à 61 % du coût élémentaire contre la référence Sysala. Décision : GO.

Recalcul des bornes avec le modèle de coût final (chemin calme f_0 + K_0 Δu, 2.6 µs
contre 55 µs pour un élément élastique actif) — `experiments/oracle_vs_online.py`,
`results/phase2/bounds_vs_online.json` :

| cas | frac. plastique moy. (part du coût) | (a) politique exacte idéale | (b) oracle tangente 1 % (0.2 %) | en ligne exact κ=1 |
|---|---|---|---|---|
| plaque entaillée | 0.080 (0.35) | 0.605 | 0.891 (0.851) | 0.517 |
| console | 0.125 (0.40) | 0.559 | 0.896 (0.857) | 0.486 |
| plaque cyclique cin. | 0.036 (0.21) | 0.737 | 0.871 (0.838) | 0.569 |
| console cyclique cin. | 0.116 (0.39) | 0.571 | 0.852 (0.798) | 0.456 |

(a) : un élément est intégré si et seulement s'il est plastique au pas courant,
connaissance parfaite, contrôle gratuit — plafond de toute politique **exacte**.
(b) : borne de Phase 0 (extrapolation des éléments plastiques incluse). L'écart (b)−(a),
de 13 à 34 points, est la part « extrapolation des éléments plastiques » ; l'écart
(a)−(en ligne), de 7 à 17 points, est le coût du contrôle en ligne.

## 2. Phase 2 — orchestrateur heuristique

Script : `experiments/phase2_campaign.py`, résultats `results/phase2/results.jsonl`
(229 lignes), tableaux `results/phase2/tables.md`, Pareto `results/phase2/pareto_*.png`.

### 2.1 Choix de conception fixés avant les campagnes (et pourquoi)

1. **Extrapolation au niveau élément**, f^e = f^e_0 + K^e_0 (u_e − u_e0), et non par
   points de Gauss (σ̂ puis B^T σ̂). Identiques mathématiquement ; la première coûte
   64 multiplications contre ≈230 flops. Mesuré : élément calme 2.6 µs (scalar) contre
   25 µs pour le chemin par points de Gauss et 55 µs pour un élément actif élastique.
   Sans ce choix, le gain en ligne était **nul** sur le premier essai (−0.8 %).
2. **Contrôle exact des éléments élastiques** par le critère de plasticité sur la
   contrainte extrapolée (c'est le prédicteur élastique du retour radial, sans la
   correction). Pour un élément ancré élastique, σ̂ est la contrainte élastique exacte :
   le contrôle est exact. Coût mesuré : 8.1 µs par point de Gauss (scalar), soit
   32 µs par élément contrôlé, **plus de la moitié d'un élément actif élastique**.
   D'où l'importance de ne pas tout contrôler.
3. **Borne de saut de contrôle** : un élément dont la marge au dernier contrôle dépasse
   κ × (variation maximale de contrainte par pas observée) × (âge + 1) n'est pas
   contrôlé. κ est le facteur de sécurité. Bug corrigé en cours de route : les éléments
   jamais contrôlés (marge infinie) étaient sautés pour toujours.
4. **Newton modifié** (réutiliser la factorisation du prédicteur quand la fraction
   active est faible) : **divergence** (taux de contraction ≈ 0.7, > 25 itérations) dès
   le premier essai avec seuil 0.3. Abandonné : la tangente convergée précédente est trop
   éloignée sur les éléments plastiques. Le gain sur la phase solveur reste hors de
   portée de cette approche.

### 2.2 Campagne A — quand et combien contrôler (`A_monitor`, 5 modes × 5 κ × 4 cas)

Éléments plastiques toujours actifs. Extraits (save scalar / u) :

| mode | κ | plaque | console | plaque cycl. cin. | console cycl. cin. |
|---|---|---|---|---|---|
| itération (chaque itération) | 0 (tout contrôler) | 0.049 / 1e-13 | 0.067 / 5e-13 | 0.069 / 9e-13 | 0.056 / 4e-12 |
| itération | 0.5 | 0.560 / 4.6e-4 | 0.515 / 2.3e-5 | 0.648 / 2.0e-4 | 0.511 / 3.6e-6 |
| **itération** | **1** | **0.517 / 9e-14** | **0.486 / 5e-13** | **0.569 / 9e-13** | **0.456 / 4e-12** |
| itération | 2 | 0.436 / 9e-14 | 0.427 / 5e-13 | 0.417 / 9e-13 | 0.346 / 4e-12 |
| prédicteur seul (it. 0) | 1 | 0.552 / 2.0e-5 | 0.496 / 6.5e-6 | 0.616 / 7.6e-5 | 0.477 / 7.9e-6 |
| prédicteur + convergence | 1 | 0.411 / 1e-10, it 127 (102) | 0.476 / 1e-11, it 107 (97) | 0.487 / 2e-9, it 252 (205) | 0.449 / 5e-10, it 231 (212) |
| convergence seule | 1 | 0.285 / 2e-10, it 184 | 0.259 / 8e-11, it 176 | 0.405 / 7e-10, it 374 | 0.208 / 1e-9, it 399 |
| post-pas (réveil au pas suivant) | 1 | 0.537 / 7.6e-4 | 0.499 / 4.7e-4 | 0.638 / 1.3e-2 | 0.458 / 6.6e-3 |

Lecture :
- **Contrôler tout, à chaque itération, annule le gain** (5–7 %). Le contrôle est le
  poste de coût dominant de la méthode.
- **κ = 1 est le plus petit facteur exact** sur les quatre cas ; κ = 0.5 manque des
  plastifications (erreurs 1e-5 à 5e-4). κ = 2 coûte 6 à 15 points pour rien.
- Contrôler après convergence seulement est **dominé** : chaque réveil oblige à
  reconverger (+80 % d'itérations). Le contrôle en cours d'itération n'ajoute aucune
  itération : les réveils sont absorbés par le Newton en cours (it identique à la
  référence sur les quatre cas).
- Le contrôle au prédicteur seul gagne 2 à 5 points pour une erreur 1e-5 (les
  plastifications survenant dans les corrections sont détectées au pas suivant).
- Le mode post-pas gagne 2 à 7 points en monotone, mais coûte 1e-2 d'erreur en
  cyclique (un pas résolu en élastique sur des éléments qui plastifient à la
  recharge).

Temps mur (Python, factorisation dominante) : +1 à +16 % pour la configuration exacte
κ = 1 ; la part élémentaire du temps est de 27–43 % (Phase 0), donc le gain mur est
borné à ce niveau ; il est positif malgré le surcoût Python de l'orchestrateur.

### 2.3 Campagne B — intégration paresseuse des éléments plastiques (`B_plastic`)

Politique « drift » : élément ancré plastique laissé calme tant que la déformation
plastique linéarisée depuis l'ancre reste < tol_α et qu'il ne décharge pas
(n : Δε ≥ 0) ; prédiction au début du pas à partir de l'incrément précédent, contrôle en
itération en filet de sécurité. Résultat : **négatif sur toute la ligne**.

| tol_α | plaque : save / u / a / it | console | plaque cycl. cin. | console cycl. cin. |
|---|---|---|---|---|
| toujours actif | 0.450 / 1e-13 / 1e-13 / 102 | 0.424 / 5e-13 / 2e-13 / 97 | 0.447 / 9e-13 / 1e-13 / 205 | 0.341 / 4e-12 / 4e-12 / 212 |
| 3e-5 | 0.394 / 4.4e-5 / 5.9e-5 / 112 | 0.436 / 2.1e-5 / 4.6e-5 / 101 | 0.437 / 8.9e-4 / 2.1e-3 / 208 | 0.342 / 1.2e-3 / 1.9e-3 / 215 |
| 1e-4 | 0.455 / 9.9e-4 / 1.9e-3 / 122 | 0.458 / 1.2e-4 / 3.0e-4 / 114 | 0.409 / 1.2e-2 / 1.1e-2 / 234 | 0.335 / 7.4e-3 / 7.9e-3 / 238 |
| 3e-4 | 0.418 / 4.2e-3 / 1.1e-2 / 124 | 0.493 / 1.8e-3 / 2.1e-3 / 121 | 0.393 / 7.8e-2 / 4.7e-2 / 259 | 0.326 / 3.3e-2 / 3.8e-2 / 284 |
| 1e-3 | 0.426 / 1.8e-2 / 1.5e-1 / 127 | 0.299 / 8.4e-3 / 7.6e-2 / 138 | 0.351 / 3.5e-1 / 2.6e-1 / 275 | 0.129 / 1.1e-1 / 1.6e-1 / 336 |
| 1e-3 sans réveil sur décharge | 0.487 / 1.7e-2 / 1.2e-1 | 0.319 / 8.4e-3 / 7.9e-2 | 0.593 / **6.0** / 0.69 / 82 | — |

Pourquoi ça échoue alors que l'oracle disait le contraire : (i) l'oracle suit la
trajectoire vraie ; en ligne, l'erreur d'extrapolation d'un élément plastique modifie
l'équilibre, donc la déformation des éléments voisins, et l'erreur se propage dans la
zone plastique, qui est précisément la zone qui pilote la réponse ; (ii) le réveil d'un
élément plastique provoque un saut de contrainte (état différé contre état extrapolé)
qui coûte des itérations de Newton (+10 à +60 %), ce qui mange le gain ; (iii) sur les
cas cycliques, l'intégration différée sur un incrément contenant une inversion de charge
est **invalide** (chemin non proportionnel, plasticité dépendante du chemin) — sans
réveil sur décharge la solution est détruite (erreur 600 %). La part du coût des
éléments plastiques (21–40 %) est en outre trop petite pour que le jeu en vaille la
chandelle. **Conclusion : les éléments plastiques restent actifs. La zone active en ligne
est la zone plastique, ni plus ni moins.**

### 2.4 Campagne D — rattrapage (`D_catchup`)

Avec la politique « drift » tol_α = 1e-3 : rattrapage complet tous les 5 pas ramène
l'erreur de 1.8e-2 à 2.7e-3 (plaque) et 8.4e-3 à 2.6e-3 (console) pour un coût de 22 %
d'éléments actifs ; en cyclique, aucun rattrapage ne récupère l'erreur (2e-1 à 4e-1).
Max-skip 3 : erreurs 2e-3 en monotone, 1e-1 en cyclique. Le rattrapage ne sauve pas la
politique paresseuse ; avec la politique exacte il n'y a rien à rattraper.

### 2.5 Campagne C — voisinage et marge de précaution (`C_spatial`, κ = 2)

Forcer actifs les voisins des éléments réveillés : coûte 10 à 27 points (plaque :
0.449 → 0.339 → 0.275 pour 0, 1, 2 couches ; plaque cyclique : 0.424 → 0.170) sans
aucun gain de précision (déjà exacte). Marge de précaution sur le critère (réveil si
f̂ > −m) : 0.02 → −0.5 pt, 0.1 → −4 pts, aucun gain. En mode post-pas, une marge de
0.02 à 0.2 ne réduit pas l'erreur (elle reste 3e-4 à 8e-3) : l'erreur vient des
éléments qui franchissent la limite *pendant* le pas, pas de ceux qui en sont proches
au début. **Conclusion : la propagation spatiale et la précaution sont inutiles avec un
contrôle exact ; la bonne granularité est l'élément.**

### 2.6 Campagne E — réveil et inversion de charge (`E_reversal`, cas cycliques)

Réveil global sur inversion du signe de Δλ : aucun effet sur la précision (déjà
exacte), −0.5 pt de coût. Le contrôle local exact traite l'inversion sans information
globale : à la décharge, tous les éléments deviennent élastiques et calmes
(fraction active 0 sur 45 pas), à la recharge la bande plastique se réveille élément
par élément au moment exact où sa contrainte extrapolée atteint la surface écrouie
(cartes `results/phase2/*_online_spacetime.png`). Le prix est dans le contrôle : sur la
plaque cyclique, 41 % des éléments sont contrôlés en moyenne par pas (contre 23 % en
monotone), parce que la borne κ utilise le taux de variation maximal du maillage, élevé
pendant la décharge.

### 2.7 Cas d'échec documenté (`experiments/failure_case.py`, `results/failure/`)

Plaque cyclique à écrouissage cinématique : exact 9e-13 ; post-pas 1.3e-2 (erreur
apparaissant au premier pas plastique et persistant) ; paresseux tol_α = 1e-3 : 0.35 ;
paresseux sans réveil sur décharge : 6.0 (la fraction active tombe à 0.02 : le solveur
« converge » sur une structure devenue quasi élastique, en 82 itérations au lieu de 205).
La carte spatiale de l'erreur finale sur α montre l'erreur concentrée dans la bande
plastique, là où l'extrapolation D_ep a traversé l'inversion.

### 2.8 Bilan Phase 2

Configuration retenue : contrôle exact en itération, κ = 1, éléments plastiques actifs,
pas de voisinage, pas de marge, pas de rattrapage (inutile), pas complet final. Gain sur
le coût élémentaire contre Sysala : **0.46 à 0.57** (scalar), 0.47 à 0.61 (vector), à
erreur ≤ 1e-12 sur les quatre cas ; itérations de Newton identiques à la référence.
Sur le temps mur de ce prototype : +1 à +16 %. Condition de sortie de la Phase 2
atteinte (gain significatif à erreur contrôlée sur au moins deux cas).

## 3. Phase 3 — apprentissage

(voir section suivante, complétée après les campagnes `phase3_learned.py`)

Scripts : `experiments/phase3_learned.py` (deux exécutions : modèles médians et
linéaire, `results/phase3/results.jsonl` ; modèles quantiles,
`results/phase3/results_quantile.jsonl` ; fusion `results_all.jsonl`), analyse
`experiments/phase3_analysis.py` (`results/phase3/tables.md`, `best.txt`,
`phase3_pareto.png`).

### 3.1 Ce qu'on demande à l'apprentissage, et pourquoi seulement cela

La Phase 2 laisse un seul levier sans perte : **quels éléments élastiques contrôler et
quand**. L'heuristique décide par la borne κ (marge / vitesse × âge). La politique
apprise prédit un **horizon** (nombre de pas avant plastification) à partir de
caractéristiques disponibles au moment du contrôle : marge f, ses deux dernières
variations, α, ‖β‖, fraction plastique et marge max du voisinage, signe et rapport des
incréments de charge, vitesse globale r, horizon heuristique −f/r, pas depuis la
dernière plastification. Déploiement identique pour toutes les politiques : un élément
contrôlé au pas k avec horizon h n'est pas recontrôlé avant k + ⌊s(h−1)⌋ + 1 (s = facteur
de sécurité), et reste contrôlé dans les itérations du pas courant. Étiquettes : pas
jusqu'à la première plastification, calculées a posteriori sur les historiques de
référence (le solveur est son propre professeur ; aucun jeu de données externe).
Entraînement sur cinq membres d'une famille de plaques entaillées (R ∈ {2.5, 3, 4, 5},
H ∈ {16, 18, 20, 24} ; 801 348 échantillons, 10 % à moins de 16 pas de la limite) ;
test sur deux membres non vus (R = 4, H = 20 ; R = 3.5, H = 22), puis sur la console et
les deux cas cycliques (H3). Modèles : régression linéaire, gradient boosting sur
log(1+h) en perte absolue (médiane) et en quantiles 10 %, 5 %, 2 % (variantes
conservatrices, l'analogue de κ côté apprentissage), 2–3 graines. Borne : **horizon
oracle** (vrai nombre de pas jusqu'à plastification, tiré de la référence). Le temps
d'inférence est compté dans le coût.

### 3.2 Qualité de prédiction (hors ligne, échantillons à moins de 16 pas de la limite)

| cas | MAE gbm médian / q10 / q05 / q02 (pas) | fraction non sûre (h prédit > h vrai) | MAE heuristique | non sûre heur. |
|---|---|---|---|---|
| plaque R4 H20 (famille, non vu) | 1.51 / 1.41 / 1.64 / 2.08 | 0.65 / 0.17 / 0.13 / 0.09 | 6.24 | 0.00 |
| plaque R3.5 H22 (famille, non vu) | 1.51 / 1.48 / 1.70 / 2.12 | 0.65 / 0.20 / 0.14 / 0.10 | 6.26 | 0.00 |
| console (autre famille) | 2.30 / 2.33 / 2.34 / 2.71 | 0.57 / 0.24 / 0.26 / 0.19 | 5.62 | 0.00 |
| plaque cyclique cin. | 1.80 / 3.02 / 2.76 / 3.29 | 0.27 / 0.08 / 0.09 / 0.08 | 6.05 | 0.00 |
| console cyclique cin. | 2.96 / 3.15 / 2.86 / 3.12 | 0.56 / 0.37 / 0.38 / 0.24 | 5.98 | 0.00 |

Le modèle appris prédit l'horizon 2 à 4 fois plus précisément que l'heuristique, et
mieux dans la famille (1.4–2.1) qu'en dehors (2.3–3.5) : la spécialisation par famille
(H3) est réelle **au niveau de la prédiction**. Mais il est non sûr sur 6 à 38 % des
échantillons proches de la limite, quand l'heuristique ne l'est jamais par
construction (elle sous-estime systématiquement).

### 3.3 Résultat en ligne (save scalar vs Sysala, contrôle en itération, plastiques actifs ; moyenne ± écart-type sur graines ; u = erreur relative max en déplacement)

| cas | heuristique κ=1 (exacte) | oracle horizon s=1 (exacte) | meilleure apprise **exacte** | meilleure apprise à u < 1e-3 | apprise gbm médian s=0.5 |
|---|---|---|---|---|---|
| plaque R4 H20 | 0.518±0.003 | 0.560±0.002 | aucune | q02 s=0.5 : 0.483 (u 3.6e-4) | 0.520 (u 1.5e-2) |
| plaque R3.5 H22 | 0.525±0.001 | 0.575±0.002 | aucune | q02 s=0.5 : 0.490 (u 2.4e-4) | 0.557 (u 1.7e-2) |
| console | 0.490±0.002 | 0.520±0.002 | aucune | q02 s=0.5 : 0.432 (u 7.4e-4) | 0.453 (u 4.2e-3) |
| plaque cyclique cin. | 0.573±0.006 | 0.693±0.003 | aucune | aucune (min 8.5e-3) | 0.651 (u 2.1e-1) |
| console cyclique cin. | 0.457±0.003 | 0.530±0.003 | aucune | aucune (min 5.2e-3) | 0.454 (u 1.1e-1) |

Lecture :
- **Aucune politique apprise n'atteint le régime exact**, quel que soit le modèle, le
  quantile ou le facteur de sécurité (24 configurations × 5 cas). Quand elle économise
  plus que l'heuristique, c'est en manquant des plastifications (erreur 2e-3 à 4e-1,
  et davantage d'itérations de Newton, +5 à +15 %, parce que les réveils tardifs
  arrivent avec un incrément accumulé).
- **La borne de l'anticipation est étroite** : l'oracle, qui anticipe parfaitement,
  gagne 3 à 12 points sur l'heuristique, le maximum étant sur la plaque cyclique
  (0.573 → 0.693), là où le contrôle est le plus coûteux (41 % des éléments).
- Pourquoi le modèle appris échoue à combler cet écart alors qu'il prédit mieux :
  (i) le déploiement transforme une erreur de prédiction en *absence de contrôle
  pendant h pas*, sans recontrôle intermédiaire — l'heuristique, elle, recontrôle dès
  que la marge est consommée au taux observé, ce qui est une garantie physique et non
  statistique ; (ii) les caractéristiques d'entraînement sont mesurées sur des états
  convergés, celles du déploiement sur des itérés ; (iii) les 6–38 % de prédictions non
  sûres tombent précisément sur les éléments qui plastifient, c'est-à-dire ceux dont
  l'omission coûte le plus ; (iv) réduire le quantile réduit l'erreur mais aussi le
  gain, jusqu'à passer sous l'heuristique (q02 s=0.5 : −3 à −6 points).
- **H3** : l'avantage de spécialisation par famille se voit dans la MAE, pas dans le
  gain déployé, qui est borné par la sûreté.

### 3.4 H2 (décision sous budget global, renforcement) — non testé, et pourquoi

Dans le régime exact il n'y a aucun budget d'erreur à distribuer le long de la
trajectoire : la seule décision est locale (contrôler ou non tel élément à tel pas) et
son coût comme sa conséquence sont locaux et immédiats. Un problème de trajectoire
n'apparaît que si l'on accepte une erreur (modes post-pas ou paresseux), et la Phase 2
montre que ces modes rapportent au mieux 2 à 7 points contre 1e-3 à 1e-2 d'erreur.
L'espace de gain qu'un apprentissage par renforcement pourrait exploiter est donc borné
par l'écart heuristique → oracle (3 à 12 points) et par ces quelques points ; il ne
justifie pas le coût d'entraînement d'une politique de trajectoire. Nous n'avons pas
mené cette expérience ; c'est un choix argumenté, pas un résultat.

### 3.5 Bilan Phase 3

Réponse chiffrée : **l'apprentissage ne bat pas l'heuristique optimisée sur aucune
configuration à exactitude égale** ; il prédit mieux (MAE ÷ 2 à 4, et mieux encore
dans sa famille) mais ne décide pas plus sûrement ; le gain maximal théorique d'une
anticipation parfaite est de 3 à 12 points de coût élémentaire, concentré sur les cas
cycliques.
