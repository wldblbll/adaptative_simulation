# Allocation adaptative de l'effort de calcul en éléments finis non linéaires : intégration paresseuse à maillage fixé, borne oracle, heuristique exacte et apport mesuré de l'apprentissage

*Version de travail (dépôt `adaptative_simulation`, branche `claude/adaptive-effort-fem-fp3i4n`). Tous les chiffres sont issus des scripts `experiments/` et des fichiers `results/`.*

## Résumé

Dans un calcul éléments finis élastoplastique incrémental, chaque itération de Newton
réintègre la loi de comportement et réassemble la contribution de tous les éléments,
alors que la plasticité n'occupe qu'une fraction du domaine. Nous étudions une
orchestration **en ligne, à maillage fixé**, qui n'intègre la loi que sur un ensemble
actif d'éléments et remplace la contribution des autres par leur linéarisation exacte
autour de leur dernier état intégré, f^e = f^e_0 + K^e_0 (u_e − u_e0). Une étude oracle
a posteriori borne le gain : 94–99 % des éléments-pas sont extrapolables à 1 % de σ_y
par l'extrapolation tangente, soit 85–90 % du coût élémentaire. Cette borne ne se
transfère pas en ligne : l'extrapolation des éléments plastiques dégrade l'équilibre,
coûte des itérations et devient invalide à l'inversion de charge. Ce qui se transfère,
et intégralement, est la part élastique : un orchestrateur déterministe qui garde les
éléments plastiques actifs et contrôle les éléments élastiques par le prédicteur
élastique, avec une borne de saut sur la marge à la limite, **reproduit la solution de
référence à 1e-12 près** en économisant **46 à 57 % du coût élémentaire** par rapport
à une référence qui réutilise déjà la tangente élastique, à nombre d'itérations de
Newton identique, sur quatre cas dont deux cycliques. Le poste de coût restant est le
contrôle. Une politique apprise (horizon avant plastification, régression par
gradient boosting entraînée sur les simulations d'une famille de géométries) est
comparée à l'heuristique réglée et à un horizon oracle : aucune politique apprise n'atteint le régime exact ; quand elle
économise plus que l'heuristique, c'est en manquant des plastifications, et l'oracle
borne le gain de l'anticipation parfaite à 3–12 points. Le gain sur le
temps total est borné par la part élémentaire du temps (27–43 % dans notre prototype),
que la méthode n'affecte pas.

## 1. Introduction

Le constat de départ est simple : dans un calcul non linéaire incrémental, l'effort de
calcul est uniforme sur le maillage à chaque pas, alors que la physique active
(plasticité, endommagement, contact) n'occupe à chaque instant qu'une fraction du
domaine, et que cette fraction bouge. L'approche dominante pour accélérer ces calculs
par apprentissage consiste à apprendre un raccourci de la physique (modèles de
substitution, opérateurs neuronaux). Ces modèles sont moins précis que les éléments
finis et ne valent que sur leur domaine d'apprentissage, ce qui les rend inadaptés à un
contexte de R&D où les technologies nouvelles n'ont pas de jeu de données.

Nous suivons une autre hypothèse : l'apprentissage ne remplace pas la physique, il
alloue l'effort. La physique reste exacte partout où elle compte, le solveur reste le
garant de la solution, et la politique s'entraîne à partir des simulations elles-mêmes.
La question de recherche est double : peut-on concentrer dynamiquement l'effort sur
les zones actives sans dégrader la solution au-delà d'une tolérance ? et l'apprentissage
apporte-t-il un gain mesurable par rapport à un critère déterministe bien réglé ?

Périmètre : le maillage est une donnée d'entrée et n'est jamais modifié. Cette contrainte
est délibérée : les règles de maillage sont propres à chaque organisation, et une
méthode qui prend le maillage tel quel est une couche greffable sur un code existant
sans toucher au préprocesseur. Corollaire à ne pas déformer : si l'utilisateur a
sur-raffiné une zone inactive, les degrés de liberté restent dans le système ; la
méthode réduit le coût du sur-raffinement inutile, elle ne le supprime pas.

Contributions : (i) une étude oracle quantifiant la borne haute du gain, pondérée par
le coût mesuré, en fonction de la tolérance, du pas de charge et de la variable
observée ; (ii) une réalisation sans modèle réduit ni substitut (linéarisation
élémentaire, intégration différée de la loi, contrôle par le prédicteur élastique) qui
reproduit exactement la référence ; (iii) une explication mesurée de pourquoi la borne
oracle ne se transfère pas ; (iv) une comparaison à information égale entre une
heuristique réglée, une politique apprise et un oracle, sur des cas cycliques conçus
pour tester l'anticipation du réveil.

## 2. État de l'art et positionnement

*Réserve : les textes intégraux des références n'ont pas pu être consultés depuis
l'environnement de travail ; les résumés seuls ont été exploités
(`docs/biblio.md`). Les affirmations sur les critères de bascule des relatifs proches
sont à vérifier avant soumission.*

**Raffinement adaptatif appris.** Yang et al. (AISTATS 2023) formulent l'AMR comme un
processus de décision markovien entraîné directement depuis la simulation ; Freymuth et
al. (NeurIPS 2023) et Foucart et al. (JCP 2023) suivent. Nous retenons l'argument
(décider sur une estimation instantanée n'optimise pas la trajectoire) et la contrainte
(pas de dataset), mais nous ne touchons pas au maillage.

**Solveurs hybrides par décomposition de domaine.** Wang, Hakimzadeh, Ruan et Goswami
(CMAME 2025) délèguent les sous-domaines non linéaires à un DeepONet et le reste aux EF
par une méthode de Schwarz, avec une zone ML qui peut s'étendre ; un cadre non
recouvrant (arXiv:2606.08796) note que « the current subdomain assignment is prescribed
a priori ». Deux différences structurelles : le sens de la délégation (nous gardons le
solveur exact sur la zone difficile et allégeons la zone facile) et la dynamique
(rétrécissement et réveil, pas seulement extension).

**Multi-pas de temps, sous-cyclage, intégrateurs asynchrones.** Belytschko & Mullen
(1977), Smolinski (1993, 1997), Lew, Marsden, Ortiz & West (2003). Parent numérique le
plus proche : un pas propre par élément. Mais en dynamique explicite, où la contrainte
est la stabilité ; en quasi-statique implicite, la question devient « faut-il réintégrer
et réassembler ».

**Newton modifié et réutilisation de tangente.** Čermák, Sysala & Valdman (2019)
préassemblent l'opérateur élastique et ne mettent à jour la tangente qu'aux points
plastiques ; Yusa et al. (2018) exploitent la concentration de la non-linéarité dans
un préconditionneur. Conséquence méthodologique : **notre référence n'est pas le
solveur naïf mais ce solveur « Sysala »** ; tout gain est mesuré par rapport à lui.

**Relatifs proches trouvés en vérifiant le créneau.** Radermacher & Reese (Comput.
Mech. 2014) : POD sélective sur les sous-domaines approximativement élastiques,
éléments finis complets ailleurs, sous-structuration *adaptative*. Kerfriden, Goury,
Rabczuk & Bordas (CMAME 2013) : ROM partitionnée concentrant l'effort autour de la zone
d'endommagement sans connaissance a priori. Gendre, Allix, Gosselet (2009) :
global/local non intrusif pour plasticité locale, zone fixée a priori. Ryckelynck (2005,
2009) : hyper-réduction avec domaine d'intégration réduit. Palchoudhary, Kerfriden et al.
(2024) : correction plastique locale de type Neuber accélérée par apprentissage.

**Positionnement honnête.** L'idée de partitionner adaptativement le domaine en zones
plastique et quasi élastique à maillage fixé n'est pas nouvelle. Ce que nous n'avons
trouvé nulle part : la mesure de la borne haute du gain ; une réalisation sans base
réduite, dont la garantie est celle du solveur (rien à entraîner) ; et la question de
l'apport d'un critère appris contre une heuristique réglée, avec le réveil comme cas
discriminant.

## 3. Étude de potentiel (oracle)

### 3.1 Méthode

Solveur de référence : Q4 déformation plane B-bar, plasticité J2, écrouissage isotrope
linéaire + Voce (et cinématique linéaire pour les cas cycliques « kin »), retour radial,
tangente cohérente, Newton-Raphson avec prédicteur tangent, pilotage en déplacement.
Validation : tangente cohérente vérifiée par différences finies (< 1e-5), retour radial
vectorisé identique à une version scalaire, patch test, cisaillement simple contre
solution fermée (1e-10), convergence quadratique observée (3e-1, 6e-2, 4e-4, 8e-8,
6e-14). Cas : plaque entaillée (3096 éléments), plaque surchargée jusqu'à la
plastification du ligament, console (2000 éléments), et versions cycliques
(0 → 1 → −0.6 → 1). Acier : E = 200 GPa, ν = 0.3, σ_y0 = 250 MPa.

Pour chaque élément et chaque pas, une politique gloutonne sur la trajectoire vraie
décide si l'élément aurait pu être extrapolé depuis son dernier pas intégré k_0 à
tolérance donnée sur la contrainte (norme des composantes planes, normalisée par σ_y0,
max sur les points de Gauss). Trois extrapolateurs : gel (σ_0), linéaire en temps,
**tangente** (σ_0 + D_0 Δε, D_0 tangente cohérente stockée). L'erreur de l'intégration
différée au réveil (loi intégrée depuis l'ancre sur l'incrément total) est aussi
mesurée. Le coût est pondéré par des micro-benchmarks sur ce code, sous deux modèles :
`vector` (NumPy vectorisé) et `scalar` (mêmes algorithmes point par point, dont le
ratio plastique/élastique, 2.9–3.0, est un proxy d'un code compilé).

### 3.2 Résultats

Le gel est inutilisable (4–48 % des éléments-pas à 1 %) car la contrainte des éléments
élastiques évolue avec la charge ; l'extrapolation linéaire en temps donne 78–88 % ;
**l'extrapolation tangente donne 94–99 %**. Ce chiffre a deux composantes. Les
éléments élastiques (72 à 97 % des éléments selon le cas et l'instant) sont
extrapolés exactement. Les éléments plastiques sont extrapolables 64 à 86 % de leurs
pas à 1 %, mais cette fraction **dépend du pas de charge** : sur la plaque, 39 % à
10 pas, 62 % à 20, 83 % à 60, 90 % à 120. Une partie de l'extrapolabilité plastique est
donc « les pas sont petits pour ces éléments ». L'intégration différée est
pratiquement exacte (erreur sur α au réveil ≤ 1.4e-6 au 99e centile à 1 %) : le retour
radial est exact sur trajet proportionnel.

Pondéré par le coût final (élément calme = f_0 + K_0 Δu, 2.6 µs contre 55 µs pour un
élément actif élastique et 160 µs pour un élément plastique, modèle scalar), la borne
oracle à 1 % est de **85 à 90 %** du coût élémentaire contre la référence Sysala. Une
borne plus utile est celle de la **politique exacte idéale** (intégrer un élément si et
seulement s'il est plastique, connaissance parfaite, contrôle gratuit) : 56 à 74 %.
L'écart entre les deux, 13 à 34 points, est la part « extrapolation des éléments
plastiques » (Fig. `results/phase0/*_potential.png`, `*_spacetime.png`).

Précaution : c'est une borne haute. L'oracle regarde la trajectoire vraie ; en ligne,
l'extrapolation modifie l'équilibre et l'erreur se propage. Deuxième précaution : la
part du temps passée au niveau élémentaire est de 27–43 % dans ce prototype 2D
(factorisation LU dominante) ; le gain sur le temps total est cette part multipliée par
le gain élémentaire. Pour des lois coûteuses (plasticité cristalline, endommagement à
nombreuses variables) la part élémentaire domine ; pour de grands modèles 3D à solveur
direct, elle est souvent minoritaire.

## 4. Méthode

### 4.1 Extrapolation élémentaire et intégration différée

À chaque itération de Newton, un élément *actif* est intégré normalement (déformation,
loi, tangente cohérente, K^e, f^e). Un élément *calme* renvoie f^e = f^e_0 + K^e_0
(u_e − u_e0), où l'indice 0 désigne son ancre (dernier état intégré), et K^e = K^e_0.
C'est exactement ∫ B^T (σ_0 + D_0 B Δu) — l'extrapolation tangente de la Phase 0 — mais
calculée en 64 multiplications au lieu d'une boucle sur les points de Gauss (≈230
flops) : élément calme 2.6 µs contre 25 µs par le chemin de Gauss. Les variables
internes restent à l'ancre. Au réveil, la loi est intégrée **une fois sur l'incrément
total** depuis l'ancre : l'état est la sortie d'un retour radial, donc admissible ; le
chemin est approché par un segment, ce qui n'est licite qu'en l'absence d'inversion
dans l'intervalle (§ 5.3). Pour un élément ancré élastique et resté élastique,
l'extrapolation est exacte.

### 4.2 Contrôle des éléments élastiques

Un élément ancré élastique est contrôlé par le prédicteur élastique du retour radial :
σ̂ = σ_0 + D_0 Δε aux points de Gauss puis f(σ̂, α, β) = ‖dev σ̂ − β‖ −
√(2/3) σ_y(α). Comme D_0 = C, σ̂ est la contrainte élastique exacte pour la
déformation courante : **si f ≤ 0, l'élément est exactement élastique et sa contribution
extrapolée est exacte ; si f > 0, il est réveillé et intégré dans l'itération en
cours.** Il n'y a donc pas d'erreur possible sur un élément contrôlé. Le contrôle coûte
8 µs par point de Gauss (scalar), 32 µs par élément, plus de la moitié d'un élément
actif élastique : contrôler tout à chaque itération annule le gain (§ 5.1).

**Borne de saut.** Après un contrôle, la marge m = −max_g f est mémorisée. Soit r le
maximum sur le maillage de la variation de contrainte (normalisée) au pas précédent.
L'élément n'est pas contrôlé tant que m > κ r (âge + 1), âge étant le nombre de pas
depuis le dernier contrôle. κ est un facteur de sécurité (κ = 1 : la marge peut être
consommée au taux maximal observé). C'est un critère composite (marge à la limite ×
vitesse d'évolution), déterministe, sans réglage autre que κ.

### 4.3 Politiques testées

Moment du contrôle : chaque itération ; prédicteur seul (itération 0) ; prédicteur +
vérification à convergence ; convergence seule (avec reconvergence si réveil) ;
post-pas (réveil au pas suivant, non intrusif). Éléments plastiques : toujours actifs,
ou intégration paresseuse avec tolérance sur la déformation plastique linéarisée depuis
l'ancre (Δα̂ = √(2/3) · 2μ n:Δε / (2μ + (2/3)(H_k + H'))) et réveil sur décharge
(n:Δε < 0), prédits au début du pas à partir de l'incrément précédent. Rattrapage
complet tous les N pas, nombre maximal de pas sautés, propagation aux voisins, marge de
précaution, réveil global sur inversion du signe de Δλ. Pas complet final obligatoire.

### 4.4 Protocole

Toute comparaison se fait à maillage, pas de charge et intégrateur identiques ; la
référence est le même code avec l'orchestrateur désactivé, en mode Sysala. Coûts
rapportés en opérations comptées (intégrations par point de Gauss, points plastiques,
K^e calculés, éléments extrapolés, points contrôlés, décisions, factorisations)
pondérées par les coûts unitaires mesurés, **et** en temps mur. Le surcoût de
l'orchestrateur (contrôle, décision, inférence) est toujours inclus. Erreurs : relative
en déplacement (max sur l'historique), RMS et énergétique en contrainte, sur la
déformation plastique cumulée au pas final, sur la réaction. Graines fixées,
configurations sérialisées dans `results/*/results.jsonl`.

## 5. Résultats de l'orchestrateur heuristique

Tableaux complets : `docs/rapport_experiences.md`, `results/phase2/tables.md` ;
Pareto : `results/phase2/pareto_u_max.png`.

### 5.1 Configuration exacte

Contrôle à chaque itération, κ = 1, éléments plastiques actifs :

| cas | fraction active moy. | contrôlés / pas (points de Gauss) | gain coût élém. scalar (vector) | temps mur | u_max | Newton it. (réf.) |
|---|---|---|---|---|---|---|
| plaque entaillée | 0.108 | 4101 (sur 12384) | 0.517 (0.514) | +14 % | 9e-14 | 102 (102) |
| console | 0.160 | 2291 (sur 8000) | 0.486 (0.471) | +14 % | 5e-13 | 97 (97) |
| plaque cyclique cin. | 0.055 | 6980 | 0.569 (0.606) | +16 % | 9e-13 | 205 (205) |
| console cyclique cin. | 0.137 | 4306 | 0.456 (0.467) | +1 % | 4e-12 | 212 (212) |

Le nombre d'itérations de Newton est identique à la référence : les réveils en cours
d'itération sont absorbés. κ = 0.5 manque des plastifications (erreurs 4e-6 à 5e-4) ;
κ = 2 coûte 6 à 15 points pour rien ; contrôler tout ramène le gain à 5–7 %. Les cartes
`results/phase2/*_online_spacetime.png` montrent l'ensemble intégré confondu avec la zone
plastique et l'ensemble contrôlé (23 % des éléments en monotone, 41 % en cyclique)
autour d'elle. **Pourquoi cette configuration l'emporte** : pour un élément élastique,
critère de contrôle et garantie sont le même objet (le prédicteur élastique), et le
saut de contrôle est borné par une quantité physique (marge / vitesse) ; il n'y a
aucune approximation à rattraper.

### 5.2 Variantes non exactes

Le contrôle au prédicteur seul gagne 2 à 5 points pour 1e-5 d'erreur. Le mode post-pas
(non intrusif) gagne 2 à 7 points en monotone pour 5e-4, mais 1e-2 en cyclique. Le
contrôle à convergence seule est dominé (+80 % d'itérations). La propagation aux
voisins coûte 10 à 27 points sans gain ; la marge de précaution coûte sans gain ; le
réveil global sur inversion n'apporte rien au contrôle local exact.

### 5.3 L'intégration paresseuse des éléments plastiques échoue en ligne

Avec tol_α de 3e-5 à 3e-3, l'erreur en déplacement va de 2e-5 à 5e-2 en monotone et
de 9e-4 à 8e-1 en cyclique, les itérations de Newton augmentent de 10 à 60 %, et le gain
n'est jamais meilleur que la configuration exacte (sauf par destruction de la solution).
Trois mécanismes, mesurés (`results/failure/`) : (i) l'erreur d'extrapolation d'un
élément plastique modifie l'équilibre et se propage dans la zone qui pilote la réponse ;
(ii) le réveil produit un saut de contrainte qui coûte des itérations ; (iii) à
l'inversion de charge, l'intégration différée sur un incrément non proportionnel est
invalide (plasticité dépendante du chemin) — sans réveil sur décharge, erreur de 600 %,
et la « convergence » se fait sur une structure devenue quasi élastique. La part
« éléments plastiques » de la borne oracle (13 à 34 points) **ne se transfère pas**. La
zone active en ligne est la zone plastique, ni plus ni moins ; ce qui se joue est le
coût du contrôle des éléments élastiques.

### 5.4 Ce qui reste : le contrôle

Entre la politique exacte idéale (contrôle gratuit : 56–74 %) et l'heuristique exacte
(46–57 %), l'écart de 7 à 17 points est le coût du contrôle sous la borne κ. C'est le
seul endroit où une meilleure décision peut encore gagner sans rien perdre : décider
*quels* éléments contrôler et *quand*. C'est la question posée à l'apprentissage.

## 6. Apport de l'apprentissage

### 6.1 Protocole

Le seul levier sans perte laissé par la Phase 2 est le choix des éléments élastiques à
contrôler et du moment. L'heuristique décide par la borne κ ; la politique apprise
prédit un horizon h (pas avant plastification) à partir de douze caractéristiques
disponibles au contrôle (marge, ses variations, écrouissage, voisinage, direction de
charge, vitesse globale, horizon heuristique, pas depuis la dernière plastification).
Déploiement identique pour toutes : pas de recontrôle avant k + ⌊s(h−1)⌋ + 1. Les
étiquettes sont calculées a posteriori sur les historiques de référence : le solveur
est son propre professeur, sans jeu de données externe. Entraînement sur cinq membres
d'une famille de plaques entaillées (801 348 échantillons), test sur deux membres non
vus puis sur la console et les cas cycliques (H3). Modèles : linéaire, gradient boosting
(médiane, quantiles 10, 5, 2 %), 2–3 graines. Borne : horizon oracle. Temps d'inférence
compté dans le coût.

### 6.2 Résultats

Hors ligne, le modèle appris prédit l'horizon 2 à 4 fois mieux que l'heuristique
(MAE 1.4–2.1 pas dans la famille, 2.3–3.5 hors famille, contre 5.6–6.3), mais il est
non sûr (h prédit > h vrai) sur 6 à 38 % des échantillons proches de la limite, quand
l'heuristique ne l'est jamais. En ligne (`results/phase3/phase3_pareto.png`) :

| cas | heuristique κ=1 (exacte) | oracle horizon (exacte) | meilleure apprise exacte | meilleure apprise à u < 1e-3 |
|---|---|---|---|---|
| plaque R4 H20 (famille) | 0.518 | 0.560 | aucune | 0.483 (u 3.6e-4) |
| plaque R3.5 H22 (famille) | 0.525 | 0.575 | aucune | 0.490 (u 2.4e-4) |
| console | 0.490 | 0.520 | aucune | 0.432 (u 7.4e-4) |
| plaque cyclique cin. | 0.573 | 0.693 | aucune | aucune |
| console cyclique cin. | 0.457 | 0.530 | aucune | aucune |

**H1 (anticiper au lieu de constater).** L'anticipation parfaite vaut 3 à 12 points
de coût élémentaire, le maximum sur le cas cyclique où le contrôle est le plus cher.
Aucune politique apprise ne les capture : sur 24 configurations × 5 cas, quand elle
économise plus que l'heuristique c'est en manquant des plastifications (erreur 2e-3 à
4e-1 et +5 à +15 % d'itérations, les réveils tardifs arrivant avec un incrément
accumulé). Le mécanisme est structurel : une erreur de prédiction devient une absence de
contrôle pendant h pas, alors que la borne κ recontrôle dès que la marge peut avoir été
consommée au taux observé — une garantie physique, pas statistique. Réduire le quantile
réduit l'erreur mais aussi le gain, jusqu'à passer sous l'heuristique.

**H3 (spécialisation par famille).** Réelle sur la qualité de prédiction (MAE plus
faible dans la famille), sans effet sur le gain déployé, borné par la sûreté.

**H2 (budget global, renforcement).** Non testé, par choix argumenté : dans le régime
exact il n'y a pas de budget d'erreur à répartir sur la trajectoire ; les modes qui
acceptent une erreur rapportent 2 à 7 points (§ 5.2) ; l'espace de gain d'une politique
de trajectoire est borné par l'écart heuristique → oracle. Il ne justifie pas
l'entraînement.

### 6.3 Réponse à la question

Sur ce problème, l'apprentissage n'apporte pas de gain mesurable par rapport à
l'heuristique réglée. La raison n'est pas que le modèle prédit mal, c'est que la
décision utile est une décision de *sûreté*, et que le critère physique (prédicteur
élastique + marge/vitesse) est déjà sûr et presque optimal : l'écart au parfait est de
3 à 12 points. Là où l'apprentissage garde une place : réduire le coût du contrôle sur
les cas où il est massif (cyclique : 41 % des éléments), à condition d'une garantie de
sûreté que la régression seule ne fournit pas — par exemple une politique apprise qui
ne fait que *retarder* le contrôle sous la borne κ, jamais au-delà.

## 7. Limites

- Borne oracle et gain réel sont deux choses : la borne inclut une extrapolation des
  éléments plastiques qui ne survit pas à la boucle de rétroaction de l'équilibre.
- Le gain porte sur le coût élémentaire ; la part de ce coût dans le temps total
  (27–43 % ici) borne le gain mur. Le Newton modifié pour économiser des factorisations
  a été essayé et diverge.
- 2D, déformation plane, J2, lois simples : le ratio plastique/élastique (3) et le
  coût du contrôle (0.6 élément élastique) sont ceux de cette loi ; une loi plus coûteuse
  augmente le gain, une loi plus simple le réduit.
- Le pas de charge est un facteur confondant de la Phase 0, pas de la Phase 2 (les
  éléments plastiques y sont actifs).
- Chargements proportionnels ; le contrôle exact ne dépend pas de cette hypothèse, mais
  l'intégration différée d'un élément resté élastique puis réveillé l'est, sur un
  segment : exact tant que l'élément était élastique.
- Modèle de coût : les coûts unitaires sont mesurés en Python (deux modèles) ; les
  rapports, pas les valeurs absolues, sont transférables, et ils fluctuent de ±2 points
  entre campagnes.
- Textes intégraux des relatifs proches non consultés.

## 8. Intégration dans un code existant

Voir `docs/integration_industrielle.md`. En bref : la couche a besoin (A) d'une boucle
élémentaire filtrable par itération, (B) d'un cache par élément (f^e_0, K^e_0, u_e0),
(C) d'un prédicteur élastique appelable sans correction, (D) d'un point d'entrée dans
l'itération de Newton pour élargir l'ensemble actif. (A)–(C) relèvent d'une surcharge de
la routine d'intégration ; (D) est le point intrusif, contournable par le mode
« convergence » (exact, +80 % d'itérations) ou « post-pas » (non intrusif, erreur 1e-3
à 1e-2). Garanties : la configuration par défaut reproduit la référence à 1e-12 ; la
politique apprise ne décide que du moment du contrôle. Familles favorables : lois
coûteuses, non-linéarité localisée et stable, cycles avec longues phases élastiques,
familles de pièces répétées. Défavorables : plasticité diffuse, grands modèles où le
solveur linéaire domine, structures uniformément proches de la limite (tout doit être
contrôlé).

## 9. Conclusion et perspectives

À la première question, la réponse est oui, avec une précision : on peut concentrer
l'effort d'intégration et d'assemblage sur la zone plastique **sans aucune dégradation**
(1e-12) en économisant 46 à 57 % du coût élémentaire par rapport à une référence qui
réutilise déjà la tangente élastique, à nombre d'itérations de Newton identique, y
compris en chargement cyclique — mais la zone active en ligne est la zone plastique,
pas moins : l'extrapolation des éléments plastiques, que l'oracle créditait de 13 à 34
points supplémentaires, ne survit pas à la rétroaction de l'équilibre ni à l'inversion
de charge. À la seconde, la réponse est non : sur ce problème, une politique apprise
ne bat pas l'heuristique réglée à exactitude égale, parce que la décision utile est une
décision de sûreté que le critère physique fournit déjà, et parce que le gain d'une
anticipation parfaite est borné à 3–12 points.

Le gain sur le temps total reste borné par la part élémentaire du temps. Perspectives,
dans l'ordre de valeur attendue : (1) lois de comportement coûteuses (plasticité
cristalline, endommagement), où cette part domine et où le ratio actif/calme est bien
supérieur à 3 ; (2) 3D et éléments à plus de degrés de liberté, où le cache K^e_0
coûte en mémoire mais le chemin calme reste un produit matrice-vecteur ; (3) étendre le
gain à la phase solveur : conserver la factorisation quand l'ensemble actif est petit
exige un Newton modifié qui, dans sa forme simple, diverge ; une mise à jour de rang
faible des blocs actifs est la voie naturelle ; (4) une politique apprise **sous
garantie** (retarder le contrôle sous la borne κ, jamais au-delà) pour les cas cycliques
où le contrôle coûte le plus ; (5) vérifier sur textes intégraux le positionnement par
rapport à Radermacher & Reese (2014) et Kerfriden et al. (2013).
