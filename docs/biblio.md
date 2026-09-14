# Revue bibliographique et positionnement

*Note méthodologique.* Cette revue a été constituée depuis un environnement dont l'accès
réseau est restreint : les sites d'éditeurs (Elsevier, Springer, arXiv, PubMed,
ResearchGate) sont bloqués en lecture directe. Les éléments ci-dessous proviennent des
résumés et extraits accessibles via les résultats de recherche, pas de la lecture des
textes intégraux. Les affirmations sur le contenu détaillé des articles (critères de
bascule, chiffres de gain) doivent donc être **revérifiées sur les textes intégraux**
avant rédaction finale de l'article. Aucune référence ci-dessous n'est inventée : chacune
correspond à un titre et une source retrouvés en ligne.

## 1. Les quatre courants identifiés dans le cahier des charges

### 1.1 Raffinement adaptatif de maillage par apprentissage (hors périmètre, mais cadre méthodologique)

- Yang, Dzanic, Petersen, Kudo, Mittal, Tomov, Camier, Zhao, Zha, Kolev, Anderson, Faissol,
  *Reinforcement Learning for Adaptive Mesh Refinement*, AISTATS 2023 (arXiv:2103.01342).
  AMR formulé comme processus de décision markovien, politique entraînée directement depuis la
  simulation, taille du modèle indépendante de la taille du maillage.
- Freymuth et al., *Swarm Reinforcement Learning for Adaptive Mesh Refinement*, NeurIPS 2023
  (arXiv:2304.00818). Chaque élément est un agent ; récompense spatiale ; message passing.
- Foucart, Charous, Lermusiaux, *Deep reinforcement learning for adaptive mesh refinement*,
  J. Comput. Phys. 2023 (arXiv:2209.12351). AMR comme POMDP local.
- Multi-Agent RL for AMR (arXiv:2211.00801) ; *Multi-Objective AMR using RL* (rapport LLNL, OSTI 1989992).

Ce que nous en retenons : l'argument « décider sur une estimation instantanée n'optimise
pas la trajectoire entière » (notre H2), et la contrainte « entraînement depuis la
simulation, sans dataset ». Ce que nous n'y faisons pas : toucher au maillage.

### 1.2 Solveurs hybrides EF / opérateurs neuronaux par décomposition de domaine

- Wang, Hakimzadeh, Ruan, Goswami, *Time-marching neural operator–FE coupling*, CMAME 2025
  (arXiv:2504.11383). Un DeepONet pré-entraîné résout les sous-domaines « coûteux »
  (concentrations de contrainte), les EF le reste, couplage de Schwarz alterné.
  **Point important pour nous** : le résumé mentionne « an adaptive subdomain evolution
  strategy enables the ML-resolved region to expand dynamically ». La décomposition n'est
  donc pas totalement figée : la zone ML peut **s'étendre** en cours de calcul.
- *A Non-Overlapping Schwarz Hybrid FE–Neural Operator Framework for Solid Mechanics on
  Irregular Domains* (arXiv:2606.08796, juin 2026). Extrait des perspectives : « The current
  subdomain assignment is prescribed a priori; an adaptive decomposition strategy, driven by
  local error indicators or nonlinearity measures, would eliminate this assumption and make
  the framework applicable to problems where critical regions are not known in advance. »
- *Hybrid coupling with operator inference and the overlapping Schwarz alternating method*
  (Sandia, arXiv:2511.20687).

Deux différences structurelles avec notre approche :
1. **Sens de la délégation inversé.** Ces travaux mettent le substitut appris sur la zone
   *difficile* (non linéaire) et gardent les EF sur la zone facile. Nous faisons l'inverse :
   le solveur exact reste sur la zone active, c'est la zone *calme* qui est traitée à bas
   coût. Notre hypothèse est que c'est la seule façon de garder une garantie de solution
   sans dataset préalable.
2. **Dynamique de la partition.** Chez Wang et al. la zone ML ne fait que s'étendre. Notre
   question inclut le rétrécissement (une zone se calme) et le réveil (décharge/recharge),
   ce qui est précisément le cas cyclique.

### 1.3 Multi-pas de temps, sous-cyclage, multirate (parent numérique le plus proche)

- Belytschko & Mullen (1977) et suite ; Smolinski, *Explicit multi-time step integration for
  first and second order FE semidiscretizations*, CMAME 1993 ; *Stability of explicit
  subcycling with linear interpolation*, CMAME 1997.
- Intégrateur variationnel asynchrone (AVI) : chaque élément possède son propre pas de
  temps, mise à jour constitutive élément par élément (Lew, Marsden, Ortiz, West 2003 ;
  extension contact élastoplastique, Acta Mech. Solida Sinica 2023).
- *Multi-temporal decomposition for elastoplastic ratcheting solids* (arXiv:2308.11821).

Différence : ces méthodes sont conçues pour la dynamique explicite où la contrainte est la
stabilité (pas critique local). En quasi-statique implicite, il n'y a pas de pas critique ;
la question devient « faut-il réintégrer la loi et réassembler ». L'AVI est le parent
conceptuel le plus proche (pas de temps *par élément*), mais dans un cadre explicite.

### 1.4 Newton modifié, quasi-Newton, réutilisation de tangente et d'assemblage

- Čermák, Sysala, Valdman, *Efficient and flexible MATLAB implementation of 2D and 3D
  elastoplastic problems*, Appl. Math. Comput. 2019 (arXiv:1805.04155). **Point clé** : la
  matrice tangente est décomposée en trois matrices creuses (opérateur élastique,
  déformation-déplacement, dérivée contrainte-déformation) ; les deux premières sont
  assemblées une fois pour toutes ; le temps d'assemblage de la tangente est proportionnel
  au **nombre de points d'intégration plastiques**.
- Yusa, Okada, Yamada, Yoshimura, *Scalable parallel elastic–plastic FE analysis using a
  quasi-Newton method with a balancing domain decomposition preconditioner* (2018) :
  exploite la concentration locale de la non-linéarité.

Conséquence pour notre ligne de base : **réassembler uniquement les blocs plastiques est
déjà de l'état de l'art.** Notre solveur de référence doit donc être comparé sous deux
formes : (a) naïf (tout est réintégré et réassemblé à chaque itération) et (b) « Sysala »
(tangente élastique préassemblée, mise à jour seulement aux points plastiques). Le gain
que nous revendiquons doit être mesuré **par rapport à (b)**, sinon il est artificiel. Ce
que (b) ne supprime pas : le calcul de la déformation, la prédiction élastique, le test
du critère à chaque point de Gauss, et l'assemblage du résidu sur tous les éléments, à
chaque itération de Newton. C'est là que se situe l'économie supplémentaire de
l'extrapolation.

## 2. Relatifs proches trouvés lors de la vérification du créneau

Le cahier des charges demandait de chercher explicitement l'activation sélective
d'éléments, le gel de zones élastiques, l'assemblage sélectif et l'intégration paresseuse.
Résultats :

- **Radermacher & Reese, *Model reduction in elastoplasticity: proper orthogonal
  decomposition combined with adaptive sub-structuring*, Comput. Mech. 54 (2014) 677–687.**
  C'est le relatif le plus proche trouvé. POD « sélective » appliquée uniquement aux
  sous-domaines à comportement approximativement élastique, éléments finis complets dans les
  sous-domaines plastiques, sous-structuration **adaptative**. Le titre même contient
  « adaptive sub-structuring ». Le critère de bascule et sa dynamique (réveil, rétrécissement)
  n'ont pas pu être vérifiés sur le texte intégral. **À lire impérativement.**
- **Kerfriden, Goury, Rabczuk, Bordas, *A partitioned model order reduction approach to
  rationalise computational expenses in nonlinear fracture mechanics*, CMAME 256 (2013)
  169–188.** Décomposition de domaine + ROM par projection ; l'effort est concentré autour de
  la zone d'endommagement ; « no a priori knowledge of the damage pattern is required ».
- Gendre, Allix, Gosselet, Comte, *Non-intrusive and exact global/local techniques for
  structural problems with local plasticity*, Comput. Mech. 2009 ; Gosselet et al. 2018
  (global/local vu comme Schwarz). Zone plastique locale traitée par un sous-modèle non
  linéaire couplé itérativement à un modèle global linéaire. La zone locale est **définie a
  priori**.
- Ryckelynck, *A priori hyperreduction method: an adaptive approach* (2005) ; *Hyper-reduction
  of mechanical models involving internal variables*, IJNME 2009. Le « domaine d'intégration
  réduit » (RID) : la loi de comportement n'est intégrée que sur quelques éléments, le reste
  est extrapolé par la base réduite. C'est une forme d'**intégration sélective de la loi**,
  mais l'extrapolation passe par une base réduite globale, pas par un état local.
- *Model order reduction of nonlinear THM systems by means of elastic and plastic domain
  sub-structuring*, Finite Elem. Anal. Des. 2024 : « the computationally expensive nonlinear
  iterative procedure is confined in the zone where plasticity is assumed to be restricted »
  — zone plastique supposée connue.
- Palchoudhary, Peter, Maurel, Ovalle, Kerfriden, *A plastic correction algorithm for
  full-field elasto-plastic FE simulations*, Comput. Mech. 2024 (arXiv:2402.06313) :
  correction plastique locale de type Neuber généralisé autour d'entailles, accélérée par
  méta-modèle et couche corrective apprise. Approche approchée, non garantie par le solveur.
- Approches ML « à l'intérieur de Newton » sans changement d'allocation spatiale :
  *Neural-Initialized Newton* (arXiv:2511.06802), *ML-accelerated time integration of
  plasticity models* (arXiv:2606.14548), COMMET (arXiv:2510.00884), RL pour le choix du pas
  de temps (Mach. Learn. Comput. Sci. Eng. 2025).

## 3. Verdict sur le créneau

**Le créneau tel que formulé dans le cahier des charges (« arbitrage en ligne, mobile, à
maillage fixé ») n'est pas vierge.** Radermacher & Reese (2014) et Kerfriden et al. (2013)
font déjà une partition adaptative élastique/plastique à maillage fixé, et Wang et al.
(2025) font croître dynamiquement la zone ML. Ce que nous n'avons trouvé nulle part, et qui
constitue la contribution repositionnée :

1. **Une étude de potentiel oracle** quantifiant, a posteriori et pondérée par le coût
   réel, la fraction d'éléments-pas extrapolables, avec sa dépendance à la tolérance et à la
   variable observée (contrainte vs déformation plastique cumulée). Nous n'avons trouvé
   aucune borne haute de ce type publiée.
2. **Une extrapolation locale par état interne** (dernier état convergé + tangente locale,
   intégration différée de la loi) sans base réduite ni substitut : la garantie est celle du
   solveur, il n'y a rien à entraîner hors ligne. Les relatifs proches passent tous par une
   ROM (POD, RID) ou un substitut.
3. **Le traitement explicite du réveil** (décharge/recharge cyclique) comme problème de
   décision, et la question de l'apport d'un critère appris **contre une heuristique
   optimisée**, avec la comparaison faite à ligne de base « Sysala » (réassemblage plastique
   seul) et non contre un solveur naïf. Nous n'avons trouvé aucune comparaison de ce type.

Le positionnement honnête pour l'article est donc : *nous ne revendiquons pas l'idée
de partitionner le domaine en zones actives et calmes, qui est connue ; nous revendiquons
la mesure de son potentiel réel, une réalisation sans modèle réduit, et la réponse à la
question de la valeur ajoutée de l'apprentissage pour la décision.*

Points à vérifier sur texte intégral avant soumission : critère de bascule et gestion du
réveil chez Radermacher & Reese ; critère algébrique chez Kerfriden et al. ; mécanisme
d'expansion chez Wang et al.
