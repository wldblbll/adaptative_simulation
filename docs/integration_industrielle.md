# Note d'intégration industrielle

Que faudrait-il pour greffer l'orchestrateur sur un code EF industriel existant ? Cette
note s'appuie sur la réalisation du prototype (`adaptfem/`), qui a été conçu pour que
l'orchestrateur soit une **couche externe à la boucle de Newton**, sans modification du
noyau élémentaire ni de la loi de comportement.

## 1. Ce que fait exactement l'orchestrateur, et ce qu'il ne fait pas

À chaque itération de Newton, le solveur demande à chaque élément sa force interne
f^e(u) et, aux itérations où la tangente est réassemblée, son bloc K^e(u). L'orchestrateur
intercepte cette demande et, pour un élément qu'il a classé *calme*, renvoie

    f^e = f^e_0 + K^e_0 (u_e − u_e,0),     K^e = K^e_0

où l'indice 0 désigne le dernier état où l'élément a été effectivement intégré
(« ancre »). Aucun point de Gauss n'est visité, aucune variable interne n'est touchée.
Quand l'élément est réveillé, la loi est intégrée **une seule fois sur l'incrément
total** depuis l'ancre (intégration différée) : l'état obtenu est celui d'un retour
radial, donc admissible ; le chemin de déformation entre l'ancre et le réveil est
approché par un segment.

Il ne fait rien d'autre : pas de changement de maillage, de pas de charge, de
tolérance de Newton, de solveur linéaire. Le maillage reste celui de l'utilisateur ; les
degrés de liberté des zones calmes restent dans le système linéaire. *La méthode réduit
le coût du sur-raffinement inutile, elle ne le supprime pas.*

## 2. Points d'accroche nécessaires dans le solveur

| besoin | forme minimale | présent dans les codes courants ? |
|---|---|---|
| A. Boucle élémentaire filtrable | itérer sur un sous-ensemble d'éléments pour l'intégration de la loi et le calcul de K^e, f^e | Oui dans la plupart des codes (boucles par groupes, éléments désactivés) ; le filtre par itération est rarement exposé |
| B. Stockage par élément de l'ancre | f^e_0 (n_dof,e), K^e_0 (dense), u_e,0 | K^e est souvent recalculé et non stocké : c'est le point le plus intrusif en mémoire (64 réels par Q4, ~600 par HEX8 à 24 ddl, ~4700 pour un HEX20) |
| C. Accès en lecture aux variables internes de l'ancre | contrainte, déformation, tangente cohérente stockée, variables d'écrouissage | Les codes stockent l'état ; la tangente cohérente par point de Gauss est stockée rarement (elle est recalculée). Sans elle, l'extrapolation tangente doit être remplacée par K^e_0 seul, ce qui suffit à l'équation d'équilibre ; la tangente par point de Gauss ne sert qu'au **contrôle** (calcul de la contrainte extrapolée pour le test du critère) |
| D. Un « prédicteur élastique » exposé | calcul de la contrainte d'essai σ̂ = σ_0 + D_0 Δε et du critère f(σ̂) sans retour radial | C'est exactement la première moitié de tout algorithme de retour radial ; il faut pouvoir l'appeler seul |
| E. Un point d'entrée dans la boucle de Newton | après le calcul du résidu, avant la décision de convergence : possibilité d'ajouter des éléments à l'ensemble actif et de recalculer leur contribution | Rarement exposé ; c'est le point d'intrusivité algorithmique |
| F. Compteurs de coût | nombre d'appels à la loi, d'éléments assemblés, de factorisations | Nécessaires pour vérifier le gain ; souvent absents |

L'intégration différée (réveil) suppose que la loi tolère un incrément « long » sans
sous-incrémentation ; les lois industrielles avec sous-pas automatiques le gèrent.

## 3. Nature de l'intrusivité

- **Couche externe, non intrusive** : impossible avec les API usuelles, car l'intercept
  se situe *à l'intérieur* d'une itération de Newton (point E). Une approche par
  « éléments utilisateur » (UEL/UMAT) permet une forme dégradée : une UMAT peut décider
  de renvoyer la tangente et la contrainte extrapolées sans retour radial (points C, D)
  — mais elle est appelée à chaque point de Gauss, donc le coût de la boucle élémentaire
  et de B^T σ reste payé. Le gain se limite alors au retour radial et à la tangente,
  c'est-à-dire à la part « loi » du coût. Pour les lois lourdes c'est l'essentiel ; pour
  J2 c'est marginal (voir les mesures de coût dans `docs/rapport_experiences.md`).
- **Surcharge de la routine d'intégration + filtre de boucle** : niveau réaliste. Modifier
  la boucle d'assemblage pour sauter les éléments calmes et prendre leur contribution
  dans le cache (points A, B), et exposer le test du critère (D).
- **Modification du noyau** : nécessaire pour le réveil à l'intérieur de l'itération
  (E). Alternative moins intrusive : réveil *après convergence* du pas puis réitération
  du pas avec l'ensemble actif élargi (mode « converged » du prototype) — exact aussi,
  au prix d'itérations supplémentaires (mesuré : +40 à +80 % d'itérations de Newton sur
  nos cas). Le mode « post-step » (réveil au pas suivant) est non intrusif mais
  introduit une erreur (un pas résolu avec une réponse élastique sur un élément qui
  plastifie) ; il n'est acceptable qu'avec un rattrapage.

## 4. Interface minimale à exposer

```
orchestrator.begin_step(k, load_increment)          -> active element set
orchestrator.monitor(iterate)                         -> elements to wake (uses trial stress only)
element.integrate(el_subset, u)                       -> stress, tangent, K^e, f^e (existing kernel)
element.linearised_force(el_subset, u)                -> f^e_0 + K^e_0 (u_e - u_e0)  (cache)
element.trial_yield(el_subset, u)                     -> f(sigma_hat) per Gauss point   (no return mapping)
solver.newton_hook_after_residual(callback)           -> allows growing the active set mid-iteration
cost.counters                                          -> law calls, K^e computed, monitored points, factorisations
```

## 5. Garanties de robustesse qu'un éditeur exigerait

1. **Reproduction exacte de la référence** dans la configuration par défaut. Le
   prototype le fait : avec le contrôle exact du critère à chaque itération et les
   éléments plastiques maintenus actifs, la solution coïncide avec la référence à
   1e-13 près sur tous les cas (`results/phase2/tables.md`). C'est la configuration à
   livrer ; les modes qui introduisent une erreur (lecture différée, intégration
   paresseuse des éléments plastiques) doivent rester expérimentaux.
2. **Pas de dépendance à une donnée apprise pour la correction** : la politique apprise
   ne décide que *quand contrôler* un élément ; un contrôle manqué se traduit par une
   détection retardée d'un pas, jamais par un état inadmissible.
3. **Pas complet final obligatoire** et rattrapage complet sur demande (tous les N pas,
   ou sur critère de dérive) pour toute configuration non exacte.
4. **Dégradation contrôlée** : si l'ensemble actif dépasse un seuil (par exemple 60 %
   des éléments), l'orchestrateur se désactive de lui-même : le surcoût de contrôle
   n'est plus justifié.
5. **Traçabilité** : journal par pas de la fraction active, des réveils, du coût de
   contrôle ; compteurs comparables à ceux du calcul de référence.

## 6. Chemin réaliste

Option activable (« lazy element integration »), désactivée par défaut, avec deux
niveaux : *exact* (défaut de l'option) et *expérimental* (paramètres de tolérance
exposés). Réglage utilisateur limité au facteur de sécurité du contrôle (κ) ; le reste
sans réglage. Une étape préalable indispensable dans tout code : mesurer la part du
temps passée au niveau élémentaire, qui borne le gain (voir § 7).

## 7. Familles de calcul qui en bénéficient, et celles qui n'en bénéficient pas

**Favorables** : lois de comportement coûteuses par point de Gauss (plasticité
cristalline, viscoplasticité avec Newton local, endommagement à nombreuses variables,
lois utilisateur), zones non linéaires localisées et stables (entailles, congés,
assemblages boulonnés, contact local), chargements cycliques avec longues phases
élastiques (fatigue), calculs répétés sur une famille de pièces. Dans ces cas la part
élémentaire domine le temps total et la fraction active est faible.

**Défavorables** : plasticité diffuse (mise en forme, crash), où l'ensemble actif tend
vers tout le domaine ; grands modèles 3D à solveur direct où la factorisation domine
(le gain élémentaire est borné par la part élémentaire, mesurée à 27–43 % sur notre
prototype 2D) ; lois élastiques ou élastiques linéaires par morceaux où l'intégration
est déjà triviale ; dynamique explicite (pas de Newton, coût dominé par le pas critique,
domaine du sous-cyclage classique). Le contrôle exact coûte une évaluation du critère
sur les éléments surveillés : si presque tous les éléments sont proches de la limite
(structure uniformément chargée près de la plastification), le contrôle coûte autant
que le calcul et le gain disparaît.
