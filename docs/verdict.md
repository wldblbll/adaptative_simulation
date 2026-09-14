# Verdict

Une page, sans détour. Chiffres issus de `results/` (scripts `experiments/`).

## Est-ce une bonne idée ?

**Oui, sous une forme plus étroite que celle du cahier des charges.** Concentrer
l'intégration de la loi et l'assemblage sur la zone plastique, en remplaçant les autres
éléments par leur linéarisation exacte f^e_0 + K^e_0 Δu et en les contrôlant par le
prédicteur élastique, fonctionne **sans aucune dégradation** de la solution (1e-12) et
sans itération de Newton supplémentaire. L'idée telle qu'écrite — extrapoler aussi les
zones « calmes mais plastiques » — ne fonctionne pas en ligne : l'oracle la créditait,
l'équilibre et l'inversion de charge la détruisent.

## Quel gain, sur quels problèmes, à quelle erreur ?

- **46 à 57 % du coût élémentaire** (modèle de coût scalaire ; 47 à 61 % en vectorisé)
  par rapport à une référence qui réutilise déjà la tangente élastique, à erreur
  ≤ 1e-12, sur plaque entaillée, console, et leurs versions cycliques à écrouissage
  cinématique. +2 à +7 points de plus pour une erreur de 1e-5 à 1e-3 (contrôle au
  prédicteur seul ou après le pas).
- Sur le **temps total** de ce prototype 2D Python : +1 à +16 %, parce que la part
  élémentaire du temps n'y est que de 27 à 43 % (factorisation dominante). Le gain mur
  est le produit du gain élémentaire par cette part : il est réel pour les lois de
  comportement coûteuses (le ratio plastique/élastique de J2 n'est que de 3) et faible
  pour les grands modèles dominés par le solveur linéaire.
- Borne haute oracle (Phase 0) : 85–90 % du coût élémentaire ; borne d'une politique
  exacte idéale (contrôle gratuit) : 56–74 %. L'écart entre les deux (13–34 points) est
  la part plastique, non transférable ; l'écart entre l'idéal et l'obtenu (7–17 points)
  est le coût du contrôle.

## Quelles limites, quels cas contre-productifs ?

- Plasticité diffuse : l'ensemble actif tend vers tout le domaine, le contrôle s'ajoute
  au coût. Structures uniformément proches de la limite : tout doit être contrôlé, le
  gain tombe à 5–7 %.
- Tout mode non exact (intégration paresseuse des éléments plastiques) est
  contre-productif : erreurs de 1e-3 à 6.0, itérations de Newton +10 à +60 %, et sur
  chargement cyclique la solution est détruite (chemin non proportionnel dans
  l'intégration différée).
- La propagation aux voisins et les marges de précaution coûtent 10 à 27 points sans
  rien apporter. Le Newton modifié pour économiser la factorisation diverge.
- Périmètre : 2D, J2, chargements proportionnels, prototype Python ; rapports de coûts
  mesurés sous deux modèles, transférables à ±2 points près ; textes intégraux des
  relatifs proches non lus.

## L'IA apporte-t-elle quelque chose, où, combien ?

**Non, pas dans ce cadre.** Une politique apprise (gradient boosting, 800 k
échantillons tirés des simulations d'une famille de pièces, sans dataset externe)
prédit l'horizon avant plastification 2 à 4 fois mieux que l'heuristique, et mieux
dans sa famille qu'en dehors (H3 réel au niveau prédiction). Mais aucune de ses 24
configurations n'atteint le régime exact sur aucun des 5 cas : quand elle économise
plus que l'heuristique réglée, c'est en manquant des plastifications (erreur 2e-3 à
4e-1). La borne de ce qu'une anticipation parfaite pourrait apporter (H1) est de **3 à
12 points** de coût élémentaire (oracle 52–69 % contre heuristique 46–57 %), concentrés
sur les cas cycliques où le contrôle coûte le plus. H2 (renforcement) n'a pas été testé,
par choix argumenté : il n'y a pas de budget d'erreur à distribuer dans le régime exact.
La raison de fond : la décision utile est une décision de *sûreté*, et le critère
physique (prédicteur élastique + marge/vitesse) est sûr par construction et presque
optimal.

## Que faudrait-il faire ensuite ?

1. Porter la méthode sur une loi de comportement coûteuse (plasticité cristalline,
   endommagement) et en 3D : c'est là que le gain élémentaire pèse sur le temps total.
2. Attaquer la phase solveur : mise à jour de rang faible de la factorisation sur les
   seuls blocs actifs, plutôt qu'un Newton modifié.
3. Si apprentissage il y a : une politique qui ne fait que *retarder* le contrôle sous
   la borne κ (jamais au-delà), évaluée sur les cas cycliques, seul endroit où il reste
   3–12 points à prendre.
4. Lire Radermacher & Reese (2014) et Kerfriden et al. (2013) en texte intégral avant
   toute soumission, et repositionner si nécessaire.
