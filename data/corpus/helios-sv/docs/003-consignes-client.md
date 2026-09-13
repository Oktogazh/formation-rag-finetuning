---
id: client-003
titre: Consignes particulières du client Helios
source: interne/clients/helios/consignes
date: 2026-05-06
categorie: client
---

## Ce qui est fourni

Chaque lot de traduction arrive avec la mémoire de traduction du compte, le
glossaire à jour et la référence de la version du produit. La mémoire fait
foi : un segment déjà validé ne se retraduit pas, il se réutilise.

## Correspondances de la mémoire

- Au-dessus de **95 %** de correspondance, le segment de la mémoire est réutilisé
  tel quel après vérification des chiffres.
- Entre **65 et 95 %**, la mémoire sert de modèle, mais le texte est retraduit.
  C'est là que se produisent les erreurs les plus coûteuses : recopier un
  segment voisin dont un chiffre ou une négation diffère.
- En dessous de **65 %**, on traduit sans regarder la mémoire, qui n'apporte que
  du bruit.

## Chiffres, codes et marqueurs

Tout chiffre présent dans la source doit se retrouver dans la cible, à
l'identique. Cela vaut pour les durées, les tailles de fichier, les codes
d'erreur HTTP et les numéros de version.

Un segment dont un chiffre a changé par rapport à la mémoire est un piège connu :
c'est la première chose que vérifie la relecture.

## Délais de publication

Helios publie sa documentation le premier mardi de chaque mois. Un lot livré
après le vendredi précédent part dans la publication suivante.
