---
id: proc-004
titre: Procédure de relecture et de post-édition
source: interne/qualite/relecture
date: 2026-02-20
categorie: procedure
---

## Les deux niveaux de post-édition

**Post-édition légère.** On corrige ce qui est faux : contresens, chiffre
altéré, terme de glossaire non respecté, marqueur perdu. On ne touche pas au
style si la phrase est compréhensible et correcte.

**Post-édition complète.** On amène le texte au niveau d'une traduction humaine :
registre, fluidité, cohérence avec le reste du document. C'est le niveau demandé
pour tout ce qui est publié sous le nom du client.

## Qui relit quoi

Un traducteur ne relit jamais sa propre production. Les lots issus d'une
traduction automatique passent systématiquement en post-édition complète par un
traducteur qui n'a pas paramétré le moteur.

## Les quatre contrôles automatiques

Avant toute relecture humaine, quatre contrôles tournent sur le lot :

1. **Terminologie** : chaque terme du glossaire présent dans la source doit
   apparaître dans sa forme imposée dans la cible.
2. **Chiffres** : les nombres de la source et de la cible doivent correspondre.
3. **Marqueurs** : `{0}`, `{1}` conservés et appariés.
4. **Adresse** : aucune forme de tutoiement dans la cible.

Un lot qui échoue à l'un des quatre contrôles ne part pas en relecture : il
retourne au moteur.

## Traçabilité

Chaque segment livré porte l'origine de sa traduction : mémoire, moteur, ou
humain. C'est ce qui permet, six mois plus tard, de savoir ce qu'il faut
reprendre quand le glossaire change.
