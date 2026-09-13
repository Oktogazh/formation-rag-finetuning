# TP 6 — Cinq crans, un seul graphe, et le verdict

**Objectif.** Assembler tout ce que vous avez construit dans un graphe LangGraph
qui **route** chaque segment vers le moyen le moins cher qui suffit, puis lire la
table finale des trois jours et décider ce qu'on mettrait en production.

**Durée.** Noyau ≈ 75 min · bonus ≈ 20 min · difficulté ★★★☆☆

## Ce que vous avez construit, en résumé

| Moyen | Appels au modèle | Ce qu'il sait faire |
|---|---|---|
| réutiliser la mémoire | **0** | rendre un segment déjà traduit, chiffres reportés |
| RAG | 1 | traduire avec les voisins et le glossaire |
| RAG + correction | 2 | rattraper une faute de terminologie ou de chiffre |
| RAG + adaptateur | 1 | la même chose, avec le style du client dans les poids |

Aucun n'est bon partout. **Le travail d'ingénierie consiste à choisir lequel
s'applique à quel segment** — et c'est exactement ce que fait un graphe.

## La règle de routage, qui vient du client

`docs/003-consignes-client.md` dit : au-dessus de 95 % de correspondance, on
réutilise le segment de la mémoire **après vérification des chiffres**. Les deux
conditions, pas une seule. Un segment très proche dont un **mot** a changé ne se
réutilise pas : c'est la faute la plus chère qu'un outil de TAO puisse produire.

Sur le corpus Helios, mesuré le 13 septembre 2026 : **41 segments sur 80**
remplissent la condition, et la réparation floue rend la référence **exacte 41
fois sur 41**, sans un seul appel au modèle.

La leçon est inconfortable et c'est la meilleure de la formation : **pour la
moitié du corpus, la bonne réponse n'est pas d'appeler un modèle de langue.**
Non pas parce que le modèle échouerait — sur ces segments-là il s'en sort bien —
mais parce qu'il coûte du temps, de l'argent et de l'incertitude pour produire
ce qu'un remplacement de texte donne exactement et instantanément.

## Prérequis

Les TP 1, 2 et 3 faits — le graphe réutilise la recherche, le prompt augmenté et
la vérification. Le TP 4 est utile mais pas obligatoire : sans adaptateur,
`--moteur base` fonctionne.

## Déroulé

**1. Le routeur** — TODO 1, dans `tp06/graphe.py`.

**2. La sortie de boucle** — TODO 2. Avec compteur, toujours.

**3. Câbler le graphe** — TODO 3.

```
analyser ──┬─(reutiliser)──────────────────────────┐
           └─(recuperer)── traduire ── verifier ──┬┴─ livrer ── FIN
                               ▲                  │
                               └──── corriger ◄───┘
```

La boucle `corriger → verifier` est ce qui distingue un graphe d'une chaîne :
une chaîne ne revient jamais en arrière.

```bash
python tp.py test tp06
python tp.py graphe --n 80
```

Regardez la répartition des chemins, et la ligne « dont N sans aucun appel ».

**4. Avec l'adaptateur du TP 4.**

```bash
python tp.py graphe --n 80 --moteur adapte
```

**5. Le verdict.**

```bash
python tp.py banc
```

C'est **votre** parcours, reconstruit depuis `resultats/` : chaque commande de
chaque TP y a déposé sa mesure. Les lignes que vous n'avez pas mesurées sont
marquées comme telles.

Comment la lire, dans cet ordre :

1. **BLEU monte-t-il d'un cran à l'autre ?** Si un cran ne fait pas mieux que le
   précédent, il ne mérite pas sa complexité. C'est vrai pour le RAG comme pour
   le fine-tuning.
2. **La colonne Appels.** Le cran le plus juste n'est pas le plus cher, et c'est
   le résultat le plus utile de ces trois jours.
3. **La ligne `nouveau` des tableaux par catégorie.** Elle rappelle ce
   qu'aucune technique ne sait faire : inventer ce qui n'est nulle part.

## Ce que ça donne — mesuré le 13 septembre 2026

`ministral-3:3b` sous Ollama, recherche dense `k=3`, les **80** segments, moteur
de base (sans adaptateur).

| | BLEU | chrF | Termino | Chiffres | s/seg | Appels |
|---|---|---|---|---|---|---|
| TP 3, chaîne seule | 70,5 | 84,7 | 100 % | 96 % | 1,14 | 1,00 |
| TP 3, + vérification | 66,8 | 83,8 | 100 % | 98 % | 1,74 | 1,04 |
| **TP 6, graphe routé** | **75,1** | **86,1** | 100 % | 98 % | **1,01** | **0,51** |

Comparez uniquement des lignes mesurées sur le **même nombre de segments** :
ces trois-là le sont, sur les 80.

Le graphe fait mieux que la chaîne **et coûte moitié moins d'appels**. Par
catégorie :

| Catégorie | BLEU | Appels | Chemin |
|---|---|---|---|
| `repetition` (21) | **100,0** | **0,00** | réutiliser |
| `piege` (20) | **100,0** | **0,00** | réutiliser |
| `fuzzy` (18) | 47,9 | 1,00 | RAG |
| `nouveau` (21) | 51,0 | 1,10 | RAG, parfois corrigé |

Répartition : 41 segments réutilisés, 37 en RAG, 2 en RAG + correction.

Deux choses à dire à voix haute devant ce tableau :

1. **Le gain vient du routage, pas d'une meilleure traduction.** Sur `fuzzy` et
   `nouveau`, le graphe fait exactement ce que faisait le TP 3, aux mêmes
   chiffres. Tout le progrès est d'avoir cessé d'appeler le modèle là où il
   n'apportait rien.
2. **Le plafond, ce sont les lignes `fuzzy` et `nouveau`**, à 48 et 51. Elles
   n'ont bougé ni au TP 3, ni au TP 4, ni ici. Ce qui n'est pas dans la mémoire
   n'y sera pas mis par une technique d'orchestration. C'est la limite honnête à
   annoncer à un client, et c'est le vrai sujet d'une formation sur le RAG.

**6. Tracer le graphe** (facultatif). Avec `LANGSMITH_TRACING=true`, relancez
trois segments : vous voyez le chemin pris par chacun. C'est la clôture visuelle
de la formation.

## Les TODO

| # | Fichier | Difficulté | Ce qu'on attend | Test |
|---|---|---|---|---|
| 1 | `tp06/graphe.py::router` | ★★ | seuil **et** chiffres | `-k todo1` |
| 2 | `tp06/graphe.py::apres_verification` | ★★ | corriger ou livrer, avec compteur | `-k todo2` |
| 3 | `tp06/graphe.py::construire_graphe` | ★★ | la topologie ci-dessus | `-k todo3` |

## Bonus

**BONUS 6** — le graphe **sans RAG** : le modèle adapté, seul, privé de voisins
et de glossaire.

```bash
python tp.py test tp06 --bonus -k bonus6
```

C'est la seule façon de répondre honnêtement à la question du dernier
après-midi : **le fine-tuning remplace-t-il le RAG ?** Mesurez, puis répondez.
La ligne `4b adaptateur sans RAG` apparaîtra dans le banc à côté des autres.

## Si ça coince

- *`test_todo3` dit qu'une arête manque* → comparez avec le schéma ci-dessus ;
  `add_conditional_edges` prend un dictionnaire qui associe la **valeur rendue**
  par la fonction au **nom du nœud**.
- *Une répétition passe quand même par le modèle* → votre routeur ne teste
  probablement que le taux, pas les chiffres.
- *Le graphe tourne en rond* → `apres_verification` ne regarde pas
  `tentatives`.
- *`--moteur adapte` dit qu'il n'y a pas d'adaptateur* → faites le TP 4, ou
  `python tp.py adaptateur --telecharger`.
