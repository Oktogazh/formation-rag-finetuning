# TP 4 — QLoRA et RAFT : apprendre au modèle à se servir du contexte

**Objectif.** Construire un jeu d'entraînement où le contexte est parfois bon et
parfois trompeur, entraîner un adaptateur LoRA sur le modèle **quantifié en
4 bits**, puis comparer le modèle de base et le modèle adapté **avec exactement
le même prompt**. Et regarder ce que le fine-tuning sait faire — et ce qu'il ne
sait pas faire.

**Durée.** Noyau ≈ 90 min (dont 20 à 30 min d'entraînement) · bonus ≈ 15 min ·
difficulté ★★★★☆

## Les trois mots, séparés

**Quantization.** Les poids passent de 16 bits à 4. Le modèle occupe quatre fois
moins de mémoire et tient sur un portable. On y perd un peu de qualité.

**LoRA.** On gèle le modèle et on entraîne, à côté, deux petites matrices par
couche. Quelques millions de paramètres au lieu de trois milliards. Le résultat,
l'« adaptateur », pèse quelques dizaines de mégaoctets et se branche ou se
débranche.

**QLoRA** = LoRA sur un modèle quantifié. C'est ce qui rend l'exercice possible
ici et maintenant.

**RAFT** (*Retrieval-Augmented Fine-Tuning*, Zhang et al., 2024) est une
quatrième idée, et c'est celle du TP : pendant l'entraînement, le contexte
contient **parfois** les bons voisins, parfois des distracteurs. Un modèle
entraîné uniquement avec le bon contexte apprend à le recopier ; le jour où la
recherche se trompe, il recopie une bêtise avec assurance.

## Ce que vous allez mesurer

La comparaison base / adapté, **par catégorie**. Regardez dans cet ordre :

1. la ligne `piege` et la colonne `Termino` — c'est là que le fine-tuning
   travaille : le style et les conventions du client entrent dans les poids ;
2. la ligne `nouveau` — **si elle progresse autant que les autres, quelque chose
   ne va pas**. Le fine-tuning n'apporte pas de connaissance nouvelle ;
3. la colonne `Tokens` — elle ne bouge pas. Le fine-tuning ne raccourcit pas le
   prompt, contrairement à ce qu'on lit souvent.

## Prérequis, selon votre machine

| Machine | À installer | Modèle utilisé | Durée |
|---|---|---|---|
| Apple Silicon, 16 Go | `pip install mlx-lm` | `mlx-community/Ministral-3-3B-Instruct-2512-4bit` | 20 à 30 min |
| NVIDIA, 8 Go de VRAM | `pip install torch --index-url https://download.pytorch.org/whl/cu124` puis `pip install transformers peft trl bitsandbytes datasets accelerate` | `mistralai/Ministral-3-3B-Instruct-2512` | 15 à 25 min |
| Processeur seul, ou Windows sans carte | rien | — | **pas d'entraînement** |

Sans accélérateur, l'entraînement prendrait des heures. Ce n'est pas un défaut
de votre machine, c'est la réalité du fine-tuning. Vous faites les TODO 1 et 2,
vous sautez `entrainer`, et vous récupérez l'adaptateur du formateur :

```bash
python tp.py adaptateur --telecharger
```

`python tp.py check` vous dit dans quelle ligne du tableau vous êtes.

## Déroulé

**1. Séparer sans fuite** — TODO 1.

Un segment de mémoire trop proche d'un segment d'évaluation est écarté des deux
jeux : l'entraîner reviendrait à donner les réponses de l'examen. Le seuil de
0,98 est un arbitrage et le code le dit : à 0,98 on écarte 50 segments sur 444 ;
à 0,95 on en écarterait 226, soit la moitié de la mémoire.

**2. Fabriquer les exemples RAFT** — TODO 2.

C'est le TODO le plus dense de la formation (★★★), et c'est normal : **la
fabrication des données est 80 % du travail de fine-tuning**, et c'est là que
se trouvent 80 % des erreurs.

```bash
python tp.py donnees
```

Puis **ouvrez `resultats/tp04/donnees/train.jsonl` et lisez trois lignes.**
C'est le seul moyen de voir ce que le modèle va réellement apprendre. Vérifiez
en particulier qu'un segment ne voit jamais sa propre traduction dans son
contexte.

**3. Régler LoRA** — TODO 3.

Quatre nombres, et ils ont un sens : rang, alpha, dropout, matrices ciblées. La
docstring les explique un par un.

**4. Entraîner.**

```bash
python tp.py entrainer
```

Regardez la perte descendre. Si elle stagne, le pas d'apprentissage est trop
petit ; si elle oscille, il est trop grand. Notez la durée : c'est une donnée
utile quand on vous demandera un budget.

**5. Mesurer.**

```bash
python tp.py comparer --n 40
```

Le tableau de deltas, catégorie par catégorie. **Relisez la ligne `nouveau`.**
Si elle a peu bougé, le fine-tuning a fait exactement ce qu'il sait faire, et
rien de plus. C'est le résultat attendu, et c'est une bonne nouvelle : cela veut
dire que votre RAG sert encore.

## Les TODO

| # | Fichier | Difficulté | Ce qu'on attend | Test |
|---|---|---|---|---|
| 1 | `tp04/donnees.py::separer` | ★★ | train / validation sans fuite, reproductible | `-k todo1` |
| 2 | `tp04/donnees.py::exemple_raft` | ★★★ | oracle ou distracteurs, au format du prompt d'inférence | `-k todo2` |
| 3 | `tp04/entrainer.py::config_lora` | ★ | rang, alpha, dropout, cibles | `-k todo3` |

Le test `test_le_prompt_d_entrainement_est_celui_de_l_inference` est le plus
important du TP et le moins spectaculaire. Si les deux formats divergent, rien
ne plante : le modèle est simplement moins bon que prévu, et personne ne sait
pourquoi.

## Bonus

**BONUS 4** — `exemple_sans_contexte` : glisser quelques exemples sans aucun
voisin. Sinon le modèle apprend qu'il y a **toujours** un bloc « Mémoire de
traduction », et le jour où la recherche ne rend rien, il est désorienté par un
prompt qu'il n'a jamais vu.

## Si ça coince

- *`entrainer` dit qu'il manque une bibliothèque* → suivez la ligne de votre
  machine dans le tableau des prérequis.
- *L'entraînement dépasse 30 minutes* → `python tp.py entrainer --epoques 1`.
- *Vous avez une clé Mistral* → `entrainer` refuse, et c'est volontaire : on ne
  fine-tune pas par l'API dans cette formation, parce que le sujet du TP est
  précisément ce qui se passe dans les poids. Prenez l'adaptateur du formateur.
- *`comparer` affiche « pas de modèle fine-tuné disponible »* → l'adaptateur
  n'est pas là. `python tp.py adaptateur --telecharger`.
