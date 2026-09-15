# TP 4 — LoRA et RAFT : apprendre au modèle à se servir du contexte

**Durée** noyau ≈ 90 min, dont ~10 min d'entraînement · **Niveau** ★★★★☆ ·
**Indépendant** des autres TP

## Le modèle change, et il faut savoir pourquoi

Fine-tuner les 3 milliards de paramètres de Ministral demande une carte
graphique que personne n'a ici. **Ce TP entraîne donc pour de vrai un modèle
assez petit pour tenir sur un processeur** : `SmolLM2-135M-Instruct`, 270 Mo.

Ses traductions sont mauvaises, et c'est une bonne nouvelle : l'écart entre
« avant » et « après » n'en est que plus lisible. Ce que le TP démontre reste
vrai à toutes les tailles — un adaptateur apprend une **forme**, pas une
**connaissance**, et la taille du modèle qu'on peut fine-tuner est une décision
matérielle.

**Ne comparez pas les chiffres de ce TP à ceux des autres.** Comparez « avant »
et « après », sur le même petit modèle.

## Lancer

```bash
python tp.py notebooks tp04
jupyter lab
```

## Prérequis

```bash
pip install torch peft
ollama pull bge-m3
```

Environ 250 Mo, aucun GPU. Le modèle se télécharge au premier usage (270 Mo).

`bge-m3` (1,2 Go) sert à chercher les termes du glossaire, comme au TP 2 : les
exemples d'entraînement portent le même glossaire filtré que l'inférence, et
c'est la condition pour que le fine-tuning serve à quelque chose. Si vous avez
déjà fait le TP 2, il est déjà là.

## Les exercices

| Type | Exercice |
|---|---|
| OBS 1 | ce que le petit modèle sait faire avant l'entraînement |
| OBS 2 | fabriquer un exemple RAFT : oracle ou distracteurs (lecture) |
| RÉG 1 | le dosage des distracteurs |
| LIRE 1 | ce que le modèle va réellement apprendre |
| **CODE 1** | régler LoRA |
| RÉG 2 | ce que le rang coûte |
| LIRE 2 | la boucle d'entraînement, quarante lignes |
| OBS 3 | entraîner |
| OBS 4 | avant, après |
| ARB 1 | qu'a appris le modèle ? |
| BONUS 4 | un exemple sans contexte |

## Ce que ça donne — mesuré le 13 septembre 2026

MacBook Apple Silicon, 355 exemples RAFT, rang 8, une époque : **4,6 minutes
d'entraînement**, perte 3,10 → 1,67, adaptateur de 1,9 Mo. Puis, sur 20
segments, avec exactement le même prompt :

| | BLEU | chrF | Termino | Tokens |
|---|---|---|---|---|
| avant | 2,1 | 16,2 | 14 % | 463 |
| après | **38,3** | **45,4** | **43 %** | 463 |

Et par catégorie, c'est là que tout se joue :

| Catégorie | avant | après |
|---|---|---|
| `repetition` | 4,4 | **67,2** |
| `piege` | 4,0 | **77,4** |
| `fuzzy` | 1,5 | 6,3 |
| `nouveau` | 1,2 | 5,5 |

**Lisez ces quatre lignes deux fois.** Là où un voisin ressemble au segment, le
modèle passe de rien à presque tout : il a appris le format, le vocabulaire
imposé et la façon de se servir du contexte. Là où il n'y a pas de voisin, il
n'a rien appris du tout — 1,2 à 5,5, c'est du bruit.

C'est la thèse du dernier après-midi, et elle est mesurée sur votre machine :
**le fine-tuning apprend une forme, pas une connaissance. Il ne remplace pas le
RAG, il le complète.**

La colonne `Tokens` ne bouge pas non plus : le fine-tuning ne raccourcit pas le
prompt, contrairement à ce qu'on lit partout.

## Si ça coince

- *L'entraînement dépasse quinze minutes* → réduisez le jeu :
  `jeu[:150]` dans la cellule d'entraînement. La démonstration tient encore.
- *`torch` refuse de s'installer* → faites les exercices CODE et LIRE, sautez
  l'entraînement, et regardez la table d'un voisin. Les exercices notés
  fonctionnent sans torch.
