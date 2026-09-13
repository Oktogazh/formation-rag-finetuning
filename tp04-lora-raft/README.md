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
```

Environ 250 Mo, aucun GPU. Le modèle se télécharge au premier usage (270 Mo).

## Les exercices

| Type | Exercice |
|---|---|
| OBS 1 | ce que le petit modèle sait faire avant l'entraînement |
| **CODE 1** | fabriquer un exemple RAFT : oracle ou distracteurs |
| RÉG 1 | le dosage des distracteurs |
| LIRE 1 | ce que le modèle va réellement apprendre |
| **CODE 2** | régler LoRA |
| RÉG 2 | ce que le rang coûte |
| LIRE 2 | la boucle d'entraînement, quarante lignes |
| OBS 2 | entraîner |
| OBS 3 | avant, après |
| ARB 1 | qu'a appris le modèle ? |
| BONUS 4 | un exemple sans contexte |

## Si ça coince

- *L'entraînement dépasse quinze minutes* → réduisez le jeu :
  `jeu[:150]` dans la cellule d'entraînement. La démonstration tient encore.
- *`torch` refuse de s'installer* → faites les exercices CODE et LIRE, sautez
  l'entraînement, et regardez la table d'un voisin. Les exercices notés
  fonctionnent sans torch.
