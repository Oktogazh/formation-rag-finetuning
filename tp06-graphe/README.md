# TP 6 — Cinq crans, un seul graphe, et le verdict

**Durée** noyau ≈ 75 min · **Niveau** ★★★☆☆ · **Indépendant** des autres TP

Router chaque segment vers le moyen le moins cher qui suffit, puis lire la table
des trois jours.

## Lancer

```bash
python tp.py notebooks tp06
jupyter lab
```

## Prérequis

Ollama avec `ministral-3:3b` et `bge-m3`. `langgraph` est déjà dans
l'environnement conda.

## Les exercices

| Type | Exercice |
|---|---|
| LIRE 1 | les sept nœuds |
| **CODE 1** | le routeur : le seuil **et** les chiffres |
| OBS 1 | le graphe tourne, et la répartition des chemins |
| RÉG 1 | le seuil de réutilisation |
| OBS 2 | et sans RAG ? |
| OBS 3 | le banc |
| ARB 1 | quel cran pour quelle catégorie ? |
| ARB 2 | ce que vous recommandez |

## Si ça coince

```bash
python tp.py test tp06 -k code1
python tp.py indice tp06 --code 1
```

- *Une répétition passe quand même par le modèle* → votre routeur ne teste
  probablement que le taux, pas les chiffres. Les deux conditions.
- *Le graphe tourne en rond* → `apres_verification` ne regarde pas
  `tentatives`. Une boucle d'agent sans compteur est un incident de production.
