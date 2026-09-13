# TP 1 — Le prompt : ce qu'il donne, et où il s'arrête

**Durée** noyau ≈ 45 min · **Niveau** ★★☆☆☆ · **Indépendant** des autres TP

Traduire 20 segments suédois avec un prompt nu, mesurer, améliorer le prompt à
la main, mesurer encore — puis trouver le point où en mettre plus cesse de payer.

## Lancer

```bash
python tp.py notebooks tp01
jupyter lab
```

Ouvrez `tp01-prompt/tp01.ipynb` et exécutez les cellules dans l'ordre.

## Prérequis

Ollama avec `ministral-3:3b`, ou une clé `MISTRAL_API_KEY` dans `.env`.
`python tp.py check` vous le dit.

## Les exercices

| Type | Exercice |
|---|---|
| OBS 1 | ce que votre machine sait faire |
| OBS 2 | le point zéro : 20 segments, prompt nu |
| LIRE 1 | où est le prompt, exactement ? |
| RÉG 1 | la température, et pourquoi toutes les mesures sont à zéro |
| **CODE 1** | écrire la consigne de style du client |
| OBS 3 | ce que la consigne change — et ce qu'elle dégrade |
| RÉG 2 | combien d'exemples faut-il montrer ? |
| OBS 4 | le mur : tout dans le prompt |
| ARB 1 | quelle métrique défendez-vous ? |
| BONUS 1 | choisir les exemples en fonction du segment |

## Si ça coince

```bash
python tp.py test tp01 -k code1     # le test de l'exercice
python tp.py indice tp01 --code 1   # le diff avec le corrigé
python tp.py rattraper tp01         # repartir du corrigé
```

- *`import commun` échoue* → la première cellule remonte à la racine du dépôt.
  Exécutez-la avant les autres.
- *Une cellule met plus de dix secondes par segment* → vous êtes sur processeur.
  Baissez `n` dans `atelier.segments(n=20)`, et demandez une clé au formateur.
