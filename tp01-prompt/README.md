# TP 1 — Le prompt : ce qu'il donne, et où il s'arrête

**Durée** noyau ≈ 65 min · **Niveau** ★★☆☆☆ · **Indépendant** des autres TP

D'abord tout à la main avec `transformers` — tokeniser, embedder, générer,
détokeniser, et voir le préprompt qu'on vous pose à votre insu. Ensuite avec
`commun/` : traduire 20 segments avec un prompt nu, mesurer, améliorer le prompt,
mesurer encore — puis trouver le point où en mettre plus cesse de payer.

## Lancer

```bash
python tp.py notebooks tp01
jupyter lab
```

Ouvrez `tp01-prompt/tp01.ipynb` et exécutez les cellules dans l'ordre.

## Prérequis

Ollama avec `ministral-3:3b`, ou une clé `MISTRAL_API_KEY` dans `.env`.
Plus, pour le chapitre 0, `transformers` et le petit modèle `SmolLM2-135M`
(270 Mo, processeur, celui du TP 4) :

```bash
python -c "from huggingface_hub import snapshot_download as d; d('HuggingFaceTB/SmolLM2-135M-Instruct')"
```

`python tp.py check` vous dit ce qui manque.

## Les exercices

| Type | Exercice |
|---|---|
| OBS 1 | ce que votre machine sait faire |
| | **Chapitre 0 — sous le capot, à la main, avec `transformers`** |
| RÉG 1 | tokeniser : du texte à une liste d'entiers (changez `MON_TEXTE`) |
| OBS 2 | la couche d'embedding : des entiers à des listes de nombres |
| OBS 3 | un pas de modèle : les logits, et le token suivant |
| LIRE 1 | la boucle de génération, écrite à la main, puis détokenisée |
| OBS 4 | le préprompt et les jetons du gabarit de conversation |
| OBS 5 | trois requêtes au vrai modèle : sans préprompt, instruction nue, préprompt |
| | **Chapitre 1 — mesurer, puis améliorer le prompt** |
| OBS 6 | le point zéro : 20 segments, prompt nu |
| LIRE 2 | où est le prompt, exactement ? |
| RÉG 2 | la température, et pourquoi toutes les mesures sont à zéro |
| **CODE 1** | écrire la consigne de style du client |
| OBS 7 | ce que la consigne change — et ce qu'elle dégrade |
| RÉG 3 | combien d'exemples faut-il montrer ? |
| OBS 8 | le mur : tout dans le prompt |

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
