# TP 3 — Orchestrer : chaîne, vérification, correction

**Durée** noyau ≈ 60 min · **Niveau** ★★★☆☆ · **Indépendant** des autres TP

Refaire le RAG avec LangChain, puis ajouter un contrôle automatique et une
boucle de correction. Et mesurer ce que ça coûte.

## Lancer

```bash
python tp.py notebooks tp03
jupyter lab
```

## Prérequis

Ollama avec `ministral-3:3b` et `bge-m3`. `langchain-core` et `langchain-ollama`
sont déjà dans l'environnement conda.

Pour l'observation du traçage, le formateur donne une clé LangSmith en séance.
Sans elle, le TP fonctionne, vous sautez un exercice.

## Les exercices

| Type | Exercice |
|---|---|
| LIRE 1 | les trois briques de LCEL |
| **CODE 1** | assembler la chaîne |
| OBS 1 | la chaîne fait-elle la même chose qu'un RAG écrit à la main ? |
| LIRE 2 | les quatre contrôles de l'agence |
| RÉG 1 | la boucle de correction, et ce qu'elle dégrade |
| RÉG 2 | et si on réparait sans appeler le modèle ? |
| OBS 2 | voir la chaîne tourner dans LangSmith |
| ARB 1 | que livrez-vous au client ? |

## Si ça coince

```bash
python tp.py test tp03 -k code1
python tp.py indice tp03 --code 1
```

- *Rien n'apparaît dans LangSmith* → neuf fois sur dix c'est la région. Un compte
  européen exige `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`, et
  sans cette ligne les traces n'arrivent jamais sans que rien ne le signale.
