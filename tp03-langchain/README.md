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

Ollama avec `ministral-3:3b` et `bge-m3`. `langchain-core`, `langchain-ollama`
et `langsmith` sont déjà dans l'environnement conda.

**Un compte LangSmith personnel**, gratuit, à créer en trois minutes — la marche
à suivre complète est dans le **RÉG 0 du notebook**, le tout premier exercice :

1. <https://smith.langchain.com> → *Sign up*. **La région (`US` ou `EU`) est
   demandée à l'inscription et n'est plus modifiable** : prenez `EU` si vous
   hésitez ;
2. *Settings* → *API Keys* → *Create API Key*, type **Personal Access Token**.
   La clé commence par `lsv2_pt_` et n'est affichée qu'une fois ;
3. collez-la **dans une variable, en haut du notebook** (`LANGSMITH_API_KEY = "…"`).

Rien à éditer à côté : le notebook n'est **jamais versionné**
(`python tp.py notebooks` le refabrique depuis `tp03.py` à chaque fois), donc
coller une clé dans une cellule ne finit ni dans un commit ni dans une pull
request. Sans compte, le TP fonctionne quand même — vous sautez OBS 2.

## Les exercices

| Type | Exercice |
|---|---|
| **RÉG 0** | régler sa clé LangSmith |
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

- *Rien n'apparaît dans LangSmith* → remontez au RÉG 0, vérifiez la clé et
  relancez sa cellule : le message dit lequel des trois problèmes vous avez
  (pas de clé, clé refusée, traçage coupé). Neuf fois sur dix c'était la
  **région** avant que `commun/tracage.py` ne la trouve tout seul en présentant
  la clé aux deux serveurs — plus la peine d'y penser.
- *Le projet est vide alors que `activer()` dit « actif »* → les traces partent
  depuis un fil d'arrière-plan. `tracage.rapport()` attend l'envoi, compte ce qui
  est arrivé et vous donne l'URL du projet.
