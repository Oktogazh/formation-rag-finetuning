# RAG et fine-tuning d'un LLM — les travaux pratiques

Support de travaux pratiques de la formation **« RAG et fine-tuning d'un LLM »**
(PLB, 3 jours). Un dépôt, six TP, **et chacun se fait indépendamment des
autres** : rater une demi-journée ou ne pas finir un TP ne vous pénalise jamais.

## Le fil rouge

Une agence de traduction reçoit la documentation technique suédoise de l'éditeur
Helios et doit livrer du français conforme à un guide de style, à un glossaire
imposé et à une mémoire de traduction de 444 segments validés.

Le sujet n'est pas la traduction : c'est **comment un modèle de langue entre
progressivement dans une chaîne de production, et ce que chaque étage lui
apporte**. On part du moyen le plus simple, on le pousse jusqu'à ce qu'il casse,
on mesure la casse, et on ne monte d'un cran que là.

Personne n'a besoin de parler suédois : tout le monde sait juger une phrase
française, et c'est ce qui compte pour mesurer.

## Installation

```bash
git clone https://github.com/Oktogazh/formation-rag-finetuning.git
cd formation-rag-finetuning
conda env create -f environment.yml
conda activate formation-rag
ollama pull ministral-3:3b && ollama pull bge-m3
python tp.py check
```

**Envoyez la sortie de `python tp.py check` au formateur.** C'est elle, et elle
seule, qui dit avant la session qui travaillera confortablement et qui aura
besoin d'une clé d'API.

Prévoyez **10 Go** d'espace libre : environnement ≈ 1 Go, `ministral-3:3b`
3,0 Go, `bge-m3` 1,2 Go, plus les modèles téléchargés en séance.

**Aucun GPU n'est nécessaire**, nulle part, y compris pour le fine-tuning du
TP 4.

### Si votre machine ne suffit pas

Le formateur vous donne une clé d'API Mistral. Sa seule présence dans `.env`
suffit : **tous les TP basculent sur l'API sans que vous changiez une ligne.**

## Les six TP

Cinq se font dans un **notebook**. Le TP 5 reste en fichiers, parce qu'on y
déploie un service et qu'on le bombarde depuis un second terminal.

```bash
python tp.py notebooks      # fabrique les cinq notebooks
jupyter lab                 # et ouvrez celui du jour
```

| Dossier | Demi-journée | Ce qu'on mesure | Durée | Niveau |
|---|---|---|---|---|
| [`tp01-prompt`](tp01-prompt/) | J1 matin | BLEU, terminologie, température, mur du contexte | 45 min | ★★☆☆☆ |
| [`tp02-rag`](tp02-rag/) | J1 après-midi | rappel de la recherche, tokens, BLEU par catégorie | 75 min | ★★★☆☆ |
| [`tp03-langchain`](tp03-langchain/) | J2 matin | anomalies restantes, appels par segment | 60 min | ★★★☆☆ |
| [`tp04-lora-raft`](tp04-lora-raft/) | J2 après-midi | avant / après entraînement, par catégorie | 90 min | ★★★★☆ |
| [`tp05-mise-en-service`](tp05-mise-en-service/) | J3 matin | p50, p95, débit, effondrement sous charge | 70 min | ★★☆☆☆ |
| [`tp06-graphe`](tp06-graphe/) | J3 après-midi | chemins empruntés, appels, table finale | 75 min | ★★★☆☆ |

## Les cinq types d'exercice

Chaque étape porte un type, pour que vous sachiez d'avance si vous allez écrire
du code. C'est ce qui permet au formateur de piloter le niveau en salle.

| Type | Ce qu'on demande | Code écrit |
|---|---|---|
| **OBS** | lancer, relever un chiffre, le noter | 0 ligne |
| **RÉG** | changer *une* valeur, relancer, expliquer l'écart | 1 ligne |
| **LIRE** | pointer dans le code ou dans une trace où se passe X | 0 ligne |
| **CODE** | compléter un `CODE n` numéroté, dix lignes au plus, un test dédié | ≤ 10 lignes |
| **ARB** | trancher et justifier par écrit, à partir de vos propres chiffres | 0 ligne |

Onze exercices `CODE` sur les six TP, et **aucun n'est sans analogue visible** :
compléter, c'est transposer, jamais inventer.

## Les règles du jeu

**1. Les TP sont indépendants.** Tout ce dont un TP a besoin vient de `commun/`,
jamais de ce qu'un autre TP vous a fait écrire. Vous pouvez les faire dans le
désordre, en sauter un, ou revenir sur un TP raté trois jours plus tard.

**2. Les tests sont rouges au départ.** C'est le dispositif.

```bash
python tp.py test           # tous les TP
python tp.py test tp02      # un seul
python tp.py test tp02 -k code3
python tp.py test tp02 --bonus
```

**3. Quand vous bloquez, deux commandes.**

```bash
python tp.py indice tp02 --code 3   # le diff avec le corrigé
python tp.py rattraper tp02         # prendre le corrigé du TP entier
```

Le corrigé complet est dans [`corrige/`](corrige/). Il est visible, et c'est
assumé : sur trois jours, l'indice ciblé vaut mieux que le stagiaire bloqué.
Mais recopier n'apprend rien — lisez, fermez, retapez.

## Ce que fait `tp.py`

```
check        ce que votre machine sait faire, et ce qu'il lui manque
notebooks    (re)fabrique les notebooks depuis les .py, et pose le bon noyau
test         les tests d'un TP, ou de tous
indice       le diff entre votre code et le corrigé
rattraper    prendre le corrigé d'un TP entier
api          TP 5 — lance le service de traduction en local
assaut       TP 5 — bombarde un service et mesure
banc         la table finale, tous crans confondus
```

### Notebook et module : comment ça marche

Chaque notebook est **apparié** à un fichier `.py` du même nom. Vous éditez le
`.ipynb`, jupytext met le `.py` à jour, et c'est le `.py` que les tests
importent et que git versionne. Les `.ipynb` ne sont pas versionnés :
`python tp.py notebooks` les refabrique quand vous voulez.

Conséquence pratique : `python tp.py test` synchronise avant de lancer pytest.
Vous n'avez rien à faire, mais si un test paraît ignorer votre dernière cellule,
enregistrez le notebook et relancez.

## Le corpus

`data/corpus/helios-sv/` — **entièrement fictif**, produit pour la formation.
Aucun document réel d'entreprise n'est dans ce dépôt, et il ne faut pas en
ajouter.

| Fichier | Contenu |
|---|---|
| `tm.jsonl` | 444 segments traduits et validés : la mémoire |
| `segments-eval.jsonl` | 80 segments à traduire, avec leur référence |
| `glossaire.jsonl` | 7 termes imposés, avec leurs traductions interdites |
| `docs/` | 6 documents de consignes : style, client, relecture, tarifs, confidentialité |

Les 80 segments d'évaluation portent une catégorie : `repetition` (21),
`piege` (20), `fuzzy` (18), `nouveau` (21). C'est elle qui rend la formation
lisible, et le détail est dans `commun/corpus.py`.

```bash
python scripts/verifier_corpus.py             # contrôle d'intégrité
python scripts/verifier_corpus.py --lister    # les 80 références, pour relecture
```

## Où lire quoi

- `commun/prompts.py` — **le gabarit de prompt unique**. Une page, et c'est le
  cœur de la formation.
- `commun/mesure.py` — ce que la formation appelle « mieux ».
- `commun/moteur.py` — Ollama, API Mistral, moteur factice, derrière une porte.
- `commun/recherche.py`, `commun/verification.py` — les versions de référence de
  ce que vous écrivez aux TP 2 et 3. C'est ce qui rend les TP indépendants.

## Licence et contact

Matériel pédagogique produit pour PLB — Alan Kersaudy, développeur et expert IA.
Le corpus est fictif ; le code est à vous, gardez-le et réutilisez-le.
