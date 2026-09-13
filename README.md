# RAG et fine-tuning d'un LLM — les travaux pratiques

Support de travaux pratiques de la formation **« RAG et fine-tuning d'un LLM »**
(PLB, 3 jours). Un seul dépôt pour les six demi-journées : vous le clonez une
fois, au jour 1, et vous changez de dossier au fil de la formation.

## Le fil rouge

Une agence de traduction reçoit la documentation technique suédoise de l'éditeur
Helios et doit livrer du français conforme à un guide de style, à un glossaire
imposé et à une mémoire de traduction de 444 segments validés.

Le sujet n'est pas la traduction : c'est **comment un modèle de langue entre
progressivement dans une chaîne de production, et ce que chaque étage lui
apporte**. On part du moyen le plus simple, on le pousse jusqu'à ce qu'il casse,
on mesure la casse, et on ne monte d'un cran que là.

| Cran | TP | Moyen | Ce qui le fait céder |
|---|---|---|---|
| 0 | 1 | prompt nu | le modèle ignore la terminologie du client |
| 1 | 1 | consigne + exemples écrits à la main | ça marche… jusqu'au mur du contexte |
| 2 | 2 | RAG : n'injecter que ce qui sert à *ce* segment | la recherche devient le plafond |
| 3 | 3 | chaîne LangChain, vérification, correction | deux appels au lieu d'un |
| 4 | 4 | QLoRA + RAFT : le style entre dans les poids | n'apporte aucune connaissance nouvelle |
| 5 | 5 | mise en service, et incident sous charge | le service n'était jamais le modèle |
| 6 | 6 | graphe routé | le verdict : quel cran pour quel segment |

La langue de travail est le **suédois vers le français**. Personne dans la salle
n'a besoin de parler suédois : tout le monde sait juger une phrase française, et
c'est ce qui compte pour mesurer.

## Installation

Six commandes, une quinzaine de minutes, dont l'essentiel en téléchargements.

```bash
git clone https://github.com/Oktogazh/formation-rag-finetuning.git
cd formation-rag-finetuning
conda env create -f environment.yml
conda activate formation-rag
ollama pull ministral-3:3b && ollama pull bge-m3
cp .env.example .env && python tp.py check
```

**Envoyez la sortie de `python tp.py check` au formateur.** C'est elle, et elle
seule, qui dit avant la session qui tournera confortablement et qui aura besoin
d'une clé d'API.

Détail des tailles : environnement conda ≈ 300 Mo, `ministral-3:3b` 3,0 Go,
`bge-m3` 1,2 Go. Prévoyez **8 Go** d'espace libre, 10 si vous ferez
l'entraînement du TP 4.

### Si votre machine ne suffit pas

Le formateur vous donne une clé d'API Mistral. Sa seule présence dans `.env`
suffit : **tous les TP basculent sur l'API sans que vous changiez une ligne**.

```
MISTRAL_API_KEY=...
```

La clé est nominative et révoquée après la formation.

## Les six TP

| Dossier | Demi-journée | Ce qu'on mesure | Durée | Niveau |
|---|---|---|---|---|
| [`tp01-prompt`](tp01-prompt/) | J1 matin | BLEU, terminologie, effet de la température, mur du contexte | 45 min | ★★☆☆☆ |
| [`tp02-rag`](tp02-rag/) | J1 après-midi | rappel de la recherche, tokens, BLEU par catégorie | 75 min | ★★★☆☆ |
| [`tp03-langchain`](tp03-langchain/) | J2 matin | anomalies restantes, appels par segment | 60 min | ★★★☆☆ |
| [`tp04-qlora-raft`](tp04-qlora-raft/) | J2 après-midi | base contre adapté, par catégorie | 90 min | ★★★★☆ |
| [`tp05-mise-en-service`](tp05-mise-en-service/) | J3 matin | p50, p95, débit, effondrement sous charge | 70 min | ★★☆☆☆ |
| [`tp06-graphe`](tp06-graphe/) | J3 après-midi | chemins empruntés, appels, table finale | 75 min | ★★★☆☆ |

Chaque TP a un **noyau**, qui tient dans la demi-journée, et un **bonus** pour
ceux qui finissent avant la pause. Le bonus n'est jamais nécessaire au TP
suivant.

## Ce que la formation démontre, mesuré

`ministral-3:3b` sous Ollama, MacBook Apple Silicon, température 0, 13 septembre
2026. C'est la table que `python tp.py banc` reconstruit à partir de **vos**
mesures à vous, à la fin du TP 6.

| Cran | n | BLEU | Termino | Tokens | s/seg | Appels |
|---|---|---|---|---|---|---|
| 0 · prompt nu | 20 | 45,5 | 14 % | 85 | 0,96 | 1,00 |
| 1 · + consigne de style | 20 | 41,3 | 36 % | 211 | 0,74 | 1,00 |
| 1 · + 3 exemples choisis | 20 | 57,3 | 57 % | 331 | 0,74 | 1,00 |
| 1b · tout dans le prompt | 8 | 68,9 | 100 % | **4 461** | **3,19** | 1,00 |
| 2 · RAG lexical k=3 | 20 | 68,2 | 100 % | 343 | 1,12 | 1,00 |
| 2 · RAG dense k=3 | 20 | **75,7** | 100 % | 343 | 1,09 | 1,00 |
| 3 · LangChain | 80 | 70,5 | 100 % | — | 1,14 | 1,00 |
| 3 · + vérification | 80 | 66,8 | 100 % | — | 1,74 | 1,04 |
| 6 · **graphe routé** | 80 | **75,1** | 100 % | 175 | 1,01 | **0,51** |

Ne comparez que des lignes de même `n`. Sur les 80 segments, le graphe du TP 6
fait mieux que la chaîne du TP 3 avec **moitié moins d'appels au modèle** — et
c'est la conclusion de trois jours : le travail n'est pas de trouver le meilleur
modèle, c'est de **savoir quand ne pas l'appeler**.

Deux résultats ne vont pas dans le sens qu'on attend, et ils sont dans les
README des TP concernés, expliqués au lieu d'être cachés : la consigne de style
fait *baisser* BLEU tout en doublant la conformité terminologique, et la boucle
de correction du TP 3 coûte quatre points de BLEU pour gagner deux points de
conformité sur les chiffres.

## Les règles du jeu

**1. Vous restez sur `master`.** Pas de branche, pas de tag, pas de
`git checkout`. Ce qui distingue un TP d'un autre est un **dossier**.

**2. Les tests sont rouges au départ.** C'est le dispositif : 18 `TODO` à
compléter, et chacun a un test qui passe du rouge au vert.

```bash
python tp.py test          # tous les TP
python tp.py test tp02     # un seul
python tp.py test tp02 -k todo3
python tp.py test tp02 --bonus
```

**3. Quand vous bloquez, deux commandes.**

```bash
python tp.py indice tp02/recherche.py --todo 2   # le diff avec le corrigé
python tp.py rattraper tp02                      # prendre le corrigé, TP entier
```

`rattraper` existe parce que les TP s'appuient les uns sur les autres : le TP 3
importe la recherche du TP 2, le TP 6 importe presque tout. Un blocage au TP 2
ne doit pas vous coûter le TP 6.

Le corrigé complet est dans [`corrige/`](corrige/), aux mêmes chemins. Il est
visible, et c'est assumé : sur trois jours, l'indice ciblé vaut mieux que le
stagiaire bloqué. Mais recopier n'apprend rien — lisez, fermez, retapez.

## Ce que fait `tp.py`

```
check        ce que votre machine peut faire, et ce qu'il lui manque
test         les tests d'un TP, ou de tous
indice       le diff entre votre fichier et le corrigé
rattraper    prendre le corrigé d'un TP entier

traduire     TP 1 — traduire 20 segments avec une variante de prompt
temperature  TP 1 — le même segment à plusieurs températures
mur          TP 1 — tout mettre dans le prompt, et voir
rag          TP 2 — recherche lexicale ou dense, k voisins
chaine       TP 3 — la même chose en LangChain, avec ou sans vérification
donnees      TP 4 — fabriquer le jeu d'entraînement RAFT
entrainer    TP 4 — entraîner l'adaptateur LoRA
comparer     TP 4 — modèle de base contre modèle adapté
adaptateur   TP 4 — récupérer l'adaptateur du formateur
api          TP 5 — lancer le service de traduction en local
assaut       TP 5 — bombarder un service et mesurer
graphe       TP 6 — le graphe route, mesure et livre
banc         TP 6 — la table finale, tous crans confondus
```

`python tp.py <commande> --help` détaille chacune.

## Le corpus

`data/corpus/helios-sv/` — **entièrement fictif**, produit pour la formation.
Aucun document réel d'entreprise n'est dans ce dépôt, et il ne faut pas en
ajouter.

| Fichier | Contenu |
|---|---|
| `tm.jsonl` | 444 segments traduits et validés : la mémoire |
| `segments-eval.jsonl` | 80 segments à traduire, avec leur référence |
| `glossaire.jsonl` | 7 termes imposés, avec leurs traductions interdites |
| `docs/` | 6 documents de consignes : style, glossaire, client, relecture, tarifs, confidentialité |

Les 80 segments d'évaluation portent une catégorie, et c'est elle qui rend la
formation lisible : `repetition` (21), `piege` (20), `fuzzy` (18),
`nouveau` (21). Le détail est dans `commun/corpus.py`.

```bash
python scripts/verifier_corpus.py             # contrôle d'intégrité
python scripts/verifier_corpus.py --lister    # les 80 références, pour relecture
```

## Où lire quoi

- `commun/prompts.py` — **le gabarit de prompt unique**. Une page, et c'est le
  cœur de la formation.
- `commun/mesure.py` — ce que la formation appelle « mieux ». BLEU, chrF,
  terminologie, vouvoiement, chiffres.
- `commun/moteur.py` — Ollama, API Mistral, adaptateur LoRA, moteur factice,
  derrière une seule porte.

## Licence et contact

Matériel pédagogique produit pour PLB — Alan Kersaudy, développeur et expert IA.
Le corpus est fictif ; le code est à vous, gardez-le et réutilisez-le.
