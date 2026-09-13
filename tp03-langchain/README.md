# TP 3 — Orchestrer : chaîne, vérification, correction

**Objectif.** Refaire le RAG du TP 2 avec LangChain, puis ajouter ce qu'une
chaîne permet et qu'un script rend pénible : un contrôle automatique et une
boucle de correction. Et mesurer ce que ça coûte.

**Durée.** Noyau ≈ 60 min · bonus ≈ 15 min · difficulté ★★★☆☆

## La question honnête, posée d'emblée

LangChain ne traduit pas mieux. **C'est de la plomberie**, et la question au
bureau sera : qu'est-ce qu'elle apporte, et à quel prix ?

Ce qu'elle apporte, et que ce TP montre :

- un assemblage **déclaratif** : la chaîne se lit comme un schéma, et chaque
  morceau se remplace sans toucher aux autres ;
- la même interface pour Ollama, Mistral, OpenAI ou autre chose. Vous avez déjà
  cette abstraction dans `commun/moteur.py`, écrite à la main en 40 lignes :
  comparez, c'est instructif dans les deux sens ;
- **le traçage gratuit** : `LANGSMITH_TRACING=true` dans `.env`, et chaque étape
  devient visible sans une ligne de code en plus.

Ce qu'elle coûte : une dépendance de plus, des abstractions à apprendre, et une
pile d'appels difficile à lire quand ça casse.

## Ce que vous allez mesurer

La colonne **Appels** apparaît ici et ne vous quittera plus. Une traduction
vérifiée puis corrigée, c'est deux appels au lieu d'un. Sur 100 000 segments par
mois, c'est la ligne que lira la direction financière.

## Prérequis

Les TP 1 et 2 faits (la chaîne réutilise la recherche et le glossaire filtré).
`langchain-core`, `langchain-ollama` et `langgraph` sont déjà dans
l'environnement conda.

## Déroulé

**1. Assembler la chaîne** — TODO 1, dans `tp03/chaine.py`.

LCEL (*LangChain Expression Language*) compose des `Runnable` avec l'opérateur
`|`. `RunnableParallel` calcule plusieurs valeurs à partir d'une entrée,
`RunnableLambda` enveloppe une fonction ordinaire, `StrOutputParser` extrait le
texte d'un message.

Le prompt reste celui de `commun/prompts.py`. On ne le réécrit **pas** en
`ChatPromptTemplate` : ce serait un deuxième format, et le TP 4 entraînera le
modèle sur le premier.

```bash
python tp.py test tp03 -k todo1
python tp.py chaine --n 20
```

Vous devez retrouver, au bruit près, les chiffres du TP 2 en recherche dense.
**Si l'écart est grand, quelque chose a changé dans le prompt sans que vous le
vouliez** — et c'est exactement le genre de dérive que ce TP doit vous apprendre
à repérer.

**2. Le contrôle** — TODO 2, dans `tp03/verification.py`.

Les quatre contrôles de la procédure de relecture de l'agence
(`docs/004-procedure-relecture.md`) existent déjà dans `commun/mesure.py`. Ils
servaient à **noter** ; ici, ils vont **déclencher**. C'est tout le passage d'un
indicateur à un garde-fou.

**3. La boucle** — TODO 3, dans `tp03/boucle.py`.

Si le contrôle trouve un défaut, on redemande au modèle en lui disant lequel.
Avec un compteur de tentatives : une boucle d'agent sans compteur est un
incident de production, pas une audace.

```bash
python tp.py chaine --verifier --n 20
```

Comparez les deux tables : les anomalies restantes baissent, les appels montent.
**Notez le facteur de coût réel** — c'est le chiffre à connaître au TP 6.

**4. Regarder la chaîne tourner.**

Le formateur vous donne une clé LangSmith pendant la séance. Dans `.env` :

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=formation-helios-<votre prénom>
```

Si le compte est européen, ajoutez
`LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`. Sans cette ligne, les
traces n'arrivent jamais et **rien ne vous le dit** : c'est le piège classique.

Relancez trois segments, ouvrez le projet. Vous voyez le prompt exact, les
voisins injectés, la réponse, la durée, les tokens. C'est l'outil du TP 5.

## Les TODO

| # | Fichier | Difficulté | Ce qu'on attend | Test |
|---|---|---|---|---|
| 1 | `tp03/chaine.py::construire_chaine` | ★★ | la chaîne LCEL complète | `-k todo1` |
| 2 | `tp03/verification.py::verifier` | ★ | les quatre contrôles en anomalies | `-k todo2` |
| 3 | `tp03/boucle.py::traduire_et_corriger` | ★★ | la boucle, avec compteur | `-k todo3` |

## Bonus

**BONUS 3** — `reparer_sans_modele` : corriger les chiffres par un remplacement
de texte, sans rappeler le modèle. Zéro appel, zéro token, et plus fiable qu'un
second appel — un modèle qui s'est trompé une fois se trompe souvent deux.

La question à se poser à chaque étape d'une chaîne : **ai-je vraiment besoin
d'un modèle de langue pour ça ?** Au TP 6, la réponse sera « non » pour la
moitié du corpus.

## Si ça coince

- *Le test todo1 dit que les voisins n'arrivent pas au modèle* → votre
  `RunnableParallel` calcule bien les voisins, mais le `RunnableLambda` suivant
  ne les passe pas à `construire`.
- *`chaine` est beaucoup plus lent que `rag`* → vous avez peut-être laissé la
  vérification active, ou `num_ctx` n'est pas réglé.
- *Rien n'apparaît dans LangSmith* → neuf fois sur dix, c'est la région (voir
  ci-dessus), ou `LANGSMITH_TRACING` absent.
