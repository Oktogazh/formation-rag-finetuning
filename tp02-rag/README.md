# TP 2 — Ne mettre dans le prompt que ce qui sert

**Objectif.** Remplacer les trois exemples choisis à la main par une recherche
automatique dans la mémoire de traduction, d'abord comme le fait un outil de TAO
depuis 1998, puis avec des vecteurs. Et ne plus injecter que les entrées de
glossaire qui concernent le segment en cours.

**Durée.** Noyau ≈ 75 min · bonus ≈ 20 min · difficulté ★★★☆☆

## L'idée à retenir avant de coder

**Une mémoire de traduction est déjà un système de recherche documentaire.**
Elle fait du *retrieval* depuis les années 1990, avec une métrique de similarité
de chaînes, et le pourcentage qu'affiche votre outil de TAO (« 87 % ») est
exactement ce ratio. Ce que la formation appelle RAG, c'est la même mécanique
avec une meilleure métrique et un modèle de langue au bout.

Vous n'apprenez donc pas une idée neuve. Vous découvrez que votre outil
quotidien en est un cas particulier.

## Ce que vous allez mesurer

Les mêmes cinq indicateurs qu'au TP 1, plus deux :

- **Tokens par segment** : il doit s'effondrer par rapport au mur du TP 1 ;
- **Rappel de la recherche** : sur quelle part des segments le bon voisin
  est-il dans les `k` retenus ? C'est le **plafond** de ce que le modèle peut
  faire. Un RAG médiocre est presque toujours un problème de recherche, pas de
  génération — et on ne le voit que si on mesure les deux séparément.

## Prérequis

Le TP 1 fait (`tp01/prompt.py::consigne_systeme` est réutilisée ici), et pour la
recherche dense :

```bash
ollama pull bge-m3
```

1,2 Go. C'est un modèle d'embeddings multilingue : il place le suédois et le
français dans le même espace vectoriel, ce qui est la condition pour chercher
un segment suédois et retrouver son voisin.

## Déroulé

**1. Charger la mémoire** — TODO 1.

444 segments, dont on ne garde que les validés. Proposer un segment non validé à
un traducteur, c'est propager une faute à l'échelle industrielle.

**2. La recherche floue, celle que vous connaissez** — TODO 2.

`difflib.SequenceMatcher` compare deux chaînes et rend un ratio entre 0 et 1.
C'est, à peu de chose près, l'algorithme de votre outil de TAO.

```bash
python tp.py test tp02 -k todo2
python tp.py rag --recherche lexicale --k 3 --n 20
```

**3. La recherche dense** — TODO 3.

Chaque segment devient un vecteur ; on compare les vecteurs. Ça retrouve des
segments qui **disent la même chose avec d'autres mots**, ce que la recherche
lexicale rate par construction.

```bash
python tp.py rag --recherche dense --k 3 --n 20
```

Le premier appel encode les 444 segments (quelques secondes) puis met l'index en
cache. Comparez avec la recherche lexicale : **le dense gagne 7,5 points de
BLEU**, pour le même nombre de tokens et le même temps.

Où gagne-t-il exactement ? Regardez par catégorie. Sur `repetition`, les deux
font 100 : quand un segment est presque identique, comparer des chaînes suffit.
L'écart se creuse sur `piege` et `fuzzy`, là où la ressemblance littérale est
trompeuse ou partielle. **C'est une conclusion d'ingénierie, pas une mode** : le
dense coûte un modèle de plus à installer et à faire tourner, et il faut savoir
ce qu'on achète avec.

**4. Filtrer le glossaire** — TODO 4.

Sur Helios, le glossaire fait 7 entrées et le filtrage en laisse une ou deux.
Sur un vrai compte client il en fait 400, et la différence devient une facture.

**5. Construire le prompt augmenté** — TODO 5.

Vous n'écrivez pas le format : `commun/prompts.py` s'en charge. À partir
d'ici et jusqu'au TP 6, **tous** les appels passent par ce gabarit unique.
C'est la condition pour que la table finale compare des crans et non des mises
en page — et au TP 4, c'est la condition pour que le fine-tuning serve à
quelque chose.

**6. Comparer au TP 1.**

```bash
python tp.py banc
```

Le RAG doit faire aussi bien ou mieux que le mur du TP 1, avec **treize fois
moins de tokens**. C'est tout l'argument.

## Ce que ça donne — mesuré le 13 septembre 2026

`ministral-3:3b` sous Ollama, MacBook Apple Silicon 16 Go, température 0,
20 segments, `k=3`.

| Cran | BLEU | chrF | Termino | Chiffres | Tokens | s/seg |
|---|---|---|---|---|---|---|
| TP 1, 3 exemples choisis | 57,3 | 73,4 | 57 % | 85 % | 331 | 0,74 |
| TP 1, le mur | 68,9 | 85,4 | 100 % | 88 % | 4 461 | 3,19 |
| **RAG lexical k=3** | 68,2 | 84,1 | 100 % | 100 % | 343 | 1,12 |
| **RAG dense k=3** | **75,7** | **88,2** | 100 % | 100 % | 343 | 1,09 |

Le RAG lexical égale le mur avec **treize fois moins de tokens**. Le RAG dense
le dépasse de 7 points. Et les deux atteignent 100 % sur la terminologie **et**
sur les chiffres, ce que le mur n'obtenait pas.

Par catégorie, en recherche dense : `repetition` 100, `piege` 100, `fuzzy` 47,
`nouveau` 61. **La ligne `fuzzy` est le vrai plafond**, et c'est la recherche
qui le fixe, pas le modèle. Retenez-la : c'est elle que le TP 4 essaiera de
faire bouger, et elle résistera.

## Les TODO

| # | Fichier | Difficulté | Ce qu'on attend | Test |
|---|---|---|---|---|
| 1 | `tp02/memoire.py::charger_memoire` | ★ | ne garder que `statut == "valide"` | `-k todo1` |
| 2 | `tp02/recherche.py::rechercher_lexical` | ★★ | les `k` plus proches par ratio de chaînes | `-k todo2` |
| 3 | `tp02/recherche.py::Index.chercher` | ★★ | les `k` plus proches par cosinus | `-k todo3` |
| 4 | `tp02/augmenter.py::glossaire_pertinent` | ★ | filtrer par début de mot | `-k todo4` |
| 5 | `tp02/augmenter.py::construire` | ★ | appeler le gabarit commun | `-k todo5` |

## Bonus

**BONUS 2** — `Index.chercher_hybride` : mélanger les deux scores. En production
on combine presque toujours les deux, parce que le lexical attrape les
références exactes (numéros de version, noms de formule) et le dense attrape les
reformulations.

```bash
python tp.py test tp02 --bonus -k bonus2
python tp.py rag --recherche hybride --k 3 --n 20   # après avoir fait le bonus
```

## Si ça coince

```bash
python tp.py indice tp02/recherche.py --todo 2
python tp.py rattraper tp02
```

- *`ollama pull bge-m3` est trop long ou impossible* → faites tout le TP en
  `--recherche lexicale`. Vous ne perdez que la comparaison de l'étape 3.
- *La recherche dense rend n'importe quoi* → vérifiez que vous comparez bien le
  vecteur de la requête à chaque vecteur de l'index, et non l'inverse.
- *Le test todo4 attend 2 entrées et j'en ai 0* → la comparaison se fait en
  minuscules, et par **début de mot** : `förfrågningar` commence par `förfråg`.
