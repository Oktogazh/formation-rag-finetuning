# TP 2 — Ne mettre dans le prompt que ce qui sert

**Durée** noyau ≈ 75 min · **Niveau** ★★★☆☆ · **Indépendant** des autres TP

Remplacer les exemples choisis à la main par une recherche automatique dans la
mémoire de traduction : d'abord comme le fait un outil de TAO, puis avec des
vecteurs.

## Lancer

```bash
python tp.py notebooks tp02
jupyter lab
```

## Prérequis

Ollama avec `ministral-3:3b`, plus le modèle d'embeddings pour la recherche
dense :

```bash
ollama pull bge-m3
```

1,2 Go, multilingue : il place le suédois et le français dans le même espace
vectoriel. Si le téléchargement échoue, tout le TP se fait en recherche
lexicale, vous ne perdez qu'un exercice.

## Les exercices

| Type | Exercice |
|---|---|
| **CODE 1** | ne garder que les segments validés |
| OBS 1 | ce qu'il y a dans la mémoire |
| **CODE 2** | la recherche floue, celle de votre outil de TAO |
| OBS 2 | le rappel de la recherche, avant toute génération |
| **CODE 3** | filtrer le glossaire sur le segment |
| **CODE 4** | le prompt augmenté |
| OBS 3 | mesurer le RAG |
| RÉG 1 | combien de voisins ? |
| RÉG 2 | lexical ou dense ? |
| LIRE 1 | où sont passés les tokens ? |
| ARB 1 | lequel recommandez-vous pour Helios ? |
| BONUS 2 | recherche hybride |

## Si ça coince

```bash
python tp.py test tp02 -k code2
python tp.py indice tp02 --code 2
python tp.py rattraper tp02
```

- *Le test `code3` attend 2 entrées et j'en ai 0* → la comparaison se fait en
  minuscules et par **début de mot** : `förfrågningar` commence par `förfråg`.
