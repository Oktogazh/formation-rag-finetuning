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
vectoriel. **Il n'est plus optionnel** : depuis que le glossaire se cherche par
vecteurs (CODE 3), les exercices 3 et 4 en dépendent.

Si le téléchargement échoue le jour même, lancez le notebook avec
`TP_EMBEDDINGS=factice` : un encodeur de secours en Python pur prend le relais.
Il hache des trigrammes au lieu de comprendre le sens, donc il sépare vers
**0,55** et non 0,75 — `GlossaireVectoriel` applique ce seuil tout seul, le
seuil étant attaché à l'encodeur. Vous ferez tous les exercices, mais pas la
démonstration multilingue d'OBS 2.

## Les exercices

| Type | Exercice |
|---|---|
| **CODE 1** | ne garder que les segments validés |
| OBS 1 | ce qu'il y a dans la mémoire |
| **CODE 2** | la recherche floue, celle de votre outil de TAO |
| **CODE 3** | chercher les termes du glossaire, par le sens |
| OBS 2 | ce que les vecteurs voient, et ce qu'ils coûtent |
| **CODE 4** | le prompt augmenté |
| OBS 3 | mesurer le RAG |
| RÉG 1 | combien de voisins ? |
| RÉG 2 | lexical ou dense ? |
| LIRE 1 | où sont passés les tokens ? |
| ARB 1 | lexical ou dense pour Helios ? |
| ARB 2 | chercher le glossaire, à quel prix ? |
| *BONUS 2* | *découper un document en chunks* |
| *OBS 4* | *le document devient une mémoire de traduction* |

## Si ça coince

```bash
python tp.py test tp02 -k code2
python tp.py indice tp02 --code 2
python tp.py rattraper tp02
```

- *Le test `code3` attend 2 entrées et j'en ai 0* → vous comparez sans doute le
  segment entier au terme. Il faut comparer **mot à mot** : le meilleur cosinus
  entre un mot du segment et le terme. Une phrase de dix mots n'est jamais
  proche d'un mot isolé.
- *`code3` me rend les 7 termes* → le seuil n'est pas appliqué, ou il est trop
  bas. `index.seuil` vaut 0,75 avec `bge-m3`.
- *Ollama ne répond pas* → `ollama serve`, puis `ollama pull bge-m3`. En
  dépannage, `TP_EMBEDDINGS=factice` avec un seuil de 0,55.
