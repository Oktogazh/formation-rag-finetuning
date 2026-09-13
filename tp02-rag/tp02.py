# %% [markdown]
# # TP 2 — Ne mettre dans le prompt que ce qui sert
#
# **Durée** : noyau ≈ 75 min · **Niveau** ★★★☆☆
#
# Le TP 1 s'est terminé sur un prompt de 9 000 tokens qui ne faisait pas mieux
# qu'un prompt de 300. La différence n'était pas la quantité, c'était la
# **pertinence**. Ici, on ne met dans le prompt que deux choses : les segments
# de mémoire qui ressemblent à celui qu'on traduit, et les entrées de glossaire
# qui y apparaissent.
#
# **Ce TP ne dépend d'aucun autre.** Si vous n'avez pas fait le TP 1, la
# consigne de style de référence est dans `commun/consignes.py`, et tout
# fonctionne.
#
# ## L'idée à retenir avant de coder
#
# **Une mémoire de traduction est déjà un système de recherche documentaire.**
# Elle fait du *retrieval* depuis les années 1990, avec une métrique de
# similarité de chaînes, et le pourcentage qu'affiche votre outil de TAO
# (« 87 % ») est exactement ce ratio. Ce que la formation appelle RAG, c'est la
# même mécanique avec une meilleure métrique et un modèle de langue au bout.
#
# Vous n'apprenez donc pas une idée neuve. Vous découvrez que votre outil
# quotidien en est un cas particulier.
#
# ## Les cinq types d'exercice
#
# **OBS** relever un chiffre · **RÉG** changer une valeur et expliquer ·
# **LIRE** pointer où se passe quelque chose · **CODE** compléter dix lignes au
# plus · **ARB** trancher par écrit.

# %% [markdown]
# ## Préparation
#
# Comme au TP 1 : on remonte à la racine du dépôt, on ouvre un moteur, on charge
# les segments d'évaluation. Deux choses en plus ici :
#
# - `charger_memoire_brute()` lit les 444 lignes de `tm.jsonl`, **sans filtre** ;
# - `CONSIGNE_STYLE` est la consigne de référence. Si vous avez écrit la vôtre
#   au TP 1, vous pouvez la coller à la place et comparer.

# %%
import os
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
os.chdir(RACINE)

from commun import atelier, corpus
from commun.consignes import CONSIGNE_STYLE
from commun.moteur import obtenir_moteur
from commun.prompts import construire_messages

moteur = obtenir_moteur()
glossaire = corpus.charger_glossaire()
memoire_brute = corpus.charger_memoire_brute()
segments = atelier.segments(n=20)

print(f"Moteur : {moteur.nom()}")
print(f"{len(memoire_brute)} segments dans la mémoire, {len(segments)} à traduire")

# %% [markdown]
# ## CODE 1 · Ne garder que ce qui est validé
#
# Un fichier de mémoire contient des segments à différents statuts. On ne
# propose à un traducteur que ce qui a été **validé** : proposer un segment non
# validé, c'est propager une faute à l'échelle industrielle.
#
# Chaque entrée a les clés `id`, `src`, `tgt`, `domaine`, `date`, `statut`.

# %%
def charger_memoire(memoire_brute: list[dict]) -> list[dict]:
    """Les segments de la mémoire utilisables."""
    # <<<CODE 1 ★ Filtrer sur le statut
    # Rendez la liste des segments dont la clé « statut » vaut exactement
    # "valide". Une ligne suffit.
    # Test : python tp.py test tp02 -k code1
    raise NotImplementedError(
        "CODE 1 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp02-rag/README.md"
    )
    # >>>CODE 1


memoire = charger_memoire(memoire_brute)
print(f"{len(memoire)} segments validés sur {len(memoire_brute)}")

# %% [markdown]
# ## OBS 1 · Ce qu'il y a dans la mémoire
#
# Avant de chercher dedans, regardez-la. La cellule affiche le nombre de
# domaines, la longueur moyenne d'un segment, et trois entrées au hasard.
#
# Notez la longueur moyenne : c'est elle qui explique pourquoi injecter trois
# voisins coûte 300 tokens et pas 3 000.

# %%
from commun.memoire import statistiques

stats = statistiques(memoire)
print(f"  {stats['segments']} segments · {stats['domaines']} domaines · "
      f"{stats['mots_par_segment']:.0f} mots par segment en moyenne\n")
for segment in memoire[:3]:
    print(f"  [{segment['domaine']}]")
    print(f"    sv  {segment['src']}")
    print(f"    fr  {segment['tgt']}")

# %% [markdown]
# ## CODE 2 · La recherche floue, celle que vous connaissez
#
# `difflib.SequenceMatcher(None, a, b).ratio()` compare deux chaînes et rend un
# nombre entre 0 et 1. C'est, à peu de chose près, l'algorithme de votre outil
# de TAO.
#
# Écrivez la recherche : pour chaque segment de la mémoire, calculez la
# similarité avec le segment à traduire, rangez du plus proche au plus lointain,
# rendez les `k` premiers.
#
# **Analogue visible** : `meilleur_voisin()` dans la cellule suivante fait
# exactement cela pour `k = 1`. Compléter, c'est transposer.

# %%
from difflib import SequenceMatcher


def similarite(a: str, b: str) -> float:
    """Le ratio de similarité de chaînes, entre 0 et 1. Fourni."""
    return SequenceMatcher(None, a, b).ratio()


def rechercher_lexical(segment_src: str, memoire: list[dict], k: int = 3) -> list[dict]:
    """Les k segments les plus proches, au sens des caractères."""
    # <<<CODE 2 ★★ La recherche floue d'un outil de TAO
    # Construisez la liste [{**s, "score": similarite(segment_src, s["src"])} ...],
    # triez-la par score décroissant, rendez les k premiers.
    # Indice : liste.sort(key=lambda s: s["score"], reverse=True)
    # Test : python tp.py test tp02 -k code2
    raise NotImplementedError(
        "CODE 2 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp02-rag/README.md"
    )
    # >>>CODE 2


def meilleur_voisin(segment_src: str, memoire: list[dict]) -> dict:
    """Le segment le plus proche, et lui seul. Fourni — c'est votre modèle."""
    meilleur = max(memoire, key=lambda s: similarite(segment_src, s["src"]))
    return {**meilleur, "score": similarite(segment_src, meilleur["src"])}


for voisin in rechercher_lexical(segments[0]["src"], memoire, k=3):
    print(f"  {voisin['score']:.3f}  {voisin['src']}")
print(f"\n  à traduire : {segments[0]['src']}")

# %% [markdown]
# ## OBS 2 · Le rappel de la recherche, avant toute génération
#
# **C'est la mesure la plus importante du TP, et elle ne fait intervenir aucun
# modèle.** Sur quelle part des segments le bon voisin est-il dans les `k`
# retenus ?
#
# Ce chiffre est le **plafond** de tout ce qui suit : ce que la recherche ne
# remonte pas, la génération ne l'inventera pas. Un RAG médiocre est presque
# toujours un problème de recherche, pas de génération — et on ne le voit que
# si on mesure les deux séparément.

# %%
from commun.recherche import rappel

for k in (1, 3, 5):
    part = rappel(segments, lambda src, k=k: rechercher_lexical(src, memoire, k))
    print(f"  k = {k} : un voisin à plus de 90 % pour {part:.0f} % des segments")

# %% [markdown]
# ## CODE 3 · Filtrer le glossaire
#
# Le glossaire d'Helios fait 7 entrées. Celui d'un vrai compte client en fait
# 400, et les coller toutes dans chaque prompt, c'est le mur du TP 1.
#
# On ne garde que les entrées dont le terme suédois apparaît dans **ce** segment.
# La comparaison se fait par **début de mot**, en minuscules : le suédois compose
# et fléchit (`förfrågan`, `förfrågningar`), une égalité stricte ne trouverait
# rien.

# %%
import re

MOTS = re.compile(r"[\wåäöÅÄÖ]+")


def glossaire_pertinent(segment_src: str, glossaire: list[dict]) -> list[dict]:
    """Les entrées de glossaire qui concernent ce segment, et elles seules."""
    mots = [m.lower() for m in MOTS.findall(segment_src)]
    # <<<CODE 3 ★ Filtrer le glossaire
    # Rendez les entrées dont le terme suédois (clé "sv", en minuscules) est le
    # DÉBUT d'au moins un des mots ci-dessus.
    # Indice : mot.startswith(terme) — et .lower() sur les deux.
    # Test : python tp.py test tp02 -k code3
    raise NotImplementedError(
        "CODE 3 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp02-rag/README.md"
    )
    # >>>CODE 3


exemple = "Hastighetsgränsen är satt till 60 förfrågningar per minut."
print(exemple)
for terme in glossaire_pertinent(exemple, glossaire):
    print(f"  {terme['sv']} → {terme['fr']}")

# %% [markdown]
# ## CODE 4 · Le prompt augmenté
#
# On assemble : la consigne de style, le glossaire filtré, les voisins retenus,
# et le segment à traduire.
#
# **Vous n'écrivez pas le format.** `commun/prompts.py` le fait, et c'est
# volontaire : à partir d'ici et jusqu'au TP 6, tous les appels passent par ce
# gabarit unique. C'est la condition pour que la table finale compare des crans
# et non des mises en page — et au TP 4, c'est la condition pour que le
# fine-tuning serve à quelque chose.
#
# Allez lire `commun/prompts.py` avant d'écrire cette cellule. C'est une page.

# %%
def construire(segment_src: str, memoire: list[dict], glossaire: list[dict],
               k: int = 3) -> list[dict]:
    """Le prompt augmenté pour ce segment."""
    voisins = rechercher_lexical(segment_src, memoire, k)
    termes = glossaire_pertinent(segment_src, glossaire)
    # <<<CODE 4 ★ Appeler le gabarit commun
    # Rendez construire_messages(...) avec, dans l'ordre : le segment source,
    # voisins=voisins, glossaire=termes, consignes=CONSIGNE_STYLE.
    # Test : python tp.py test tp02 -k code4
    raise NotImplementedError(
        "CODE 4 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp02-rag/README.md"
    )
    # >>>CODE 4


atelier.montrer_prompt(construire(segments[0]["src"], memoire, glossaire))

# %% [markdown]
# ## OBS 3 · Mesurer le RAG
#
# Mêmes segments qu'au TP 1, même modèle, même température. La seule chose qui
# change est ce qu'il y a dans le prompt.
#
# Regardez trois colonnes : **Tokens** (comparez au 4 461 du mur), **Termino**
# (le glossaire filtré fait son travail) et **BLEU** par catégorie, juste en
# dessous.

# %%
rag_lexical = atelier.mesurer(
    "tp02-rag-lexical",
    segments,
    atelier.traducteur(moteur, lambda src: construire(src, memoire, glossaire, k=3)),
    titre="RAG — recherche lexicale, k=3",
)

# %% [markdown]
# ## RÉG 1 · Combien de voisins ?
#
# **Changez la valeur de `K`** ci-dessous — essayez 1, puis 5, puis 10 — et
# regardez ce qui bouge : BLEU, les tokens, le temps.
#
# Vous cherchez le point où ajouter un voisin ne rapporte plus rien mais coûte
# toujours. C'est le même raisonnement qu'au mur du TP 1, à une autre échelle.

# %%
K = 5

rag_k = atelier.mesurer(
    f"tp02-rag-k{K}",
    segments,
    atelier.traducteur(moteur, lambda src: construire(src, memoire, glossaire, k=K)),
    titre=f"RAG — recherche lexicale, k={K}",
    enregistrer=False,
)
atelier.comparer(("k=3", rag_lexical), (f"k={K}", rag_k))

# %% [markdown]
# ## RÉG 2 · Lexical ou dense ?
#
# La recherche dense transforme chaque segment en vecteur avec un modèle
# d'embeddings multilingue (`bge-m3`, tiré par Ollama), et compare les vecteurs.
# Elle retrouve des segments qui **disent la même chose avec d'autres mots**, ce
# que la recherche lexicale rate par construction.
#
# Le premier appel encode les 444 segments — quelques secondes — puis met
# l'index en cache.
#
# **Changez `METHODE`** entre `"lexicale"` et `"dense"`. Regardez la table par
# catégorie : l'écart ne se fait pas là où on l'attend.

# %%
from commun.recherche import chercheur

METHODE = "dense"

chercher = chercheur(memoire, METHODE, k=3)
rag_dense = atelier.mesurer(
    "tp02-rag-dense",
    segments,
    atelier.traducteur(moteur, lambda src: construire_messages(
        src, voisins=chercher(src), glossaire=glossaire_pertinent(src, glossaire),
        consignes=CONSIGNE_STYLE)),
    titre=f"RAG — recherche {METHODE}, k=3",
)
atelier.par_categorie(("lexical", rag_lexical), ("dense", rag_dense))

# %% [markdown]
# ## LIRE 1 · Où sont passés les tokens ?
#
# Comparez la colonne **Tokens** de vos mesures avec celle du mur du TP 1
# (≈ 4 500 tokens par segment).
#
# **Répondez à trois questions, en regardant le prompt affiché plus haut :**
#
# 1. combien d'entrées de glossaire le prompt contient-il, et combien le
#    glossaire complet en compte-t-il ?
# 2. combien de voisins, sur 444 segments de mémoire ?
# 3. sur 100 000 segments par mois, quel est le rapport de coût entre le mur et
#    votre RAG ?
#
# La troisième réponse est la seule ligne de ce TP qu'une direction financière
# lira.

# %%
atelier.comparer(("RAG lexical k=3", rag_lexical), ("RAG dense k=3", rag_dense))

# %% [markdown]
# ## ARB 1 · Votre arbitrage
#
# Lexical ou dense, pour le compte Helios ?
#
# Le dense coûte un modèle de plus à installer (1,2 Go) et à faire tourner. Le
# lexical ne coûte rien et n'a aucune dépendance. **Écrivez en trois phrases**
# lequel vous recommandez, avec le chiffre de votre propre table qui le
# justifie. Un choix d'ingénierie se défend avec une mesure, pas avec une mode.
#
# > *Votre réponse :*
# >
# >
#
# ---
#
# ## Pour aller plus loin, s'il reste du temps
#
# En production on combine presque toujours les deux : le lexical attrape les
# références exactes (numéros de version, noms de formule), le dense attrape les
# reformulations.

# %%
def rechercher_hybride(segment_src: str, memoire: list[dict], index, k: int = 3,
                       alpha: float = 0.5) -> list[dict]:
    """BONUS — mélanger le score dense et le score lexical."""
    denses = {s["id"]: s["score"] for s in index.chercher(segment_src, k=len(memoire))}
    # <<<BONUS 2 ★★ Recherche hybride
    # Pour chaque segment de la mémoire, calculez
    # alpha * denses[segment["id"]] + (1 - alpha) * similarite(segment_src, segment["src"])
    # rangez par score décroissant, rendez les k premiers — même forme de
    # sortie que rechercher_lexical.
    # Test : python tp.py test tp02 --bonus -k bonus2
    raise NotImplementedError(
        "BONUS 2 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp02-rag/README.md"
    )
    # >>>BONUS 2
