# %% [markdown]
# # TP 2 — Ne mettre dans le prompt que ce qui sert
#
# **Durée** : noyau ≈ 75 min · **Niveau** ★★★☆☆
#
# Le TP 1 s'est terminé sur un prompt de 9 000 tokens qui ne faisait pas mieux
# qu'un prompt de 300. La différence n'était pas la quantité, c'était la
# **pertinence**. Ici, on ne met dans le prompt que deux choses : les segments
# de mémoire qui ressemblent à celui qu'on traduit, et les entrées de glossaire
# qui le concernent.
#
# Les deux se cherchent, et vous écrirez les deux recherches : celle des
# voisins, et celle des termes. La seconde se fait **par vecteurs**, comme le
# reste du RAG — c'est le même geste appliqué à un autre corpus.
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
# ## CODE 3 · Chercher les termes du glossaire, par le sens
#
# Le glossaire d'Helios fait 7 entrées. Celui d'un vrai compte client en fait
# 400, et les coller toutes dans chaque prompt, c'est le mur du TP 1. On ne
# garde donc que les entrées qui concernent **ce** segment.
#
# La façon évidente est de comparer les chaînes : garder les termes qui sont le
# **début** d'un mot du segment. Ça marche, c'est en trois lignes, et c'est ce
# que fait `commun/augmenter.py` pour les TP 3, 4 et 6.
#
# **Mais le suédois compose.** `testslutpunkten` contient `slutpunkt` sans
# commencer par lui : un préfixe ne voit que les composés dont le terme est le
# premier morceau. C'est une limite de forme, pas de réglage — aucun seuil ne
# la lève.
#
# On cherche donc les termes comme on cherche les voisins : **avec des
# vecteurs**. Un mot du segment et un terme du glossaire deviennent deux points,
# et on regarde s'ils sont proches.
#
# `GlossaireVectoriel` encode les 7 termes une fois à la construction, et garde
# en cache le vecteur de chaque mot déjà vu. Il vous donne :
#
# - `index.glossaire` — les 7 entrées, dans l'ordre ;
# - `index.vecteurs` — leurs 7 vecteurs, dans le même ordre ;
# - `index.vecteurs_des_mots(mots)` — les vecteurs de ces mots, cache compris ;
# - `index.seuil` — le seuil au-delà duquel on retient un terme.

# %%
from commun.augmenter import GlossaireVectoriel, mots_de
from commun.embeddings import cosinus

index_glossaire = GlossaireVectoriel(glossaire)
print(f"Encodeur : {index_glossaire.nom()} · seuil {index_glossaire.seuil}")


def glossaire_pertinent(segment_src: str, index: GlossaireVectoriel,
                        seuil: float | None = None) -> list[dict]:
    """Les entrées de glossaire qui concernent ce segment, et elles seules."""
    seuil = index.seuil if seuil is None else seuil
    vecteurs = index.vecteurs_des_mots(mots_de(segment_src))
    # <<<CODE 3 ★★ Chercher les termes par le sens
    # Pour chaque terme du glossaire et son vecteur — zip(index.glossaire,
    # index.vecteurs) — prenez le MEILLEUR cosinus entre ce vecteur et les
    # vecteurs des mots du segment. Gardez les termes dont ce score atteint le
    # seuil, sous la forme {**terme, "score": score}, du plus sûr au moins sûr.
    # Indice : max((cosinus(v, vecteur) for v in vecteurs), default=0.0)
    # Test : python tp.py test tp02 -k code3
    raise NotImplementedError(
        "CODE 3 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp02-rag/README.md"
    )
    # >>>CODE 3


exemple = "Hastighetsgränsen är satt till 60 förfrågningar per minut."
print(f"\n{exemple}")
for terme in glossaire_pertinent(exemple, index_glossaire):
    print(f"  {terme['score']:.3f}  {terme['sv']} → {terme['fr']}")

# %% [markdown]
# ## OBS 2 · Ce que les vecteurs voient, et ce qu'ils coûtent
#
# Trois choses à relever, dans l'ordre.
#
# **1. Le mot agglutiné.** Le suédois compose sans espace ni trait d'union :
# `Testslutpunkten`, c'est `test` + `slutpunkt` + le suffixe défini `-en`. Le
# terme du glossaire est enfoui entre un préfixe et un suffixe, et pourtant son
# vecteur reste proche de celui du mot entier — les scores ci-dessous le
# montrent.
#
# **2. Le suédois et le français ne sont pas au même endroit.** On encode ici
# les deux côtés du glossaire — alors que **la recherche, elle, ne voit que le
# suédois** : elle compare les mots suédois du segment aux termes suédois du
# glossaire, jamais à leur traduction. Les deux colonnes disent la même chose,
# et pourtant le cosinus n'est pas de 1 : un vecteur encode **une chaîne dans
# une langue**, pas un sens pur.
#
# `abonnemang` → `formule` tombe à 0,50, et en partant de « formule » on
# retombe sur `förfråg`. **Cela ne gêne en rien ce TP** : `abonnemang` est
# retrouvé dans les 21 segments qui l'attendent, à 0,895. C'est un
# avertissement pour le jour où vous chercherez *entre* deux langues — la
# qualité de l'alignement se vérifie terme par terme, elle ne se suppose pas.
#
# **3. Le cache.** Les 80 segments font 645 mots, mais beaucoup moins de mots
# **distincts**. Sans cache, on paierait un encodage par occurrence.

# %%
# --- 1. le terme retrouvé à l'intérieur d'un mot agglutiné -----------------
tous = corpus.charger_evaluation()          # les 80, pas l'échantillon de 20
compose = next(s for s in tous if "Testslutpunkten" in s["src"])
print(f"  {compose['src']}")
print(f"    scores : "
      f"{[(t['sv'], round(t['score'], 3)) for t in glossaire_pertinent(compose['src'], index_glossaire)]}")

# --- 2. la source et la cible du glossaire, encodées toutes les deux -------
vecteurs_fr = index_glossaire.encodeur.encoder([t["fr"] for t in glossaire])
print(f"\n  {'suédois':<18} {'français':<22} {'cos(sv, fr)':>11}")
for terme, vecteur_sv, vecteur_fr in zip(glossaire, index_glossaire.vecteurs, vecteurs_fr):
    print(f"  {terme['sv']:<18} {terme['fr']:<22} {cosinus(vecteur_sv, vecteur_fr):>11.3f}")

print("\n  Hors du chemin de recherche — ce TP cherche sv → sv :")
print("  en partant du français, retombe-t-on sur son suédois ?")
for i, terme in enumerate(glossaire):
    classe = sorted(zip((cosinus(vecteurs_fr[i], v) for v in index_glossaire.vecteurs),
                        (t["sv"] for t in glossaire)), reverse=True)
    verdict = "oui" if classe[0][1] == terme["sv"] else f"NON → {classe[0][1]}"
    print(f"    {terme['fr']:<22} {verdict}")

# --- 3. ce que le cache économise -----------------------------------------
import time

index_neuf = GlossaireVectoriel(glossaire)

depart = time.time()
for segment in tous:
    glossaire_pertinent(segment["src"], index_neuf)
premier, encodes_1, evites_1 = time.time() - depart, index_neuf.encodes, index_neuf.evites

depart = time.time()
for segment in tous:
    glossaire_pertinent(segment["src"], index_neuf)
second = time.time() - depart
encodes_2 = index_neuf.encodes - encodes_1

mots_par_passage = encodes_1 + evites_1
print(f"\n  {len(tous)} segments, {mots_par_passage} mots par passage")
print(f"    passage 1 : {encodes_1:>4} mots encodés, {evites_1:>4} pris au cache"
      f"  ({100 * evites_1 / mots_par_passage:.0f} %)   {premier:.2f} s")
print(f"    passage 2 : {encodes_2:>4} mots encodés, {mots_par_passage:>4} pris au cache"
      f"  (100 %)   {second:.2f} s   ×{premier / max(second, 1e-9):.0f}")

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
def construire(segment_src: str, memoire: list[dict], index: GlossaireVectoriel,
               k: int = 3) -> list[dict]:
    """Le prompt augmenté pour ce segment."""
    voisins = rechercher_lexical(segment_src, memoire, k)
    termes = glossaire_pertinent(segment_src, index)
    # <<<CODE 4 ★ Appeler le gabarit commun
    # Rendez construire_messages(...) avec, dans l'ordre : le segment source,
    # voisins=voisins, glossaire=termes, consignes=CONSIGNE_STYLE.
    # Test : python tp.py test tp02 -k code4
    raise NotImplementedError(
        "CODE 4 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp02-rag/README.md"
    )
    # >>>CODE 4


atelier.montrer_prompt(construire(segments[0]["src"], memoire, index_glossaire))

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
    atelier.traducteur(moteur, lambda src: construire(src, memoire, index_glossaire, k=3)),
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
    atelier.traducteur(moteur, lambda src: construire(src, memoire, index_glossaire, k=K)),
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
        src, voisins=chercher(src), glossaire=glossaire_pertinent(src, index_glossaire),
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
# lira. Comptez-y l'encodage des mots : il se paie une fois par mot distinct,
# pas une fois par segment — c'est le chiffre d'OBS 2.

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
# ## ARB 2 · Et le glossaire, fallait-il des vecteurs ?
#
# Vous venez d'écrire deux filtres de glossaire : un début de mot en trois
# lignes, et une recherche vectorielle qui demande un encodeur, un cache et un
# seuil. Sur les 80 segments, l'écart tient en **un** mot composé.
#
# **Deux phrases :** ce gain vaut-il cette dépendance pour Helios ? Et votre
# réponse changerait-elle pour un client dont le glossaire fait 400 entrées et
# dont la langue compose autant que le suédois — l'allemand, le finnois ?
#
# La bonne réponse n'est pas la même dans les deux cas, et c'est tout l'intérêt
# de la question.
#
# > *Votre réponse :*
# >
# >
