# %% [markdown]
# # TP 6 — Cinq crans, un seul graphe, et le verdict
#
# **Durée** : noyau ≈ 75 min · **Niveau** ★★★☆☆
#
# **Ce TP ne dépend d'aucun autre.** Tout ce que les TP précédents ont construit
# existe en version de référence dans `commun/` : la recherche, le prompt
# augmenté, la vérification. Vous pouvez faire ce TP sans avoir fait les cinq
# autres.
#
# ## Ce qu'on assemble
#
# La formation a produit plusieurs moyens de traduire un segment, du moins cher
# au plus cher :
#
# | Moyen | Appels au modèle | Ce qu'il sait faire |
# |---|---|---|
# | réutiliser la mémoire | **0** | rendre un segment déjà traduit, chiffres reportés |
# | RAG | 1 | traduire avec les voisins et le glossaire |
# | RAG + correction | 2 | rattraper une faute de terminologie ou de chiffre |
#
# Aucun n'est bon partout. **Le travail d'ingénierie consiste à choisir lequel
# s'applique à quel segment** — et c'est exactement ce que fait un graphe.
#
# ## La règle de routage, qui vient du client
#
# `docs/003-consignes-client.md` dit : au-dessus de 95 % de correspondance, on
# réutilise le segment de la mémoire **après vérification des chiffres**. Les
# deux conditions, pas une seule. Un segment très proche dont un **mot** a changé
# ne se réutilise pas : c'est la faute la plus chère qu'un outil de TAO puisse
# produire.

# %% [markdown]
# ## Préparation
#
# `StateGraph` est la brique de LangGraph : un dictionnaire d'état qui passe de
# nœud en nœud, chaque nœud le lisant et y ajoutant quelque chose. Tout est là,
# il n'y a pas d'objet caché — c'est ce qui le rend lisible, et traçable.

# %%
import os
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
os.chdir(RACINE)

from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from commun import atelier, corpus
from commun.augmenter import GlossaireVectoriel, construire
from commun.consignes import CONSIGNE_STYLE
from commun.memoire import charger_memoire
from commun.moteur import obtenir_moteur
from commun.prompts import nettoyer_sortie
from commun.recherche import chercheur, meilleur_voisin
from commun.verification import (difference_chiffres_seulement, reparation_floue, verifier)

SEUIL_REUTILISATION = 0.95
MAX_TENTATIVES = 2

memoire = charger_memoire()
glossaire = corpus.charger_glossaire()
index_glossaire = GlossaireVectoriel(glossaire)
segments = atelier.segments(n=20)
moteur = obtenir_moteur()
chercher = chercheur(memoire, "dense", k=3)

print(f"Moteur : {moteur.nom()} · {len(memoire)} segments de mémoire")


def ajouter(ancien: list, nouveau: list) -> list:
    """Réducteur : les listes s'accumulent au lieu de s'écraser."""
    return (ancien or []) + (nouveau or [])


class Etat(TypedDict, total=False):
    segment: str
    reference: str
    voisin: dict
    voisins: list
    glossaire: list
    taux_memoire: float
    traduction: str
    anomalies: list
    tentatives: int
    chemin: Annotated[list, ajouter]
    appels: int
    tokens_entree: int

# %% [markdown]
# ## LIRE 1 · Les sept nœuds
#
# Un nœud reçoit l'état et rend **ce qu'il veut y ajouter**, rien d'autre. Les
# sept sont fournis ci-dessous, et ils reprennent les TP 2 et 3.
#
# **Le nœud intéressant est `reutiliser`.** Il ne fait aucun appel au modèle : il
# recopie le segment de la mémoire et reporte les chiffres de la source. C'est
# la « réparation floue » qu'un outil de TAO fait depuis vingt-cinq ans.
#
# **Trois questions, en lisant le code :**
#
# 1. quel nœud incrémente `appels`, et lequel ne l'incrémente jamais ?
# 2. pourquoi `chemin` s'accumule-t-il au lieu de s'écraser, contrairement à
#    `traduction` ?
# 3. `corriger` renvoie vers `verifier`. Qu'est-ce qui empêche cette boucle de
#    tourner indéfiniment ?

# %%
def analyser(etat):
    voisin = meilleur_voisin(etat["segment"], memoire)
    return {"taux_memoire": voisin["score"], "voisin": voisin, "chemin": ["analyser"]}


def reutiliser(etat):
    voisin = etat["voisin"]
    return {"traduction": reparation_floue(voisin["src"], voisin["tgt"], etat["segment"]),
            "chemin": ["reutiliser"]}


def recuperer(etat):
    return {"voisins": chercher(etat["segment"]),
            "glossaire": index_glossaire.chercher(etat["segment"]),
            "chemin": ["recuperer"]}


def traduire(etat):
    messages = construire(etat["segment"], etat["voisins"], etat["glossaire"], CONSIGNE_STYLE)
    reponse = moteur.generer(messages)
    return {"traduction": nettoyer_sortie(reponse.texte),
            "appels": etat.get("appels", 0) + 1,
            "tokens_entree": etat.get("tokens_entree", 0) + reponse.tokens_entree,
            "chemin": ["traduire"]}


def controler(etat):
    return {"anomalies": [str(a) for a in verifier(etat["segment"], etat["traduction"], glossaire)],
            "tentatives": etat.get("tentatives", 0) + 1,
            "chemin": ["verifier"]}


def corriger(etat):
    remarques = "\n".join(f"- {a}" for a in etat["anomalies"])
    messages = construire(etat["segment"], etat["voisins"], etat["glossaire"], CONSIGNE_STYLE)
    messages += [
        {"role": "assistant", "content": etat["traduction"]},
        {"role": "user", "content": f"Le contrôle a relevé :\n{remarques}\n"
                                    f"Rends la traduction corrigée, et rien d'autre."},
    ]
    reponse = moteur.generer(messages)
    return {"traduction": nettoyer_sortie(reponse.texte),
            "appels": etat.get("appels", 0) + 1,
            "tokens_entree": etat.get("tokens_entree", 0) + reponse.tokens_entree,
            "chemin": ["corriger"]}


def livrer(etat):
    return {"chemin": ["livrer"]}


NOEUDS = {"analyser": analyser, "reutiliser": reutiliser, "recuperer": recuperer,
          "traduire": traduire, "verifier": controler, "corriger": corriger, "livrer": livrer}
print(f"{len(NOEUDS)} nœuds : {', '.join(NOEUDS)}")

# %% [markdown]
# ## CODE 1 · Le routeur
#
# Après l'analyse, une décision : réutiliser la mémoire, ou passer par le modèle ?
#
# La règle est celle du client, et elle a **deux** conditions. Un segment très
# proche dont un mot a changé ne se réutilise pas.
#
# `difference_chiffres_seulement(a, b)` est fourni : il rend `True` si les deux
# segments ne diffèrent que par des nombres.

# %%
def router(etat) -> str:
    """Après analyse : "reutiliser" ou "recuperer" ?"""
    # <<<CODE 1 ★★ Le routeur
    # Rendez "reutiliser" si les DEUX conditions sont remplies :
    # - etat["taux_memoire"] >= SEUIL_REUTILISATION
    # - difference_chiffres_seulement(etat["segment"], etat["voisin"]["src"])
    # Sinon "recuperer". Les deux conditions, pas une seule.
    # Test : python tp.py test tp06 -k code1
    raise NotImplementedError(
        "CODE 1 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp06-graphe/README.md"
    )
    # >>>CODE 1


def apres_verification(etat) -> str:
    """Après contrôle : corriger, ou livrer ? Fourni."""
    if etat["anomalies"] and etat["tentatives"] < MAX_TENTATIVES:
        return "corriger"
    return "livrer"


for cas in [(0.98, "Lösenordet förnyas var 14:e dag.", "Lösenordet förnyas var 180:e dag."),
            (0.96, "Certifikatet förnyas var 30:e dag.", "Certifikatet förnyas var 30:e månad."),
            (0.62, "Fakturan skickas varje vecka.", "Underhållsarbeten meddelas 30 dagar.")]:
    taux, voisin, segment = cas
    print(f"  {taux}  {segment[:46]:48} → {router({'taux_memoire': taux, 'voisin': {'src': voisin}, 'segment': segment})}")

# %% [markdown]
# ## Câbler le graphe
#
# **Fourni.** Lisez le schéma, puis les huit lignes qui le réalisent :
#
# ```
# analyser ──┬─(reutiliser)──────────────────────────┐
#            └─(recuperer)── traduire ── verifier ──┬┴─ livrer ── FIN
#                                ▲                  │
#                                └──── corriger ◄───┘
# ```
#
# La boucle `corriger → verifier` est ce qui distingue un graphe d'une chaîne :
# une chaîne ne revient jamais en arrière.

# %%
def construire_graphe(avec_rag: bool = True):
    graphe = StateGraph(Etat)
    for nom, fonction in NOEUDS.items():
        if not avec_rag and nom in ("recuperer", "reutiliser"):
            continue
        graphe.add_node(nom, fonction)
    graphe.set_entry_point("analyser")
    if avec_rag:
        graphe.add_conditional_edges("analyser", router,
                                     {"reutiliser": "reutiliser", "recuperer": "recuperer"})
        graphe.add_edge("recuperer", "traduire")
        graphe.add_edge("reutiliser", "livrer")
    else:
        graphe.add_edge("analyser", "traduire")
    graphe.add_edge("traduire", "verifier")
    graphe.add_conditional_edges("verifier", apres_verification,
                                 {"corriger": "corriger", "livrer": "livrer"})
    graphe.add_edge("corriger", "verifier")
    graphe.add_edge("livrer", END)
    return graphe.compile()


graphe = construire_graphe()
print(graphe.get_graph().draw_mermaid())

# %% [markdown]
# ## OBS 1 · Le graphe tourne
#
# Regardez la **répartition des chemins** en bas de la sortie, et la ligne
# « sans aucun appel ».
#
# Sur ce corpus, la moitié des segments sortent exacts sans que le modèle soit
# appelé une seule fois. La leçon est inconfortable et c'est la meilleure de la
# formation : **pour une bonne part du travail, la bonne réponse n'est pas
# d'appeler un modèle de langue.**

# %%
from commun.mesure import Sortie


def traduire_par_graphe(segment: dict) -> Sortie:
    etat = {"segment": segment["src"], "reference": segment["tgt"], "voisins": [],
            "glossaire": [], "anomalies": [], "tentatives": 0, "chemin": [],
            "appels": 0, "tokens_entree": 0}
    final = graphe.invoke(etat)
    return Sortie(texte=final["traduction"], tokens_entree=final.get("tokens_entree", 0),
                  appels=final.get("appels", 0), chemin=tuple(final.get("chemin", [])),
                  anomalies=final.get("anomalies", []))


routes = atelier.mesurer("tp06-graphe", segments, traduire_par_graphe, titre="Graphe routé")

chemins = {}
for r in routes:
    cle = ("reutiliser" if "reutiliser" in r["chemin"]
           else "rag + correction" if "corriger" in r["chemin"] else "rag")
    chemins[cle] = chemins.get(cle, 0) + 1
sans_appel = [r for r in routes if r["appels"] == 0]
print("\n  Chemins empruntés")
for cle, nombre in sorted(chemins.items(), key=lambda kv: -kv[1]):
    print(f"    {nombre:3} segments  {cle}")
print(f"\n  Appels au modèle : {sum(r['appels'] for r in routes)} pour {len(routes)} segments")
if sans_appel:
    exacts = sum(1 for r in sans_appel if r["hypothese"] == r["reference"])
    print(f"  Dont {len(sans_appel)} sans aucun appel, exacts {exacts} fois sur {len(sans_appel)}.")

# %% [markdown]
# ## RÉG 1 · Le seuil de réutilisation
#
# **Changez `SEUIL`** — 0.85, 0.95, 0.99 — et relancez.
#
# Vous arbitrez entre deux coûts : trop bas, on réutilise des segments qui ne
# conviennent pas et on livre des fautes ; trop haut, on paie le modèle pour des
# segments que la mémoire traitait gratuitement.
#
# Notez le seuil que vous retiendriez pour Helios, et le chiffre qui le justifie.

# %%
SEUIL = 0.85

SEUIL_REUTILISATION = SEUIL
graphe = construire_graphe()
variante = atelier.mesurer(f"tp06-seuil-{SEUIL}", segments, traduire_par_graphe,
                           titre=f"Graphe routé — seuil {SEUIL}", enregistrer=False)
atelier.comparer(("seuil 0.95", routes), (f"seuil {SEUIL}", variante))
SEUIL_REUTILISATION = 0.95
graphe = construire_graphe()

# %% [markdown]
# ## OBS 2 · Et sans RAG ?
#
# La question du dernier après-midi : **le fine-tuning remplace-t-il le RAG ?**
# On ne peut y répondre qu'en mesurant le modèle **privé de contexte**.
#
# Le même graphe, sans les nœuds `recuperer` et `reutiliser` : on analyse, on
# traduit, on vérifie, on livre.

# %%
graphe_sans_rag = construire_graphe(avec_rag=False)
graphe, complet = graphe_sans_rag, graphe
sans_rag = atelier.mesurer("tp06-sans-rag", segments, traduire_par_graphe,
                           titre="Sans RAG — le modèle seul")
graphe = complet
atelier.par_categorie(("graphe routé", routes), ("sans RAG", sans_rag))

# %% [markdown]
# ## OBS 3 · Le banc
#
# `banc` relit `resultats/` et reconstruit **votre** parcours. Les lignes que
# vous n'avez pas mesurées sont marquées « non mesuré », et ce n'est pas grave :
# les TP sont indépendants, personne ne fait tout.
#
# **Ne comparez que des lignes de même `n`.** Une mesure sur 8 segments et une
# sur 20 ne se comparent pas, et c'est la faute la plus fréquente.

# %%
# !python tp.py banc

# %% [markdown]
# ## ARB 1 · Quel cran pour quel segment ?
#
# **Écrivez, pour chacune des quatre catégories** (`repetition`, `piege`,
# `fuzzy`, `nouveau`), le cran que vous mettriez en production, et le chiffre de
# votre table qui le justifie.
#
# > *Votre réponse :*
# >
# >
#
# ## ARB 2 · Ce que vous recommandez
#
# Vous présentez à un responsable qui n'était pas là. **Trois phrases** :
# ce que vous recommandez pour le compte Helios, ce que ça coûte par rapport à
# aujourd'hui, et **ce que vous ne savez pas encore** — parce que c'est cette
# dernière phrase qui distingue une recommandation d'une promesse.
#
# > *Votre réponse :*
# >
# >
