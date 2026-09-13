"""TP 6 — l'etat qui circule dans le graphe. Fourni.

Un graphe LangGraph, c'est un dictionnaire qui passe de noeud en noeud, chaque
noeud le lisant et y ajoutant quelque chose. Tout est la : il n'y a pas d'objet
cache, pas de contexte implicite. C'est ce qui le rend lisible — et tracable.
"""

from __future__ import annotations

from typing import Annotated, TypedDict


def _ajouter(ancien: list, nouveau: list) -> list:
    """Reducteur : les listes s'accumulent au lieu de s'ecraser."""
    return (ancien or []) + (nouveau or [])


class EtatTraduction(TypedDict, total=False):
    segment: str
    reference: str
    categorie: str

    taux_memoire: float
    voisin: dict
    voisins: list
    glossaire: list

    traduction: str
    anomalies: list
    tentatives: int

    chemin: Annotated[list, _ajouter]
    appels: int
    tokens_entree: int
    tokens_sortie: int


def etat_initial(segment: dict) -> EtatTraduction:
    return {
        "segment": segment["src"],
        "reference": segment.get("tgt", ""),
        "categorie": segment.get("categorie", ""),
        "taux_memoire": 0.0,
        "voisins": [],
        "glossaire": [],
        "traduction": "",
        "anomalies": [],
        "tentatives": 0,
        "chemin": [],
        "appels": 0,
        "tokens_entree": 0,
        "tokens_sortie": 0,
    }
