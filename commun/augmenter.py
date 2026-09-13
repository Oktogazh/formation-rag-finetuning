"""Construire le prompt augmenté : glossaire filtré, voisins, segment.

Vous réécrivez ces deux fonctions au TP 2. Celles d'ici servent aux autres TP.
"""

from __future__ import annotations

import re

from commun.prompts import construire_messages

_MOTS = re.compile(r"[\wåäöÅÄÖ]+")


def glossaire_pertinent(segment_src: str, glossaire: list[dict]) -> list[dict]:
    """Les entrées de glossaire qui concernent ce segment, et elles seules.

    Comparaison par **début de mot** : le suédois compose et fléchit
    (``förfrågan``, ``förfrågningar``), une égalité stricte ne trouverait rien.
    """
    mots = [m.lower() for m in _MOTS.findall(segment_src)]
    return [t for t in glossaire if any(mot.startswith(t["sv"].lower()) for mot in mots)]


def construire(segment_src: str, voisins: list[dict], glossaire: list[dict],
               consignes: str = "") -> list[dict]:
    """Le prompt augmenté, au gabarit unique de ``commun/prompts.py``."""
    return construire_messages(segment_src, voisins=voisins, glossaire=glossaire,
                               consignes=consignes)
