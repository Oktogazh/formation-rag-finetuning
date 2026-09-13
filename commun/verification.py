"""Les quatre contrôles de la procédure de relecture de l'agence.

``commun/mesure.py`` les utilise pour **noter**. Ici, ils servent à
**déclencher** : une anomalie relance le modèle, ou déclenche une réparation.
C'est tout le passage d'un indicateur à un garde-fou.

Vous réécrivez ``verifier`` au TP 3. Celui d'ici sert au TP 6.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from commun.mesure import balises, chiffres, terminologie, vouvoiement


@dataclass
class Anomalie:
    type: str
    detail: str

    def __str__(self) -> str:
        return f"{self.type} : {self.detail}"


def verifier(segment_src: str, traduction: str, glossaire: list[dict]) -> list[Anomalie]:
    """Les défauts de cette traduction. Liste vide = elle passe."""
    anomalies: list[Anomalie] = []
    _, _, fautes = terminologie(segment_src, traduction, glossaire)
    for faute in fautes:
        anomalies.append(Anomalie("terminologie", faute))
    if not chiffres(segment_src, traduction):
        anomalies.append(
            Anomalie("chiffres", "les nombres de la source ne sont pas tous repris à l'identique"))
    if vouvoiement(traduction) is False:
        anomalies.append(Anomalie("tutoiement", "le guide de style impose le vouvoiement"))
    if not balises(segment_src, traduction):
        anomalies.append(Anomalie("balises", "les marqueurs {0}, {1} doivent être conservés"))
    return anomalies


def reparer_chiffres(segment_src: str, traduction: str) -> str:
    """Reporte les nombres de la source, sans appeler le modèle.

    Zéro appel, zéro token, et plus fiable qu'un second appel : un modèle qui
    s'est trompé une fois se trompe souvent deux.
    """
    source = re.findall(r"\d+", segment_src)
    produits = re.findall(r"\d+", traduction)
    if len(source) != len(produits) or source == produits:
        return traduction
    morceaux, i = [], 0
    for part in re.split(r"(\d+)", traduction):
        if part.isdigit() and i < len(source):
            morceaux.append(source[i]); i += 1
        else:
            morceaux.append(part)
    return "".join(morceaux)


def reparation_floue(voisin_src: str, voisin_tgt: str, source: str) -> str:
    """Reporte les nombres de la source dans la traduction d'un voisin proche."""
    anciens = re.findall(r"\d+", voisin_src)
    nouveaux = re.findall(r"\d+", source)
    if len(anciens) != len(nouveaux):
        return voisin_tgt
    morceaux, i = [], 0
    for part in re.split(r"(\d+)", voisin_tgt):
        if part.isdigit():
            if i < len(anciens) and part == anciens[i]:
                morceaux.append(nouveaux[i]); i += 1
            elif part in anciens:
                morceaux.append(nouveaux[anciens.index(part)])
            else:
                morceaux.append(part)
        else:
            morceaux.append(part)
    return "".join(morceaux)


def difference_chiffres_seulement(source: str, voisin: str) -> bool:
    """Vrai si les deux segments ne diffèrent que par des nombres."""
    from difflib import SequenceMatcher

    mots_source = re.findall(r"\w+", source.lower())
    mots_voisin = re.findall(r"\w+", voisin.lower())
    differences = [
        (mots_voisin[i1:i2], mots_source[j1:j2])
        for etiquette, i1, i2, j1, j2 in SequenceMatcher(None, mots_voisin, mots_source).get_opcodes()
        if etiquette != "equal"
    ]
    if not differences:
        return True
    return all(all(mot.isdigit() for mot in avant + apres) for avant, apres in differences)
