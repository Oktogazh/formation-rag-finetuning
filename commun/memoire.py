"""La mémoire de traduction, chargée et filtrée. Référence partagée.

Une mémoire contient des segments à différents statuts. On ne propose à un
traducteur que ce qui a été **validé** : proposer un segment non validé, c'est
propager une faute à l'échelle industrielle.
"""

from __future__ import annotations

from commun.corpus import charger_memoire_brute


def charger_memoire() -> list[dict]:
    """Les 444 segments validés de ``tm.jsonl``."""
    return [segment for segment in charger_memoire_brute() if segment["statut"] == "valide"]


def statistiques(memoire: list[dict]) -> dict:
    domaines: dict[str, int] = {}
    for segment in memoire:
        domaines[segment["domaine"]] = domaines.get(segment["domaine"], 0) + 1
    longueurs = [len(s["src"].split()) for s in memoire]
    return {
        "segments": len(memoire),
        "domaines": len(domaines),
        "mots_par_segment": sum(longueurs) / max(len(longueurs), 1),
    }
