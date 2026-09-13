"""TP 2 — la memoire de traduction, c'est-a-dire la base documentaire.

Une chose a dire avant d'ecrire une ligne : **une memoire de traduction est
deja un systeme de recherche documentaire**. Elle fait du retrieval depuis les
annees 1990, avec une metrique de similarite de chaines. Ce que la formation
appelle RAG, c'est la meme mecanique avec une meilleure metrique et un
generateur au bout.

Vous n'apprenez donc pas une idee neuve. Vous voyez que votre outil quotidien
est un cas particulier de ce qu'on va construire.
"""

from __future__ import annotations

from commun.corpus import charger_memoire_brute


def charger_memoire() -> list[dict]:
    """Les segments de la memoire **utilisables**.

    Un fichier de memoire contient des segments a differents statuts. On ne
    propose a un traducteur que ce qui a ete valide : proposer un segment
    non valide, c'est propager une faute a l'echelle industrielle.
    """
    # <<<TODO 1 ★ Ne garder que les segments valides
    # charger_memoire_brute() rend les 444 entrees du fichier tm.jsonl.
    # Chaque entree a les cles : id, src, tgt, domaine, date, statut.
    # Rendez la liste de celles dont le statut vaut exactement "valide".
    # Test : python tp.py test tp02 -k todo1
    raise NotImplementedError(
        "TODO 1 — a completer. Consigne juste au-dessus, "
        "explications dans tp02-rag/README.md"
    )
    # >>>TODO 1


def statistiques(memoire: list[dict]) -> dict:
    """De quoi regarder ce qu'on vient de charger. Fourni."""
    domaines: dict[str, int] = {}
    for segment in memoire:
        domaines[segment["domaine"]] = domaines.get(segment["domaine"], 0) + 1
    longueurs = [len(s["src"].split()) for s in memoire]
    return {
        "segments": len(memoire),
        "domaines": len(domaines),
        "mots_par_segment": sum(longueurs) / max(len(longueurs), 1),
        "plus_gros_domaine": max(domaines.items(), key=lambda kv: kv[1]) if domaines else None,
    }
