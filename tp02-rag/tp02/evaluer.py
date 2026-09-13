"""TP 2 — mesurer le RAG contre le prompt du TP 1. Fourni, rien a completer."""

from __future__ import annotations

from commun import corpus, mesure, rapport
from commun.mesure import Sortie
from commun.moteur import obtenir_moteur
from commun.prompts import nettoyer_sortie


def traducteur_rag(moteur, chercher, glossaire, consignes: str = ""):
    """La fonction ``segment -> Sortie`` du cran 2."""
    from tp02.augmenter import construire, glossaire_pertinent

    def traduire(segment: dict) -> Sortie:
        voisins = chercher(segment["src"])
        termes = glossaire_pertinent(segment["src"], glossaire)
        messages = construire(segment["src"], voisins, termes, consignes)
        reponse = moteur.generer(messages)
        return Sortie(
            texte=nettoyer_sortie(reponse.texte),
            tokens_entree=reponse.tokens_entree,
            tokens_sortie=reponse.tokens_sortie,
            secondes=reponse.secondes,
            chemin=("rag",),
        )

    return traduire


def evaluer_rag(recherche: str = "lexicale", k: int = 3, n: int = 80) -> list[dict]:
    from tp01.prompt import consigne_systeme
    from tp02.memoire import charger_memoire, statistiques
    from tp02.recherche import chercheur

    memoire = charger_memoire()
    stats = statistiques(memoire)
    print(f"\nMemoire : {stats['segments']} segments valides, "
          f"{stats['domaines']} domaines, "
          f"{stats['mots_par_segment']:.0f} mots par segment en moyenne")

    segments = corpus.charger_evaluation(n=n)
    glossaire = corpus.charger_glossaire()
    moteur = obtenir_moteur()
    chercher = chercheur(memoire, recherche, k)

    print(f"TP 2 — recherche {recherche}, k={k} · {len(segments)} segments · {moteur.nom()}")
    resultats = mesure.evaluer(
        segments,
        traducteur_rag(moteur, chercher, glossaire, consigne_systeme()),
    )
    print()
    print(rapport.table(resultats, f"RAG — recherche {recherche}, k={k}"))
    fautes = rapport.fautes_frequentes(resultats)
    if fautes:
        print("\n" + fautes)

    rappel = _rappel_de_la_recherche(segments, chercher)
    print(f"""
  Rappel de la recherche : le bon voisin est dans les {k} retenus pour
  {rappel:.0f} % des segments. C'est le plafond de ce que le modele peut faire :
  ce que la recherche ne remonte pas, la generation ne l'inventera pas.""")
    print("\nEnregistre :", rapport.enregistrer(
        f"tp02-rag-{recherche}-k{k}", resultats,
        {"moteur": moteur.nom(), "recherche": recherche, "k": k, "rappel": rappel},
    ))
    return resultats


def _rappel_de_la_recherche(segments, chercher) -> float:
    """Part des segments pour lesquels un voisin a plus de 90 % de similarite.

    Mesure la recherche **seule**, sans le modele. Un RAG mediocre est presque
    toujours un probleme de recherche, pas de generation, et on ne le voit que
    si on les mesure separement.
    """
    from tp02.recherche import similarite

    trouves = 0
    for segment in segments:
        voisins = chercher(segment["src"])
        if voisins and max(similarite(segment["src"], v["src"]) for v in voisins) >= 0.90:
            trouves += 1
    return 100.0 * trouves / max(len(segments), 1)
