"""TP 6 — faire tourner le graphe sur les 80 segments et regarder les chemins. Fourni."""

from __future__ import annotations

import time

from commun import corpus, mesure, rapport
from commun.mesure import Sortie
from commun.moteur import obtenir_moteur


def evaluer_graphe(n: int = 80, moteur_traduction: str = "base", k: int = 3,
                   avec_rag: bool = True) -> list[dict]:
    from tp01.prompt import consigne_systeme
    from tp02.memoire import charger_memoire
    from tp02.recherche import chercheur
    from tp06.etat import etat_initial
    from tp06.graphe import construire_graphe
    from tp06.noeuds import construire_noeuds

    memoire = charger_memoire()
    glossaire = corpus.charger_glossaire()
    segments = corpus.charger_evaluation(n=n)
    moteur = obtenir_moteur(moteur_traduction)
    noeuds = construire_noeuds(
        memoire, glossaire, moteur, chercheur(memoire, "dense", k), consigne_systeme()
    )
    graphe = construire_graphe(noeuds, avec_rag=avec_rag)

    etiquette = f"graphe {'routé' if avec_rag else 'sans RAG'} · moteur {moteur_traduction}"
    print(f"\nTP 6 — {etiquette} · {len(segments)} segments · {moteur.nom()}")

    def traduire(segment: dict) -> Sortie:
        debut = time.perf_counter()
        final = graphe.invoke(etat_initial(segment))
        return Sortie(
            texte=final["traduction"],
            tokens_entree=final.get("tokens_entree", 0),
            tokens_sortie=final.get("tokens_sortie", 0),
            secondes=time.perf_counter() - debut,
            appels=final.get("appels", 0),
            chemin=tuple(final.get("chemin", [])),
            anomalies=final.get("anomalies", []),
        )

    resultats = mesure.evaluer(segments, traduire)
    print()
    print(rapport.table(resultats, etiquette))

    chemins: dict[str, int] = {}
    for r in resultats:
        cle = "reutiliser" if "reutiliser" in r["chemin"] else (
            "rag + correction" if "corriger" in r["chemin"] else "rag"
        )
        chemins[cle] = chemins.get(cle, 0) + 1
    appels = sum(r["appels"] for r in resultats)

    print("\n  Chemins empruntes")
    for cle, nombre in sorted(chemins.items(), key=lambda kv: -kv[1]):
        print(f"    {nombre:3} segments  {cle}")
    print(f"\n  Appels au modele : {appels} pour {len(resultats)} segments "
          f"({appels / len(resultats):.2f} par segment)")

    sans_appel = [r for r in resultats if r["appels"] == 0]
    if sans_appel:
        exacts = sum(1 for r in sans_appel if r["hypothese"] == r["reference"])
        print(f"  Dont {len(sans_appel)} sans aucun appel, exacts {exacts} fois sur "
              f"{len(sans_appel)}.")

    cran = "tp06-graphe" if avec_rag else "tp06-sans-rag"
    print("\nEnregistre :", rapport.enregistrer(
        cran, resultats, {"moteur": moteur.nom(), "chemins": chemins, "appels": appels}))
    return resultats
