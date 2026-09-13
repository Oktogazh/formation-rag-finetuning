"""TP 3 — mesurer la chaine, avec et sans controle. Fourni."""

from __future__ import annotations

import time

from commun import corpus, mesure, rapport
from commun.mesure import Sortie
from commun.prompts import nettoyer_sortie


def evaluer_chaine(avec_verification: bool = False, k: int = 3, n: int = 80) -> list[dict]:
    from tp01.prompt import consigne_systeme
    from tp02.memoire import charger_memoire
    from tp02.recherche import chercheur
    from tp03.chaine import chaine_de_correction, construire_chaine, modele_langchain
    from tp03.boucle import traduire_et_corriger
    from tp03.verification import verifier

    memoire = charger_memoire()
    glossaire = corpus.charger_glossaire()
    segments = corpus.charger_evaluation(n=n)
    chercher = chercheur(memoire, "dense", k)
    modele = modele_langchain()
    chaine = construire_chaine(chercher, glossaire, modele, consigne_systeme())
    correction = chaine_de_correction(modele)

    etiquette = "chaine + verification" if avec_verification else "chaine seule"
    print(f"\nTP 3 — {etiquette} · {len(segments)} segments · {type(modele).__name__}")

    def traduire(segment: dict) -> Sortie:
        debut = time.perf_counter()
        if avec_verification:
            sortie = traduire_et_corriger(segment["src"], chaine, correction, glossaire)
        else:
            sortie = Sortie(texte=nettoyer_sortie(chaine.invoke(segment["src"])), appels=1,
                            chemin=("chaine",))
            # On controle quand meme, sans corriger : sinon la ligne « anomalies »
            # afficherait zero parce qu'on n'a rien cherche, et la comparaison
            # avec --verifier ne voudrait rien dire.
            sortie.anomalies = [
                str(a) for a in verifier(segment["src"], sortie.texte, glossaire)
            ]
        sortie.secondes = time.perf_counter() - debut
        return sortie

    resultats = mesure.evaluer(segments, traduire)
    print()
    print(rapport.table(resultats, f"LangChain — {etiquette}"))
    fautes = rapport.fautes_frequentes(resultats)
    if fautes:
        print("\n" + fautes)

    appels = sum(r["appels"] for r in resultats)
    restantes = sum(len(r["anomalies"]) for r in resultats)
    etiquette_anomalies = "Anomalies restantes  " if avec_verification else "Anomalies detectees  "
    print(f"""
  Appels au modele      {appels} pour {len(resultats)} segments ({appels / len(resultats):.2f} par segment)
  {etiquette_anomalies} {restantes}

  Traçage : posez LANGSMITH_TRACING=true et LANGSMITH_API_KEY dans .env, puis
  relancez. Chaque etape de la chaine apparait dans LangSmith sans une ligne de
  code en plus. C'est l'outil du TP 5.""")
    cran = "tp03-chaine-verifiee" if avec_verification else "tp03-chaine"
    print("\nEnregistre :", rapport.enregistrer(cran, resultats, {"k": k}))
    return resultats
