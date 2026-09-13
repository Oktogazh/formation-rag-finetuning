"""TP 5 — le meme service, sur votre machine, avec et sans garde. Fourni.

Lancez-le dans un terminal, bombardez-le depuis un autre :

    python tp.py api                    # avec la garde que vous avez ecrite
    python tp.py api --sans-garde       # le defaut du Space, reproduit ici

    python tp.py assaut --url http://localhost:8000 --clients 8 --duree 30

Comparez les deux. C'est la meme machine, le meme modele, le meme assaut : la
seule difference est trente lignes de garde.
"""

from __future__ import annotations

import time

DESCRIPTION = "Service de traduction suedois vers francais — Helios"


def construire_app(avec_garde: bool = True):
    """L'application FastAPI. Importe FastAPI tard : il n'est pas requis ailleurs."""
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel

    from commun.moteur import obtenir_moteur
    from commun.prompts import construire_messages, nettoyer_sortie
    from tp05.reparer import Delai, Garde, Sature

    app = FastAPI(title=DESCRIPTION)
    moteur = obtenir_moteur()
    garde = Garde(concurrence_max=1, file_max=4, timeout_s=60.0) if avec_garde else None
    app.state.compteur = 0

    class Demande(BaseModel):
        segment: str
        voisins: list[dict] = []

    def traduire_bloquant(segment: str, voisins: list[dict]) -> dict:
        debut = time.perf_counter()
        reponse = moteur.generer(construire_messages(segment, voisins))
        return {
            "traduction": nettoyer_sortie(reponse.texte),
            "secondes": time.perf_counter() - debut,
            "tokens": reponse.tokens_sortie,
        }

    @app.get("/sante")
    def sante():
        return {
            "statut": "ok",
            "moteur": moteur.nom(),
            "garde": garde.etat() if garde else None,
            "requetes": app.state.compteur,
        }

    @app.post("/traduire")
    async def traduire(demande: Demande, requete: Request):
        app.state.compteur += 1
        if garde is None:
            # Le defaut du Space, reproduit exprès : on appelle le modele dans
            # la boucle asyncio, sans limite et sans delai maximum. Le serveur
            # ne peut plus rien faire d'autre pendant ce temps, /sante compris.
            return traduire_bloquant(demande.segment, demande.voisins)
        try:
            return await garde.executer(traduire_bloquant, demande.segment, demande.voisins)
        except Sature as erreur:
            raise HTTPException(status_code=503, detail=str(erreur)) from erreur
        except Delai as erreur:
            raise HTTPException(status_code=504, detail=str(erreur)) from erreur

    return app


def lancer(port: int = 8000, repare: bool = True) -> int:
    try:
        import uvicorn
    except ImportError:
        print("  pip install fastapi uvicorn")
        return 2
    etat = "avec garde" if repare else "SANS garde (le defaut du Space)"
    print(f"""
  Service de traduction — {etat}
  http://localhost:{port}/sante        etat du service
  http://localhost:{port}/docs         interface d'essai

  Dans un autre terminal :
      python tp.py assaut --url http://localhost:{port} --clients 8 --duree 30
""")
    uvicorn.run(construire_app(repare), host="127.0.0.1", port=port, log_level="warning")
    return 0
