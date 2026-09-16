"""TP 5 — le meme service, sur votre machine, avec et sans garde. Fourni.

Deux facons de le lancer, et les deux servent :

**Depuis un terminal**, quand vous voulez le voir vivre pendant que vous le
bombardez depuis un autre :

    python tp.py api                    # avec la garde de reference
    python tp.py api --sans-garde       # le defaut du Space, reproduit ici

    python tp.py assaut --url http://localhost:8000 --clients 8 --duree 30

**Depuis le notebook**, quand vous voulez juste une cible a mesurer sans ouvrir
un second terminal :

    serveur = demarrer_en_fond(port=8000, repare=False)
    ...
    serveur.arreter()

Comparez les deux modes. C'est la meme machine, le meme modele, le meme assaut :
la seule difference est trente lignes de garde.
"""

# Pas de « from __future__ import annotations » ici, et ce n'est pas un oubli :
# il transformerait les annotations en chaines de caracteres, et FastAPI ne
# saurait plus resoudre « demande: Demande » — la classe est definie DANS
# construire_app(). Il prendrait alors « demande » pour un parametre d'URL et
# repondrait 422 a chaque requete, sans que rien n'explique pourquoi.

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
DESCRIPTION = "Service de traduction suedois vers francais — Helios"


def construire_app(avec_garde: bool = True):
    """L'application FastAPI. Importe FastAPI tard : il n'est pas requis ailleurs."""
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel

    from commun.moteur import obtenir_moteur
    from commun.prompts import construire_messages, nettoyer_sortie
    from service.reparer import Delai, Garde, Sature

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


# ---------------------------------------------------------------------------
# Le meme serveur, lance depuis le notebook
# ---------------------------------------------------------------------------
class ServeurEnFond:
    """Un service lance dans un autre processus, et l'URL pour le joindre.

    Pourquoi un autre processus : un serveur, ca boucle. Lance dans la cellule,
    il figerait le notebook — et un notebook fige ne mesure rien.
    """

    def __init__(self, processus: subprocess.Popen, port: int, repare: bool, journal: Path):
        self.processus = processus
        self.port = port
        self.repare = repare
        self.chemin_journal = journal
        self.url = f"http://127.0.0.1:{port}"

    def sante(self) -> dict:
        with urllib.request.urlopen(self.url + "/sante", timeout=10) as reponse:
            import json

            return json.loads(reponse.read())

    def journal(self, lignes: int = 20) -> None:
        """Les dernieres lignes du serveur. A lire quand il ne demarre pas."""
        texte = self.chemin_journal.read_text(encoding="utf-8", errors="replace")
        print("\n".join(texte.splitlines()[-lignes:]) or "  (journal vide)")

    def arreter(self) -> None:
        if self.processus.poll() is None:
            self.processus.terminate()
            try:
                self.processus.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self.processus.kill()
        print(f"  Service arrete (port {self.port}).")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.arreter()


def demarrer_en_fond(port: int = 8000, repare: bool = True,
                     attente: float = 120.0) -> ServeurEnFond:
    """Lance ``python tp.py api`` dans un autre processus et attend qu'il reponde."""
    journal = RACINE / "resultats" / f"tp05-serveur-{port}.log"
    journal.parent.mkdir(exist_ok=True)
    commande = [sys.executable, "tp.py", "api", "--port", str(port)]
    if not repare:
        commande.append("--sans-garde")
    with journal.open("w", encoding="utf-8") as sortie:
        processus = subprocess.Popen(commande, cwd=RACINE, stdout=sortie,
                                     stderr=subprocess.STDOUT)
    serveur = ServeurEnFond(processus, port, repare, journal)

    limite = time.perf_counter() + attente
    while time.perf_counter() < limite:
        if processus.poll() is not None:
            serveur.journal()
            raise RuntimeError(
                f"Le service s'est arrete tout de suite. Journal : {journal}")
        try:
            etat = serveur.sante()
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.5)
            continue
        mode = "avec garde" if repare else "SANS garde (le defaut du Space)"
        print(f"  Service pret sur {serveur.url}  —  {mode}, moteur {etat['moteur']}")
        return serveur

    serveur.arreter()
    raise TimeoutError(f"Le service n'a pas repondu en {attente:.0f} s. Journal : {journal}")
