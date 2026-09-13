"""Service de traduction Helios — le serveur du TP 5.

Deploye par le formateur sur un Space HuggingFace en CPU gratuit. Les stagiaires
ne le modifient pas : ils le bombardent, regardent ses traces dans LangSmith, et
reparent **une copie locale** (``tp05/serveur_local.py``).

------------------------------------------------------------------------------
DEFAUT VOLONTAIRE — ne pas « reparer » ce fichier
------------------------------------------------------------------------------
Le point d'entree /traduire est declare avec « def » et non « async def », il
appelle le modele de facon bloquante, il n'a ni limite de concurrence, ni file
d'attente bornee, ni delai maximum, et uvicorn tourne avec un seul processus.

Consequence, et c'est l'exercice : a un client le service repond en une
vingtaine de secondes ; a quatre clients simultanes, FastAPI execute les
handlers synchrones dans un pool de threads, les quatre requetes se disputent
deux coeurs, et la latence de chacune est multipliee au lieu d'etre partagee.
Personne n'obtient de reponse rapide, et rien dans les journaux ne dit pourquoi
— sauf les traces LangSmith.

C'est le defaut le plus frequent en production, et le moins visible en
developpement : sur la machine de celui qui l'a ecrit, avec un seul client, tout
allait bien.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, Request
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel

DEPOT_GGUF = os.environ.get("DEPOT_GGUF", "mistralai/Ministral-3-3B-Instruct-2512-GGUF")
FICHIER_GGUF = os.environ.get("FICHIER_GGUF", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf")
PROJET = os.environ.get("LANGSMITH_PROJECT", "formation-helios")
TRACAGE = os.environ.get("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")

SYSTEME = (
    "Tu es traducteur technique du suedois vers le francais pour l'editeur de "
    "logiciels Helios.\n"
    "Tu rends uniquement la traduction francaise du segment demande, sans "
    "commentaire, sans guillemets et sans repeter le suedois."
)

# --- LangSmith ---------------------------------------------------------------
# Sans LANGSMITH_TRACING, @traceable est inerte : le service fonctionne, il
# n'envoie simplement rien. C'est le repli du jour J si LangSmith est
# indisponible, et il se teste avant la session.
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):
        def decorateur(fonction):
            return fonction
        return decorateur


print(f"Telechargement de {FICHIER_GGUF} …", flush=True)
CHEMIN = hf_hub_download(repo_id=DEPOT_GGUF, filename=FICHIER_GGUF)
MODELE = Llama(model_path=CHEMIN, n_ctx=4096, n_threads=2, verbose=False)
print("Modele charge.", flush=True)

app = FastAPI(title="Traduction Helios — sv vers fr")


class Demande(BaseModel):
    segment: str
    voisins: list[dict] = []


def _messages(segment: str, voisins: list[dict]) -> list[dict]:
    blocs = []
    if voisins:
        lignes = ["### Memoire de traduction"]
        for voisin in voisins:
            lignes += [f"sv: {voisin['src']}", f"fr: {voisin['tgt']}"]
        blocs.append("\n".join(lignes))
    blocs.append(f"### Segment a traduire\nsv: {segment}\nfr:")
    return [
        {"role": "system", "content": SYSTEME},
        {"role": "user", "content": "\n\n".join(blocs)},
    ]


@traceable(run_type="llm", name="traduire")
def _appeler_modele(segment: str, voisins: list[dict], clients: str) -> dict:
    """L'appel au modele, trace. ``clients`` vient de l'en-tete X-Clients.

    C'est cette metadonnee qui permet a la salle de retrouver ses propres
    requetes dans LangSmith : chaque stagiaire lit la ligne qui porte le nombre
    de clients qu'il a demande.
    """
    debut = time.perf_counter()
    reponse = MODELE.create_chat_completion(
        messages=_messages(segment, voisins), temperature=0.0, max_tokens=256
    )
    secondes = time.perf_counter() - debut
    return {
        "traduction": reponse["choices"][0]["message"]["content"].strip(),
        "secondes": secondes,
        "tokens": reponse["usage"]["completion_tokens"],
        "clients_declares": clients,
    }


@app.get("/sante")
def sante():
    """Dit si le tracage est actif. C'est le seul moyen de le savoir sans LangSmith."""
    return {
        "statut": "ok",
        "modele": FICHIER_GGUF,
        "tracing": TRACAGE,
        "projet": PROJET if TRACAGE else None,
        "endpoint": os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    }


@app.post("/traduire")
def traduire(demande: Demande, requete: Request):  # DEFAUT VOLONTAIRE : « def », pas « async def »
    clients = requete.headers.get("X-Clients", "?")
    return _appeler_modele(demande.segment, demande.voisins, clients)


if __name__ == "__main__":
    import uvicorn

    # DEFAUT VOLONTAIRE : un seul processus, aucune limite de concurrence.
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)
