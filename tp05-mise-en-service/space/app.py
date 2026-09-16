"""Service de traduction Helios — le serveur du TP 5.

Deploye par le formateur sur un Space HuggingFace en CPU gratuit. Les stagiaires
ne le modifient pas : ils le bombardent, regardent ses traces dans LangSmith, et
reparent **une copie locale** (``service/serveur_local.py``), dans le
notebook ``tp05.ipynb``.

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

import asyncio
import json
import os
import time
import urllib.error
import urllib.request

import gradio as gr
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Ollama, comme le reste de la formation (commun/moteur.py) : le GGUF servi
# directement par llama-cpp-python s'est heurte a un bug reel de chargement du
# vocabulaire ("invalid gguf type for tokenizer.ggml.scores"), present meme
# dans la derniere version publiee de llama-cpp-python au 16/09/2026. Ollama
# charge le meme modele sans probleme : c'est lui qui fait deja tourner les six
# TP, ce Space s'aligne dessus plutot que de garder une deuxieme dependance.
MODELE_OLLAMA = os.environ.get("OLLAMA_MODELE", "ministral-3:3b")
URL_OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
PROJET = os.environ.get("LANGSMITH_PROJECT", "formation-helios")
TRACAGE = os.environ.get("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")

# L'adresse que l'interface affiche dans ses exemples copiables. Elle ne sert
# qu'a ca : le service ne s'appelle jamais lui-meme par cette URL.
ADRESSE_PUBLIQUE = os.environ.get(
    "ADRESSE_PUBLIQUE", "https://oktogazh-formation-rag-finetuning.hf.space"
)

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


def _attendre_ollama(url: str, tentatives: int = 60) -> None:
    for _ in range(tentatives):
        try:
            urllib.request.urlopen(f"{url}/api/tags", timeout=2)
            return
        except urllib.error.URLError:
            time.sleep(1)
    raise RuntimeError(f"Ollama ne repond pas sur {url} apres {tentatives} s.")


print(f"Attente d'Ollama sur {URL_OLLAMA} …", flush=True)
_attendre_ollama(URL_OLLAMA)
print(f"Ollama pret, modele {MODELE_OLLAMA}.", flush=True)

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
    charge = {
        "model": MODELE_OLLAMA,
        "messages": _messages(segment, voisins),
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 256, "seed": 1234},
    }
    requete = urllib.request.Request(
        f"{URL_OLLAMA}/api/chat",
        data=json.dumps(charge).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    debut = time.perf_counter()
    with urllib.request.urlopen(requete, timeout=120) as reponse:
        donnees = json.loads(reponse.read())
    secondes = time.perf_counter() - debut
    return {
        "traduction": donnees["message"]["content"].strip(),
        "secondes": secondes,
        "tokens": donnees.get("eval_count", 0),
        "clients_declares": clients,
    }


@app.get("/sante")
def sante():
    """Dit si le tracage est actif. C'est le seul moyen de le savoir sans LangSmith."""
    return {
        "statut": "ok",
        "modele": MODELE_OLLAMA,
        "tracing": TRACAGE,
        "projet": PROJET if TRACAGE else None,
        "endpoint": os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    }


@app.post("/traduire")
def traduire(demande: Demande, requete: Request):  # DEFAUT VOLONTAIRE : « def », pas « async def »
    clients = requete.headers.get("X-Clients", "?")
    return _appeler_modele(demande.segment, demande.voisins, clients)


# =============================================================================
# FRONT-END — l'interface, montee sur le meme service que l'API
# =============================================================================
# Deux choses vivent dans ce processus, et tout le TP 5 tient dans leur
# distinction :
#
#   POST /traduire   le BACK-END. Une API JSON, sans interface. C'est lui que le
#                    notebook bombarde, c'est lui qui appelle le modele, et c'est
#                    lui qui porte le defaut volontaire decrit en tete de fichier.
#   GET  /           le FRONT-END. La page ci-dessous. Elle n'a aucune
#                    intelligence propre : elle POSTe sur /traduire, par HTTP,
#                    exactement comme le font curl et le notebook.
#
# L'interface est donc un client de l'API parmi d'autres, pas une deuxieme
# implementation. C'est volontaire : si elle appelait _appeler_modele()
# directement, la frontiere disparaitrait, et avec elle ce que la salle doit
# voir — qu'un service se mesure a son API, pas a sa page d'accueil.

PORT = int(os.environ.get("PORT", "7860"))
URL_INTERNE = f"http://127.0.0.1:{PORT}"

EXEMPLES = [
    "Abonnemanget Företag tillåter 28 förfrågningar per minut.",
    "Du kan bjuda in upp till 180 administratörer per konto.",
    "Slutpunkten för version 10 tas bort om 12 månader.",
]


def _poster_sur_api(segment: str) -> dict:
    """Appelle le back-end par HTTP. Bloquant, d'ou le passage en fil separe."""
    requete = urllib.request.Request(
        f"{URL_INTERNE}/traduire",
        data=json.dumps({"segment": segment, "voisins": []}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Clients": "interface"},
    )
    with urllib.request.urlopen(requete, timeout=180) as reponse:
        return json.loads(reponse.read())


async def _traduire_depuis_interface(segment: str):
    segment = (segment or "").strip()
    if not segment:
        return "", "Entrez un segment suédois."
    debut = time.perf_counter()
    try:
        # to_thread : le fil de l'interface ne doit pas bloquer celui qui sert
        # l'API, sinon le service se repondrait a lui-meme en file indienne.
        donnees = await asyncio.to_thread(_poster_sur_api, segment)
    except Exception as erreur:  # noqa: BLE001 — l'erreur est le resultat affiche
        return "", f"Le back-end n'a pas répondu : {erreur}"
    attente = time.perf_counter() - debut
    detail = (
        f"{attente:.1f} s de bout en bout · {donnees.get('secondes', 0):.1f} s "
        f"côté modèle · {donnees.get('tokens', 0)} tokens générés"
    )
    return donnees.get("traduction", ""), detail


with gr.Blocks(title="Traduction Helios — sv vers fr") as interface:
    gr.Markdown(
        "# Traduction Helios · suédois → français\n"
        f"Ministral 3B sur processeur. **Cette page est le _front-end_** : elle "
        f"n'appelle pas le modèle, elle poste sur `POST /traduire`, qui est le "
        f"_back-end_. Les deux tournent dans ce Space, et c'est le back-end "
        f"que le TP 5 mesure."
    )
    with gr.Row():
        with gr.Column():
            entree = gr.Textbox(
                label="Segment suédois",
                placeholder="Abonnemanget Bas tillåter 20 förfrågningar per minut.",
                lines=4,
            )
            bouton = gr.Button("Traduire", variant="primary")
            gr.Examples(examples=EXEMPLES, inputs=entree, label="Exemples du corpus Helios")
        with gr.Column():
            sortie = gr.Textbox(label="Traduction française", lines=4)
            mesure = gr.Markdown("")

    gr.Markdown(
        "---\n"
        "### Le même service, sans cette page\n"
        "L'interface ci-dessus est un client comme un autre. Les deux appels "
        "suivants font exactement ce que fait le bouton :\n"
    )
    with gr.Accordion("En ligne de commande", open=False):
        gr.Code(
            f"curl -s -X POST {ADRESSE_PUBLIQUE}/traduire \\\n"
            f"  -H 'Content-Type: application/json' -H 'X-Clients: 1' \\\n"
            f"  -d '{{\"segment\":\"{EXEMPLES[0]}\"}}'\n\n"
            f"curl -s {ADRESSE_PUBLIQUE}/sante",
            language="shell",
        )
    with gr.Accordion("Dans le notebook du TP 5", open=False):
        gr.Code(
            f'URL_SERVICE = "{ADRESSE_PUBLIQUE}"',
            language="python",
        )
        gr.Markdown(
            "Collez cette ligne en **RÉG 0** de `tp05.ipynb`. L'assaut du TP "
            "frappe `/traduire` en direct — il ne passe jamais par cette page."
        )

    bouton.click(_traduire_depuis_interface, inputs=entree, outputs=[sortie, mesure])
    entree.submit(_traduire_depuis_interface, inputs=entree, outputs=[sortie, mesure])

# Montee sur "/" APRES /traduire et /sante : FastAPI essaie ses routes dans
# l'ordre d'enregistrement, les deux routes d'API gardent donc la priorite.
app = gr.mount_gradio_app(app, interface, path="/")


if __name__ == "__main__":
    import uvicorn

    # DEFAUT VOLONTAIRE : un seul processus, aucune limite de concurrence.
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)
