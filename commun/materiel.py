"""Ce que la machine du stagiaire peut faire, et ce qu'il lui manque.

C'est le premier exercice du TP 1, et c'est aussi ce que le formateur demande a
J-3 : la sortie de ``python tp.py check`` dit, avant la session, qui tournera en
local et qui aura besoin d'une cle d'API.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from importlib.util import find_spec

URL_OLLAMA = os.environ.get("TP_OLLAMA_URL", "http://localhost:11434")


def memoire_go() -> float | None:
    try:
        if sys.platform == "darwin":
            sortie = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            return int(sortie.stdout.strip()) / 1024**3
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as f:
                for ligne in f:
                    if ligne.startswith("MemTotal:"):
                        return int(ligne.split()[1]) / 1024**2
        if sys.platform.startswith("win"):
            import ctypes

            class Etat(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            etat = Etat()
            etat.dwLength = ctypes.sizeof(Etat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(etat))
            return etat.ullTotalPhys / 1024**3
    except Exception:
        return None
    return None


def accelerateur() -> str:
    """``mps``, ``cuda`` ou ``cpu``, **sans importer torch** (qu'on n'installe pas)."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    if shutil.which("nvidia-smi"):
        try:
            sortie = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if sortie.returncode == 0 and sortie.stdout.strip():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def ollama() -> dict:
    try:
        with urllib.request.urlopen(f"{URL_OLLAMA}/api/tags", timeout=5) as reponse:
            modeles = [m["name"] for m in json.loads(reponse.read()).get("models", [])]
        return {"joignable": True, "modeles": modeles}
    except (urllib.error.URLError, OSError, ValueError) as erreur:
        return {"joignable": False, "modeles": [], "erreur": str(erreur)}


def petit_modele_present() -> bool:
    """Le petit modèle du chapitre 0 du TP 1 (et du TP 4) est-il téléchargé ?"""
    if find_spec("transformers") is None:
        return False
    try:
        from commun.petit_modele import est_telecharge

        return est_telecharge()
    except Exception:
        return False


def diagnostic() -> dict:
    from commun.embeddings import MODELE_EMBEDDINGS, encodeur_demande
    from commun.moteur import MODELE_OLLAMA, moteur_demande

    etat_ollama = ollama()
    modeles = etat_ollama["modeles"]
    return {
        "systeme": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "accelerateur": accelerateur(),
        "memoire_go": memoire_go(),
        "ollama_joignable": etat_ollama["joignable"],
        "ollama_modeles": modeles,
        "modele_present": any(m.split(":")[0] == MODELE_OLLAMA.split(":")[0] for m in modeles),
        "embeddings_present": any(
            m.split(":")[0] == MODELE_EMBEDDINGS.split(":")[0] for m in modeles
        ),
        "modele_attendu": MODELE_OLLAMA,
        "embeddings_attendu": MODELE_EMBEDDINGS,
        "moteur_retenu": moteur_demande(),
        "encodeur_retenu": encodeur_demande(),
        "cle_mistral": bool(os.environ.get("MISTRAL_API_KEY")),
        "cle_langsmith": bool(os.environ.get("LANGSMITH_API_KEY")),
        "jupyter": find_spec("jupyterlab") is not None or find_spec("notebook") is not None,
        "jupytext": find_spec("jupytext") is not None,
        "transformers": find_spec("transformers") is not None,
        "petit_modele_present": petit_modele_present(),
        "torch": find_spec("torch") is not None,
        "peft": find_spec("peft") is not None,
        "sentence_transformers": find_spec("sentence_transformers") is not None,
        "url_space": os.environ.get("TP_URL_SPACE", ""),
    }


def palier(etat: dict | None = None) -> str:
    """``api``, ``local`` ou ``local-lent``. Determine ce qu'on peut promettre."""
    etat = etat or diagnostic()
    if etat["moteur_retenu"] == "mistral-api":
        return "api"
    if etat["accelerateur"] == "cpu":
        return "local-lent"
    return "local"


def conseils(etat: dict | None = None) -> list[str]:
    """Les commandes à taper pour compléter l'installation, et elles seules."""
    etat = etat or diagnostic()
    a_faire = []
    if not etat["ollama_joignable"] and not etat["cle_mistral"]:
        a_faire.append("Installer Ollama : https://ollama.com/download  puis  ollama serve")
    if etat["ollama_joignable"]:
        if not etat["modele_present"]:
            a_faire.append(f"ollama pull {etat['modele_attendu']}      # 3,0 Go, le traducteur")
        if not etat["embeddings_present"]:
            a_faire.append(f"ollama pull {etat['embeddings_attendu']}                 # 1,2 Go, TP 2")
    if not etat["jupyter"] or not etat["jupytext"]:
        a_faire.append("conda env update -f environment.yml   # Jupyter manque")
    if not etat["petit_modele_present"]:
        a_faire.append(
            "python -c \"from huggingface_hub import snapshot_download as d; "
            "d('HuggingFaceTB/SmolLM2-135M-Instruct')\"\n"
            "    # 270 Mo. Le petit modèle du chapitre 0 du TP 1, et du TP 4.\n"
            "    Aucun GPU : il tourne sur le processeur.")
    if not (etat["torch"] and etat["peft"]):
        a_faire.append(
            "pip install torch peft                 # TP 4 seulement, ~250 Mo.\n"
            "    Aucun GPU n'est nécessaire : le TP 4 entraîne un petit modèle sur\n"
            "    le processeur, et cela prend quelques minutes.")
    # Pas de LangSmith ici : ce n'est pas un prerequis, c'est un reglage du
    # premier exercice du TP 3 (RÉG 0), fait dans le notebook lui-meme.
    if etat["accelerateur"] == "cpu" and not etat["cle_mistral"]:
        a_faire.append(
            "Votre machine n'a pas d'accélérateur. Tout fonctionne, mais comptez\n"
            "    trois à cinq fois plus de temps. Demandez une clé d'API au formateur\n"
            "    si l'attente devient pénible : MISTRAL_API_KEY=... dans .env")
    return a_faire
