"""Les vecteurs du TP 2, derriere une seule porte, comme le moteur.

Trois implementations :

1. ``TP_EMBEDDINGS=factice`` — sacs de trigrammes de caracteres, haches sur 256
   dimensions, normalises. Aucun reseau, aucun poids, deterministe. C'est ce que
   les tests utilisent, et c'est suffisant pour verifier qu'une recherche
   fonctionne : deux phrases proches ont des vecteurs proches.
2. **Ollama** avec ``bge-m3`` (defaut) — un vrai modele d'embeddings
   multilingue, qui place le suedois et le francais dans le meme espace. C'est
   ce qui permet de chercher un segment suedois et de retrouver son voisin.
3. ``sentence-transformers`` — si vous l'avez installe, on s'en sert. Ce n'est
   pas un prerequis de la formation : ca tire ``torch``, soit 2,5 Go.

Le TP 2 vous fera comparer la recherche **lexicale** (celle d'un outil de TAO
depuis 1998) et la recherche **dense** (celle-ci). La conclusion n'est pas
donnee d'avance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request

URL_OLLAMA = os.environ.get("TP_OLLAMA_URL", "http://localhost:11434")
MODELE_EMBEDDINGS = os.environ.get("TP_OLLAMA_EMBEDDINGS", "bge-m3")
DIMENSIONS_FACTICE = 256


class ErreurEmbeddings(RuntimeError):
    pass


class EncodeurFactice:
    """Sac de trigrammes de caracteres hache. Pas un modele, mais pas du bruit.

    Deux phrases qui partagent des morceaux de mots ont des vecteurs proches :
    c'est assez pour que les tests verifient qu'une recherche remonte bien le
    bon voisin, sans telecharger un modele.
    """

    def nom(self) -> str:
        return "factice:trigrammes"

    def encoder(self, textes: list[str]) -> list[list[float]]:
        return [self._vecteur(t) for t in textes]

    @staticmethod
    def _vecteur(texte: str) -> list[float]:
        texte = re.sub(r"\s+", " ", texte.lower().strip())
        vecteur = [0.0] * DIMENSIONS_FACTICE
        for i in range(max(1, len(texte) - 2)):
            trigramme = texte[i:i + 3]
            case = int(hashlib.md5(trigramme.encode("utf-8")).hexdigest()[:8], 16)
            vecteur[case % DIMENSIONS_FACTICE] += 1.0
        norme = math.sqrt(sum(v * v for v in vecteur)) or 1.0
        return [v / norme for v in vecteur]


class EncodeurOllama:
    def __init__(self, modele: str = MODELE_EMBEDDINGS, url: str = URL_OLLAMA):
        self.modele = modele
        self.url = url.rstrip("/")

    def nom(self) -> str:
        return f"ollama:{self.modele}"

    def encoder(self, textes: list[str]) -> list[list[float]]:
        vecteurs: list[list[float]] = []
        for debut in range(0, len(textes), 64):
            lot = textes[debut:debut + 64]
            requete = urllib.request.Request(
                f"{self.url}/api/embed",
                data=json.dumps({"model": self.modele, "input": lot}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(requete, timeout=300) as reponse:
                    donnees = json.loads(reponse.read())
            except urllib.error.URLError as erreur:
                raise ErreurEmbeddings(
                    f"Ollama ne repond pas ({erreur}).\n"
                    f"  ollama pull {self.modele}   puis   ollama serve"
                ) from erreur
            if "error" in donnees:
                raise ErreurEmbeddings(
                    f"Ollama : {donnees['error']}\n  ollama pull {self.modele}"
                )
            vecteurs.extend(donnees["embeddings"])
        return [_normaliser(v) for v in vecteurs]


class EncodeurSentenceTransformers:  # pragma: no cover - chemin optionnel
    def __init__(self, modele: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer

        self.modele_nom = modele
        self.modele = SentenceTransformer(modele)

    def nom(self) -> str:
        return f"sentence-transformers:{self.modele_nom}"

    def encoder(self, textes: list[str]) -> list[list[float]]:
        vecteurs = self.modele.encode(textes, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecteurs]


def _normaliser(vecteur):
    norme = math.sqrt(sum(float(v) * float(v) for v in vecteur)) or 1.0
    return [float(v) / norme for v in vecteur]


def cosinus(a, b) -> float:
    """Le cosinus de deux vecteurs **deja normalises** : un produit scalaire.

    Les trois encodeurs d'ici rendent des vecteurs de norme 1. Le cosinus se
    reduit donc a une multiplication et une somme, sans division.
    """
    return sum(x * y for x, y in zip(a, b))


def encodeur_demande() -> str:
    choix = os.environ.get("TP_EMBEDDINGS", "auto").lower()
    if choix in ("factice", "ollama", "sentence-transformers"):
        return choix
    return "ollama"


def obtenir_encodeur():
    choix = encodeur_demande()
    if choix == "factice":
        return EncodeurFactice()
    if choix == "sentence-transformers":
        return EncodeurSentenceTransformers()
    return EncodeurOllama()
