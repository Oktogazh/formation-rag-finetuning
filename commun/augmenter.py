"""Construire le prompt augmenté : glossaire cherché, voisins, segment.

Le glossaire se cherche **par vecteurs**, comme tout le reste du RAG. On encode
les termes une fois, on encode les mots du segment, et on garde les termes dont
un mot s'approche à plus de ``seuil``.

Pourquoi pas une comparaison de chaînes : le suédois agglutine.
``testslutpunkten``, c'est ``test`` + ``slutpunkt`` + le suffixe défini ``-en``.
Le terme est enfoui au milieu du mot, et un vecteur l'y retrouve.

**Le seuil appartient à l'encodeur, pas à la tâche.** ``bge-m3`` sépare bien
vers 0,75 ; l'encodeur factice des tests, qui hache des trigrammes, sépare vers
0,55. ``SEUILS`` associe donc un seuil à chaque encodeur, et un ``seuil=``
explicite l'emporte toujours. C'est ce qui évite qu'un appel sans argument
tombe silencieusement sur le mauvais réglage.

Vous écrivez cette recherche au TP 2. Celle d'ici sert aux TP 3, 4 et 6.
"""

from __future__ import annotations

import re

from commun.embeddings import cosinus, obtenir_encodeur
from commun.prompts import construire_messages

_MOTS = re.compile(r"[\wåäöÅÄÖ]+")

SEUIL_GLOSSAIRE = 0.75
SEUILS = {"factice:trigrammes": 0.55}


def mots_de(segment_src: str) -> list[str]:
    """Les mots du segment, en minuscules, sans ponctuation."""
    return [m.lower() for m in _MOTS.findall(segment_src)]


def seuil_pour(encodeur) -> float:
    """Le seuil étalonné pour cet encodeur, 0,75 par défaut."""
    return SEUILS.get(encodeur.nom(), SEUIL_GLOSSAIRE)


class GlossaireVectoriel:
    """Le glossaire encodé une fois, plus un cache des mots déjà vus.

    Sans cache, filtrer 80 segments redemanderait un vecteur pour chaque mot de
    chaque segment — les mêmes ``per``, ``till``, ``du`` des centaines de fois.
    Le cache ramène l'encodage au nombre de mots **distincts** du corpus.

    Les compteurs ``encodes`` et ``evites`` sont là pour que le TP 2 montre ce
    que le cache économise, au lieu de l'affirmer.
    """

    def __init__(self, glossaire: list[dict], encodeur=None,
                 seuil: float | None = None):
        self.glossaire = glossaire
        self.encodeur = encodeur or obtenir_encodeur()
        self.seuil = seuil_pour(self.encodeur) if seuil is None else seuil
        self.vecteurs = self.encodeur.encoder([t["sv"] for t in glossaire])
        self._cache: dict[str, list[float]] = {}
        self.encodes = 0
        self.evites = 0

    def nom(self) -> str:
        return self.encodeur.nom()

    def vecteurs_des_mots(self, mots: list[str]) -> list[list[float]]:
        """Les vecteurs de ces mots. Un seul appel réseau pour les inconnus."""
        inconnus = [m for m in dict.fromkeys(mots) if m not in self._cache]
        if inconnus:
            for mot, vecteur in zip(inconnus, self.encodeur.encoder(inconnus)):
                self._cache[mot] = vecteur
        self.encodes += len(inconnus)
        self.evites += len(mots) - len(inconnus)
        return [self._cache[m] for m in mots]

    def scores(self, segment_src: str) -> list[float]:
        """Pour chaque terme du glossaire, son meilleur score sur les mots."""
        vecteurs = self.vecteurs_des_mots(mots_de(segment_src))
        return [max((cosinus(v, terme) for v in vecteurs), default=0.0)
                for terme in self.vecteurs]

    def chercher(self, segment_src: str, seuil: float | None = None) -> list[dict]:
        """Les entrées dont un mot du segment s'approche à plus de ``seuil``."""
        seuil = self.seuil if seuil is None else seuil
        retenus = [{**terme, "score": score}
                   for terme, score in zip(self.glossaire, self.scores(segment_src))
                   if score >= seuil]
        retenus.sort(key=lambda t: t["score"], reverse=True)
        return retenus


def construire(segment_src: str, voisins: list[dict], glossaire: list[dict],
               consignes: str = "") -> list[dict]:
    """Le prompt augmenté, au gabarit unique de ``commun/prompts.py``."""
    return construire_messages(segment_src, voisins=voisins, glossaire=glossaire,
                               consignes=consignes)
