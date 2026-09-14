"""Construire le prompt augmenté : glossaire filtré, voisins, segment.

Le glossaire se filtre de deux façons, et le TP 2 vous fait écrire la seconde.

**Par début de mot** (``glossaire_pertinent``). Trois lignes, aucune dépendance.
C'est ce qui sert aux TP 3, 4 et 6, pour qu'ils tournent sans modèle
d'embeddings. C'est aussi ce que mesure ``commun.mesure.termes_attendus``.

**Par vecteurs** (``GlossaireVectoriel``). On encode les termes une fois, on
encode les mots du segment, on garde les termes dont un mot s'approche à plus
de ``SEUIL_GLOSSAIRE``. Mesuré sur les 80 segments d'évaluation avec ``bge-m3``
(14 septembre 2026) : 53 termes attendus sur 53 retrouvés, deux déclenchements
en plus dont un que le début de mot **ne peut pas** trouver — ``slutpunkt``
dans ``testslutpunkten``, où le terme n'est pas en tête du mot composé.

Le suédois compose : ``testslutpunkten``, ``hastighetsgränsen``. Un préfixe ne
voit que les composés dont le terme est le premier morceau. Un vecteur les voit
tous — au prix d'un encodeur, et d'un seuil à étalonner.

**Le seuil appartient à l'encodeur, pas à la tâche.** ``bge-m3`` sépare bien
vers 0,75 ; l'encodeur factice des tests, qui hache des trigrammes, sépare vers
0,55. Le même code, deux réglages.
"""

from __future__ import annotations

import re

from commun.embeddings import cosinus, obtenir_encodeur
from commun.prompts import construire_messages

_MOTS = re.compile(r"[\wåäöÅÄÖ]+")

SEUIL_GLOSSAIRE = 0.75


def mots_de(segment_src: str) -> list[str]:
    """Les mots du segment, en minuscules, sans ponctuation ni chiffres isolés."""
    return [m.lower() for m in _MOTS.findall(segment_src)]


def glossaire_pertinent(segment_src: str, glossaire: list[dict]) -> list[dict]:
    """Les entrées de glossaire qui concernent ce segment, par début de mot.

    Comparaison par **début de mot** : le suédois compose et fléchit
    (``förfrågan``, ``förfrågningar``), une égalité stricte ne trouverait rien.
    """
    mots = mots_de(segment_src)
    return [t for t in glossaire if any(mot.startswith(t["sv"].lower()) for mot in mots)]


class GlossaireVectoriel:
    """Le glossaire encodé une fois, plus un cache des mots déjà vus.

    Sans cache, filtrer 80 segments redemanderait un vecteur pour chaque mot de
    chaque segment — les mêmes ``per``, ``till``, ``du`` des centaines de fois.
    Le cache ramène l'encodage au nombre de mots **distincts** du corpus.

    Les compteurs ``encodes`` et ``evites`` sont là pour que le TP 2 montre ce
    que le cache économise, au lieu de l'affirmer.
    """

    def __init__(self, glossaire: list[dict], encodeur=None,
                 seuil: float = SEUIL_GLOSSAIRE):
        self.glossaire = glossaire
        self.encodeur = encodeur or obtenir_encodeur()
        self.seuil = seuil
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
