"""Retrouver, dans 444 segments, ceux qui servent. Référence partagée.

Deux façons, et la formation vous fait mesurer les deux.

**Lexicale.** On compare les chaînes de caractères. C'est, à peu de chose près,
ce que fait l'outil de TAO d'une agence de traduction, et le pourcentage qu'il
affiche (« 87 % ») est ce ratio-là.

**Dense.** On transforme chaque segment en vecteur avec un modèle d'embeddings,
et on compare les vecteurs. Cela retrouve des segments qui disent la même chose
avec d'autres mots.

Vous réécrivez ces deux fonctions au TP 2. Celles d'ici servent aux TP 3, 4 et 6,
pour qu'ils tournent même si vous n'avez pas fait le TP 2.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

from commun import DOSSIER_RESULTATS


def similarite(a: str, b: str) -> float:
    """Ratio de similarité de chaînes, entre 0 et 1."""
    return SequenceMatcher(None, a, b).ratio()


def rechercher_lexical(segment_src: str, memoire: list[dict], k: int = 3) -> list[dict]:
    """Les ``k`` segments les plus proches, au sens des caractères."""
    scores = [{**s, "score": similarite(segment_src, s["src"])} for s in memoire]
    scores.sort(key=lambda s: s["score"], reverse=True)
    return scores[:k]


def meilleur_voisin(segment_src: str, memoire: list[dict]) -> dict:
    """Le segment le plus proche, et lui seul."""
    meilleur = max(memoire, key=lambda s: similarite(segment_src, s["src"]))
    return {**meilleur, "score": similarite(segment_src, meilleur["src"])}


class Index:
    """La mémoire encodée en vecteurs, une fois pour toutes, puis mise en cache."""

    def __init__(self, segments: list[dict], vecteurs: list[list[float]], encodeur):
        self.segments = segments
        self.vecteurs = vecteurs
        self.encodeur = encodeur

    @classmethod
    def construire(cls, memoire: list[dict], encodeur=None, cache: bool = True) -> "Index":
        from commun.embeddings import obtenir_encodeur

        encodeur = encodeur or obtenir_encodeur()
        chemin = Path(DOSSIER_RESULTATS) / "index" / f"{encodeur.nom().replace(':', '-')}.json"
        sources = [s["src"] for s in memoire]
        if cache and chemin.exists():
            donnees = json.loads(chemin.read_text(encoding="utf-8"))
            if donnees.get("sources") == sources:
                return cls(memoire, donnees["vecteurs"], encodeur)
        print(f"  Encodage de {len(sources)} segments avec {encodeur.nom()}…", flush=True)
        vecteurs = encodeur.encoder(sources)
        if cache:
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.write_text(json.dumps({"sources": sources, "vecteurs": vecteurs}),
                              encoding="utf-8")
        return cls(memoire, vecteurs, encodeur)

    def chercher(self, segment_src: str, k: int = 3) -> list[dict]:
        """Les ``k`` segments dont le vecteur est le plus proche de la requête.

        Les vecteurs sont normalisés, donc le cosinus se réduit au produit
        scalaire : une multiplication et une somme.
        """
        requete = self.encodeur.encoder([segment_src])[0]
        scores = [
            {**segment, "score": sum(a * b for a, b in zip(requete, vecteur))}
            for segment, vecteur in zip(self.segments, self.vecteurs)
        ]
        scores.sort(key=lambda s: s["score"], reverse=True)
        return scores[:k]

    def chercher_hybride(self, segment_src: str, k: int = 3, alpha: float = 0.5) -> list[dict]:
        """Mélange des deux scores. ``alpha`` est le poids du dense."""
        denses = {s["id"]: s["score"] for s in self.chercher(segment_src, k=len(self.segments))}
        melange = [
            {**segment,
             "score": alpha * denses[segment["id"]]
             + (1 - alpha) * similarite(segment_src, segment["src"])}
            for segment in self.segments
        ]
        melange.sort(key=lambda s: s["score"], reverse=True)
        return melange[:k]


def chercheur(memoire: list[dict], methode: str = "lexicale", k: int = 3):
    """Rend une fonction ``segment_src -> voisins``."""
    if methode == "lexicale":
        return lambda src: rechercher_lexical(src, memoire, k)
    index = Index.construire(memoire)
    if methode == "hybride":
        return lambda src: index.chercher_hybride(src, k)
    return lambda src: index.chercher(src, k)


def rappel(segments, chercher, seuil: float = 0.90) -> float:
    """Part des segments pour lesquels un voisin dépasse ``seuil`` de similarité.

    Mesure la recherche **seule**, sans le modèle. C'est le plafond de tout ce
    qui suit : ce que la recherche ne remonte pas, la génération ne l'inventera
    pas.
    """
    trouves = 0
    for segment in segments:
        voisins = chercher(segment["src"])
        if voisins and max(similarite(segment["src"], v["src"]) for v in voisins) >= seuil:
            trouves += 1
    return 100.0 * trouves / max(len(segments), 1)
