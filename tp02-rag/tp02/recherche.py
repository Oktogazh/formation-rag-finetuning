"""TP 2 — retrouver, dans 444 segments, les trois qui servent.

Deux facons, et le TP vous fait mesurer les deux :

**Lexicale.** On compare les chaines de caracteres. C'est exactement ce que fait
l'outil de TAO d'une agence de traduction, et le pourcentage qu'il affiche
(« 87 % ») est ce ratio-la. C'est rapide, ca ne demande aucun modele, et ca
marche tres bien quand le nouveau segment ressemble a un ancien.

**Dense.** On transforme chaque segment en vecteur avec un modele d'embeddings,
et on compare les vecteurs. Ca retrouve des segments qui **disent la meme chose
avec d'autres mots** — ce que la recherche lexicale rate par construction.

Laquelle gagne ? Sur ce corpus, la reponse n'est pas celle qu'on vous vend
d'habitude. Mesurez, puis concluez.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

from commun import DOSSIER_RESULTATS


def similarite(a: str, b: str) -> float:
    """Le ratio de similarite de chaines, entre 0 et 1. Fourni."""
    return SequenceMatcher(None, a, b).ratio()


def rechercher_lexical(segment_src: str, memoire: list[dict], k: int = 3) -> list[dict]:
    """Les ``k`` segments de la memoire les plus proches, au sens des caracteres.

    Chaque resultat est le segment de la memoire, enrichi d'une cle ``score``.
    """
    # <<<TODO 2 ★★ La recherche floue d'un outil de TAO
    # Pour chaque segment de la memoire, calculez similarite(segment_src, s["src"]),
    # rangez du plus proche au plus lointain, et rendez les k premiers — chacun
    # sous la forme {**s, "score": <la similarite>}.
    # Analogue : meilleur_voisin() juste en dessous fait exactement ca pour k=1.
    # Test : python tp.py test tp02 -k todo2
    raise NotImplementedError(
        "TODO 2 — a completer. Consigne juste au-dessus, "
        "explications dans tp02-rag/README.md"
    )
    # >>>TODO 2


def meilleur_voisin(segment_src: str, memoire: list[dict]) -> dict:
    """Le segment le plus proche, et lui seul. Fourni — c'est votre modele."""
    meilleur = max(memoire, key=lambda s: similarite(segment_src, s["src"]))
    return {**meilleur, "score": similarite(segment_src, meilleur["src"])}


class Index:
    """Les segments de la memoire, transformes en vecteurs une fois pour toutes.

    Encoder 444 segments prend quelques secondes ; on ne le refait pas a chaque
    segment traduit. L'index est mis en cache sur le disque : le deuxieme appel
    est instantane.
    """

    def __init__(self, segments: list[dict], vecteurs: list[list[float]], encodeur):
        self.segments = segments
        self.vecteurs = vecteurs
        self.encodeur = encodeur

    @classmethod
    def construire(cls, memoire: list[dict], encodeur=None, cache: bool = True) -> "Index":
        """Encode toute la memoire. Fourni."""
        from commun.embeddings import obtenir_encodeur

        encodeur = encodeur or obtenir_encodeur()
        chemin = Path(DOSSIER_RESULTATS) / "tp02" / f"index-{encodeur.nom().replace(':', '-')}.json"
        sources = [s["src"] for s in memoire]
        if cache and chemin.exists():
            donnees = json.loads(chemin.read_text(encoding="utf-8"))
            if donnees.get("sources") == sources:
                return cls(memoire, donnees["vecteurs"], encodeur)
        print(f"  Encodage de {len(sources)} segments avec {encodeur.nom()}…", flush=True)
        vecteurs = encodeur.encoder(sources)
        if cache:
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.write_text(
                json.dumps({"sources": sources, "vecteurs": vecteurs}), encoding="utf-8"
            )
        return cls(memoire, vecteurs, encodeur)

    def chercher(self, segment_src: str, k: int = 3) -> list[dict]:
        """Les ``k`` segments dont le vecteur est le plus proche de la requete.

        Les vecteurs rendus par ``commun.embeddings`` sont **normalises** : le
        cosinus se reduit donc au produit scalaire. C'est une multiplication et
        une somme, rien de plus.
        """
        # <<<TODO 3 ★★ La recherche dense
        # 1. Encodez la requete : self.encodeur.encoder([segment_src])[0].
        # 2. Pour chaque segment i, le score est le produit scalaire entre ce
        # vecteur et self.vecteurs[i] — sum(a * b for a, b in zip(...)).
        # 3. Rangez du meilleur au moins bon, rendez les k premiers, chacun
        # sous la forme {**segment, "score": <le cosinus>}.
        # Analogue : rechercher_lexical() ci-dessus, meme forme de sortie.
        # Test : python tp.py test tp02 -k todo3
        raise NotImplementedError(
            "TODO 3 — a completer. Consigne juste au-dessus, "
            "explications dans tp02-rag/README.md"
        )
        # >>>TODO 3

    def chercher_hybride(self, segment_src: str, k: int = 3, alpha: float = 0.5) -> list[dict]:
        """BONUS — melanger les deux scores.

        En production, on combine presque toujours les deux : le lexical attrape
        les references exactes (numeros de version, noms de formule), le dense
        attrape les reformulations. ``alpha`` est le poids du dense.
        """
        # <<<BONUS 2 ★★ Recherche hybride
        # Calculez pour chaque segment un score melange :
        # alpha * <cosinus dense> + (1 - alpha) * <similarite lexicale>
        # puis rendez les k meilleurs, comme les deux fonctions ci-dessus.
        # Indice : self.chercher(segment_src, k=len(self.segments)) vous donne
        # tous les scores denses d'un coup.
        # Test : python tp.py test tp02 --bonus -k bonus2
        raise NotImplementedError(
            "BONUS 2 — a completer. Consigne juste au-dessus, "
            "explications dans tp02-rag/README.md"
        )
        # >>>BONUS 2


def chercheur(memoire: list[dict], methode: str = "lexicale", k: int = 3):
    """Rend une fonction ``segment_src -> voisins``. Fourni."""
    if methode == "lexicale":
        return lambda src: rechercher_lexical(src, memoire, k)
    index = Index.construire(memoire)
    if methode == "hybride":
        return lambda src: index.chercher_hybride(src, k)
    return lambda src: index.chercher(src, k)
