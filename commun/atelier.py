"""Les raccourcis des notebooks : mesurer, comparer, afficher.

Ce module existe pour que les cellules restent courtes. Une cellule de trois
lignes se lit ; une cellule de trente lignes se saute. Tout ce qui est ici est
fourni et commenté — ouvrez-le si vous voulez voir comment c'est fait.
"""

from __future__ import annotations

import time

from commun import corpus, mesure, rapport
from commun.mesure import Sortie
from commun.prompts import nettoyer_sortie


def traducteur(moteur, construire, temperature: float = 0.0):
    """Fabrique la fonction ``segment -> Sortie`` attendue par ``mesurer``.

    ``construire`` reçoit le segment suédois et rend la liste de messages.
    C'est le seul point de variation entre deux crans de la formation : le
    reste du dispositif est identique, et c'est ce qui rend la comparaison
    honnête.
    """

    def traduire(segment: dict) -> Sortie:
        reponse = moteur.generer(construire(segment["src"]), temperature=temperature)
        return Sortie(
            texte=nettoyer_sortie(reponse.texte),
            tokens_entree=reponse.tokens_entree,
            tokens_sortie=reponse.tokens_sortie,
            secondes=reponse.secondes,
            appels=1,
        )

    return traduire


def mesurer(nom: str, segments, traduire, *, titre: str = "", silencieux: bool = False,
            enregistrer: bool = True) -> list[dict]:
    """Traduit les segments, affiche la table, enregistre la mesure.

    ``nom`` sert de clé : ``python tp.py banc`` retrouvera cette ligne. Deux
    mesures portant le même nom, la plus récente gagne.
    """
    debut = time.perf_counter()
    resultats = mesure.evaluer(segments, traduire, silencieux=silencieux)
    print()
    print(rapport.table(resultats, titre or nom))
    fautes = rapport.fautes_frequentes(resultats)
    if fautes:
        print("\n" + fautes)
    print(f"\n  {len(segments)} segments en {time.perf_counter() - debut:.0f} s")
    if enregistrer:
        print("  Enregistré :", rapport.enregistrer(nom, resultats))
    return resultats


def comparer(*mesures) -> None:
    """Aligne plusieurs mesures sur une seule table. ``(nom, resultats)`` chacune."""
    print(f"{'':28} {'BLEU':>7} {'chrF':>7} {'Termino':>8} {'Chiffres':>9} "
          f"{'Tokens':>7} {'s/seg':>6} {'Appels':>7}")
    print("-" * 82)
    for nom, resultats in mesures:
        a = rapport.agreger(resultats)
        termino = f"{a['termino']:7.0f}%" if a.get("termino") is not None else "      -"
        print(f"{nom:28} {a['bleu']:7.1f} {a['chrf']:7.1f} {termino:>8} "
              f"{a['chiffres']:8.0f}% {a['tokens_entree']:7.0f} {a['secondes']:6.2f} "
              f"{a['appels']:7.2f}")


def par_categorie(*mesures) -> None:
    """La même comparaison, mais ligne par catégorie. C'est là que ça se joue."""
    noms = [nom for nom, _ in mesures]
    print(f"{'catégorie':12} " + " ".join(f"{n[:14]:>14}" for n in noms))
    print("-" * (13 + 15 * len(noms)))
    for categorie in rapport.CATEGORIES:
        cellules = []
        for _, resultats in mesures:
            lot = [r for r in resultats if r["categorie"] == categorie]
            cellules.append(f"{rapport.agreger(lot)['bleu']:14.1f}" if lot else f"{'-':>14}")
        print(f"{categorie:12} " + " ".join(cellules))


def segments(n: int | None = None, categorie: str | None = None) -> list[dict]:
    """Les segments d'évaluation, équilibrés entre les quatre catégories."""
    return corpus.charger_evaluation(categorie=categorie, n=n)


def montrer_prompt(messages: list[dict], largeur: int = 100) -> None:
    """Affiche un prompt tel que le modèle le reçoit. À lire au moins une fois."""
    for message in messages:
        print(f"┌─ {message['role'].upper()} " + "─" * (largeur - len(message["role"]) - 4))
        for ligne in message["content"].splitlines():
            print("│ " + ligne[:largeur - 2])
        print("└" + "─" * largeur)
