#!/usr/bin/env python
"""Controle le corpus. Le lancer apres toute modification de data/corpus/.

    python scripts/verifier_corpus.py
    python scripts/verifier_corpus.py --lister    # les 80 references, a relire

Ce que le script garantit, et pourquoi chaque point compte :

* **schema** : les TP lisent ces cles telles quelles, une cle manquante casse
  quatre TP sur six ;
* **glossaire propre** : une traduction imposee ne doit pas figurer dans ses
  propres interdits, sinon la mesure de terminologie se contredit ;
* **chiffres coherents** : dans un segment, les nombres de la source se
  retrouvent dans la cible. C'est la base du controle automatique du TP 3 ;
* **taux_memoire exact** : la valeur stockee doit etre celle que le code
  recalcule, sinon le routeur du TP 6 route sur une donnee fausse ;
* **pas de fuite** : aucun segment d'evaluation n'est present a l'identique dans
  la memoire.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CORPUS = RACINE / "data" / "corpus" / "helios-sv"
CATEGORIES = {"repetition": 21, "piege": 20, "fuzzy": 18, "nouveau": 21}


def lire(nom: str) -> list[dict]:
    return [json.loads(l) for l in (CORPUS / nom).read_text(encoding="utf-8").splitlines() if l.strip()]


def controler() -> list[str]:
    fautes: list[str] = []
    memoire, evaluation = lire("tm.jsonl"), lire("segments-eval.jsonl")
    glossaire = lire("glossaire.jsonl")

    for segment in memoire:
        if set(segment) < {"id", "src", "tgt", "domaine", "date", "statut"}:
            fautes.append(f"tm {segment.get('id')} : cles manquantes")
    for segment in evaluation:
        if set(segment) < {"id", "src", "tgt", "domaine", "categorie", "taux_memoire"}:
            fautes.append(f"eval {segment.get('id')} : cles manquantes")

    compte: dict[str, int] = {}
    for segment in evaluation:
        compte[segment["categorie"]] = compte.get(segment["categorie"], 0) + 1
    if compte != CATEGORIES:
        fautes.append(f"categories : {compte} au lieu de {CATEGORIES}")

    latin = re.compile(r"^[A-Za-zÀ-ÿ0-9'’ \-]+$")
    for terme in glossaire:
        for interdit in terme["interdits"]:
            if not latin.match(interdit):
                fautes.append(f"glossaire {terme['sv']} : interdit douteux « {interdit} »")
            if terme["fr"].lower() in interdit.lower():
                fautes.append(f"glossaire {terme['sv']} : « {interdit} » contient la bonne réponse")

    nombres = re.compile(r"\d+")
    for segment in memoire + evaluation:
        if sorted(nombres.findall(segment["src"])) != sorted(nombres.findall(segment["tgt"])):
            fautes.append(f"{segment['id']} : les nombres divergent entre source et cible")

    sources = [s["src"] for s in memoire]
    for segment in evaluation:
        if segment["src"] in sources:
            fautes.append(f"{segment['id']} : present a l'identique dans la memoire (fuite)")
        reel = max(SequenceMatcher(None, segment["src"], s).ratio() for s in sources)
        if abs(reel - segment["taux_memoire"]) > 0.005:
            fautes.append(
                f"{segment['id']} : taux_memoire {segment['taux_memoire']} au lieu de {reel:.4f}"
            )

    identifiants = [s["id"] for s in memoire + evaluation]
    if len(identifiants) != len(set(identifiants)):
        fautes.append("identifiants dupliques")

    documents = sorted((CORPUS / "docs").glob("*.md"))
    if len(documents) != 6:
        fautes.append(f"{len(documents)} documents au lieu de 6")

    return fautes


def principal(argv=None) -> int:
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument("--lister", action="store_true",
                           help="imprimer les 80 references francaises, pour relecture humaine")
    args = analyseur.parse_args(argv)

    if args.lister:
        print("Les 80 références françaises — à relire avant la session.")
        print("Toute la mesure de la formation se compare à ces phrases.\n")
        for segment in lire("segments-eval.jsonl"):
            print(f"{segment['id']}  [{segment['categorie']:10}]  {segment['src']}")
            print(f"{'':26}{segment['tgt']}\n")
        return 0

    fautes = controler()
    if fautes:
        print(f"{len(fautes)} probleme(s) :")
        for faute in fautes:
            print(f"    {faute}")
        return 1
    memoire, evaluation = lire("tm.jsonl"), lire("segments-eval.jsonl")
    print(f"Corpus helios-sv conforme : {len(memoire)} segments de memoire, "
          f"{len(evaluation)} d'evaluation, {len(lire('glossaire.jsonl'))} termes, 6 documents.")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
