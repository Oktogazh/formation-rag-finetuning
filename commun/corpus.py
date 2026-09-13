"""Acces au corpus ``helios-sv``.

Un editeur suedois fictif, Helios, publie sa documentation technique. L'agence
la traduit vers le francais. Elle dispose de quatre choses, et ce sont les
quatre fichiers du corpus :

==========================  ===============================================
``docs/*.md``               les consignes : guide de style, glossaire impose,
                            consignes client, procedure de relecture, tarifs,
                            confidentialite
``glossaire.jsonl``         7 termes dont la traduction est imposee, chacun
                            avec ses traductions **interdites**
``tm.jsonl``                444 segments deja traduits et valides : la
                            memoire de traduction
``segments-eval.jsonl``     80 segments a traduire, avec leur reference
==========================  ===============================================

Les 80 segments d'evaluation portent une ``categorie``, et c'est elle qui rend
la formation lisible :

``repetition`` (21)
    un voisin quasi identique existe dans la memoire, seuls les chiffres
    different, et le chiffre de la source est plausible.
``piege`` (20)
    meme situation, mais le chiffre de la source est **surprenant** (« tous les
    365 mois »). Un modele a qui on montre le voisin recopie son chiffre. C'est
    la faute que le controle des chiffres attrape.
``fuzzy`` (18)
    un voisin partiel existe (60 a 90 % de similarite), il aide sans suffire.
``nouveau`` (21)
    aucun voisin utile. Ni la memoire ni le RAG n'y peuvent grand-chose.
"""

from __future__ import annotations

import json
from pathlib import Path

from commun import RACINE

CORPUS = RACINE / "data" / "corpus" / "helios-sv"


def lire_jsonl(chemin: str | Path) -> list[dict]:
    """Lit un fichier JSON Lines et rend une liste de dictionnaires."""
    chemin = Path(chemin)
    if not chemin.is_absolute():
        chemin = CORPUS / chemin
    if not chemin.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {chemin}\n"
            "Lancez la commande depuis la racine du depot cloné."
        )
    with chemin.open(encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]


def manifeste() -> dict:
    with (CORPUS / "manifest.json").open(encoding="utf-8") as f:
        return json.load(f)


def charger_glossaire() -> list[dict]:
    """Les 7 termes imposes. Cles : ``sv``, ``fr``, ``interdits``."""
    return lire_jsonl("glossaire.jsonl")


def charger_memoire_brute() -> list[dict]:
    """Les 444 segments de la memoire, sans filtre.

    Le TP 2 vous fera ecrire le chargement filtre : c'est la premiere chose
    qu'on fait avec une memoire de traduction, et ca merite d'etre tape une
    fois.
    """
    return lire_jsonl("tm.jsonl")


def charger_evaluation(categorie: str | None = None, n: int | None = None) -> list[dict]:
    """Les 80 segments a traduire, filtrables par categorie.

    ``n`` prend les ``n`` premiers **en gardant les quatre categories
    representees** : on ne veut pas mesurer 20 segments qui seraient tous des
    ``repetition``, le resultat serait flatteur et faux.
    """
    segments = lire_jsonl("segments-eval.jsonl")
    if categorie:
        segments = [s for s in segments if s["categorie"] == categorie]
    if n is not None and n < len(segments):
        segments = _echantillon_equilibre(segments, n)
    return segments


def _echantillon_equilibre(segments: list[dict], n: int) -> list[dict]:
    par_categorie: dict[str, list[dict]] = {}
    for s in segments:
        par_categorie.setdefault(s["categorie"], []).append(s)
    choisis: list[dict] = []
    i = 0
    while len(choisis) < n:
        avance = False
        for categorie in sorted(par_categorie):
            lot = par_categorie[categorie]
            if i < len(lot) and len(choisis) < n:
                choisis.append(lot[i])
                avance = True
        if not avance:
            break
        i += 1
    ordre = {s["id"]: k for k, s in enumerate(segments)}
    return sorted(choisis, key=lambda s: ordre[s["id"]])


def charger_docs() -> list[dict]:
    """Les six documents de consignes, en-tete YAML separe du corps.

    Rend une liste de ``{"id", "titre", "categorie", "date", "corps"}``.
    """
    docs = []
    for chemin in sorted((CORPUS / "docs").glob("*.md")):
        texte = chemin.read_text(encoding="utf-8")
        entete: dict[str, str] = {}
        corps = texte
        if texte.startswith("---"):
            _, brut, corps = texte.split("---", 2)
            for ligne in brut.strip().splitlines():
                if ":" in ligne:
                    cle, valeur = ligne.split(":", 1)
                    entete[cle.strip()] = valeur.strip()
        docs.append({**entete, "fichier": chemin.name, "corps": corps.strip()})
    return docs


def texte_des_consignes() -> str:
    """Les six documents concatenes. C'est le « tout dans le prompt » du TP 1."""
    return "\n\n".join(f"# {d.get('titre', d['fichier'])}\n{d['corps']}" for d in charger_docs())
