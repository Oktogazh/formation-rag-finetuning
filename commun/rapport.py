"""Afficher et enregistrer une mesure.

Une seule table dans toute la formation, avec toujours les memes colonnes et
dans le meme ordre :

    BLEU · chrF · Termino · Vous · Chiffres · Tokens · s/seg · Appels

BLEU en premier parce que c'est la metrique du metier. Les deux scores sont
calcules **au niveau du corpus** pour chaque ligne : c'est la seule facon
correcte de lire BLEU.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from commun import DOSSIER_RESULTATS
from commun.mesure import bleu_corpus, chrf_corpus

CATEGORIES = ["repetition", "piege", "fuzzy", "nouveau"]


def agreger(resultats: list[dict], groupe: str | None = None) -> dict:
    if not resultats:
        return {}
    hypotheses = [r["hypothese"] for r in resultats]
    references = [r["reference"] for r in resultats]
    attendus = sum(r["termes_attendus"] for r in resultats)
    ok = sum(r["termes_ok"] for r in resultats)
    juges = [r for r in resultats if r["vouvoiement"] is not None]
    return {
        "groupe": groupe or "total",
        "n": len(resultats),
        "bleu": bleu_corpus(hypotheses, references),
        "chrf": chrf_corpus(hypotheses, references),
        "termes_ok": ok,
        "termes_attendus": attendus,
        "termino": (100.0 * ok / attendus) if attendus else None,
        "vous": (100.0 * sum(1 for r in juges if r["vouvoiement"]) / len(juges)) if juges else None,
        "chiffres": 100.0 * sum(1 for r in resultats if r["chiffres"]) / len(resultats),
        "tokens_entree": sum(r["tokens_entree"] for r in resultats) / len(resultats),
        "secondes": sum(r["secondes"] for r in resultats) / len(resultats),
        "appels": sum(r["appels"] for r in resultats) / len(resultats),
    }


def _pourcent(valeur) -> str:
    return "   -  " if valeur is None else f"{valeur:5.0f}%"


def _tokens(valeur) -> str:
    """Un zero franc veut dire « non compté », pas « gratuit ». On le dit."""
    return "      -" if not valeur else f"{valeur:7.0f}"


def table(resultats: list[dict], titre: str = "") -> str:
    lignes = []
    if titre:
        lignes += [titre, "=" * len(titre)]
    lignes.append(
        f"{'':12} {'n':>3}  {'BLEU':>6} {'chrF':>6}  {'Termino':>7} {'Vous':>6} "
        f"{'Chiffres':>8}  {'Tokens':>7} {'s/seg':>6} {'Appels':>6}"
    )
    lignes.append("-" * 84)
    groupes = [(c, [r for r in resultats if r["categorie"] == c]) for c in CATEGORIES]
    for nom, lot in groupes:
        if not lot:
            continue
        a = agreger(lot, nom)
        lignes.append(
            f"{nom:12} {a['n']:3}  {a['bleu']:6.1f} {a['chrf']:6.1f}  "
            f"{_pourcent(a['termino']):>7} {_pourcent(a['vous']):>6} {_pourcent(a['chiffres']):>8}  "
            f"{_tokens(a['tokens_entree'])} {a['secondes']:6.2f} {a['appels']:6.2f}"
        )
    lignes.append("-" * 84)
    t = agreger(resultats)
    lignes.append(
        f"{'TOTAL':12} {t['n']:3}  {t['bleu']:6.1f} {t['chrf']:6.1f}  "
        f"{_pourcent(t['termino']):>7} {_pourcent(t['vous']):>6} {_pourcent(t['chiffres']):>8}  "
        f"{_tokens(t['tokens_entree'])} {t['secondes']:6.2f} {t['appels']:6.2f}"
    )
    return "\n".join(lignes)


def fautes_frequentes(resultats: list[dict], limite: int = 5) -> str:
    compte: dict[str, int] = {}
    for r in resultats:
        for faute in r["termes_fautes"]:
            compte[faute] = compte.get(faute, 0) + 1
    if not compte:
        return ""
    lignes = ["Fautes de terminologie les plus frequentes :"]
    for faute, n in sorted(compte.items(), key=lambda kv: -kv[1])[:limite]:
        lignes.append(f"   {n:3} x  {faute}")
    return "\n".join(lignes)


def enregistrer(cran: str, resultats: list[dict], meta: dict | None = None) -> str:
    """Ecrit la mesure dans ``resultats/`` pour que ``tp.py banc`` la retrouve."""
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    horodatage = time.strftime("%Y%m%d-%H%M%S")
    chemin = DOSSIER_RESULTATS / f"{cran}-{horodatage}.jsonl"
    entete = {
        "_meta": True,
        "cran": cran,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(resultats),
        "synthese": agreger(resultats),
        **(meta or {}),
    }
    with chemin.open("w", encoding="utf-8") as f:
        f.write(json.dumps(entete, ensure_ascii=False) + "\n")
        for r in resultats:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(chemin.relative_to(DOSSIER_RESULTATS.parent))


def lire_mesures() -> list[dict]:
    """Toutes les mesures enregistrees, la plus recente par cran."""
    if not DOSSIER_RESULTATS.exists():
        return []
    par_cran: dict[str, dict] = {}
    for chemin in sorted(DOSSIER_RESULTATS.glob("*.jsonl")):
        try:
            with chemin.open(encoding="utf-8") as f:
                entete = json.loads(f.readline())
        except (ValueError, OSError):
            continue
        if entete.get("_meta"):
            par_cran[entete["cran"]] = entete
    return list(par_cran.values())
