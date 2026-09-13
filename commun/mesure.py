"""Mesurer une traduction. C'est la piece la plus importante du depot.

Cinq indicateurs, et l'ordre dans lequel ils sont affiches est le meme partout :

``BLEU``
    ``sacrebleu``, au niveau du **corpus**. C'est la metrique de reference de la
    traduction automatique depuis 2002 et c'est celle que l'industrie utilise.
    Elle compare les n-grammes de mots produits a ceux de la reference.
``chrF``
    la meme idee, mais sur des n-grammes de **caracteres**. On la garde a cote
    de BLEU parce que nos segments sont courts : BLEU y est brutale (un mot de
    difference coute tres cher), chrF bouge de facon plus lisible.
``Termino``
    part des termes du glossaire correctement rendus. **C'est l'indicateur qui
    fait vivre un client.** Une traduction a BLEU 80 qui ecrit « abonnement »
    au lieu de « formule » est refusee en relecture.
``Vous``
    le vouvoiement, impose par le guide de style.
``Chiffres``
    les nombres de la source se retrouvent-ils dans la traduction ? C'est le
    controle le moins couteux du monde et celui qui rattrape les fautes les plus
    graves.

Une mise en garde qu'il faut dire aux stagiaires : **BLEU n'est pas la qualite**.
C'est une correlation, mesuree sur des corpus entiers. Sur une phrase, elle est
bruitee ; deux traductions egalement justes peuvent differer de 30 points. On
l'utilise pour **comparer deux systemes sur les memes 80 segments**, jamais pour
juger une phrase.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

try:
    import sacrebleu
except ImportError as erreur:  # pragma: no cover
    raise ImportError(
        "sacrebleu est introuvable. Activez l'environnement de la formation :\n"
        "    conda activate formation-rag"
    ) from erreur

# Tarifs releves le 13 septembre 2026 sur la page publique de Mistral AI.
# A reverifier la semaine de la session : ces chiffres perissent vite.
TARIF = {
    "date_releve": "2026-09-13",
    "modele": "ministral-3b",
    "eur_par_million_tokens_entree": 0.04,
    "eur_par_million_tokens_sortie": 0.04,
}

_MOTS = re.compile(r"[\wåäöÅÄÖ]+", re.UNICODE)
_TUTOIEMENT = re.compile(r"\b(tu|toi|ton|ta|tes)\b", re.IGNORECASE)
_VOUVOIEMENT = re.compile(r"\b(vous|votre|vos)\b", re.IGNORECASE)
_NOMBRES = re.compile(r"\d+")
_BALISES = re.compile(r"\{\d+\}|<[^>]{1,40}>|%[sd]")


@dataclass
class Sortie:
    """Ce qu'un systeme de traduction rend, quel que soit son cran.

    ``appels`` compte les appels au modele : c'est la colonne qui dit combien
    coute un cran. Le TP 6 en fait la demonstration — la moitie des segments
    n'en demande aucun.
    """

    texte: str
    tokens_entree: int = 0
    tokens_sortie: int = 0
    secondes: float = 0.0
    appels: int = 1
    chemin: tuple = ()
    anomalies: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Les indicateurs, un par un
# --------------------------------------------------------------------------
def bleu_corpus(hypotheses, references) -> float:
    """BLEU au niveau du corpus. **C'est le chiffre qui compare deux crans.**"""
    if not hypotheses:
        return 0.0
    return sacrebleu.corpus_bleu(list(hypotheses), [list(references)]).score


def bleu_segment(hypothese: str, reference: str) -> float:
    """BLEU sur une phrase, lissee. Indicatif : a lire, pas a conclure."""
    return sacrebleu.sentence_bleu(hypothese, [reference], smooth_method="exp").score


def chrf_corpus(hypotheses, references) -> float:
    if not hypotheses:
        return 0.0
    return sacrebleu.corpus_chrf(list(hypotheses), [list(references)]).score


def chrf_segment(hypothese: str, reference: str) -> float:
    return sacrebleu.sentence_chrf(hypothese, [reference]).score


def termes_attendus(source: str, glossaire) -> list[dict]:
    """Entrees de glossaire dont le terme suedois apparait dans la source.

    Comparaison par **prefixe** : le suedois compose et flechit (``forfragan``,
    ``forfragningar``), une egalite stricte ne trouverait rien.
    """
    mots = [m.lower() for m in _MOTS.findall(source)]
    retenus = []
    for terme in glossaire:
        cle = terme["sv"].lower()
        if any(mot.startswith(cle) for mot in mots):
            retenus.append(terme)
    return retenus


def terminologie(source: str, hypothese: str, glossaire) -> tuple[int, int, list[str]]:
    """(attendus, respectes, fautes). Une faute nomme le terme fautif."""
    attendus = termes_attendus(source, glossaire)
    minuscule = hypothese.lower()
    ok, fautes = 0, []
    for terme in attendus:
        if terme["fr"].lower() in minuscule:
            ok += 1
        else:
            interdit = next(
                (i for i in terme.get("interdits", []) if i.lower() in minuscule), None
            )
            if interdit:
                fautes.append(f"« {interdit} » au lieu de « {terme['fr']} »")
            else:
                fautes.append(f"« {terme['fr']} » absent")
    return len(attendus), ok, fautes


def vouvoiement(hypothese: str) -> bool | None:
    """True si vouvoie, False si tutoie, None si la phrase n'adresse personne."""
    if _TUTOIEMENT.search(hypothese):
        return False
    if _VOUVOIEMENT.search(hypothese):
        return True
    return None


def chiffres(source: str, hypothese: str) -> bool:
    return sorted(_NOMBRES.findall(source)) == sorted(_NOMBRES.findall(hypothese))


def balises(source: str, hypothese: str) -> bool:
    return sorted(_BALISES.findall(source)) == sorted(_BALISES.findall(hypothese))


def cout_eur(tokens_entree: int, tokens_sortie: int) -> float:
    return (
        tokens_entree * TARIF["eur_par_million_tokens_entree"]
        + tokens_sortie * TARIF["eur_par_million_tokens_sortie"]
    ) / 1_000_000


# --------------------------------------------------------------------------
# Evaluer un systeme entier
# --------------------------------------------------------------------------
def evaluer(segments, traduire, *, glossaire=None, silencieux: bool = False) -> list[dict]:
    """Fait traduire ``segments`` par ``traduire`` et mesure tout.

    ``traduire`` est une fonction ``segment (dict) -> Sortie``. N'importe quel
    cran de la formation s'y plie : un appel de modele nu, une chaine LangChain,
    un graphe LangGraph, ou meme une reutilisation de la memoire sans modele.
    C'est ce qui rend les six TP comparables entre eux.
    """
    from commun.corpus import charger_glossaire

    glossaire = charger_glossaire() if glossaire is None else glossaire
    resultats = []
    debut = time.perf_counter()
    for rang, segment in enumerate(segments, 1):
        sortie = traduire(segment)
        if isinstance(sortie, str):
            sortie = Sortie(texte=sortie, appels=1)
        hypothese = sortie.texte.strip()
        attendus, ok, fautes = terminologie(segment["src"], hypothese, glossaire)
        resultats.append(
            {
                "id": segment["id"],
                "categorie": segment["categorie"],
                "src": segment["src"],
                "reference": segment["tgt"],
                "hypothese": hypothese,
                "bleu": bleu_segment(hypothese, segment["tgt"]),
                "chrf": chrf_segment(hypothese, segment["tgt"]),
                "termes_attendus": attendus,
                "termes_ok": ok,
                "termes_fautes": fautes,
                "vouvoiement": vouvoiement(hypothese),
                "chiffres": chiffres(segment["src"], hypothese),
                "balises": balises(segment["src"], hypothese),
                "tokens_entree": sortie.tokens_entree,
                "tokens_sortie": sortie.tokens_sortie,
                "secondes": sortie.secondes,
                "appels": sortie.appels,
                "chemin": list(sortie.chemin),
                "anomalies": list(sortie.anomalies),
            }
        )
        if not silencieux:
            ecoule = time.perf_counter() - debut
            reste = ecoule / rang * (len(segments) - rang)
            print(
                f"  {rang:3}/{len(segments)}  {segment['id']}  {segment['categorie']:10}"
                f"  reste ~{reste:4.0f} s",
                flush=True,
            )
    return resultats
