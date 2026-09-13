"""TP 3 — traduire, verifier, corriger, et compter ce que ca coute.

C'est le cran 3 de l'escalade. Le TP 2 produisait une traduction et la livrait.
Ici on la **controle**, et si elle echoue on redemande au modele de la reparer,
en lui disant precisement ce qui ne va pas.

Le gain est reel : les fautes de terminologie et de chiffres tombent. Le prix
l'est aussi : deux appels au lieu d'un sur les segments qui echouent, donc un
cout et une latence qui montent. La table de mesure affiche la colonne
``Appels`` pour cette raison, et c'est elle qu'il faudra regarder au TP 6.
"""

from __future__ import annotations

from commun.mesure import Sortie
from commun.prompts import nettoyer_sortie
from tp03.verification import verifier


def traduire_et_corriger(segment_src: str, chaine, chaine_correction,
                         glossaire: list[dict], max_tentatives: int = 2) -> Sortie:
    """Traduit, verifie, corrige tant qu'il reste des defauts et des tentatives.

    Rend une ``Sortie`` dont ``appels`` compte **tous** les appels au modele et
    ``anomalies`` liste ce qui restait au bout.
    """
    traduction = nettoyer_sortie(chaine.invoke(segment_src))
    appels = 1
    # <<<TODO 3 ★★ La boucle de correction
    #! Tant qu'il reste des anomalies ET qu'on n'a pas atteint max_tentatives :
    #!   1. anomalies = verifier(segment_src, traduction, glossaire) ;
    #!   2. si la liste est vide, on sort ;
    #!   3. sinon on appelle chaine_correction.invoke({
    #!          "src": segment_src,
    #!          "traduction": traduction,
    #!          "anomalies": "\\n".join(f"- {a}" for a in anomalies),
    #!      }), on nettoie la reponse avec nettoyer_sortie(), on incremente
    #!      appels, et on recommence.
    #! Attention : appels compte le premier appel deja fait ci-dessus.
    #! Test : python tp.py test tp03 -k todo3
    anomalies = verifier(segment_src, traduction, glossaire)
    while anomalies and appels < max_tentatives:
        traduction = nettoyer_sortie(
            chaine_correction.invoke(
                {
                    "src": segment_src,
                    "traduction": traduction,
                    "anomalies": "\n".join(f"- {a}" for a in anomalies),
                }
            )
        )
        appels += 1
        anomalies = verifier(segment_src, traduction, glossaire)
    # >>>TODO 3
    return Sortie(
        texte=traduction,
        appels=appels,
        chemin=("chaine",) + (("correction",) if appels > 1 else ()),
        anomalies=[str(a) for a in anomalies],
    )
