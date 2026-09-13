"""TP 2 — construire le prompt a partir de ce qu'on a trouve.

Le TP 1 s'est termine sur un prompt de 9 000 tokens qui ne faisait pas mieux
qu'un prompt de 600. La difference n'etait pas la quantite, c'etait la
**pertinence**. Ici on ne met dans le prompt que deux choses :

* les ``k`` segments de memoire les plus proches de **ce** segment ;
* les entrees de glossaire dont le terme apparait dans **ce** segment.

Sur le corpus Helios, le glossaire complet fait 7 entrees et le filtrage en
laisse une ou deux. Sur un vrai compte client il en fait 400, et c'est la que la
difference devient une facture.
"""

from __future__ import annotations

from commun.prompts import construire_messages


def glossaire_pertinent(segment_src: str, glossaire: list[dict]) -> list[dict]:
    """Les entrees de glossaire qui concernent ce segment, et elles seules.

    Le suedois compose et flechit : ``forfragan``, ``forfragningar`` et
    ``forfragan`` sont le meme terme. Une egalite stricte ne trouverait rien —
    on compare donc par **debut de mot**.
    """
    import re

    mots = [m.lower() for m in re.findall(r"[\wåäöÅÄÖ]+", segment_src)]
    # <<<TODO 4 ★ Filtrer le glossaire
    # Rendez la liste des entrees du glossaire dont le terme suedois (cle "sv",
    # en minuscules) est le **debut** d'au moins un des mots ci-dessus.
    # Indice : mot.startswith(terme) — et n'oubliez pas .lower() sur les deux.
    # Exemple : "Hastighetsgränsen är satt till 60 förfrågningar per minut."
    # doit rendre 2 entrees (hastighetsgräns et förfråg).
    # Test : python tp.py test tp02 -k todo4
    raise NotImplementedError(
        "TODO 4 — a completer. Consigne juste au-dessus, "
        "explications dans tp02-rag/README.md"
    )
    # >>>TODO 4


def construire(segment_src: str, voisins: list[dict], glossaire: list[dict],
               consignes: str = "") -> list[dict]:
    """Le prompt augmente : glossaire filtre, voisins retenus, segment.

    On n'ecrit pas le format ici. ``commun/prompts.py`` le fait, et c'est
    volontaire : a partir de maintenant et jusqu'au TP 6, **tous** les appels
    passent par ce gabarit. C'est la condition pour que la table finale compare
    des crans et non des mises en page — et au TP 4, c'est la condition pour que
    le fine-tuning serve a quelque chose.
    """
    # <<<TODO 5 ★ Appeler le gabarit commun
    # Rendez construire_messages(...) avec, dans l'ordre :
    # le segment source, les voisins, le glossaire deja filtre, les consignes.
    # Allez lire commun/prompts.py avant : c'est une page, et c'est le coeur
    # de la formation.
    # Test : python tp.py test tp02 -k todo5
    raise NotImplementedError(
        "TODO 5 — a completer. Consigne juste au-dessus, "
        "explications dans tp02-rag/README.md"
    )
    # >>>TODO 5
