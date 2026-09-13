"""TP 3 — controler la traduction avant de la livrer.

La procedure de relecture de l'agence (``docs/004-procedure-relecture.md``)
impose quatre controles automatiques avant qu'un humain regarde : terminologie,
chiffres, marqueurs, tutoiement. Ils sont deja ecrits dans ``commun/mesure.py``
— ils servent a mesurer la formation. Ici, ils changent de role : ils ne notent
plus, **ils declenchent une correction**.

C'est le passage d'un indicateur a un garde-fou, et c'est ce qui distingue une
demonstration d'un systeme de production.
"""

from __future__ import annotations

from dataclasses import dataclass

from commun.mesure import balises, chiffres, terminologie, vouvoiement


@dataclass
class Anomalie:
    type: str
    detail: str

    def __str__(self) -> str:
        return f"{self.type} : {self.detail}"


def verifier(segment_src: str, traduction: str, glossaire: list[dict]) -> list[Anomalie]:
    """Les defauts de cette traduction. Liste vide = elle passe."""
    anomalies: list[Anomalie] = []
    # <<<TODO 2 ★ Les quatre controles
    #! Ajoutez une Anomalie a la liste pour chacun des cas suivants :
    #!   - terminologie(segment_src, traduction, glossaire) rend des fautes
    #!     (3e valeur) : une Anomalie("terminologie", <la faute>) par faute ;
    #!   - chiffres(segment_src, traduction) est faux : Anomalie("chiffres", ...) ;
    #!   - vouvoiement(traduction) est False : Anomalie("tutoiement", ...) ;
    #!   - balises(segment_src, traduction) est faux : Anomalie("balises", ...).
    #! Les quatre fonctions sont importees en haut du fichier. Ecrivez des
    #! details lisibles : ils seront envoyes au modele pour qu'il corrige.
    #! Test : python tp.py test tp03 -k todo2
    _, _, fautes = terminologie(segment_src, traduction, glossaire)
    for faute in fautes:
        anomalies.append(Anomalie("terminologie", faute))
    if not chiffres(segment_src, traduction):
        anomalies.append(
            Anomalie("chiffres", "les nombres de la source ne sont pas tous repris à l'identique")
        )
    if vouvoiement(traduction) is False:
        anomalies.append(Anomalie("tutoiement", "le guide de style impose le vouvoiement"))
    if not balises(segment_src, traduction):
        anomalies.append(Anomalie("balises", "les marqueurs {0}, {1} doivent être conservés"))
    # >>>TODO 2
    return anomalies


def reparer_sans_modele(segment_src: str, traduction: str) -> str:
    """BONUS — reparer les chiffres sans rappeler le modele.

    Une anomalie de chiffres se corrige par un remplacement de texte. Cela coute
    zero appel, zero token, zero seconde — et c'est plus fiable qu'un second
    appel, parce qu'un modele qui s'est trompe une fois se trompe souvent deux.

    La question a se poser a chaque etape d'une chaine : **ai-je vraiment besoin
    d'un modele de langue pour ca ?**
    """
    import re

    source = re.findall(r"\d+", segment_src)
    produits = re.findall(r"\d+", traduction)
    # <<<BONUS 3 ★★ Reparation locale des chiffres
    #! Si les deux listes ont la meme longueur mais des valeurs differentes,
    #! remplacez dans la traduction, dans l'ordre, chaque nombre produit par le
    #! nombre de la source. Sinon, rendez la traduction inchangee.
    #! Indice : re.split(r"(\d+)", traduction) vous donne les morceaux et les
    #! nombres en alternance.
    #! Test : python tp.py test tp03 --bonus -k bonus3
    if len(source) != len(produits) or source == produits:
        return traduction
    morceaux, i = [], 0
    for part in re.split(r"(\d+)", traduction):
        if part.isdigit() and i < len(source):
            morceaux.append(source[i])
            i += 1
        else:
            morceaux.append(part)
    return "".join(morceaux)
    # >>>BONUS 3
