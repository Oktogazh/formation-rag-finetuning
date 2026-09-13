"""LE gabarit de prompt de la formation. Un seul, et c'est voulu.

Des le TP 2, tous les appels au modele passent par ``construire_messages``.
Pourquoi une seule fonction pour toute la formation :

* **Pour comparer.** Deux crans ne sont comparables que si la seule chose qui
  change entre eux est ce qu'on veut mesurer. Si chaque TP reecrivait son
  prompt, la table finale du TP 6 ne voudrait rien dire.
* **Pour le fine-tuning.** Au TP 4, le modele est entraine sur des exemples
  fabriques avec **ce** gabarit. S'il voit un autre format a l'inference, une
  partie de ce qu'il a appris ne sert a rien. C'est l'erreur classique du
  fine-tuning maison, et elle est invisible : le modele ne plante pas, il est
  juste moins bon que prevu.

Le TP 1 fait exception : le stagiaire y ecrit son propre prompt, a la main.
C'est le sujet du TP 1.
"""

from __future__ import annotations

SYSTEME = (
    "Tu es traducteur technique du suedois vers le francais pour l'editeur de "
    "logiciels Helios.\n"
    "Tu rends uniquement la traduction francaise du segment demande, sans "
    "commentaire, sans guillemets et sans repeter le suedois."
)


def bloc_glossaire(glossaire) -> str:
    if not glossaire:
        return ""
    lignes = ["### Glossaire impose"]
    for terme in glossaire:
        interdits = ", ".join(terme.get("interdits", []))
        ligne = f"- {terme['sv']} -> {terme['fr']}"
        if interdits:
            ligne += f" (jamais : {interdits})"
        lignes.append(ligne)
    return "\n".join(lignes)


def bloc_memoire(voisins) -> str:
    if not voisins:
        return ""
    lignes = ["### Memoire de traduction"]
    for voisin in voisins:
        lignes.append(f"sv: {voisin['src']}")
        lignes.append(f"fr: {voisin['tgt']}")
    return "\n".join(lignes)


def construire_messages(
    segment_src: str,
    voisins=(),
    glossaire=(),
    consignes: str = "",
) -> list[dict]:
    """Construit les messages envoyes au modele.

    Parameters
    ----------
    segment_src : le segment suedois a traduire.
    voisins     : segments de la memoire retenus par la recherche (TP 2),
                  ``[{"src": ..., "tgt": ...}, ...]``.
    glossaire   : entrees de glossaire jugees pertinentes (TP 2).
    consignes   : texte libre ajoute au message systeme (guide de style).

    L'ordre des blocs ne change jamais : glossaire, memoire, segment. Le segment
    est **en dernier**, juste avant la reponse attendue, parce que c'est la
    position dont les modeles tiennent le mieux compte.
    """
    systeme = SYSTEME
    if consignes:
        systeme = f"{SYSTEME}\n\n{consignes.strip()}"

    blocs = [bloc for bloc in (bloc_glossaire(glossaire), bloc_memoire(voisins)) if bloc]
    blocs.append(f"### Segment a traduire\nsv: {segment_src}\nfr:")

    return [
        {"role": "system", "content": systeme},
        {"role": "user", "content": "\n\n".join(blocs)},
    ]


def nettoyer_sortie(texte: str) -> str:
    """Enleve ce qu'un modele bavard ajoute autour de la traduction.

    Un modele de 3 milliards de parametres repond souvent « fr: ... », ou
    ajoute « Voici la traduction : ». On ne peut pas mesurer ca : on nettoie,
    et on nettoie **de la meme facon pour tous les crans**, sinon on mesure la
    qualite du nettoyage.
    """
    texte = texte.strip()
    for prefixe in ("fr:", "FR:", "Traduction :", "Traduction:", "Voici la traduction :"):
        if texte.startswith(prefixe):
            texte = texte[len(prefixe):].strip()
    ligne = next((l.strip() for l in texte.splitlines() if l.strip()), "")
    if len(ligne) >= 2 and ligne[0] in "\"'«" and ligne[-1] in "\"'»":
        ligne = ligne[1:-1].strip()
    return ligne
