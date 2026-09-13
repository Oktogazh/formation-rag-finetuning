"""TP 6 — les noeuds du graphe. Tous fournis : ils reprennent les TP 2, 3 et 4.

Un noeud recoit l'etat et rend ce qu'il veut y ajouter. Rien d'autre.

Le noeud interessant est ``reutiliser``. Il ne fait **aucun appel au modele** :
il recopie le segment de la memoire et reporte les chiffres de la source. C'est
la « reparation floue » qu'un outil de TAO fait depuis vingt-cinq ans, et sur ce
corpus elle rend la reference **exacte** pour les 41 segments concernes. Le
modele, lui, se trompe regulierement sur ces memes segments : quand on lui
montre un voisin qui dit « 14 jours » et une source qui dit « 180 jours », il
recopie parfois 14.

C'est la lecon du dernier TP, et elle est inconfortable : **la meilleure
reponse, pour la moitie du corpus, n'est pas d'appeler un modele de langue.**
"""

from __future__ import annotations

import re

SEUIL_REUTILISATION = 0.95
MAX_TENTATIVES = 2


def difference_chiffres_seulement(source: str, voisin: str) -> bool:
    """Vrai si les deux segments ne different que par des nombres. Fourni.

    C'est la condition du guide de l'agence (``docs/003-consignes-client.md``) :
    au-dessus de 95 % de correspondance, on reutilise **apres verification des
    chiffres**. Si autre chose qu'un nombre a change, la reutilisation est
    dangereuse et il faut traduire.
    """
    from difflib import SequenceMatcher

    mots_source = re.findall(r"\w+", source.lower())
    mots_voisin = re.findall(r"\w+", voisin.lower())
    comparateur = SequenceMatcher(None, mots_voisin, mots_source)
    differences = [
        (mots_voisin[i1:i2], mots_source[j1:j2])
        for etiquette, i1, i2, j1, j2 in comparateur.get_opcodes()
        if etiquette != "equal"
    ]
    if not differences:
        return True
    return all(all(mot.isdigit() for mot in avant + apres) for avant, apres in differences)


def reparation_floue(voisin_src: str, voisin_tgt: str, source: str) -> str:
    """Reporte les nombres de la source dans la traduction du voisin. Fourni."""
    anciens = re.findall(r"\d+", voisin_src)
    nouveaux = re.findall(r"\d+", source)
    if len(anciens) != len(nouveaux):
        return voisin_tgt
    morceaux, i = [], 0
    for part in re.split(r"(\d+)", voisin_tgt):
        if part.isdigit():
            if i < len(anciens) and part == anciens[i]:
                morceaux.append(nouveaux[i])
                i += 1
            elif part in anciens:
                morceaux.append(nouveaux[anciens.index(part)])
            else:
                morceaux.append(part)
        else:
            morceaux.append(part)
    return "".join(morceaux)


def construire_noeuds(memoire, glossaire, moteur, chercher, consignes=""):
    """Fabrique les sept noeuds, fermes sur la memoire, le glossaire et le moteur."""
    from commun.prompts import nettoyer_sortie
    from tp02.augmenter import construire, glossaire_pertinent
    from tp02.recherche import meilleur_voisin
    from tp03.verification import verifier

    def analyser(etat):
        voisin = meilleur_voisin(etat["segment"], memoire)
        return {
            "taux_memoire": voisin["score"],
            "voisin": voisin,
            "chemin": ["analyser"],
        }

    def reutiliser(etat):
        voisin = etat["voisin"]
        return {
            "traduction": reparation_floue(voisin["src"], voisin["tgt"], etat["segment"]),
            "chemin": ["reutiliser"],
        }

    def recuperer(etat):
        return {
            "voisins": chercher(etat["segment"]),
            "glossaire": glossaire_pertinent(etat["segment"], glossaire),
            "chemin": ["recuperer"],
        }

    def traduire(etat):
        messages = construire(etat["segment"], etat["voisins"], etat["glossaire"], consignes)
        reponse = moteur.generer(messages)
        return {
            "traduction": nettoyer_sortie(reponse.texte),
            "appels": etat.get("appels", 0) + 1,
            "tokens_entree": etat.get("tokens_entree", 0) + reponse.tokens_entree,
            "tokens_sortie": etat.get("tokens_sortie", 0) + reponse.tokens_sortie,
            "chemin": ["traduire"],
        }

    def controler(etat):
        anomalies = verifier(etat["segment"], etat["traduction"], glossaire)
        return {
            "anomalies": [str(a) for a in anomalies],
            "tentatives": etat.get("tentatives", 0) + 1,
            "chemin": ["verifier"],
        }

    def corriger(etat):
        remarques = "\n".join(f"- {a}" for a in etat["anomalies"])
        messages = construire(etat["segment"], etat["voisins"], etat["glossaire"], consignes)
        messages.append({"role": "assistant", "content": etat["traduction"]})
        messages.append(
            {
                "role": "user",
                "content": f"Le controle automatique a relevé :\n{remarques}\n"
                           f"Rends la traduction corrigée, et rien d'autre.",
            }
        )
        reponse = moteur.generer(messages)
        return {
            "traduction": nettoyer_sortie(reponse.texte),
            "appels": etat.get("appels", 0) + 1,
            "tokens_entree": etat.get("tokens_entree", 0) + reponse.tokens_entree,
            "tokens_sortie": etat.get("tokens_sortie", 0) + reponse.tokens_sortie,
            "chemin": ["corriger"],
        }

    def livrer(etat):
        return {"chemin": ["livrer"]}

    return {
        "analyser": analyser,
        "reutiliser": reutiliser,
        "recuperer": recuperer,
        "traduire": traduire,
        "verifier": controler,
        "corriger": corriger,
        "livrer": livrer,
    }
