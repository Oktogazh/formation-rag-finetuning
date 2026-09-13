"""TP 1 — ce que vous mettez dans le prompt, et ce que ca change.

Trois variantes, mesurees sur les memes segments :

``nu``        le modele recoit le segment suedois, rien d'autre.
``consigne``  on ajoute une consigne de style, ecrite par vous (TODO 1).
``exemples``  on ajoute trois traductions deja validees, choisies par vous
              dans la memoire (TODO 2).

Le **format** du prompt est fourni par ``commun/prompts.py`` et ne change pas de
tout le TP — sinon on ne comparerait pas trois prompts, on comparerait trois
formats. Ce qui est a vous, c'est le **contenu**.
"""

from __future__ import annotations

from commun.mesure import Sortie
from commun.prompts import construire_messages, nettoyer_sortie


def consigne_systeme() -> str:
    """La consigne de style envoyee au modele, en francais, en quelques lignes.

    Le client Helios impose quatre choses, et elles sont ecrites dans
    ``data/corpus/helios-sv/docs/`` — allez les lire, c'est la moitie du
    travail d'un traducteur professionnel :

    1. le **vouvoiement**, sans exception (``001-guide-de-style.md``) ;
    2. un **registre neutre**, ni familier ni commercial ;
    3. les **noms de produit et de formules ne se traduisent pas** : une
       formule « Företag » reste « Företag » (``003-consignes-client.md``) ;
    4. les **chiffres de la source se reportent tels quels**, meme quand ils
       surprennent.
    """
    # <<<TODO 1 ★ Ecrire la consigne de style
    # Rendez une chaine de caracteres qui dit au modele les quatre regles
    # listees dans la docstring ci-dessus. Ecrivez-la comme vous la diriez a
    # un traducteur humain qui debute : des phrases, pas des mots-cles.
    # Le test verifie qu'elle parle des quatre sujets et qu'elle fait plus de
    # 120 caracteres. Il ne verifie pas votre style : c'est le modele qui juge,
    # et vous verrez le resultat dans la table de mesure.
    # Test : python tp.py test tp01 -k todo1
    raise NotImplementedError(
        "TODO 1 — a completer. Consigne juste au-dessus, "
        "explications dans tp01-prompt/README.md"
    )
    # >>>TODO 1


def exemples_manuels() -> list[dict]:
    """Trois traductions deja validees, copiees depuis la memoire.

    C'est ce qu'on appelle du *few-shot* : au lieu de decrire le style, on le
    montre. Les trois exemples doivent venir de ``data/corpus/helios-sv/tm.jsonl``
    — ce sont des traductions validees par l'agence, pas des inventions.

    Rendez une liste de ``{"src": "<suedois>", "tgt": "<francais>"}``.
    """
    # <<<TODO 2 ★ Choisir trois exemples dans la memoire
    # Ouvrez data/corpus/helios-sv/tm.jsonl, choisissez trois segments et
    # recopiez-les ici. Choisissez-les : trois segments qui se ressemblent
    # n'apprennent qu'une chose au modele.
    # Analogue de format : le bloc « Memoire de traduction » que
    # commun/prompts.py fabrique a partir de cette liste.
    # Test : python tp.py test tp01 -k todo2
    raise NotImplementedError(
        "TODO 2 — a completer. Consigne juste au-dessus, "
        "explications dans tp01-prompt/README.md"
    )
    # >>>TODO 2


def exemples_cibles(domaine: str) -> list[dict]:
    """BONUS — trois exemples du **meme domaine** que le segment a traduire.

    Choisir les exemples au hasard, c'est du few-shot. Les choisir **en
    fonction du segment**, c'est deja du RAG : vous venez d'ecrire le TP 2 en
    quatre lignes.
    """
    from commun.corpus import charger_memoire_brute

    # <<<BONUS 1 ★ Selectionner les exemples par domaine
    # Rendez les trois premiers segments de la memoire dont le champ
    # « domaine » vaut exactement l'argument recu.
    # Analogue : charger_memoire_brute() rend la liste complete ; chaque
    # entree a les cles src, tgt, domaine, date, statut.
    # Test : python tp.py test tp01 --bonus -k bonus1
    raise NotImplementedError(
        "BONUS 1 — a completer. Consigne juste au-dessus, "
        "explications dans tp01-prompt/README.md"
    )
    # >>>BONUS 1


def messages_pour(segment_src: str, variante: str = "nu") -> list[dict]:
    """Le prompt de la variante demandee. Fourni : lisez-le, il est court."""
    if variante == "nu":
        return construire_messages(segment_src)
    if variante == "consigne":
        return construire_messages(segment_src, consignes=consigne_systeme())
    if variante == "exemples":
        return construire_messages(
            segment_src, voisins=exemples_manuels(), consignes=consigne_systeme()
        )
    raise ValueError(f"Variante inconnue : {variante}")


def traducteur(moteur, variante: str = "nu"):
    """Rend la fonction ``segment -> Sortie`` attendue par ``commun.mesure.evaluer``."""

    def traduire(segment: dict) -> Sortie:
        reponse = moteur.generer(messages_pour(segment["src"], variante))
        return Sortie(
            texte=nettoyer_sortie(reponse.texte),
            tokens_entree=reponse.tokens_entree,
            tokens_sortie=reponse.tokens_sortie,
            secondes=reponse.secondes,
            appels=1,
            chemin=(variante,),
        )

    return traduire
