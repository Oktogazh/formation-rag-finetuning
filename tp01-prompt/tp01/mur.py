"""TP 1 — le mur du prompt.

L'exercice precedent a montre que le prompt ameliore la traduction. La suite
logique, celle que toutes les equipes tentent, est : « mettons-y tout ». Tout,
ici, c'est le glossaire complet, les six documents de consignes et soixante
exemples de memoire, soit environ 9 000 tokens.

On le fait, et on mesure. Trois choses apparaissent :

1. le prompt est **15 fois plus long**, donc 15 fois plus cher, a chaque
   segment, pour toujours ;
2. la latence suit ;
3. la qualite, elle, **ne suit pas** — elle stagne, et souvent elle baisse. Un
   modele de 3 milliards de parametres traite mal le milieu d'un contexte long
   (« lost in the middle »), et 57 exemples sur 60 ne concernent pas le segment
   qu'il a sous les yeux.

C'est ce constat qui rend le TP 2 necessaire : la question n'est pas
« combien mettre dans le prompt », c'est **« lequel »**.
"""

from __future__ import annotations

from commun import corpus, mesure, rapport
from commun.mesure import Sortie
from commun.moteur import obtenir_moteur
from commun.prompts import construire_messages, nettoyer_sortie

NOMBRE_EXEMPLES = 60


def messages_du_mur(segment_src: str) -> list[dict]:
    """Tout ce qu'on a, dans le prompt, sans rien choisir."""
    return construire_messages(
        segment_src,
        voisins=corpus.charger_memoire_brute()[:NOMBRE_EXEMPLES],
        glossaire=corpus.charger_glossaire(),
        consignes=corpus.texte_des_consignes(),
    )


def mesurer_le_mur(n: int = 20) -> list[dict]:
    from tp01.prompt import traducteur

    segments = corpus.charger_evaluation(n=n)
    moteur = obtenir_moteur()

    print(f"\nTP 1 — le mur · {len(segments)} segments · {moteur.nom()}")
    print("  1/2  rappel : le prompt de la variante « exemples »")
    reference = mesure.evaluer(segments, traducteur(moteur, "exemples"), silencieux=True)

    print("  2/2  le mur : glossaire complet + 6 documents + 60 exemples")

    def traduire(segment):
        reponse = moteur.generer(messages_du_mur(segment["src"]))
        return Sortie(
            texte=nettoyer_sortie(reponse.texte),
            tokens_entree=reponse.tokens_entree,
            tokens_sortie=reponse.tokens_sortie,
            secondes=reponse.secondes,
            chemin=("mur",),
        )

    mur = mesure.evaluer(segments, traduire)

    print()
    print(rapport.table(reference, "Avant — consigne + 3 exemples choisis"))
    print()
    print(rapport.table(mur, "Apres — tout dans le prompt"))

    avant, apres = rapport.agreger(reference), rapport.agreger(mur)
    facteur = apres["tokens_entree"] / max(avant["tokens_entree"], 1)
    lenteur = apres["secondes"] / max(avant["secondes"], 1e-6)
    print(f"""
  Tokens par segment   {avant['tokens_entree']:.0f}  ->  {apres['tokens_entree']:.0f}   (x{facteur:.1f})
  Secondes par segment {avant['secondes']:.2f}  ->  {apres['secondes']:.2f}   (x{lenteur:.1f})
  BLEU                 {avant['bleu']:.1f}  ->  {apres['bleu']:.1f}
  Terminologie         {avant['termino'] or 0:.0f}%  ->  {apres['termino'] or 0:.0f}%

  Sur 100 000 segments par mois, le facteur x{facteur:.1f} sur les tokens est la
  seule ligne de ce tableau qu'une direction financiere lira.""")
    rapport.enregistrer("tp01-mur", mur, {"moteur": moteur.nom()})
    return mur
