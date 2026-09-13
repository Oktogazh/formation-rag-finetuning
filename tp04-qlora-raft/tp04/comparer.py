"""TP 4 — le modele de base contre le modele adapte, meme prompt, memes segments.

La comparaison n'a de sens que si **tout le reste est identique** : meme
recherche, meme k, meme gabarit de prompt, meme temperature. C'est pour cela que
ce fichier reutilise le traducteur du TP 2 sans le modifier, en ne changeant que
le moteur.

Ce qu'il faut regarder dans le tableau, et dans cet ordre :

1. la ligne ``piege`` et la colonne ``Termino`` — c'est la que le fine-tuning
   travaille : le style et les conventions du client entrent dans les poids ;
2. la ligne ``nouveau`` — **si elle progresse autant que les autres, quelque
   chose ne va pas.** Le fine-tuning n'apporte pas de connaissance nouvelle : un
   segment sans voisin reste un segment sans voisin ;
3. la colonne ``Tokens`` — elle ne bouge pas. Le fine-tuning ne raccourcit pas
   le prompt, contrairement a ce qu'on lit souvent.

C'est la thèse du dernier après-midi : **RAG et fine-tuning ne s'opposent pas**,
ils repondent a deux questions differentes. Le RAG apporte ce que le modele ne
sait pas. Le fine-tuning lui apprend comment se comporter.
"""

from __future__ import annotations

from commun import corpus, rapport
from commun.mesure import evaluer
from commun.moteur import obtenir_moteur


def comparer_base_et_adaptateur(k: int = 3, n: int = 80) -> dict:
    from tp01.prompt import consigne_systeme
    from tp02.evaluer import traducteur_rag
    from tp02.memoire import charger_memoire
    from tp02.recherche import chercheur

    memoire = charger_memoire()
    glossaire = corpus.charger_glossaire()
    segments = corpus.charger_evaluation(n=n)
    chercher = chercheur(memoire, "lexicale", k)
    consignes = consigne_systeme()

    mesures = {}
    for role, etiquette in (("base", "modele de base"), ("adapte", "modele + adaptateur")):
        moteur = obtenir_moteur(role)
        print(f"\n{etiquette} ({moteur.nom()}) · {len(segments)} segments")
        mesures[role] = evaluer(
            segments, traducteur_rag(moteur, chercher, glossaire, consignes)
        )
        print()
        print(rapport.table(mesures[role], f"RAG + {etiquette}"))

    print("\n\nCe que l'adaptateur a change")
    print("=" * 76)
    print(f"{'':12} {'BLEU':>14} {'chrF':>14} {'Termino':>14} {'Chiffres':>14}")
    for categorie in rapport.CATEGORIES + ["total"]:
        def lot(role):
            r = mesures[role]
            return r if categorie == "total" else [x for x in r if x["categorie"] == categorie]

        avant, apres = rapport.agreger(lot("base")), rapport.agreger(lot("adapte"))
        if not avant:
            continue

        def delta(cle):
            a, b = avant.get(cle), apres.get(cle)
            if a is None or b is None:
                return "      -       "
            return f"{a:5.1f} -> {b:5.1f}"

        print(f"{categorie:12} {delta('bleu'):>14} {delta('chrf'):>14} "
              f"{delta('termino'):>14} {delta('chiffres'):>14}")

    rapport.enregistrer("tp04-adaptateur", mesures["adapte"], {"k": k})
    print("""
  Relisez la ligne « nouveau ». Si elle a peu bouge, le fine-tuning a fait
  exactement ce qu'il sait faire, et rien de plus. C'est le resultat attendu,
  et c'est une bonne nouvelle : cela veut dire que votre RAG sert encore.""")
    return mesures
