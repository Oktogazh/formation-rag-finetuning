"""TP 1 — la temperature, en trois minutes et sans TODO.

On traduit **le meme segment** plusieurs fois, a plusieurs temperatures. Il n'y
a rien a completer : il y a quelque chose a voir, et une conclusion a tirer.

Ce que vous devez observer :

* a ``T = 0`` les trois sorties sont **identiques** — c'est la seule valeur qui
  permet de comparer deux systemes, et c'est pour ca que toutes les mesures de
  la formation sont a 0 ;
* a partir de ``T = 0,7`` la terminologie decroche et le registre part ;
* pour une traduction, la temperature n'est pas un reglage de creativite, c'est
  **un taux de defaut**. Traduction, extraction, classification : 0. Production
  de variantes qu'un humain va trier : plus haut, et c'est legitime.
"""

from __future__ import annotations

from commun.corpus import charger_evaluation, charger_glossaire
from commun.mesure import chiffres, terminologie, vouvoiement
from commun.moteur import obtenir_moteur
from commun.prompts import construire_messages, nettoyer_sortie


def comparer(identifiant: str, valeurs=(0.0, 0.3, 0.7, 1.2), repetitions: int = 3) -> None:
    segments = {s["id"]: s for s in charger_evaluation()}
    if identifiant not in segments:
        raise SystemExit(f"Segment inconnu : {identifiant}")
    segment = segments[identifiant]
    glossaire = charger_glossaire()
    moteur = obtenir_moteur()

    print(f"\nSegment {identifiant} ({segment['categorie']})")
    print(f"  sv  {segment['src']}")
    print(f"  ref {segment['tgt']}\n")

    for temperature in valeurs:
        print(f"  T = {temperature}")
        vues = []
        for _ in range(repetitions):
            reponse = moteur.generer(
                construire_messages(segment["src"]), temperature=temperature
            )
            texte = nettoyer_sortie(reponse.texte)
            marques = []
            if texte in vues:
                marques.append("identique")
            _, ok, fautes = terminologie(segment["src"], texte, glossaire)
            if fautes:
                marques.append(f"termino : {fautes[0]}")
            if not chiffres(segment["src"], texte):
                marques.append("chiffre faux")
            if vouvoiement(texte) is False:
                marques.append("tutoie")
            vues.append(texte)
            suffixe = f"   <- {', '.join(marques)}" if marques else ""
            print(f"      {texte}{suffixe}")
        print()

    print("  A retenir : T = 0 est la seule valeur reproductible. Toutes les")
    print("  mesures de la formation sont faites a 0, et c'est pour cette raison.")
