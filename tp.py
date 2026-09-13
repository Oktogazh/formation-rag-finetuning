#!/usr/bin/env python
"""Le point d'entree unique de la formation.

    python tp.py check          ce que votre machine peut faire
    python tp.py test           les tests : rouges au depart, verts a la fin
    python tp.py indice tp02/recherche.py --todo 2
    python tp.py rattraper tp02

Puis, TP par TP :

    TP 1   traduire · temperature · mur
    TP 2   rag
    TP 3   chaine
    TP 4   donnees · entrainer · comparer · adaptateur
    TP 5   api · assaut
    TP 6   graphe · banc

``python tp.py <commande> --help`` detaille chaque commande.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parent
DOSSIERS_TP = {
    "tp01": "tp01-prompt",
    "tp02": "tp02-rag",
    "tp03": "tp03-langchain",
    "tp04": "tp04-qlora-raft",
    "tp05": "tp05-mise-en-service",
    "tp06": "tp06-graphe",
}

for _dossier in [RACINE, *(RACINE / d for d in DOSSIERS_TP.values())]:
    if str(_dossier) not in sys.path:
        sys.path.insert(0, str(_dossier))

try:
    from dotenv import load_dotenv

    load_dotenv(RACINE / ".env")
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Outils d'affichage
# ---------------------------------------------------------------------------
def titre(texte: str) -> None:
    print(f"\n{texte}\n{'=' * len(texte)}")


def avertir_factice() -> None:
    from commun.moteur import moteur_demande

    if moteur_demande() == "factice":
        print(
            "\n  ! MOTEUR FACTICE : les chiffres ci-dessous ne mesurent rien.\n"
            "    Il imite les defauts d'un modele, il ne les reproduit pas.\n"
            "    Retirez TP_MOTEUR=factice de votre .env pour mesurer pour de vrai.\n"
        )


def dossier_tp(nom: str) -> Path:
    cle = nom.strip("/").split("/")[0].split("-")[0]
    if cle not in DOSSIERS_TP:
        raise SystemExit(f"TP inconnu : {nom}. Attendu : {', '.join(DOSSIERS_TP)}")
    return RACINE / DOSSIERS_TP[cle]


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    from commun import materiel

    etat = materiel.diagnostic()
    titre("Votre machine")
    print(f"  Systeme            {etat['systeme']} ({etat['machine']})")
    print(f"  Python             {etat['python']}")
    print(f"  Accelerateur       {etat['accelerateur']}")
    memoire = f"{etat['memoire_go']:.0f} Go" if etat["memoire_go"] else "inconnue"
    print(f"  Memoire            {memoire}")

    titre("Ce dont la formation a besoin")
    def ligne(nom, ok, detail=""):
        print(f"  [{'x' if ok else ' '}] {nom:<28} {detail}")

    ligne("Ollama joignable", etat["ollama_joignable"],
          ", ".join(etat["ollama_modeles"]) or "aucun modele")
    ligne(f"Modele {etat['modele_attendu']}", etat["modele_present"])
    ligne(f"Embeddings {etat['embeddings_attendu']}", etat["embeddings_present"])
    ligne("Cle API Mistral", etat["cle_mistral"], "(repli si la machine ne suit pas)")
    ligne("Cle LangSmith", etat["cle_langsmith"], "(TP 3, 5 et 6, facultatif)")
    ligne("mlx-lm", etat["mlx"], "(TP 4 sur Apple Silicon)")
    ligne("peft + bitsandbytes", etat["peft"] and etat["bitsandbytes"], "(TP 4 sur NVIDIA)")

    titre("Ce qui sera utilise")
    print(f"  Moteur             {etat['moteur_retenu']}")
    print(f"  Embeddings         {etat['encodeur_retenu']}")
    palier = materiel.palier(etat)
    duree = {"api": "≈ 1 s", "local": "≈ 3 a 5 s", "local-lent": "≈ 10 a 20 s"}[palier]
    print(f"  Palier             {palier}   ({duree} par segment traduit)")

    a_faire = materiel.conseils(etat)
    if a_faire:
        titre("A faire avant la formation")
        for conseil in a_faire:
            print(f"    {conseil}")

    pret = (etat["ollama_joignable"] and etat["modele_present"]) or etat["cle_mistral"]
    print()
    if pret:
        print("  Vous etes pret. Envoyez cette sortie au formateur.")
        return 0
    print("  Il manque de quoi generer du texte : voir « A faire » ci-dessus.")
    return 1


# ---------------------------------------------------------------------------
# test / indice / rattraper
# ---------------------------------------------------------------------------
def cmd_test(args) -> int:
    if args.corrige:
        return _test_corrige(args)
    commande = [sys.executable, "-m", "pytest"]
    if args.tp:
        commande.append(str(dossier_tp(args.tp).relative_to(RACINE) / "tests"))
    if args.bonus:
        commande.append("--bonus")
    if args.modele:
        commande.append("--modele")
    commande += args.pytest_args
    return subprocess.call(commande, cwd=RACINE)


def _test_corrige(args) -> int:
    """Monte le corrige dans une copie jetable et y lance la suite complete."""
    with tempfile.TemporaryDirectory(prefix="tp-corrige-") as temporaire:
        copie = Path(temporaire) / "depot"
        shutil.copytree(
            RACINE, copie,
            ignore=shutil.ignore_patterns(".git", "resultats", "__pycache__", ".pytest_cache"),
        )
        for cle, dossier in DOSSIERS_TP.items():
            source = copie / "corrige" / dossier / cle
            cible = copie / dossier / cle
            if source.exists():
                shutil.rmtree(cible, ignore_errors=True)
                shutil.copytree(source, cible)
        print(f"Suite complete sur le corrige ({copie})")
        commande = [sys.executable, "-m", "pytest", "--bonus"]
        if args.tp:
            commande.append(str(Path(DOSSIERS_TP[args.tp.split("-")[0]]) / "tests"))
        commande += args.pytest_args
        return subprocess.call(commande, cwd=copie)


def _paire(chemin: str) -> tuple[Path, Path]:
    """(fichier d'exercice, fichier corrige) a partir de « tp02/recherche.py »."""
    relatif = Path(chemin)
    if relatif.parts and relatif.parts[0] in DOSSIERS_TP.values():
        relatif = Path(*relatif.parts[1:])
    cle = relatif.parts[0]
    dossier = dossier_tp(cle)
    exercice = dossier / relatif
    corrige = RACINE / "corrige" / dossier.name / relatif
    if not corrige.exists():
        raise SystemExit(
            f"Pas de corrige pour {chemin}.\n"
            f"Attendu : {corrige.relative_to(RACINE)}"
        )
    return exercice, corrige


def cmd_indice(args) -> int:
    import difflib

    exercice, corrige = _paire(args.fichier)
    texte_exercice = exercice.read_text(encoding="utf-8").splitlines(keepends=True)
    # Les lignes « #! » sont la consigne telle qu'elle est stockee dans le
    # corrige ; dans votre fichier elles sont deja des commentaires ordinaires.
    # On les normalise, sinon le diff serait plein de marqueurs qui ne vous
    # apprennent rien.
    texte_corrige = [
        re.sub(r"^(\s*)#!\s?", r"\1# ", ligne)
        for ligne in corrige.read_text(encoding="utf-8").splitlines(keepends=True)
    ]

    if args.todo:
        def extraire(lignes):
            debut = fin = None
            for i, ligne in enumerate(lignes):
                if re.search(rf"<<<TODO {args.todo}\b", ligne):
                    debut = i
                if debut is not None and re.search(rf">>>TODO {args.todo}\b", ligne):
                    fin = i + 1
                    break
            if debut is None:
                raise SystemExit(f"TODO {args.todo} introuvable dans {args.fichier}")
            return lignes[debut:fin]

        texte_exercice, texte_corrige = extraire(texte_exercice), extraire(texte_corrige)

    diff = list(difflib.unified_diff(
        texte_exercice, texte_corrige,
        fromfile=f"le votre  {args.fichier}", tofile=f"le corrige {args.fichier}", n=2,
    ))
    if not diff:
        print("Aucune difference : votre version fait deja ce que fait le corrige.")
        return 0
    print("".join(diff))
    print("\n  Recopier n'apprend rien. Lisez, fermez, retapez.")
    return 0


def cmd_rattraper(args) -> int:
    cle = args.tp.split("-")[0]
    dossier = dossier_tp(cle)
    source = RACINE / "corrige" / dossier.name / cle
    cible = dossier / cle
    if not source.exists():
        raise SystemExit(f"Pas de corrige pour {args.tp}")
    fichiers = sorted(source.glob("*.py"))
    print(f"Remplacer {len(fichiers)} fichier(s) de {cible.relative_to(RACINE)} par le corrige :")
    for f in fichiers:
        print(f"    {f.name}")
    print("\n  Votre travail sur ces fichiers sera perdu.")
    if not args.oui:
        reponse = input("  Continuer ? [o/N] ").strip().lower()
        if reponse not in ("o", "oui", "y"):
            print("  Annule.")
            return 1
    for f in fichiers:
        shutil.copy2(f, cible / f.name)
    print(f"\n  Fait. Verifiez : python tp.py test {cle}")
    return 0


# ---------------------------------------------------------------------------
# TP 1
# ---------------------------------------------------------------------------
def cmd_traduire(args) -> int:
    from commun import corpus, mesure, rapport
    from commun.moteur import obtenir_moteur
    from tp01.prompt import traducteur

    avertir_factice()
    segments = corpus.charger_evaluation(n=args.n)
    moteur = obtenir_moteur()
    titre(f"TP 1 — variante « {args.variante} » · {len(segments)} segments · {moteur.nom()}")
    resultats = mesure.evaluer(segments, traducteur(moteur, args.variante))
    print()
    print(rapport.table(resultats, f"Variante {args.variante}"))
    fautes = rapport.fautes_frequentes(resultats)
    if fautes:
        print("\n" + fautes)
    print("\nEnregistre :", rapport.enregistrer(f"tp01-{args.variante}", resultats,
                                                {"moteur": moteur.nom()}))
    return 0


def cmd_temperature(args) -> int:
    from tp01.temperature import comparer

    avertir_factice()
    comparer(args.segment, [float(v) for v in args.valeurs.split(",")], args.repetitions)
    return 0


def cmd_mur(args) -> int:
    from tp01.mur import mesurer_le_mur

    avertir_factice()
    mesurer_le_mur(n=args.n)
    return 0


# ---------------------------------------------------------------------------
# TP 2
# ---------------------------------------------------------------------------
def cmd_rag(args) -> int:
    from tp02.evaluer import evaluer_rag

    avertir_factice()
    evaluer_rag(recherche=args.recherche, k=args.k, n=args.n)
    return 0


# ---------------------------------------------------------------------------
# TP 3
# ---------------------------------------------------------------------------
def cmd_chaine(args) -> int:
    from tp03.evaluer import evaluer_chaine

    avertir_factice()
    evaluer_chaine(avec_verification=args.verifier, k=args.k, n=args.n)
    return 0


# ---------------------------------------------------------------------------
# TP 4
# ---------------------------------------------------------------------------
def cmd_donnees(args) -> int:
    from tp04.donnees import construire_jeux

    construire_jeux(k=args.k, p_oracle=args.p_oracle)
    return 0


def cmd_entrainer(args) -> int:
    from tp04.entrainer import entrainer

    return entrainer(epoques=args.epoques, rang=args.rang)


def cmd_comparer(args) -> int:
    from tp04.comparer import comparer_base_et_adaptateur

    avertir_factice()
    comparer_base_et_adaptateur(k=args.k, n=args.n)
    return 0


def cmd_adaptateur(args) -> int:
    from tp04.entrainer import telecharger_adaptateur

    if args.telecharger:
        return telecharger_adaptateur(args.depot)
    print("Rien a faire. Utilisez --telecharger pour recuperer l'adaptateur du formateur.")
    return 0


# ---------------------------------------------------------------------------
# TP 5
# ---------------------------------------------------------------------------
def cmd_api(args) -> int:
    from tp05.serveur_local import lancer

    return lancer(port=args.port, repare=not args.sans_garde)


def cmd_assaut(args) -> int:
    from tp05.assaut import campagne

    url = args.url or os.environ.get("TP_URL_SPACE", "")
    if not url:
        raise SystemExit(
            "Aucune URL. Donnez --url, ou posez TP_URL_SPACE dans .env.\n"
            "  Service local : python tp.py api   puis   --url http://localhost:8000"
        )
    campagne(url, [int(c) for c in args.clients.split(",")], args.duree)
    return 0


# ---------------------------------------------------------------------------
# TP 6
# ---------------------------------------------------------------------------
def cmd_graphe(args) -> int:
    from tp06.evaluer import evaluer_graphe

    avertir_factice()
    evaluer_graphe(n=args.n, moteur_traduction=args.moteur, k=args.k)
    return 0


def cmd_banc(args) -> int:
    from tp06.banc import afficher_banc

    afficher_banc()
    return 0


def cmd_resultats(args) -> int:
    from tp06.banc import afficher_banc

    afficher_banc()
    return 0


# ---------------------------------------------------------------------------
# Analyse des arguments
# ---------------------------------------------------------------------------
def construire_analyseur() -> argparse.ArgumentParser:
    analyseur = argparse.ArgumentParser(
        prog="python tp.py",
        description="Formation « RAG et fine-tuning d'un LLM » — traduction suedois vers francais.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commencez par : python tp.py check",
    )
    sous = analyseur.add_subparsers(dest="commande", metavar="<commande>")

    p = sous.add_parser("check", help="ce que votre machine peut faire")
    p.set_defaults(fonction=cmd_check)

    p = sous.add_parser("test", help="lancer les tests d'un TP, ou de tous")
    p.add_argument("tp", nargs="?", help="tp01 … tp06 ; vide = tous")
    p.add_argument("--corrige", action="store_true", help="jouer la suite sur le corrige")
    p.add_argument("--bonus", action="store_true", help="inclure les exercices bonus")
    p.add_argument("--modele", action="store_true", help="inclure les tests qui appellent un moteur")
    p.add_argument("pytest_args", nargs="*", help="arguments passes a pytest (ex : -k todo2)")
    p.set_defaults(fonction=cmd_test)

    p = sous.add_parser("indice", help="le diff entre votre fichier et le corrige")
    p.add_argument("fichier", help="ex : tp02/recherche.py")
    p.add_argument("--todo", type=int, help="se limiter a ce TODO")
    p.set_defaults(fonction=cmd_indice)

    p = sous.add_parser("rattraper", help="prendre le corrige d'un TP entier")
    p.add_argument("tp", help="tp01 … tp06")
    p.add_argument("--oui", action="store_true", help="ne pas demander confirmation")
    p.set_defaults(fonction=cmd_rattraper)

    p = sous.add_parser("traduire", help="TP 1 — traduire avec un prompt")
    p.add_argument("--variante", choices=["nu", "consigne", "exemples"], default="nu")
    p.add_argument("--n", type=int, default=20, help="nombre de segments (defaut 20)")
    p.set_defaults(fonction=cmd_traduire)

    p = sous.add_parser("temperature", help="TP 1 — le meme segment a plusieurs temperatures")
    p.add_argument("--segment", default="ev-005", help="identifiant du segment (defaut ev-005)")
    p.add_argument("--valeurs", default="0,0.3,0.7,1.2")
    p.add_argument("--repetitions", type=int, default=3)
    p.set_defaults(fonction=cmd_temperature)

    p = sous.add_parser("mur", help="TP 1 — tout mettre dans le prompt, et voir")
    p.add_argument("--n", type=int, default=20)
    p.set_defaults(fonction=cmd_mur)

    p = sous.add_parser("rag", help="TP 2 — ne mettre que ce qui sert")
    p.add_argument("--recherche", choices=["lexicale", "dense", "hybride"], default="lexicale",
                   help="« hybride » n'existe qu'une fois le bonus 2 fait")
    p.add_argument("--k", type=int, default=3, help="nombre de voisins injectes")
    p.add_argument("--n", type=int, default=80)
    p.set_defaults(fonction=cmd_rag)

    p = sous.add_parser("chaine", help="TP 3 — la meme chose avec LangChain")
    p.add_argument("--verifier", action="store_true", help="ajouter verification et correction")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n", type=int, default=80)
    p.set_defaults(fonction=cmd_chaine)

    p = sous.add_parser("donnees", help="TP 4 — fabriquer le jeu d'entrainement RAFT")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--p-oracle", dest="p_oracle", type=float, default=0.8)
    p.set_defaults(fonction=cmd_donnees)

    p = sous.add_parser("entrainer", help="TP 4 — entrainer l'adaptateur LoRA")
    p.add_argument("--epoques", type=int, default=2)
    p.add_argument("--rang", type=int, default=16)
    p.set_defaults(fonction=cmd_entrainer)

    p = sous.add_parser("comparer", help="TP 4 — modele de base contre modele adapte")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n", type=int, default=80)
    p.set_defaults(fonction=cmd_comparer)

    p = sous.add_parser("adaptateur", help="TP 4 — recuperer l'adaptateur du formateur")
    p.add_argument("--telecharger", action="store_true")
    p.add_argument("--depot", default="Oktogazh/helios-sv-lora", help="depot HuggingFace")
    p.set_defaults(fonction=cmd_adaptateur)

    p = sous.add_parser("api", help="TP 5 — lancer le service de traduction en local")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--sans-garde", action="store_true",
                   help="reproduire le defaut du Space : aucune protection")
    p.set_defaults(fonction=cmd_api)

    p = sous.add_parser("assaut", help="TP 5 — bombarder un service et mesurer")
    p.add_argument("--url", help="defaut : TP_URL_SPACE")
    p.add_argument("--clients", default="1,2,4,8")
    p.add_argument("--duree", type=int, default=30, help="secondes par palier")
    p.set_defaults(fonction=cmd_assaut)

    p = sous.add_parser("graphe", help="TP 6 — le graphe route, mesure et livre")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--moteur", choices=["base", "adapte"], default="base")
    p.set_defaults(fonction=cmd_graphe)

    p = sous.add_parser("banc", help="TP 6 — la table finale, tous crans confondus")
    p.set_defaults(fonction=cmd_banc)

    p = sous.add_parser("resultats", help="synonyme de « banc »")
    p.set_defaults(fonction=cmd_resultats)
    return analyseur


def principal(argv=None) -> int:
    analyseur = construire_analyseur()
    args = analyseur.parse_args(argv)
    if not getattr(args, "fonction", None):
        analyseur.print_help()
        return 0
    try:
        return args.fonction(args)
    except NotImplementedError as erreur:
        print(f"\n  TODO a faire : {erreur}\n"
              f"  Indice : python tp.py indice <fichier> --todo <n>")
        return 3
    except KeyboardInterrupt:
        print("\n  Interrompu.")
        return 130


if __name__ == "__main__":
    sys.exit(principal())
