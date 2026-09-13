#!/usr/bin/env python
"""Le point d'entrée unique de la formation.

    python tp.py check                  ce que votre machine sait faire
    python tp.py notebooks              (re)fabrique les notebooks depuis les .py
    python tp.py test tp02              les tests d'un TP — rouges au départ
    python tp.py indice tp02 --code 3   le diff entre votre code et le corrigé
    python tp.py rattraper tp02         reprendre le corrigé d'un TP entier

Les six TP sont **indépendants**. Rater le TP 2 n'interdit pas le TP 3 : chaque
notebook s'appuie sur ``commun/``, jamais sur ce qu'un autre TP vous a fait
écrire.

Cinq TP se font dans un notebook. Le TP 5 reste en fichiers, parce qu'on y
déploie un service et qu'on le bombarde depuis un second terminal :

    python tp.py api                    lance le service de traduction en local
    python tp.py assaut --clients 1,4,8 bombarde un service et mesure
    python tp.py banc                   la table finale, tous crans confondus
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent
DOSSIERS = {
    "tp01": "tp01-prompt",
    "tp02": "tp02-rag",
    "tp03": "tp03-langchain",
    "tp04": "tp04-lora-raft",
    "tp05": "tp05-mise-en-service",
    "tp06": "tp06-graphe",
}
NOTEBOOKS = ("tp01", "tp02", "tp03", "tp04", "tp06")

for _dossier in (RACINE, RACINE / "tp05-mise-en-service"):
    if str(_dossier) not in sys.path:
        sys.path.insert(0, str(_dossier))

try:
    from dotenv import load_dotenv

    load_dotenv(RACINE / ".env")
except ImportError:  # pragma: no cover
    pass


def titre(texte: str) -> None:
    print(f"\n{texte}\n{'=' * len(texte)}")


def cle_tp(nom: str) -> str:
    cle = nom.strip("/").split("/")[0].split("-")[0]
    if cle not in DOSSIERS:
        raise SystemExit(f"TP inconnu : {nom}. Attendu : {', '.join(DOSSIERS)}")
    return cle


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    from commun import materiel

    etat = materiel.diagnostic()
    titre("Votre machine")
    print(f"  Système            {etat['systeme']} ({etat['machine']})")
    print(f"  Python             {etat['python']}")
    print(f"  Accélérateur       {etat['accelerateur']}")
    memoire = f"{etat['memoire_go']:.0f} Go" if etat["memoire_go"] else "inconnue"
    print(f"  Mémoire            {memoire}")

    titre("Ce dont la formation a besoin")

    def ligne(nom, ok, detail=""):
        print(f"  [{'x' if ok else ' '}] {nom:<30} {detail}")

    ligne("Ollama joignable", etat["ollama_joignable"],
          ", ".join(etat["ollama_modeles"]) or "aucun modèle")
    ligne(f"Modèle {etat['modele_attendu']}", etat["modele_present"], "TP 1 à 6")
    ligne(f"Embeddings {etat['embeddings_attendu']}", etat["embeddings_present"], "TP 2, 3 et 6")
    ligne("Jupyter", etat["jupyter"], "TP 1, 2, 3, 4 et 6")
    ligne("torch + peft", etat["torch"] and etat["peft"], "TP 4 seulement")
    ligne("Clé API Mistral", etat["cle_mistral"], "repli si la machine ne suit pas")
    ligne("Clé LangSmith", etat["cle_langsmith"], "TP 3, 5 et 6, facultatif")

    titre("Ce qui sera utilisé")
    print(f"  Moteur             {etat['moteur_retenu']}")
    print(f"  Embeddings         {etat['encodeur_retenu']}")
    palier = materiel.palier(etat)
    duree = {"api": "≈ 1 s", "local": "≈ 1 à 3 s", "local-lent": "≈ 5 à 15 s"}[palier]
    print(f"  Palier             {palier}   ({duree} par segment traduit)")

    a_faire = materiel.conseils(etat)
    if a_faire:
        titre("À faire avant la formation")
        for conseil in a_faire:
            print(f"    {conseil}")

    pret = (etat["ollama_joignable"] and etat["modele_present"]) or etat["cle_mistral"]
    print()
    if pret:
        print("  Vous êtes prêt. Envoyez cette sortie au formateur.")
        return 0
    print("  Il manque de quoi générer du texte : voir « À faire » ci-dessus.")
    return 1


# ---------------------------------------------------------------------------
# notebooks
# ---------------------------------------------------------------------------
def _jupytext(*arguments) -> int:
    try:
        import jupytext  # noqa: F401
    except ImportError:
        print("  jupytext est absent :  pip install jupytext")
        return 2
    return subprocess.call([sys.executable, "-m", "jupytext", *arguments], cwd=RACINE)


NOYAU = "formation-rag"


def _installer_noyau() -> None:
    """Enregistre le noyau Jupyter de l'environnement de la formation.

    Sans ça, Jupyter propose le noyau « python3 » — qui est souvent celui d'un
    autre environnement conda, sans langchain ni langgraph. Le notebook échoue
    alors sur son premier import, et c'est incompréhensible pour qui débute.
    """
    subprocess.call([sys.executable, "-m", "ipykernel", "install", "--user",
                     "--name", NOYAU, "--display-name", "Formation RAG"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cmd_notebooks(args) -> int:
    _installer_noyau()
    cles = [cle_tp(args.tp)] if args.tp else list(NOTEBOOKS)
    for cle in cles:
        if cle not in NOTEBOOKS:
            print(f"  {cle} n'est pas un notebook : il se fait en fichiers.")
            continue
        source = RACINE / DOSSIERS[cle] / f"{cle}.py"
        cible = source.with_suffix(".ipynb")
        if cible.exists() and not args.forcer:
            print(f"  {cible.relative_to(RACINE)} existe déjà — --forcer pour l'écraser")
            continue
        if _jupytext("--to", "ipynb", "--set-kernel", NOYAU,
                     str(source.relative_to(RACINE))) != 0:
            return 2
    print(f"""
  Ouvrez-les :  jupyter lab

  Le noyau « Formation RAG » est déjà sélectionné dans chaque notebook. S'il ne
  l'est pas, choisissez-le en haut à droite : le noyau « python3 » est souvent
  celui d'un autre environnement, et les imports échoueraient.""")
    return 0


def _synchroniser() -> None:
    """Fait redescendre les modifications du notebook dans le module .py.

    Sans ça, le test importerait la version d'avant vos dernières cellules, et
    la première question de la salle serait « mon test échoue alors que ma
    cellule marche ».
    """
    for cle in NOTEBOOKS:
        carnet = RACINE / DOSSIERS[cle] / f"{cle}.ipynb"
        if carnet.exists():
            _jupytext("--sync", "--quiet", str(carnet.relative_to(RACINE)))


# ---------------------------------------------------------------------------
# test / indice / rattraper
# ---------------------------------------------------------------------------
def cmd_test(args) -> int:
    if args.corrige:
        return _test_corrige(args)
    _synchroniser()
    commande = [sys.executable, "-m", "pytest"]
    if args.tp:
        commande.append(str(Path(DOSSIERS[cle_tp(args.tp)]) / "tests"))
    if args.bonus:
        commande.append("--bonus")
    if args.modele:
        commande.append("--modele")
    commande += args.pytest_args
    return subprocess.call(commande, cwd=RACINE)


def _test_corrige(args) -> int:
    with tempfile.TemporaryDirectory(prefix="tp-corrige-") as temporaire:
        copie = Path(temporaire) / "depot"
        shutil.copytree(RACINE, copie, ignore=shutil.ignore_patterns(
            ".git", "resultats", "__pycache__", ".pytest_cache", "*.ipynb"))
        for source in (copie / "corrige").rglob("*.py"):
            cible = copie / source.relative_to(copie / "corrige")
            cible.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, cible)
        print(f"Suite complète sur le corrigé ({copie})")
        commande = [sys.executable, "-m", "pytest", "--bonus"]
        if args.tp:
            commande.append(str(Path(DOSSIERS[cle_tp(args.tp)]) / "tests"))
        commande += args.pytest_args
        return subprocess.call(commande, cwd=copie)


def _paire(chemin: str) -> tuple[Path, Path]:
    cle = cle_tp(chemin)
    morceaux = chemin.strip("/").split("/")
    if len(morceaux) > 1 and morceaux[-1].endswith(".py"):
        relatif = Path(DOSSIERS[cle]) / Path(*morceaux[1:])
    else:
        relatif = Path(DOSSIERS[cle]) / f"{cle}.py"
    exercice, corrige = RACINE / relatif, RACINE / "corrige" / relatif
    if not corrige.exists():
        raise SystemExit(f"Pas de corrigé pour {chemin}. Attendu : corrige/{relatif}")
    return exercice, corrige


def cmd_indice(args) -> int:
    import difflib

    _synchroniser()
    exercice, corrige = _paire(args.fichier)
    a = exercice.read_text(encoding="utf-8").splitlines(keepends=True)
    # Les lignes « #> » sont la consigne telle que le corrigé la stocke ; chez
    # vous ce sont déjà des commentaires ordinaires. On normalise, sinon le diff
    # serait plein de marqueurs qui n'apprennent rien.
    b = [re.sub(r"^(\s*)#>\s?", r"\1# ", l)
         for l in corrige.read_text(encoding="utf-8").splitlines(keepends=True)]

    if args.code:
        def extraire(lignes):
            debut = fin = None
            for i, ligne in enumerate(lignes):
                if re.search(rf"<<<(CODE|BONUS) {args.code}\b", ligne):
                    debut = i
                if debut is not None and re.search(rf">>>(CODE|BONUS) {args.code}\b", ligne):
                    fin = i + 1
                    break
            if debut is None:
                raise SystemExit(f"CODE {args.code} introuvable dans {args.fichier}")
            return lignes[debut:fin]

        a, b = extraire(a), extraire(b)

    diff = list(difflib.unified_diff(a, b, fromfile=f"le vôtre  {args.fichier}",
                                     tofile=f"le corrigé {args.fichier}", n=2))
    if not diff:
        print("Aucune différence : votre version fait déjà ce que fait le corrigé.")
        return 0
    print("".join(diff))
    print("\n  Recopier n'apprend rien. Lisez, fermez, retapez.")
    return 0


def cmd_rattraper(args) -> int:
    cle = cle_tp(args.tp)
    dossier = RACINE / DOSSIERS[cle]
    sources = sorted((RACINE / "corrige" / DOSSIERS[cle]).rglob("*.py"))
    if not sources:
        raise SystemExit(f"Pas de corrigé pour {args.tp}")
    print(f"Remplacer le travail de {DOSSIERS[cle]} par le corrigé :")
    for source in sources:
        print(f"    {source.relative_to(RACINE / 'corrige')}")
    print("\n  Votre travail sur ces fichiers sera perdu.")
    if not args.oui and input("  Continuer ? [o/N] ").strip().lower() not in ("o", "oui", "y"):
        print("  Annulé.")
        return 1
    for source in sources:
        cible = RACINE / source.relative_to(RACINE / "corrige")
        cible.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, cible)
    if cle in NOTEBOOKS:
        carnet = dossier / f"{cle}.ipynb"
        if carnet.exists():
            carnet.unlink()
        _jupytext("--to", "ipynb", "--set-kernel", NOYAU,
                  str((dossier / f"{cle}.py").relative_to(RACINE)))
    print(f"\n  Fait. Vérifiez : python tp.py test {cle}")
    return 0


# ---------------------------------------------------------------------------
# TP 5 et table finale
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
            "  Service local : python tp.py api   puis   --url http://localhost:8000")
    campagne(url, [int(c) for c in args.clients.split(",")], args.duree)
    return 0


def cmd_banc(args) -> int:
    from commun.banc import afficher_banc

    afficher_banc()
    return 0


# ---------------------------------------------------------------------------
def construire_analyseur() -> argparse.ArgumentParser:
    analyseur = argparse.ArgumentParser(
        prog="python tp.py",
        description="Formation « RAG et fine-tuning d'un LLM » — traduction suédois vers français.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commencez par : python tp.py check")
    sous = analyseur.add_subparsers(dest="commande", metavar="<commande>")

    p = sous.add_parser("check", help="ce que votre machine sait faire")
    p.set_defaults(fonction=cmd_check)

    p = sous.add_parser("notebooks", help="(re)fabriquer les notebooks depuis les .py")
    p.add_argument("tp", nargs="?", help="tp01 … tp06 ; vide = tous")
    p.add_argument("--forcer", action="store_true", help="écraser un notebook existant")
    p.set_defaults(fonction=cmd_notebooks)

    p = sous.add_parser("test", help="lancer les tests d'un TP, ou de tous")
    p.add_argument("tp", nargs="?", help="tp01 … tp06 ; vide = tous")
    p.add_argument("--corrige", action="store_true", help="jouer la suite sur le corrigé")
    p.add_argument("--bonus", action="store_true", help="inclure les exercices bonus")
    p.add_argument("--modele", action="store_true", help="inclure les tests qui appellent un moteur")
    p.add_argument("pytest_args", nargs="*", help="arguments passés à pytest (ex : -k code2)")
    p.set_defaults(fonction=cmd_test)

    p = sous.add_parser("indice", help="le diff entre votre code et le corrigé")
    p.add_argument("fichier", help="ex : tp02  ou  tp05/assaut.py")
    p.add_argument("--code", type=int, help="se limiter à ce CODE")
    p.set_defaults(fonction=cmd_indice)

    p = sous.add_parser("rattraper", help="prendre le corrigé d'un TP entier")
    p.add_argument("tp", help="tp01 … tp06")
    p.add_argument("--oui", action="store_true", help="ne pas demander confirmation")
    p.set_defaults(fonction=cmd_rattraper)

    p = sous.add_parser("api", help="TP 5 — lancer le service de traduction en local")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--sans-garde", action="store_true",
                   help="reproduire le défaut du Space : aucune protection")
    p.set_defaults(fonction=cmd_api)

    p = sous.add_parser("assaut", help="TP 5 — bombarder un service et mesurer")
    p.add_argument("--url", help="défaut : TP_URL_SPACE")
    p.add_argument("--clients", default="1,2,4,8")
    p.add_argument("--duree", type=int, default=30, help="secondes par palier")
    p.set_defaults(fonction=cmd_assaut)

    p = sous.add_parser("banc", help="la table finale, tous crans confondus")
    p.set_defaults(fonction=cmd_banc)
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
        print(f"\n  À compléter : {erreur}\n"
              f"  Indice : python tp.py indice <tp> --code <n>")
        return 3
    except KeyboardInterrupt:
        print("\n  Interrompu.")
        return 130


if __name__ == "__main__":
    sys.exit(principal())
