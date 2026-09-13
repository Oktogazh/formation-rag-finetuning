#!/usr/bin/env python
"""Fabrique les fichiers d'exercice à partir du corrigé.

Le corrigé est la source de vérité. Ce que le stagiaire édite en est **dérivé** :
on remplace le contenu de chaque bloc marqué par un ``NotImplementedError``, et
on garde la consigne.

    python scripts/faire_les_squelettes.py              # (re)génère
    python scripts/faire_les_squelettes.py --verifier   # contrôle, sans écrire

Convention dans le corrigé :

    # <<<CODE 2 ★★ Titre court
    #> une ligne de consigne
    #> un analogue à regarder
    #> Test : python tp.py test tp02 -k code2
    <le code de la solution>
    # >>>CODE 2

Les lignes ``#>`` deviennent des commentaires ordinaires dans le squelette ; le
code entre elles et le marqueur de fin disparaît. Un ``BONUS`` s'écrit pareil.

Pourquoi ``#>`` et pas ``#!`` : jupytext échappe ``#!`` en ``# #!`` à
l'aller-retour notebook, et le corrigé cesserait de correspondre à l'exercice.

**Un bug se corrige dans ``corrige/``, jamais dans le fichier d'exercice.**
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DEBUT = re.compile(r"^(\s*)# <<<(CODE|BONUS) (\d+)\b(.*)$")
FIN = re.compile(r"^\s*# >>>(CODE|BONUS) (\d+)\b")


def squelette(source: str, nom_tp: str) -> str:
    lignes = source.splitlines(keepends=True)
    sortie: list[str] = []
    i = 0
    while i < len(lignes):
        debut = DEBUT.match(lignes[i])
        if not debut:
            sortie.append(lignes[i])
            i += 1
            continue
        marge, genre, numero, _ = debut.groups()
        sortie.append(lignes[i])
        i += 1
        while i < len(lignes) and lignes[i].lstrip().startswith("#>"):
            texte = lignes[i].lstrip()[2:].lstrip()
            sortie.append(f"{marge}# {texte}" if texte.strip() else f"{marge}#\n")
            i += 1
        while i < len(lignes) and not FIN.match(lignes[i]):
            i += 1
        if i >= len(lignes):
            raise SystemExit(f"Bloc {genre} {numero} jamais refermé dans {nom_tp}")
        sortie.append(
            f'{marge}raise NotImplementedError(\n'
            f'{marge}    "{genre} {numero} — à compléter. La consigne est juste au-dessus, "\n'
            f'{marge}    "le détail dans {nom_tp}/README.md"\n'
            f'{marge})\n'
        )
        sortie.append(lignes[i])
        i += 1
    return "".join(sortie)


def paires() -> list[tuple[Path, Path, str]]:
    trouvees = []
    for fichier in sorted((RACINE / "corrige").rglob("*.py")):
        relatif = fichier.relative_to(RACINE / "corrige")
        trouvees.append((fichier, RACINE / relatif, relatif.parts[0]))
    return trouvees


def principal(argv=None) -> int:
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument("--verifier", action="store_true",
                           help="contrôler que les exercices correspondent au corrigé")
    args = analyseur.parse_args(argv)

    ecarts, ecrits, blocs = [], 0, 0
    for corrige, exercice, nom_tp in paires():
        source = corrige.read_text(encoding="utf-8")
        blocs += sum(1 for ligne in source.splitlines() if DEBUT.match(ligne))
        attendu = squelette(source, nom_tp)
        if args.verifier:
            actuel = exercice.read_text(encoding="utf-8") if exercice.exists() else ""
            if actuel != attendu:
                ecarts.append(exercice.relative_to(RACINE))
        else:
            exercice.parent.mkdir(parents=True, exist_ok=True)
            if not exercice.exists() or exercice.read_text(encoding="utf-8") != attendu:
                exercice.write_text(attendu, encoding="utf-8", newline="\n")
                ecrits += 1

    if args.verifier:
        if ecarts:
            print("Le corrigé et les fichiers d'exercice ont divergé :")
            for chemin in ecarts:
                print(f"    {chemin}")
            print("\n  Corrigez dans corrige/, puis relancez sans --verifier.")
            return 1
        print(f"Corrigé et exercices en phase ({len(paires())} fichiers, {blocs} blocs).")
        return 0
    print(f"{ecrits} fichier(s) régénéré(s) sur {len(paires())}, {blocs} blocs à compléter.")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
