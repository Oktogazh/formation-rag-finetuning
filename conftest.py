"""Configuration partagee par toute la suite de tests.

Deux garanties, et elles comptent plus que le reste du fichier :

1. **Aucun test ne telecharge ni n'appelle un modele.** On force
   ``TP_MOTEUR=factice`` : ``commun.moteur.obtenir_moteur()`` rend alors un
   moteur deterministe, ecrit en Python pur, qui ne sort pas de la machine.
2. **Les tests marques ``modele`` sont ignores**, sauf ``--modele``. Ce sont les
   seuls qui parlent a Ollama ou a une API.

Et une commodite : les tests marques ``bonus`` ne sont joues qu'avec ``--bonus``.
Le rouge que vous voyez par defaut est donc exactement le **noyau** du TP, celui
qui tient dans la demi-journee. Les bonus sont la pour ceux qui finissent tot.
"""

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("TP_MOTEUR", "factice")
os.environ.setdefault("TP_EMBEDDINGS", "factice")
os.environ["LANGSMITH_TRACING"] = "false"


def pytest_addoption(parser):
    parser.addoption(
        "--modele",
        action="store_true",
        default=False,
        help="lancer aussi les tests qui appellent un vrai moteur (Ollama, API Mistral)",
    )
    parser.addoption(
        "--bonus",
        action="store_true",
        default=False,
        help="lancer aussi les tests des exercices bonus",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--modele"):
        saut = pytest.mark.skip(reason="test de moteur reel ; ajoutez --modele pour le lancer")
        for item in items:
            if "modele" in item.keywords:
                item.add_marker(saut)
    if not config.getoption("--bonus"):
        saut = pytest.mark.skip(reason="exercice bonus ; ajoutez --bonus pour le lancer")
        for item in items:
            if "bonus" in item.keywords:
                item.add_marker(saut)


# ---------------------------------------------------------------------------
# Charger les fonctions d'un notebook sans exécuter ses cellules de mesure
# ---------------------------------------------------------------------------
RACINE = Path(__file__).resolve().parent
DOSSIERS = {
    "tp01": "tp01-prompt", "tp02": "tp02-rag", "tp03": "tp03-langchain",
    "tp04": "tp04-lora-raft", "tp05": "tp05-mise-en-service", "tp06": "tp06-graphe",
}
GARDES = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def charger_exercice(cle: str) -> types.ModuleType:
    """Rend un module fait des seules définitions du notebook ``cle``.

    Un notebook mélange deux choses : des fonctions que vous écrivez, et des
    cellules qui les font tourner sur 80 segments. Les tests ne veulent que les
    premières — sinon lancer la suite relancerait toutes les mesures, et un
    TODO non fait empêcherait de tester les autres.

    On lit donc le fichier, on garde les imports, les fonctions, les classes et
    les constantes en MAJUSCULES, et on jette le reste.

    **Conséquence pour qui écrit un exercice** : une fonction testée prend ses
    données en argument. Si elle capture une variable de cellule en minuscules
    (``glossaire``, ``memoire``), le test lèvera un NameError — et ce serait un
    défaut du sujet, pas du stagiaire.
    """
    chemin = RACINE / DOSSIERS[cle] / f"{cle}.py"
    if not chemin.exists():
        pytest.skip(f"{chemin.name} absent")
    arbre = ast.parse(chemin.read_text(encoding="utf-8"), filename=str(chemin))
    gardes = []
    for noeud in arbre.body:
        if isinstance(noeud, GARDES):
            gardes.append(noeud)
        elif isinstance(noeud, ast.Assign) and all(
            isinstance(c, ast.Name) and c.id.isupper() for c in noeud.targets
        ):
            gardes.append(noeud)
    arbre.body = gardes
    module = types.ModuleType(cle)
    module.__file__ = str(chemin)
    sys.modules.setdefault(cle, module)
    exec(compile(arbre, str(chemin), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def exercice(request):
    """Le module de l'exercice, déduit du dossier du fichier de test."""
    cle = Path(request.fspath).parent.parent.name.split("-")[0]
    return charger_exercice(cle)
