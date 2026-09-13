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

import os

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
