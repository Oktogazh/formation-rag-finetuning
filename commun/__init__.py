"""Socle partage par les six TP.

Rien ici n'est a completer : ce sont les pieces sur lesquelles les exercices
s'appuient. Vous pouvez les lire, et vous devriez — surtout ``prompts.py`` et
``mesure.py``, qui decident de ce que la formation appelle « mieux ».
"""

from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_RESULTATS = RACINE / "resultats"

__all__ = ["RACINE", "DOSSIER_RESULTATS"]
