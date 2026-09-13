"""Tests du TP 3. Le modèle est remplacé par un faux."""

import pytest

from commun import corpus
from commun.recherche import rechercher_lexical
from commun.verification import Anomalie, verifier

pytest.importorskip("langchain_core", reason="langchain-core absent")
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

SEGMENT = "Abonnemanget Bas tillåter 20 förfrågningar per minut."
JUSTE = "La formule Bas autorise 20 requêtes par minute."


@pytest.fixture(scope="module")
def glossaire():
    return corpus.charger_glossaire()


@pytest.fixture(scope="module")
def chercher():
    memoire = corpus.charger_memoire_brute()
    return lambda src: rechercher_lexical(src, memoire, 2)


@pytest.mark.code
def test_code1_la_chaine_traduit(exercice, glossaire, chercher):
    modele = FakeListChatModel(responses=[JUSTE])
    chaine = exercice.construire_chaine(chercher, glossaire, modele, "Vouvoyez.")
    assert chaine.invoke(SEGMENT) == JUSTE


@pytest.mark.code
def test_code1_les_voisins_et_le_glossaire_arrivent_au_modele(exercice, glossaire, chercher):
    vus = {}

    def espion(entree):
        vus["entree"] = entree
        return AIMessage(content=JUSTE)

    exercice.construire_chaine(chercher, glossaire, RunnableLambda(espion),
                               "Vouvoyez.").invoke(SEGMENT)
    texte = str(vus["entree"])
    assert "Memoire de traduction" in texte, "les voisins doivent arriver jusqu'au modèle"
    assert "Glossaire impose" in texte, "le glossaire filtré aussi"
    assert SEGMENT in texte


# --- socle ------------------------------------------------------------------
def test_le_modele_langchain_suit_la_regle_du_socle(exercice, monkeypatch):
    monkeypatch.setenv("TP_MOTEUR", "factice")
    assert isinstance(exercice.modele_langchain(), FakeListChatModel)


def test_la_verification_attrape_les_quatre_defauts(glossaire):
    assert any(a.type == "terminologie" for a in verifier(
        SEGMENT, "La formule Bas autorise 20 demandes par minute.", glossaire))
    assert verifier(SEGMENT, JUSTE, glossaire) == []
    assert any(a.type == "chiffres" for a in verifier(
        SEGMENT, "La formule Bas autorise 50 requêtes par minute.", glossaire))
    assert any(a.type == "tutoiement" for a in verifier(
        "Du kan skapa.", "Tu peux créer une formule.", glossaire))


def test_une_anomalie_se_lit_comme_une_phrase():
    assert str(Anomalie("chiffres", "180 attendu")) == "chiffres : 180 attendu"


def test_la_reparation_locale_remet_les_bons_chiffres():
    from commun.verification import reparer_chiffres

    assert reparer_chiffres("var 180:e dag", "tous les 14 jours") == "tous les 180 jours"
    assert reparer_chiffres("inga siffror", "aucune") == "aucune"
