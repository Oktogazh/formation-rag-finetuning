"""Tests du TP 3. Aucun n'appelle un vrai modele : tout passe par un faux."""

import pytest

from commun import corpus
from tp02.memoire import charger_memoire
from tp02.recherche import rechercher_lexical
from tp03.verification import Anomalie, reparer_sans_modele, verifier

langchain_core = pytest.importorskip("langchain_core", reason="langchain-core absent")
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

SEGMENT = "Abonnemanget Bas tillåter 20 förfrågningar per minut."
FAUX = "La formule Bas autorise 20 demandes par minute."
JUSTE = "La formule Bas autorise 20 requêtes par minute."


@pytest.fixture(scope="module")
def glossaire():
    return corpus.charger_glossaire()


@pytest.fixture
def chercher():
    memoire = charger_memoire() if _todo1_fait() else corpus.charger_memoire_brute()
    return lambda src: rechercher_lexical(src, memoire, 2)


def _todo1_fait() -> bool:
    try:
        charger_memoire()
        return True
    except NotImplementedError:
        return False


@pytest.mark.todo
def test_todo1_la_chaine_traduit_et_voit_les_voisins(glossaire, chercher):
    from tp03.chaine import construire_chaine

    modele = FakeListChatModel(responses=[JUSTE])
    chaine = construire_chaine(chercher, glossaire, modele, "Vouvoyez.")
    assert chaine.invoke(SEGMENT) == JUSTE

    vus = {}

    def espion(entree):
        vus["entree"] = entree
        return AIMessage(content=JUSTE)

    construire_chaine(chercher, glossaire, RunnableLambda(espion), "Vouvoyez.").invoke(SEGMENT)
    texte = str(vus["entree"])
    assert "Memoire de traduction" in texte, "les voisins doivent arriver jusqu'au modele"
    assert "Glossaire impose" in texte, "le glossaire filtre aussi"
    assert SEGMENT in texte


@pytest.mark.todo
def test_todo2_la_verification_attrape_les_quatre_defauts(glossaire):
    anomalies = verifier(SEGMENT, FAUX, glossaire)
    assert any(a.type == "terminologie" for a in anomalies)
    assert verifier(SEGMENT, JUSTE, glossaire) == []

    chiffre_faux = "La formule Bas autorise 50 requêtes par minute."
    assert any(a.type == "chiffres" for a in verifier(SEGMENT, chiffre_faux, glossaire))

    tutoie = "Tu peux créer une formule."
    assert any(a.type == "tutoiement" for a in verifier("Du kan skapa.", tutoie, glossaire))

    balise = any(
        a.type == "balises"
        for a in verifier("Kod {0} returneras.", "Le code est renvoyé.", glossaire)
    )
    assert balise, "les marqueurs {0} doivent etre conserves"


@pytest.mark.todo
def test_todo3_la_boucle_corrige_puis_s_arrete(glossaire, chercher):
    from tp03.boucle import traduire_et_corriger
    from tp03.chaine import chaine_de_correction, construire_chaine

    modele = FakeListChatModel(responses=[FAUX, JUSTE])
    chaine = construire_chaine(chercher, glossaire, modele, "")
    sortie = traduire_et_corriger(SEGMENT, chaine, chaine_de_correction(modele), glossaire)
    assert sortie.appels == 2, "un appel pour traduire, un pour corriger"
    assert sortie.texte == JUSTE
    assert sortie.anomalies == []


@pytest.mark.todo
def test_todo3_la_boucle_ne_tourne_pas_indefiniment(glossaire, chercher):
    from tp03.boucle import traduire_et_corriger
    from tp03.chaine import chaine_de_correction, construire_chaine

    modele = FakeListChatModel(responses=[FAUX])
    chaine = construire_chaine(chercher, glossaire, modele, "")
    sortie = traduire_et_corriger(
        SEGMENT, chaine, chaine_de_correction(modele), glossaire, max_tentatives=2
    )
    assert sortie.appels == 2
    assert sortie.anomalies, "on livre avec le defaut plutot que de boucler"


@pytest.mark.bonus
@pytest.mark.todo
def test_bonus3_la_reparation_locale_remet_les_bons_chiffres():
    assert reparer_sans_modele("var 180:e dag", "tous les 14 jours") == "tous les 180 jours"
    assert reparer_sans_modele("var 180:e dag", "tous les 180 jours") == "tous les 180 jours"
    assert reparer_sans_modele("inga siffror", "aucune") == "aucune"


# --- socle ------------------------------------------------------------------
def test_le_modele_langchain_suit_la_regle_du_socle(monkeypatch):
    from tp03.chaine import modele_langchain

    monkeypatch.setenv("TP_MOTEUR", "factice")
    assert isinstance(modele_langchain(), FakeListChatModel)


def test_une_anomalie_se_lit_comme_une_phrase():
    assert str(Anomalie("chiffres", "180 attendu")) == "chiffres : 180 attendu"
