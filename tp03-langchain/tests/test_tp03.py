"""Tests du TP 3. Le modèle est remplacé par un faux."""

import os

import pytest

from commun import corpus
from commun.augmenter import GlossaireVectoriel
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
def index_glossaire(glossaire):
    """L'encodeur factice du conftest, avec le seuil qui lui correspond."""
    return GlossaireVectoriel(glossaire)


@pytest.fixture(scope="module")
def chercher():
    memoire = corpus.charger_memoire_brute()
    return lambda src: rechercher_lexical(src, memoire, 2)


@pytest.mark.code
def test_code1_la_chaine_traduit(exercice, index_glossaire, chercher):
    modele = FakeListChatModel(responses=[JUSTE])
    chaine = exercice.construire_chaine(chercher, index_glossaire, modele, "Vouvoyez.")
    assert chaine.invoke(SEGMENT) == JUSTE


@pytest.mark.code
def test_code1_les_voisins_et_le_glossaire_arrivent_au_modele(exercice, index_glossaire,
                                                            chercher):
    vus = {}

    def espion(entree):
        vus["entree"] = entree
        return AIMessage(content=JUSTE)

    exercice.construire_chaine(chercher, index_glossaire, RunnableLambda(espion),
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


# --- socle : le traçage (commun/tracage.py) ---------------------------------
#
# Aucun de ces tests ne sort de la machine : la sonde de région est remplacée
# par une fonction, et la relecture de .env est neutralisée — sinon la clé
# personnelle du formateur ferait passer ou échouer la suite selon le poste.
from commun import tracage  # noqa: E402


@pytest.fixture
def tracage_isole(monkeypatch):
    monkeypatch.setattr(tracage, "_relire_env", lambda: None)
    for variable in ("LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT",
                     "LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    return monkeypatch


def test_sans_cle_le_tracage_se_coupe_au_lieu_de_planter(tracage_isole):
    etat = tracage.activer(silencieux=True)
    assert not etat.actif
    assert "cle" in etat.raison
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_false_est_respecte(tracage_isole):
    tracage_isole.setenv("LANGSMITH_API_KEY", "lsv2_pt_peu_importe")
    tracage_isole.setenv("LANGSMITH_TRACING", "false")
    etat = tracage.activer(silencieux=True)
    assert not etat.actif and "false" in etat.raison


def test_une_cle_refusee_partout_ne_laisse_pas_le_tracage_allume(tracage_isole):
    tracage_isole.setattr(tracage, "_cle_valide_sur", lambda serveur, cle: False)
    tracage_isole.setenv("LANGSMITH_API_KEY", "lsv2_pt_fausse")
    etat = tracage.activer(silencieux=True)
    assert not etat.actif
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_la_cle_reglee_en_argument_est_le_chemin_normal(tracage_isole):
    """RÉG 0 : la clé se règle en variable de notebook, pas dans .env."""
    tracage_isole.setattr(tracage, "_cle_valide_sur",
                          lambda serveur, cle: serveur == tracage.SERVEURS["US"])
    etat = tracage.activer(cle="lsv2_pt_reglee_en_cellule",
                           projet="formation-helios-test", silencieux=True)
    assert etat.actif
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_pt_reglee_en_cellule"
    assert os.environ["LANGSMITH_PROJECT"] == "formation-helios-test"


def test_la_region_se_trouve_toute_seule(tracage_isole):
    """Le piège numéro un du TP : un compte européen, un endpoint américain."""
    tracage_isole.setattr(tracage, "_cle_valide_sur",
                          lambda serveur, cle: serveur == tracage.SERVEURS["EU"])
    tracage_isole.setenv("LANGSMITH_API_KEY", "lsv2_pt_europeenne")
    etat = tracage.activer(projet="formation-helios-test", silencieux=True)
    assert etat.actif and etat.region == "EU"
    assert os.environ["LANGSMITH_ENDPOINT"] == tracage.SERVEURS["EU"]
    assert os.environ["LANGSMITH_PROJECT"] == "formation-helios-test"
    assert etat.application == tracage.APPLICATIONS["EU"]


def test_un_endpoint_pose_a_la_main_court_circuite_la_sonde(tracage_isole):
    def interdit(serveur, cle):
        raise AssertionError("la sonde ne doit pas tourner si l'endpoint est imposé")

    tracage_isole.setattr(tracage, "_cle_valide_sur", interdit)
    tracage_isole.setenv("LANGSMITH_ENDPOINT", tracage.SERVEURS["US"])
    assert tracage.trouver_region("lsv2_pt_peu_importe") == ("US", tracage.SERVEURS["US"])


def test_le_decorateur_inerte_se_pose_des_deux_facons():
    """Sans le paquet langsmith, @traceable doit s'effacer, pas casser le notebook."""
    @tracage._traceable_inerte
    def nue(x):
        return x + 1

    @tracage._traceable_inerte(name="avec-arguments")
    def parametree(x):
        return x + 2

    assert nue(1) == 2 and parametree(1) == 3
