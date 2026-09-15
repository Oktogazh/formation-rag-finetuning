"""Tests du TP 4. Aucun n'entraîne quoi que ce soit."""

import random

import pytest

from commun import corpus
from commun.augmenter import GlossaireVectoriel
from commun.recherche import rechercher_lexical


@pytest.fixture(scope="module")
def memoire():
    return corpus.charger_memoire_brute()


@pytest.fixture(scope="module")
def index_glossaire():
    """L'encodeur factice du conftest, avec le seuil qui lui correspond."""
    return GlossaireVectoriel(corpus.charger_glossaire())


@pytest.mark.code
def test_code1_l_exemple_raft_a_la_bonne_forme(exercice, memoire, index_glossaire):
    segment = memoire[5]
    exemple = exercice.exemple_raft(segment, memoire, index_glossaire,
                                    k=3, p_oracle=1.0, rng=random.Random(1))
    messages = exemple["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[-1]["content"] == segment["tgt"]
    assert segment["src"] in messages[1]["content"]
    assert messages[1]["content"].count("sv:") == 4, "trois voisins, plus le segment"
    assert segment["tgt"] not in messages[1]["content"], (
        "le segment ne doit jamais voir sa propre traduction dans son contexte")


@pytest.mark.code
def test_code1_les_distracteurs_ne_sont_pas_les_vrais_voisins(exercice, memoire,
                                                              index_glossaire):
    segment = memoire[5]
    autres = [s for s in memoire if s["id"] != segment["id"]]
    proches = {s["id"] for s in rechercher_lexical(segment["src"], autres, k=10)}
    contexte = exercice.exemple_raft(segment, memoire, index_glossaire, k=3,
                                     p_oracle=0.0, rng=random.Random(7))["messages"][1]["content"]
    injectes = {s["id"] for s in autres if s["src"] in contexte}
    assert injectes, "il doit y avoir des voisins, même faux"
    assert not (injectes & proches), "avec p_oracle=0, aucun vrai voisin ne doit passer"


@pytest.mark.code
def test_code2_la_configuration_lora_est_coherente(exercice):
    config = exercice.config_lora(rang=8)
    assert set(config) == {"rang", "alpha", "dropout", "cibles"}
    assert config["alpha"] == 2 * config["rang"]
    assert 0 <= config["dropout"] < 0.5
    assert "q_proj" in config["cibles"] and "v_proj" in config["cibles"]
    assert exercice.config_lora(rang=16)["alpha"] == 32


@pytest.mark.bonus
@pytest.mark.code
def test_bonus4_l_exemple_sans_contexte_n_a_pas_de_memoire(exercice, memoire,
                                                           index_glossaire):
    exemple = exercice.exemple_sans_contexte(memoire[2], index_glossaire)
    contexte = exemple["messages"][1]["content"]
    assert "Memoire de traduction" not in contexte
    assert exemple["messages"][-1]["content"] == memoire[2]["tgt"]


# --- socle ------------------------------------------------------------------
def test_le_prompt_d_entrainement_est_celui_de_l_inference(memoire):
    """Le test le plus important du TP 4, et le moins spectaculaire.

    Si le format d'entraînement dérive du format d'inférence, le modèle apprend
    quelque chose qu'on ne lui redemandera jamais. Rien ne plante, la qualité
    est juste moins bonne que prévu, et personne ne sait pourquoi.
    """
    from commun.prompts import construire_messages

    messages = construire_messages(memoire[11]["src"], voisins=memoire[:3],
                                   glossaire=corpus.charger_glossaire(), consignes="Vouvoyez.")
    assert messages[0]["role"] == "system"
    assert messages[1]["content"].rstrip().endswith("fr:")
    assert "### Glossaire impose" in messages[1]["content"]


def test_le_petit_modele_est_bien_petit():
    from commun.petit_modele import MODELE_PETIT

    assert "135M" in MODELE_PETIT or "SmolLM" in MODELE_PETIT, (
        "le TP 4 doit tourner sans GPU : le modèle reste petit")
