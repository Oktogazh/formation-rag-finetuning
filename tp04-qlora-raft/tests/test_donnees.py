"""Tests du TP 4. Aucun n'entraine quoi que ce soit."""

import random

import pytest

from commun import corpus
from commun.prompts import construire_messages
from tp04.donnees import exemple_raft, exemple_sans_contexte, separer
from tp04.entrainer import config_entrainement, config_lora


@pytest.fixture(scope="module")
def memoire():
    return corpus.charger_memoire_brute()


@pytest.fixture(scope="module")
def glossaire():
    return corpus.charger_glossaire()


@pytest.mark.todo
def test_todo1_la_separation_ne_laisse_pas_fuiter_l_evaluation(memoire):
    entrainement, validation, ecartes = separer(memoire)
    assert len(entrainement) + len(validation) + len(ecartes) == len(memoire)
    assert len(ecartes) == 50, "50 segments sont trop proches de l'evaluation (seuil 0,98)"
    assert len(validation) == 39 and len(entrainement) == 355
    assert not ({s["id"] for s in entrainement} & {s["id"] for s in validation})

    from tp02.recherche import similarite

    sources_eval = [s["src"] for s in corpus.charger_evaluation()]
    pire = max(
        max(similarite(s["src"], e) for e in sources_eval) for s in entrainement[:60]
    )
    assert pire < 0.98


@pytest.mark.todo
def test_todo1_la_separation_est_reproductible(memoire):
    premier = [s["id"] for s in separer(memoire)[0]]
    second = [s["id"] for s in separer(memoire)[0]]
    assert premier == second, "la graine doit rendre le partage identique d'un appel a l'autre"


@pytest.mark.todo
def test_todo2_l_exemple_raft_a_la_bonne_forme(memoire, glossaire):
    segment = memoire[5]
    exemple = exemple_raft(segment, memoire, glossaire, k=3, p_oracle=1.0,
                           rng=random.Random(1))
    messages = exemple["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[-1]["content"] == segment["tgt"]
    assert segment["src"] in messages[1]["content"]
    assert messages[1]["content"].count("sv:") == 4, "trois voisins, plus le segment"
    assert segment["tgt"] not in messages[1]["content"], (
        "le segment ne doit jamais voir sa propre traduction dans le contexte"
    )


@pytest.mark.todo
def test_todo2_les_distracteurs_ne_sont_pas_les_vrais_voisins(memoire, glossaire):
    from tp02.recherche import rechercher_lexical

    segment = memoire[5]
    autres = [s for s in memoire if s["id"] != segment["id"]]
    proches = {s["id"] for s in rechercher_lexical(segment["src"], autres, k=10)}
    exemple = exemple_raft(segment, memoire, glossaire, k=3, p_oracle=0.0,
                           rng=random.Random(7))
    contexte = exemple["messages"][1]["content"]
    injectes = {s["id"] for s in autres if s["src"] in contexte}
    assert injectes, "il doit y avoir des voisins, meme faux"
    assert not (injectes & proches), "avec p_oracle=0, aucun vrai voisin ne doit passer"


@pytest.mark.todo
def test_todo3_la_configuration_lora_est_coherente():
    config = config_lora(rang=16)
    assert set(config) == {"rang", "alpha", "dropout", "cibles"}
    assert config["alpha"] == 2 * config["rang"]
    assert 0 <= config["dropout"] < 0.5
    assert "q_proj" in config["cibles"] and "v_proj" in config["cibles"]
    assert config_lora(rang=8)["alpha"] == 16


@pytest.mark.bonus
@pytest.mark.todo
def test_bonus4_l_exemple_sans_contexte_n_a_pas_de_memoire(memoire, glossaire):
    exemple = exemple_sans_contexte(memoire[2], glossaire)
    contexte = exemple["messages"][1]["content"]
    assert "Memoire de traduction" not in contexte
    assert exemple["messages"][-1]["content"] == memoire[2]["tgt"]


# --- socle ------------------------------------------------------------------
def test_le_prompt_d_entrainement_est_celui_de_l_inference(memoire, glossaire):
    """Le test le plus important du TP 4, et le moins spectaculaire.

    Si le format d'entrainement derive du format d'inference, le modele apprend
    quelque chose qu'on ne lui redemandera jamais. Rien ne plante, la qualite
    est juste moins bonne que prevu — et personne ne sait pourquoi.
    """
    segment = memoire[11]
    voisins = memoire[:3]
    attendu = construire_messages(segment["src"], voisins, glossaire, "Vouvoyez.")
    assert attendu[0]["role"] == "system"
    assert attendu[1]["content"].rstrip().endswith("fr:")
    assert "### Glossaire impose" in attendu[1]["content"]


def test_la_configuration_d_entrainement_calcule_ses_iterations():
    config = config_entrainement(epoques=2, exemples=355)
    assert config["iterations"] == (355 // 4) * 2
    assert config["pas_apprentissage"] == pytest.approx(2e-4)
