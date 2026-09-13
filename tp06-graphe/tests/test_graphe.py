"""Tests du TP 6."""

import pytest

from commun import corpus
from commun.moteur import MoteurFactice
from tp02.recherche import rechercher_lexical
from tp06.etat import etat_initial
from tp06.graphe import apres_verification, router
from tp06.noeuds import construire_noeuds, difference_chiffres_seulement, reparation_floue

pytest.importorskip("langgraph", reason="langgraph absent")


def etat(taux, voisin_src, segment):
    return {"taux_memoire": taux, "voisin": {"src": voisin_src}, "segment": segment}


@pytest.mark.todo
def test_todo1_le_routeur_reutilise_quand_seuls_les_chiffres_changent():
    assert router(etat(0.98, "Lösenordet förnyas var 14:e dag.",
                       "Lösenordet förnyas var 180:e dag.")) == "reutiliser"


@pytest.mark.todo
def test_todo1_le_routeur_refuse_de_reutiliser_si_un_mot_change():
    assert router(etat(0.96, "Certifikatet förnyas var 30:e dag.",
                       "Certifikatet förnyas var 30:e månad.")) == "recuperer"


@pytest.mark.todo
def test_todo1_le_routeur_passe_par_le_modele_quand_la_memoire_est_loin():
    assert router(etat(0.62, "Fakturan skickas varje vecka.",
                       "Underhållsarbeten meddelas 30 dagar i förväg.")) == "recuperer"
    assert router(etat(0.949, "Lösenordet förnyas var 14:e dag.",
                       "Lösenordet förnyas var 18:e dag.")) == "recuperer"


@pytest.mark.todo
def test_todo2_la_boucle_de_correction_a_un_compteur():
    assert apres_verification({"anomalies": ["chiffres : faux"], "tentatives": 1}) == "corriger"
    assert apres_verification({"anomalies": [], "tentatives": 1}) == "livrer"
    assert apres_verification({"anomalies": ["chiffres : faux"], "tentatives": 2}) == "livrer"


@pytest.mark.todo
def test_todo3_le_graphe_a_la_bonne_topologie(noeuds):
    from tp06.graphe import construire_graphe

    graphe = construire_graphe(noeuds)
    aretes = {(a.source, a.target) for a in graphe.get_graph().edges}
    for arete in [("analyser", "reutiliser"), ("analyser", "recuperer"),
                  ("recuperer", "traduire"), ("traduire", "verifier"),
                  ("verifier", "corriger"), ("verifier", "livrer"),
                  ("corriger", "verifier"), ("reutiliser", "livrer")]:
        assert arete in aretes, f"arete manquante : {arete[0]} -> {arete[1]}"


@pytest.mark.todo
def test_todo3_une_repetition_ne_coute_aucun_appel(noeuds, segments):
    from tp06.graphe import construire_graphe

    graphe = construire_graphe(noeuds)
    repetition = next(s for s in segments if s["categorie"] == "repetition")
    final = graphe.invoke(etat_initial(repetition))
    assert "reutiliser" in final["chemin"]
    assert final["appels"] == 0, "la memoire suffit : aucun appel au modele"
    assert final["traduction"] == repetition["tgt"], "et la reponse est exacte"


@pytest.mark.todo
def test_todo3_un_segment_nouveau_passe_par_le_modele(noeuds, segments):
    from tp06.graphe import construire_graphe

    graphe = construire_graphe(noeuds)
    nouveau = next(s for s in segments if s["categorie"] == "nouveau")
    final = graphe.invoke(etat_initial(nouveau))
    assert "recuperer" in final["chemin"] and "traduire" in final["chemin"]
    assert final["appels"] >= 1


@pytest.mark.bonus
@pytest.mark.todo
def test_bonus6_le_graphe_sans_rag_n_a_pas_de_recuperation(noeuds, segments):
    from tp06.graphe import construire_graphe

    graphe = construire_graphe(noeuds, avec_rag=False)
    final = graphe.invoke(etat_initial(segments[0]))
    assert "recuperer" not in final["chemin"]
    assert "reutiliser" not in final["chemin"]
    assert final["appels"] >= 1
    assert final["voisins"] == []


# --- socle ------------------------------------------------------------------
@pytest.fixture(scope="module")
def segments():
    return corpus.charger_evaluation()


@pytest.fixture(scope="module")
def noeuds(segments):
    memoire = corpus.charger_memoire_brute()
    glossaire = corpus.charger_glossaire()
    return construire_noeuds(
        memoire, glossaire, MoteurFactice(),
        lambda src: rechercher_lexical(src, memoire, 3), "Vouvoyez.",
    )


def test_la_reparation_floue_reporte_les_chiffres():
    assert reparation_floue("var 14:e dag", "tous les 14 jours", "var 180:e dag") == \
        "tous les 180 jours"
    assert reparation_floue("inga siffror", "aucun", "inga siffror") == "aucun"


def test_la_difference_se_limite_ou_non_aux_chiffres():
    assert difference_chiffres_seulement("var 180:e dag", "var 14:e dag") is True
    assert difference_chiffres_seulement("var 30:e månad", "var 30:e dag") is False


def test_l_etat_initial_est_complet(segments):
    etat = etat_initial(segments[0])
    assert etat["segment"] == segments[0]["src"]
    assert etat["appels"] == 0 and etat["chemin"] == [] and etat["tentatives"] == 0


def test_le_banc_supporte_l_absence_de_mesures(monkeypatch, tmp_path):
    import commun.rapport as rapport
    from tp06.banc import afficher_banc

    monkeypatch.setattr(rapport, "DOSSIER_RESULTATS", tmp_path)
    assert afficher_banc() == []
