"""Tests du TP 6."""

import pytest

from commun import corpus
from commun.verification import difference_chiffres_seulement, reparation_floue

pytest.importorskip("langgraph", reason="langgraph absent")


def etat(taux, voisin_src, segment):
    return {"taux_memoire": taux, "voisin": {"src": voisin_src}, "segment": segment}


@pytest.mark.code
def test_code1_le_routeur_reutilise_quand_seuls_les_chiffres_changent(exercice):
    assert exercice.router(etat(0.98, "Lösenordet förnyas var 14:e dag.",
                                "Lösenordet förnyas var 180:e dag.")) == "reutiliser"


@pytest.mark.code
def test_code1_le_routeur_refuse_si_un_mot_a_change(exercice):
    assert exercice.router(etat(0.96, "Certifikatet förnyas var 30:e dag.",
                                "Certifikatet förnyas var 30:e månad.")) == "recuperer"


@pytest.mark.code
def test_code1_le_routeur_passe_par_le_modele_quand_la_memoire_est_loin(exercice):
    assert exercice.router(etat(0.62, "Fakturan skickas varje vecka.",
                                "Underhållsarbeten meddelas 30 dagar.")) == "recuperer"
    assert exercice.router(etat(0.949, "Lösenordet förnyas var 14:e dag.",
                                "Lösenordet förnyas var 18:e dag.")) == "recuperer"


# --- socle ------------------------------------------------------------------
def test_la_sortie_de_boucle_a_un_compteur(exercice):
    assert exercice.apres_verification({"anomalies": ["x"], "tentatives": 1}) == "corriger"
    assert exercice.apres_verification({"anomalies": [], "tentatives": 1}) == "livrer"
    assert exercice.apres_verification({"anomalies": ["x"], "tentatives": 2}) == "livrer"


def test_le_graphe_a_la_bonne_topologie(exercice):
    aretes = {(a.source, a.target) for a in exercice.construire_graphe().get_graph().edges}
    for arete in [("analyser", "reutiliser"), ("analyser", "recuperer"),
                  ("recuperer", "traduire"), ("traduire", "verifier"),
                  ("verifier", "corriger"), ("verifier", "livrer"),
                  ("corriger", "verifier"), ("reutiliser", "livrer")]:
        assert arete in aretes, f"arête manquante : {arete[0]} → {arete[1]}"


def test_le_graphe_sans_rag_n_a_ni_recuperation_ni_reutilisation(exercice):
    noeuds = set(exercice.construire_graphe(avec_rag=False).get_graph().nodes)
    assert "recuperer" not in noeuds and "reutiliser" not in noeuds
    assert {"analyser", "traduire", "verifier", "livrer"} <= noeuds


def test_la_reparation_floue_reporte_les_chiffres():
    assert reparation_floue("var 14:e dag", "tous les 14 jours", "var 180:e dag") == \
        "tous les 180 jours"
    assert difference_chiffres_seulement("var 180:e dag", "var 14:e dag") is True
    assert difference_chiffres_seulement("var 30:e månad", "var 30:e dag") is False


def test_le_banc_supporte_l_absence_de_mesures(monkeypatch, tmp_path):
    import commun.rapport as rapport
    from commun.banc import afficher_banc

    monkeypatch.setattr(rapport, "DOSSIER_RESULTATS", tmp_path)
    assert afficher_banc() == []
