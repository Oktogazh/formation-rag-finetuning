"""Tests du TP 1. Rouges tant que les TODO ne sont pas faits, verts apres."""

import pytest

from commun import corpus, mesure, moteur, prompts
from tp01.prompt import consigne_systeme, exemples_cibles, exemples_manuels, messages_pour

MOTS_CLES = {
    "vouvoiement": ("vous", "vouvoi"),
    "registre": ("registre", "neutre", "professionnel", "familier", "familière", "familiere"),
    "noms propres": ("nom", "formule", "produit", "marque", "företag", "foretag"),
    "chiffres": ("chiffre", "nombre", "nombres", "numerique", "numérique"),
}


@pytest.mark.todo
def test_todo1_consigne_couvre_les_quatre_regles():
    consigne = consigne_systeme()
    assert isinstance(consigne, str)
    assert len(consigne) >= 120, "une consigne de moins de 120 caracteres ne dit pas grand-chose"
    minuscule = consigne.lower()
    manquants = [
        sujet for sujet, mots in MOTS_CLES.items()
        if not any(mot in minuscule for mot in mots)
    ]
    assert not manquants, f"la consigne ne parle pas de : {', '.join(manquants)}"


@pytest.mark.todo
def test_todo2_exemples_viennent_de_la_memoire():
    exemples = exemples_manuels()
    assert len(exemples) == 3, "trois exemples, ni deux ni dix"
    memoire = corpus.charger_memoire_brute()
    sources = {s["src"] for s in memoire}
    cibles = {s["tgt"] for s in memoire}
    for exemple in exemples:
        assert set(exemple) >= {"src", "tgt"}, "chaque exemple a les cles src et tgt"
        assert exemple["src"] in sources, (
            f"« {exemple['src'][:40]}… » n'est pas dans tm.jsonl : recopiez, n'inventez pas"
        )
        assert exemple["tgt"] in cibles, (
            "la traduction doit etre celle de la memoire, accents compris"
        )
    evaluation = {s["src"] for s in corpus.charger_evaluation()}
    assert not evaluation & {e["src"] for e in exemples}, (
        "un exemple pris dans le jeu d'evaluation fausse toute la mesure"
    )


@pytest.mark.todo
def test_todo1_et_2_le_prompt_grossit_avec_les_variantes():
    segment = "Abonnemanget Bas tillåter 20 förfrågningar per minut."
    tailles = [
        len(str(messages_pour(segment, variante)))
        for variante in ("nu", "consigne", "exemples")
    ]
    assert tailles[0] < tailles[1] < tailles[2]


@pytest.mark.bonus
@pytest.mark.todo
def test_bonus1_exemples_cibles_partagent_le_domaine():
    exemples = exemples_cibles("cle_rotation")
    assert len(exemples) == 3
    assert {e["domaine"] for e in exemples} == {"cle_rotation"}


# --- socle : verts des le depart, ils valident le materiel fourni ------------
def test_le_prompt_nu_contient_le_segment_et_rien_de_plus():
    messages = messages_pour("Fakturan skickas varje vecka.", "nu")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Fakturan skickas varje vecka." in messages[1]["content"]
    assert "Memoire de traduction" not in messages[1]["content"]
    assert "Glossaire" not in messages[1]["content"]


def test_la_mesure_attrape_un_terme_interdit():
    glossaire = corpus.charger_glossaire()
    source = "Hastighetsgränsen är satt till 60 förfrågningar per minut."
    attendus, ok, fautes = mesure.terminologie(
        source, "La limite de vitesse est fixée à 60 demandes par minute.", glossaire
    )
    assert attendus == 2 and ok == 0 and len(fautes) == 2


def test_la_mesure_attrape_un_chiffre_et_un_tutoiement():
    assert mesure.chiffres("var 180:e dag", "tous les 180 jours") is True
    assert mesure.chiffres("var 180:e dag", "tous les 14 jours") is False
    assert mesure.vouvoiement("Tu peux créer une formule.") is False
    assert mesure.vouvoiement("Vous pouvez créer une formule.") is True
    assert mesure.vouvoiement("La facture est envoyée.") is None


def test_le_moteur_factice_est_deterministe():
    machine = moteur.MoteurFactice()
    messages = prompts.construire_messages("Fakturan skickas varje vecka.")
    premier = machine.generer(messages).texte
    for _ in range(3):
        assert machine.generer(messages).texte == premier
