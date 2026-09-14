"""Tests du TP 1. Rouges tant que l'exercice CODE n'est pas fait."""

import pytest

from commun import corpus, mesure, moteur, prompts

SUJETS = {
    "vouvoiement": ("vous", "vouvoi"),
    "registre": ("registre", "neutre", "professionnel", "familier", "familière", "familiere"),
    "noms propres": ("nom", "formule", "produit", "marque", "företag", "foretag"),
    "chiffres": ("chiffre", "nombre", "nombres", "numérique", "numerique"),
}


@pytest.mark.code
def test_code1_la_consigne_couvre_les_quatre_regles(exercice):
    consigne = exercice.consigne_systeme()
    assert isinstance(consigne, str)
    assert len(consigne) >= 120, "une consigne de moins de 120 caractères ne dit pas grand-chose"
    minuscule = consigne.lower()
    manquants = [s for s, mots in SUJETS.items() if not any(m in minuscule for m in mots)]
    assert not manquants, f"la consigne ne parle pas de : {', '.join(manquants)}"


# --- socle : verts dès le départ, ils valident ce qui est fourni ------------
def test_le_prompt_nu_contient_le_segment_et_rien_de_plus():
    messages = prompts.construire_messages("Fakturan skickas varje vecka.")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Fakturan skickas varje vecka." in messages[1]["content"]
    assert "Memoire de traduction" not in messages[1]["content"]
    assert messages[1]["content"].rstrip().endswith("fr:")


def test_la_mesure_attrape_un_terme_interdit():
    glossaire = corpus.charger_glossaire()
    source = "Hastighetsgränsen är satt till 60 förfrågningar per minut."
    attendus, ok, fautes = mesure.terminologie(
        source, "La limite de vitesse est fixée à 60 demandes par minute.", glossaire)
    assert (attendus, ok) == (2, 0) and len(fautes) == 2


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
    assert all(machine.generer(messages).texte == premier for _ in range(3))
