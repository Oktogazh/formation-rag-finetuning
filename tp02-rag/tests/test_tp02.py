"""Tests du TP 2. Aucun n'appelle de modèle."""

import pytest

from commun import corpus
from commun.recherche import similarite

PROCHE = "Abonnemanget Företag tillåter 28 förfrågningar per minut."
GLOSSAIRE = "Hastighetsgränsen är satt till 60 förfrågningar per minut."


@pytest.fixture(scope="module")
def memoire():
    return corpus.charger_memoire_brute()


@pytest.mark.code
def test_code1_la_memoire_ne_garde_que_les_segments_valides(exercice, memoire):
    valides = exercice.charger_memoire(memoire)
    assert len(valides) == 444
    assert {s["statut"] for s in valides} == {"valide"}


@pytest.mark.code
def test_code2_la_recherche_lexicale_remonte_le_bon_voisin(exercice, memoire):
    voisins = exercice.rechercher_lexical(PROCHE, memoire, k=3)
    assert len(voisins) == 3
    scores = [v["score"] for v in voisins]
    assert scores == sorted(scores, reverse=True), "les voisins doivent être rangés"
    assert scores[0] >= 0.95, "le plus proche a plus de 95 % de similarité"
    attendu = max(memoire, key=lambda s: similarite(PROCHE, s["src"]))
    assert voisins[0]["src"] == attendu["src"]


@pytest.mark.code
def test_code3_le_glossaire_est_filtre_sur_le_segment(exercice):
    glossaire = corpus.charger_glossaire()
    retenus = exercice.glossaire_pertinent(GLOSSAIRE, glossaire)
    assert {t["sv"] for t in retenus} == {"hastighetsgräns", "förfråg"}
    assert exercice.glossaire_pertinent("Fakturan skickas varje vecka.", glossaire) == []


@pytest.mark.code
def test_code4_le_prompt_augmente_contient_voisins_et_glossaire(exercice, memoire):
    messages = exercice.construire(PROCHE, memoire, corpus.charger_glossaire(), k=2)
    texte = messages[1]["content"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "### Glossaire impose" in texte and "### Memoire de traduction" in texte
    assert texte.count("sv:") == 3, "deux voisins, plus le segment à traduire"
    assert texte.rstrip().endswith("fr:"), "le segment à traduire vient en dernier"
    assert "vous" in messages[0]["content"].lower(), "la consigne de style doit être là"


@pytest.mark.bonus
@pytest.mark.code
def test_bonus2_la_recherche_hybride_melange_les_deux_scores(exercice, memoire):
    from commun.embeddings import EncodeurFactice
    from commun.recherche import Index

    echantillon = memoire[:40]
    index = Index.construire(echantillon, encodeur=EncodeurFactice(), cache=False)
    cible = echantillon[3]
    voisins = exercice.rechercher_hybride(cible["src"], echantillon, index, k=3, alpha=0.5)
    assert voisins[0]["id"] == cible["id"]
    assert voisins[0]["score"] == pytest.approx(1.0, abs=1e-3)


# --- socle ------------------------------------------------------------------
def test_le_corpus_respecte_son_schema(memoire):
    assert len(memoire) == 444
    for segment in memoire[:20]:
        assert set(segment) >= {"id", "src", "tgt", "domaine", "date", "statut"}
    evaluation = corpus.charger_evaluation()
    categories = {}
    for segment in evaluation:
        categories[segment["categorie"]] = categories.get(segment["categorie"], 0) + 1
    assert categories == {"repetition": 21, "piege": 20, "fuzzy": 18, "nouveau": 21}


def test_le_rappel_se_mesure_sans_modele():
    from commun.recherche import rappel, rechercher_lexical

    memoire = corpus.charger_memoire_brute()
    segments = corpus.charger_evaluation(n=8)
    part = rappel(segments, lambda src: rechercher_lexical(src, memoire, 3))
    assert 0.0 <= part <= 100.0
