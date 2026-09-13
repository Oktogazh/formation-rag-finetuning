"""Tests du TP 2."""

import pytest

from commun import corpus
from commun.embeddings import EncodeurFactice
from tp02.augmenter import construire, glossaire_pertinent
from tp02.memoire import charger_memoire
from tp02.recherche import Index, rechercher_lexical, similarite

SEGMENT_PROCHE = "Abonnemanget Företag tillåter 28 förfrågningar per minut."
SEGMENT_GLOSSAIRE = "Hastighetsgränsen är satt till 60 förfrågningar per minut."


@pytest.fixture(scope="module")
def memoire():
    return corpus.charger_memoire_brute()


@pytest.mark.todo
def test_todo1_la_memoire_ne_garde_que_les_segments_valides():
    memoire = charger_memoire()
    assert len(memoire) == 444
    assert {s["statut"] for s in memoire} == {"valide"}


@pytest.mark.todo
def test_todo2_la_recherche_lexicale_remonte_le_bon_voisin(memoire):
    voisins = rechercher_lexical(SEGMENT_PROCHE, memoire, k=3)
    assert len(voisins) == 3
    scores = [v["score"] for v in voisins]
    assert scores == sorted(scores, reverse=True), "les voisins doivent etre ranges"
    assert scores[0] >= 0.95, "le voisin le plus proche a plus de 95 % de similarite"
    assert voisins[0]["src"] == max(memoire, key=lambda s: similarite(SEGMENT_PROCHE, s["src"]))["src"]


@pytest.mark.todo
def test_todo3_la_recherche_dense_remonte_le_segment_identique(memoire):
    echantillon = memoire[:40]
    index = Index.construire(echantillon, encodeur=EncodeurFactice(), cache=False)
    cible = echantillon[7]
    voisins = index.chercher(cible["src"], k=3)
    assert len(voisins) == 3
    assert voisins[0]["id"] == cible["id"], "un segment doit se retrouver lui-meme en premier"
    assert voisins[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert voisins[0]["score"] >= voisins[1]["score"] >= voisins[2]["score"]


@pytest.mark.todo
def test_todo4_le_glossaire_est_filtre_sur_le_segment():
    glossaire = corpus.charger_glossaire()
    retenus = glossaire_pertinent(SEGMENT_GLOSSAIRE, glossaire)
    assert {t["sv"] for t in retenus} == {"hastighetsgräns", "förfråg"}
    assert glossaire_pertinent("Fakturan skickas varje vecka.", glossaire) == []


@pytest.mark.todo
def test_todo5_le_prompt_augmente_contient_voisins_et_glossaire(memoire):
    voisins = rechercher_lexical(SEGMENT_PROCHE, memoire, k=2)
    glossaire = glossaire_pertinent(SEGMENT_PROCHE, corpus.charger_glossaire())
    messages = construire(SEGMENT_PROCHE, voisins, glossaire, "Vouvoyez.")
    texte = messages[1]["content"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Vouvoyez." in messages[0]["content"]
    assert "### Glossaire impose" in texte and "### Memoire de traduction" in texte
    for voisin in voisins:
        assert voisin["tgt"] in texte
    assert texte.rstrip().endswith("fr:"), "le segment a traduire vient en dernier"


@pytest.mark.bonus
@pytest.mark.todo
def test_bonus2_la_recherche_hybride_melange_les_deux_scores(memoire):
    echantillon = memoire[:40]
    index = Index.construire(echantillon, encodeur=EncodeurFactice(), cache=False)
    cible = echantillon[3]
    voisins = index.chercher_hybride(cible["src"], k=3, alpha=0.5)
    assert voisins[0]["id"] == cible["id"]
    assert voisins[0]["score"] == pytest.approx(1.0, abs=1e-3)


# --- socle ------------------------------------------------------------------
def test_le_corpus_respecte_son_schema(memoire):
    assert len(memoire) == 444
    for segment in memoire[:20]:
        assert set(segment) >= {"id", "src", "tgt", "domaine", "date", "statut"}
    evaluation = corpus.charger_evaluation()
    assert len(evaluation) == 80
    categories = {}
    for segment in evaluation:
        categories[segment["categorie"]] = categories.get(segment["categorie"], 0) + 1
    assert categories == {"repetition": 21, "piege": 20, "fuzzy": 18, "nouveau": 21}


def test_l_encodeur_factice_rend_des_vecteurs_normalises():
    vecteurs = EncodeurFactice().encoder(["Lösenordet förnyas var 30:e dag.", "Fakturan skickas."])
    for vecteur in vecteurs:
        assert sum(v * v for v in vecteur) == pytest.approx(1.0, abs=1e-6)
    proche = sum(a * b for a, b in zip(vecteurs[0], vecteurs[0]))
    loin = sum(a * b for a, b in zip(vecteurs[0], vecteurs[1]))
    assert proche > loin
