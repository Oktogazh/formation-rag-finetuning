"""Tests du TP 2. Aucun n'appelle de modèle."""

import pytest

from commun import corpus
from commun.augmenter import GlossaireVectoriel
from commun.embeddings import EncodeurFactice
from commun.recherche import similarite

PROCHE = "Abonnemanget Företag tillåter 28 förfrågningar per minut."
GLOSSAIRE = "Hastighetsgränsen är satt till 60 förfrågningar per minut."

# L'encodeur des tests hache des trigrammes : il sépare vers 0,55, là où
# bge-m3 sépare vers 0,75. Le seuil appartient à l'encodeur, pas à la tâche.
SEUIL_FACTICE = 0.55


@pytest.fixture(scope="module")
def memoire():
    return corpus.charger_memoire_brute()


@pytest.fixture(scope="module")
def index_glossaire():
    return GlossaireVectoriel(corpus.charger_glossaire(),
                              encodeur=EncodeurFactice(), seuil=SEUIL_FACTICE)


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
def test_code3_le_glossaire_est_cherche_par_vecteurs(exercice, index_glossaire):
    retenus = exercice.glossaire_pertinent(GLOSSAIRE, index_glossaire)
    assert {t["sv"] for t in retenus} == {"hastighetsgräns", "förfråg"}
    scores = [t["score"] for t in retenus]
    assert scores == sorted(scores, reverse=True), "les termes doivent être rangés"
    assert all(s >= index_glossaire.seuil for s in scores), "sous le seuil, on ne garde pas"
    assert exercice.glossaire_pertinent("Fakturan skickas varje vecka.", index_glossaire) == []


@pytest.mark.code
def test_code3_le_mot_compose_que_le_prefixe_ne_voit_pas(exercice, index_glossaire):
    """``testslutpunkten`` contient ``slutpunkt`` sans commencer par lui."""
    from commun.augmenter import glossaire_pertinent as par_debut_de_mot

    compose = "Testslutpunkten förbrukar inte din kvot."
    assert par_debut_de_mot(compose, corpus.charger_glossaire()) == []
    retenus = exercice.glossaire_pertinent(compose, index_glossaire)
    assert "slutpunkt" in {t["sv"] for t in retenus}


@pytest.mark.code
def test_code4_le_prompt_augmente_contient_voisins_et_glossaire(exercice, memoire,
                                                                index_glossaire):
    messages = exercice.construire(PROCHE, memoire, index_glossaire, k=2)
    texte = messages[1]["content"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "### Glossaire impose" in texte and "### Memoire de traduction" in texte
    assert texte.count("sv:") == 3, "deux voisins, plus le segment à traduire"
    assert texte.rstrip().endswith("fr:"), "le segment à traduire vient en dernier"
    assert "vous" in messages[0]["content"].lower(), "la consigne de style doit être là"


# --- socle ------------------------------------------------------------------
def test_le_cache_du_glossaire_evite_de_reencoder(index_glossaire):
    """Deux fois le même segment ne coûte qu'un encodage de ses mots."""
    phrase = "Du kan skapa upp till 5 integrationer per abonnemang."
    index_glossaire.chercher(phrase)
    encodes = index_glossaire.encodes
    index_glossaire.chercher(phrase)
    assert index_glossaire.encodes == encodes, "le second passage ne réencode rien"


def test_le_glossaire_encode_ses_deux_langues(index_glossaire):
    """Source et cible disent la même chose sans avoir le même vecteur."""
    from commun.embeddings import cosinus

    glossaire = corpus.charger_glossaire()
    vecteurs_fr = index_glossaire.encodeur.encoder([t["fr"] for t in glossaire])
    for terme, sv, fr in zip(glossaire, index_glossaire.vecteurs, vecteurs_fr):
        proximite = cosinus(sv, fr)
        if terme["sv"].lower() == terme["fr"].lower():
            assert proximite == pytest.approx(1.0, abs=1e-6), terme["sv"]
        else:
            assert proximite < 1.0, f"{terme['sv']} et {terme['fr']} ne sont pas le même point"



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
