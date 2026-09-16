"""Tests du TP 5. Un faux serveur local, aucune connexion sortante.

Les exercices sont dans le notebook ``tp05.py`` : la fixture ``exercice`` en
charge les fonctions et les classes sans exécuter les cellules de mesure. Le
paquet ``service/`` est fourni — les quelques tests de fin le contrôlent, ils
sont verts au clonage.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from service.reparer import Delai, Sature

SEGMENTS = [{"src": "Fakturan skickas varje vecka.", "id": "t1"}]
LENTEUR = 0.15


class Poseur(BaseHTTPRequestHandler):
    """Un serveur qui met LENTEUR secondes a repondre, une requete a la fois.

    ``HTTPServer`` est mono-fil : il reproduit en miniature le defaut du Space.
    A deux clients, chacun attend son tour, et la latence double.
    """

    def do_POST(self):
        longueur = int(self.headers.get("Content-Length", 0))
        self.rfile.read(longueur)
        time.sleep(LENTEUR)
        corps = json.dumps({"traduction": "La facture est envoyée chaque semaine."}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def serveur():
    httpd = HTTPServer(("127.0.0.1", 0), Poseur)
    fil = threading.Thread(target=httpd.serve_forever, daemon=True)
    fil.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.mark.code
def test_code1_l_assaut_mesure_des_latences(exercice, serveur):
    mesure = exercice.assaillir(serveur, clients=1, duree=1.0, segments=SEGMENTS)
    assert mesure["clients"] == 1
    assert mesure["requetes"] >= 2, "en une seconde, un client doit passer plusieurs requetes"
    assert mesure["erreurs"] == 0
    assert mesure["p50"] == pytest.approx(LENTEUR, abs=0.12)
    assert mesure["debit"] > 0
    assert set(mesure) >= {"p50", "p95", "max", "debit", "erreurs", "codes", "requetes"}


@pytest.mark.code
def test_code1_la_latence_monte_quand_les_clients_se_multiplient(exercice, serveur):
    seul = exercice.assaillir(serveur, clients=1, duree=1.5, segments=SEGMENTS)
    quatre = exercice.assaillir(serveur, clients=4, duree=1.5, segments=SEGMENTS)
    assert quatre["p50"] > seul["p50"] * 1.5, (
        "un serveur mono-fil fait attendre : c'est tout l'exercice"
    )
    assert quatre["debit"] < seul["debit"] * 2.5, "le debit ne suit pas les clients"


@pytest.mark.code
def test_code2_la_garde_refuse_au_dela_de_la_file(exercice):
    def lent(valeur):
        time.sleep(0.2)
        return valeur

    async def scenario():
        garde = exercice.Garde(concurrence_max=1, file_max=2, timeout_s=5)
        resultats = await asyncio.gather(
            *(garde.executer(lent, i) for i in range(6)), return_exceptions=True
        )
        return garde, resultats

    garde, resultats = asyncio.run(scenario())
    acceptes = [r for r in resultats if not isinstance(r, Exception)]
    refuses = [r for r in resultats if isinstance(r, Sature)]
    assert len(acceptes) == 3, "une en cours, deux en file : trois passent"
    assert len(refuses) == 3, "les autres sont refusees tout de suite"
    assert garde.en_attente == 0, "la file doit se vider"
    assert garde.refusees == 3


@pytest.mark.code
def test_code2_la_garde_abandonne_ce_qui_traine(exercice):
    def tres_lent(valeur):
        time.sleep(0.5)
        return valeur

    async def scenario():
        garde = exercice.Garde(concurrence_max=1, file_max=4, timeout_s=0.05)
        with pytest.raises(Delai):
            await garde.executer(tres_lent, 1)
        return garde

    assert asyncio.run(scenario()).expirees == 1


@pytest.mark.bonus
@pytest.mark.code
def test_bonus5_le_palier_de_saturation_se_trouve_par_dichotomie(exercice, monkeypatch):
    monkeypatch.setattr(
        exercice, "assaillir",
        lambda url, clients, duree, segments, **kw: {"p95": 0.5 * clients, "clients": clients},
    )
    assert exercice.palier_sature("http://faux", seuil_p95=2.0, segments=SEGMENTS,
                                  duree=0.01, maximum=16) == 4


# --- socle ------------------------------------------------------------------
def test_le_faux_serveur_repond(serveur):
    import urllib.request

    requete = urllib.request.Request(
        serveur + "/traduire", data=b"{}", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(requete, timeout=5) as reponse:
        assert reponse.status == 200


def test_l_assaut_de_reference_mesure_la_meme_chose(serveur):
    """Le socle, celui de « python tp.py assaut » : il doit rester juste."""
    from service.assaut import assaillir

    mesure = assaillir(serveur, clients=2, duree=1.0, segments=SEGMENTS)
    assert mesure["requetes"] >= 2 and mesure["erreurs"] == 0
    assert mesure["p50"] > LENTEUR, "deux clients sur un serveur mono-fil : chacun attend"


def test_la_garde_de_reference_refuse_et_expose_son_etat():
    from service.reparer import Garde

    etat = Garde(concurrence_max=2, file_max=3, timeout_s=10).etat()
    assert etat["concurrence_max"] == 2 and etat["file_max"] == 3
    assert etat["refusees"] == 0 and etat["expirees"] == 0
