"""TP 5 — envoyer plus de requetes qu'un service ne peut en traiter. Fourni.

Ce module est le client d'assaut **en ligne de commande** :

    python tp.py assaut --clients 1,2,4,8 --duree 30

C'est la version de reference. Dans le notebook ``tp05.py``, vous ecrivez la
votre (CODE 1) : quarante lignes, et c'est en l'ecrivant qu'on comprend
pourquoi une mesure de charge se fait a plusieurs fils d'execution et pas en
boucle.

Ce qu'on mesure, et pourquoi ce ne sont pas des moyennes :

``p50``   la latence mediane. La moitie des requetes sont plus rapides.
``p95``   la latence que 95 % des requetes ne depassent pas. **C'est le chiffre
          qui figure dans un engagement de service**, parce que c'est celui que
          vos utilisateurs mecontents vivent.
``max``   le pire cas observe.
``debit`` requetes terminees par seconde.

Une moyenne cache exactement ce qu'on cherche ici : quand un service sature, la
moyenne monte doucement pendant que le p95 explose.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def appel(url: str, segment: dict, clients: int, timeout: float) -> dict:
    """Un appel, chronometre.

    L'en-tete ``X-Clients`` ne sert a rien au service : il sert a **vous**.
    Le serveur le recopie dans sa trace LangSmith, et c'est comme ca que vous
    retrouvez vos propres requetes dans un projet ou toute la salle ecrit.
    """
    charge = json.dumps({"segment": segment["src"]}).encode("utf-8")
    requete = urllib.request.Request(
        url.rstrip("/") + "/traduire",
        data=charge,
        headers={"Content-Type": "application/json", "X-Clients": str(clients)},
    )
    debut = time.perf_counter()
    try:
        with urllib.request.urlopen(requete, timeout=timeout) as reponse:
            reponse.read()
            return {"secondes": time.perf_counter() - debut, "code": reponse.status}
    except urllib.error.HTTPError as erreur:
        return {"secondes": time.perf_counter() - debut, "code": erreur.code}
    except Exception as erreur:
        return {"secondes": time.perf_counter() - debut, "code": 0, "erreur": str(erreur)}


def _centile(valeurs: list[float], part: float) -> float:
    """Le centile d'une liste triee. Rend 0 sur une liste vide."""
    if not valeurs:
        return 0.0
    if len(valeurs) == 1:
        return valeurs[0]
    return valeurs[min(int(len(valeurs) * part), len(valeurs) - 1)]


def resumer(appels: list[dict], clients: int, ecoule: float) -> dict:
    """Les chiffres d'un palier. Partage avec le notebook.

    Deux series, et il faut les deux : **toutes** les requetes, et les seules
    **servies** (code 200). Des qu'un service refuse — et un service qui se
    protege refuse —, la latence de l'ensemble ne veut plus rien dire : un refus
    en cinq millisecondes ferait passer le p95 global pour excellent. Le chiffre
    a tenir est ``p95_ok``, celui des requetes effectivement servies.
    """
    latences = sorted(a["secondes"] for a in appels)
    servies = sorted(a["secondes"] for a in appels if a["code"] == 200)
    codes: dict[int, int] = {}
    for a in appels:
        codes[a["code"]] = codes.get(a["code"], 0) + 1
    return {
        "ok": len(servies),
        "p50_ok": statistics.median(servies) if servies else 0.0,
        "p95_ok": _centile(servies, 0.95),
        "debit_ok": len(servies) / max(ecoule, 1e-6),
        "clients": clients,
        "requetes": len(appels),
        "p50": statistics.median(latences) if latences else 0.0,
        "p95": _centile(latences, 0.95),
        "max": max(latences) if latences else 0.0,
        "debit": len(appels) / max(ecoule, 1e-6),
        "erreurs": sum(1 for a in appels if a["code"] != 200),
        "codes": codes,
    }


def assaillir(url: str, clients: int, duree: float, segments: list[dict],
              timeout: float = 120.0) -> dict:
    """``clients`` fils d'execution bombardent ``url`` pendant ``duree`` secondes."""
    appels: list[dict] = []
    verrou = threading.Lock()
    fin = time.perf_counter() + duree

    def client(rang: int) -> None:
        i = rang
        while time.perf_counter() < fin:
            resultat = appel(url, segments[i % len(segments)], clients, timeout)
            with verrou:
                appels.append(resultat)
            i += clients

    debut = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as executeur:
        list(executeur.map(client, range(clients)))
    return resumer(appels, clients, time.perf_counter() - debut)


def entete() -> None:
    """La ligne de titre de la table des paliers. Partage avec le notebook."""
    print(f"  {'Clients':>7} {'Requetes':>9} {'Servies':>8} {'p50':>8} {'p95':>8} "
          f"{'p95 ok':>8} {'req/s ok':>9} {'Erreurs':>8}")
    print("  " + "-" * 74)


def ligne(mesure: dict) -> None:
    """Un palier, une ligne. Partage avec le notebook.

    ``p95 ok`` et ``req/s ok`` ne comptent que les requetes servies (code 200) :
    c'est ce qu'un service promet, une fois qu'il sait refuser.
    """
    print(f"  {mesure['clients']:>7} {mesure['requetes']:>9} {mesure['ok']:>8} "
          f"{mesure['p50']:>7.1f}s {mesure['p95']:>7.1f}s {mesure['p95_ok']:>7.1f}s "
          f"{mesure['debit_ok']:>9.2f} {mesure['erreurs']:>8}")
    if mesure["codes"]:
        details = ", ".join(f"{code}: {n}" for code, n in sorted(mesure["codes"].items()))
        print(f"  {'':>7} codes HTTP : {details}")


def verdict(mesures: list[dict]) -> None:
    """Ce que la table dit, en clair. Partage avec le notebook."""
    if len(mesures) < 2:
        return
    reference, dernier = mesures[0], mesures[-1]
    facteur_p95 = dernier["p95_ok"] / max(reference["p95_ok"], 1e-6)
    facteur_debit = dernier["debit_ok"] / max(reference["debit_ok"], 1e-6)
    print(f"""
  De {reference['clients']} a {dernier['clients']} clients :
     p95    x{facteur_p95:.1f}
     debit  x{facteur_debit:.1f}

  Si le debit ne monte pas alors que les clients se multiplient, le service ne
  traite qu'une requete a la fois : les autres attendent. C'est le defaut a
  trouver, et il n'est pas dans le modele — il est dans la facon dont le service
  a ete ecrit.""")


def campagne(url: str, paliers=(1, 2, 4, 8), duree: float = 30.0) -> list[dict]:
    """Monte en charge palier par palier et affiche la table."""
    from commun.corpus import charger_evaluation

    segments = charger_evaluation(n=8)
    print(f"\nAssaut de {url}")
    print(f"  {len(segments)} segments en boucle, {duree:.0f} s par palier\n")
    entete()
    mesures = []
    for clients in paliers:
        mesure = assaillir(url, clients, duree, segments)
        mesures.append(mesure)
        ligne(mesure)
    verdict(mesures)
    return mesures
