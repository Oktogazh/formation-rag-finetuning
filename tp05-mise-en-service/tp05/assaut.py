"""TP 5 — envoyer plus de requetes qu'un service ne peut en traiter.

Le service du formateur tourne sur un Space HuggingFace, sur deux coeurs de
processeur, sans carte graphique. Il repond correctement a un client. La
question du TP est : **que se passe-t-il a quatre ?**

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


def _appel(url: str, segment: dict, clients: int, timeout: float) -> dict:
    """Un appel, chronometre. Fourni."""
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


def assaillir(url: str, clients: int, duree: float, segments: list[dict],
              timeout: float = 120.0) -> dict:
    """``clients`` fils d'execution bombardent ``url`` pendant ``duree`` secondes.

    Rend ``{"clients", "requetes", "p50", "p95", "max", "debit", "erreurs", "codes"}``.
    """
    appels: list[dict] = []
    verrou = threading.Lock()
    fin = time.perf_counter() + duree

    # <<<TODO 1 ★★ Lancer l'assaut
    # Ecrivez la fonction interne « client(rang) » : tant que
    # time.perf_counter() < fin, elle appelle _appel(url, <un segment, en
    # tournant sur la liste>, clients, timeout) et range le resultat dans
    # « appels » — sous le verrou, parce que plusieurs fils ecrivent.
    # Puis lancez « clients » exemplaires de cette fonction avec
    # ThreadPoolExecutor(max_workers=clients) et attendez-les tous.
    # Indice : segments[i % len(segments)] pour tourner sur les segments.
    # Indice : list(executeur.map(client, range(clients))) attend la fin.
    # Test : python tp.py test tp05 -k todo1
    raise NotImplementedError(
        "TODO 1 — a completer. Consigne juste au-dessus, "
        "explications dans tp05-mise-en-service/README.md"
    )
    # >>>TODO 1

    latences = sorted(a["secondes"] for a in appels)
    codes: dict[int, int] = {}
    for a in appels:
        codes[a["code"]] = codes.get(a["code"], 0) + 1
    return {
        "clients": clients,
        "requetes": len(appels),
        "p50": statistics.median(latences) if latences else 0.0,
        "p95": latences[int(len(latences) * 0.95)] if len(latences) > 1 else (latences or [0.0])[0],
        "max": max(latences) if latences else 0.0,
        "debit": len(appels) / max(ecoule, 1e-6),
        "erreurs": sum(1 for a in appels if a["code"] != 200),
        "codes": codes,
    }


def campagne(url: str, paliers=(1, 2, 4, 8), duree: float = 30.0) -> list[dict]:
    """Monte en charge palier par palier et affiche la table. Fourni."""
    from commun.corpus import charger_evaluation

    segments = charger_evaluation(n=8)
    print(f"\nAssaut de {url}")
    print(f"  {len(segments)} segments en boucle, {duree:.0f} s par palier\n")
    print(f"  {'Clients':>7} {'Requetes':>9} {'p50':>8} {'p95':>8} {'max':>8} "
          f"{'req/s':>7} {'Erreurs':>8}")
    print("  " + "-" * 60)
    mesures = []
    for clients in paliers:
        mesure = assaillir(url, clients, duree, segments)
        mesures.append(mesure)
        print(f"  {mesure['clients']:>7} {mesure['requetes']:>9} {mesure['p50']:>7.1f}s "
              f"{mesure['p95']:>7.1f}s {mesure['max']:>7.1f}s {mesure['debit']:>7.2f} "
              f"{mesure['erreurs']:>8}")
        if mesure["codes"]:
            details = ", ".join(f"{code}: {n}" for code, n in sorted(mesure["codes"].items()))
            print(f"  {'':>7} codes HTTP : {details}")

    if len(mesures) > 1:
        reference, dernier = mesures[0], mesures[-1]
        facteur_p95 = dernier["p95"] / max(reference["p95"], 1e-6)
        facteur_debit = dernier["debit"] / max(reference["debit"], 1e-6)
        print(f"""
  De {reference['clients']} a {dernier['clients']} clients :
     p95    x{facteur_p95:.1f}
     debit  x{facteur_debit:.1f}

  Si le debit ne monte pas alors que les clients se multiplient, le service ne
  traite qu'une requete a la fois : les autres attendent. C'est le defaut a
  trouver, et il n'est pas dans le modele — il est dans la facon dont le service
  a ete ecrit.

  Ouvrez LangSmith maintenant : chaque requete y est tracee, avec sa duree et le
  nombre de clients declare dans l'en-tete X-Clients.""")
    return mesures


def palier_sature(url: str, seuil_p95: float, duree: float = 20.0, maximum: int = 16) -> int:
    """BONUS — trouver le nombre de clients a partir duquel p95 depasse un seuil.

    C'est la mesure qu'on fournit a un client qui demande « combien
    d'utilisateurs ce service supporte-t-il ». La reponse n'est jamais un
    nombre d'utilisateurs : c'est un nombre d'appels simultanes, pour un
    engagement de latence donne.
    """
    from commun.corpus import charger_evaluation

    segments = charger_evaluation(n=4)
    # <<<BONUS 5 ★★ Recherche du palier de saturation
    # Par dichotomie entre 1 et « maximum » : cherchez le plus grand nombre de
    # clients dont le p95 reste sous seuil_p95. Utilisez assaillir(url, n,
    # duree, segments) a chaque essai et rendez ce nombre (0 si meme 1 client
    # depasse deja le seuil).
    # Test : python tp.py test tp05 --bonus -k bonus5
    raise NotImplementedError(
        "BONUS 5 — a completer. Consigne juste au-dessus, "
        "explications dans tp05-mise-en-service/README.md"
    )
    # >>>BONUS 5
