"""Le tracage LangSmith, derriere une seule porte.

Une chaine LangChain est tracee **sans une ligne de code** : trois variables
d'environnement suffisent. Le probleme n'est donc pas d'instrumenter le code,
c'est que **l'echec est silencieux**. Mauvaise region, cle revoquee, faute de
frappe : la chaine tourne, rend la bonne traduction, et rien n'arrive dans
LangSmith. Aucune exception, aucun avertissement. En seance, on y perd un
quart d'heure et on accuse le reseau.

Ce module existe pour rendre cet echec bruyant. ``activer()`` :

1. prend la cle **en argument** — ``tracage.activer(cle=LANGSMITH_API_KEY)`` —
   une variable de cellule que le stagiaire regle au debut du notebook. Pas de
   fichier a ouvrir a cote : le notebook n'est jamais versionne (voir
   ``.gitignore``), coller une cle dedans ne sort donc pas du poste. ``.env``
   reste lu en repli, pour qui prefere ce chemin ;
2. **trouve la region tout seul** en presentant la cle aux deux serveurs. Un
   compte europeen exige ``https://eu.api.smith.langchain.com`` ; sans cette
   ligne les traces partent vers les Etats-Unis, ou la cle est inconnue, et le
   SDK avale l'erreur. C'est le piege numero un du TP ;
3. pose les variables que LangChain lit, et **dit ce qu'il a fait**.

Ensuite ``vider_la_file()`` et ``compter_traces()`` repondent a la seule
question qui compte : est-ce que mes traces sont arrivees ?

Aucun appel reseau a l'import. Tout est dans les fonctions, et ``activer()``
sans cle ne sort pas de la machine.
"""

from __future__ import annotations

import datetime as _dt
import os
import pathlib
from dataclasses import dataclass, field

SERVEURS = {
    "US": "https://api.smith.langchain.com",
    "EU": "https://eu.api.smith.langchain.com",
}
APPLICATIONS = {
    "US": "https://smith.langchain.com",
    "EU": "https://eu.smith.langchain.com",
}
PROJET_PAR_DEFAUT = "formation-helios"
DELAI_SONDE = 8.0


@dataclass
class Etat:
    """Ce que ``activer()`` a reellement obtenu. Affichable tel quel."""

    actif: bool = False
    region: str = ""
    serveur: str = ""
    projet: str = ""
    application: str = ""
    raison: str = ""
    conseils: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.actif:
            lignes = [f"  Tracage       inactif — {self.raison}"]
            lignes += [f"      {c}" for c in self.conseils]
            return "\n".join(lignes)
        return (f"  Tracage       actif\n"
                f"  Region        {self.region}   ({self.serveur})\n"
                f"  Projet        {self.projet}\n"
                f"  A ouvrir      {self.application}/projects")


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------
def racine() -> pathlib.Path:
    """Le dossier du depot, trouve en remontant jusqu'a ``commun/``."""
    return pathlib.Path(__file__).resolve().parent.parent


def _relire_env() -> None:
    """Relit ``.env`` en ecrasant ce qui est deja en memoire.

    ``override=True`` est volontaire : dans un notebook, le stagiaire colle sa
    cle dans ``.env`` **apres** avoir lance la premiere cellule. Sans ecrasement
    il relancerait la cellule sans effet, et conclurait que le fichier n'est pas
    lu. C'est precisement la confusion qu'on veut eviter ici.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover — python-dotenv est dans l'environnement
        return
    load_dotenv(racine() / ".env", override=True)


# ---------------------------------------------------------------------------
# Quelle region ?
# ---------------------------------------------------------------------------
def _cle_valide_sur(serveur: str, cle: str) -> bool:
    """La cle est-elle reconnue par ce serveur ? Une requete, sans exception."""
    try:
        from langsmith import Client
    except ImportError:  # pragma: no cover
        return False
    try:
        client = Client(api_url=serveur, api_key=cle, timeout_ms=int(DELAI_SONDE * 1000))
        next(iter(client.list_projects(limit=1)), None)
        return True
    except Exception:
        return False


def trouver_region(cle: str) -> tuple[str, str]:
    """Rend ``(region, serveur)`` pour cette cle, ou ``("", "")`` si aucune.

    On essaie les deux serveurs. C'est une requete de plus au demarrage, contre
    un quart d'heure perdu a chercher pourquoi le projet reste vide : le change
    est bon. Un ``LANGSMITH_ENDPOINT`` deja pose dans ``.env`` court-circuite la
    sonde — si vous savez, on ne discute pas.
    """
    impose = os.environ.get("LANGSMITH_ENDPOINT", "").strip().rstrip("/")
    if impose:
        for region, serveur in SERVEURS.items():
            if impose == serveur:
                return region, serveur
        return "sur mesure", impose
    for region, serveur in SERVEURS.items():
        if _cle_valide_sur(serveur, cle):
            return region, serveur
    return "", ""


# ---------------------------------------------------------------------------
# activer
# ---------------------------------------------------------------------------
def activer(cle: str | None = None, projet: str | None = None,
            silencieux: bool = False) -> Etat:
    """Branche le tracage et rend l'etat obtenu. Sans cle : ne fait rien, et le dit.

    ``cle`` et ``projet``, donnes en argument, l'emportent sur ``.env`` et
    l'environnement — c'est le chemin normal : une variable regle au debut du
    notebook, avant la premiere cellule qui traduit. Sans argument, la fonction
    retombe sur ``.env`` puis sur l'environnement, pour qui a prefere ce chemin.

    A appeler **avant la premiere invocation** d'une chaine : LangChain decide
    d'ouvrir une trace au moment ou il est appele, pas au moment ou il est
    importe, mais le client LangSmith, lui, est mis en cache des le premier
    envoi. Activer trop tard, c'est perdre les premieres cellules.
    """
    _relire_env()
    if cle:
        os.environ["LANGSMITH_API_KEY"] = cle.strip()
        os.environ["LANGSMITH_TRACING"] = "true"
    if projet:
        os.environ["LANGSMITH_PROJECT"] = projet.strip()

    etat = Etat(projet=os.environ.get("LANGSMITH_PROJECT") or PROJET_PAR_DEFAUT)

    cle_active = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if not cle_active:
        os.environ["LANGSMITH_TRACING"] = "false"
        etat.raison = "aucune cle"
        etat.conseils = [
            "Le TP entier fonctionne sans : vous sautez l'exercice OBS 2.",
            "Pour l'activer : reglez LANGSMITH_API_KEY dans la cellule au-dessus.",
        ]
        if not silencieux:
            print(etat)
        return etat

    if os.environ.get("LANGSMITH_TRACING", "true").strip().lower() in ("false", "0", "no"):
        os.environ["LANGSMITH_TRACING"] = "false"
        etat.raison = "LANGSMITH_TRACING=false"
        etat.conseils = ["Passez la variable a « true », puis relancez cette cellule."]
        if not silencieux:
            print(etat)
        return etat

    region, serveur = trouver_region(cle_active)
    if not serveur:
        os.environ["LANGSMITH_TRACING"] = "false"
        etat.raison = "cle refusee par les deux serveurs (US et EU)"
        etat.conseils = [
            "Une cle personnelle commence par « lsv2_pt_ ». Verifiez le copier-coller :",
            "un espace ou un retour a la ligne colle en fin de cle suffit a la casser.",
            "Si elle est correcte, regenerez-en une : Settings -> API Keys.",
        ]
        if not silencieux:
            print(etat)
        return etat

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = serveur
    os.environ["LANGSMITH_PROJECT"] = etat.projet
    # LangChain lit encore les anciens noms dans certaines versions. Les poser
    # tous les deux coute deux lignes et evite une panne qu'on ne saurait pas lire.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = serveur

    etat.actif = True
    etat.region = region
    etat.serveur = serveur
    etat.application = APPLICATIONS.get(region, APPLICATIONS["US"])
    if not silencieux:
        print(etat)
    return etat


# ---------------------------------------------------------------------------
# Verifier que ca arrive
# ---------------------------------------------------------------------------
def vider_la_file(delai: float = 20.0) -> None:
    """Attend que les traces en attente soient parties.

    LangChain envoie les traces depuis un fil d'execution en arriere-plan. Dans
    un notebook, la cellule rend la main **avant** l'envoi : on ouvre LangSmith,
    on ne voit rien, on croit que ca ne marche pas. Cette fonction est la pour
    que le « rien » signifie vraiment quelque chose.
    """
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:  # pragma: no cover
        pass
    try:
        from langsmith import Client

        Client().flush()
    except Exception:  # pragma: no cover
        pass


def compter_traces(minutes: int = 15, projet: str | None = None) -> int:
    """Combien de traces racines sont arrivees dans le projet, recemment.

    C'est la reponse chiffree a « est-ce que ca marche ? ». Zero apres avoir
    fait tourner la chaine veut dire que quelque chose est casse, et c'est
    exactement le genre de constat qu'on veut pouvoir faire en dix secondes.
    """
    nom = projet or os.environ.get("LANGSMITH_PROJECT") or PROJET_PAR_DEFAUT
    try:
        from langsmith import Client

        depuis = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=minutes)
        runs = Client().list_runs(project_name=nom, start_time=depuis, is_root=True)
        return sum(1 for _ in runs)
    except Exception as erreur:
        print(f"  Lecture impossible : {type(erreur).__name__} — {erreur}")
        return 0


def lien_projet(projet: str | None = None) -> str:
    """L'URL exacte du projet, une fois qu'il existe. Sinon, la page des projets."""
    nom = projet or os.environ.get("LANGSMITH_PROJECT") or PROJET_PAR_DEFAUT
    application = APPLICATIONS["EU"] if "eu." in os.environ.get(
        "LANGSMITH_ENDPOINT", "") else APPLICATIONS["US"]
    try:
        from langsmith import Client

        return Client().read_project(project_name=nom).url
    except Exception:
        return f"{application}/projects"


def rapport(minutes: int = 15) -> None:
    """Vide la file, compte, et affiche ou aller regarder. Le geste de fin de cellule."""
    if os.environ.get("LANGSMITH_TRACING") != "true":
        print("  Tracage inactif : rien n'a ete envoye.")
        return
    vider_la_file()
    nombre = compter_traces(minutes)
    print(f"  {nombre} trace(s) arrivee(s) dans « {os.environ.get('LANGSMITH_PROJECT')} » "
          f"ces {minutes} dernieres minutes")
    print(f"  {lien_projet()}")
    if nombre == 0:
        print("      Zero alors que la chaine a tourne : relancez tracage.activer(),")
        print("      il vous dira lequel des trois problemes vous avez.")


# ---------------------------------------------------------------------------
# Le decorateur, meme quand LangSmith est absent
# ---------------------------------------------------------------------------
def _traceable_inerte(*arguments, **nommes):
    """Remplace ``@traceable`` quand le paquet manque : ne fait rien, ne casse rien."""
    if len(arguments) == 1 and callable(arguments[0]) and not nommes:
        return arguments[0]

    def decorateur(fonction):
        return fonction

    return decorateur


try:  # pragma: no cover — depend de l'environnement, pas du code
    from langsmith import traceable
except ImportError:  # pragma: no cover
    traceable = _traceable_inerte


__all__ = ["activer", "traceable", "vider_la_file", "compter_traces", "lien_projet",
           "rapport", "trouver_region", "Etat", "SERVEURS", "APPLICATIONS"]
