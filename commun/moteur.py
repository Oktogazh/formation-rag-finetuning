"""Le modele de langue, derriere une seule porte.

Toute la formation appelle ``obtenir_moteur()`` et rien d'autre. Quatre
implementations se cachent derriere, choisies dans cet ordre :

1. ``TP_MOTEUR=factice`` — un moteur ecrit en Python pur, sans reseau, sans
   poids, deterministe. C'est celui des tests, et c'est pour lui que la suite
   passe en trois secondes sans telecharger 3 Go.
2. ``MISTRAL_API_KEY`` presente — l'API Mistral. **La seule presence de la cle
   suffit** : si votre machine ne tient pas le modele local, le formateur vous
   donne une cle, vous la collez dans ``.env``, et les six TP fonctionnent sans
   changer une ligne.
3. sinon — **Ollama en local**, ``ministral-3:3b``. C'est le chemin par defaut
   et celui qu'on souhaite : le modele tourne sur votre machine, rien ne sort.
4. ``role="adapte"`` (TP 4 et 6) — le modele **plus l'adaptateur LoRA** que
   vous avez entraine. MLX sur Apple Silicon, PEFT ailleurs.

Une regle qui vaut pour tout le depot : **temperature 0 par defaut**. Une mesure
qui n'est pas reproductible n'est pas une mesure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

MODELE_OLLAMA = os.environ.get("TP_OLLAMA_MODELE", "ministral-3:3b")
MODELE_MISTRAL = os.environ.get("TP_MISTRAL_MODELE", "ministral-3b-latest")
URL_OLLAMA = os.environ.get("TP_OLLAMA_URL", "http://localhost:11434")
FENETRE = 8192


@dataclass
class Reponse:
    texte: str
    tokens_entree: int = 0
    tokens_sortie: int = 0
    secondes: float = 0.0
    moteur: str = ""


class Moteur(Protocol):
    def generer(self, messages: list[dict], temperature: float = 0.0,
                max_tokens: int = 256) -> Reponse: ...
    def nom(self) -> str: ...


class ErreurMoteur(RuntimeError):
    """Une erreur qui dit quoi taper pour la reparer."""


# ---------------------------------------------------------------------------
# 1. Ollama
# ---------------------------------------------------------------------------
class MoteurOllama:
    """Ollama en local. ``num_ctx`` est explicite, et ce detail compte.

    Par defaut Ollama tronque le contexte a 4096 tokens **sans rien dire**. Le
    TP 1 se termine par un prompt de 9 000 tokens : sans ce reglage, la moitie
    du prompt serait jetee en silence et l'exercice du « mur » ne mesurerait
    rien du tout.
    """

    def __init__(self, modele: str = MODELE_OLLAMA, url: str = URL_OLLAMA):
        self.modele = modele
        self.url = url.rstrip("/")

    def nom(self) -> str:
        return f"ollama:{self.modele}"

    def generer(self, messages, temperature=0.0, max_tokens=256) -> Reponse:
        charge = {
            "model": self.modele,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": temperature,
                "num_ctx": FENETRE,
                "num_predict": max_tokens,
                "seed": 1234,
            },
        }
        requete = urllib.request.Request(
            f"{self.url}/api/chat",
            data=json.dumps(charge).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        debut = time.perf_counter()
        try:
            with urllib.request.urlopen(requete, timeout=300) as reponse:
                donnees = json.loads(reponse.read())
        except urllib.error.URLError as erreur:
            raise ErreurMoteur(
                f"Ollama ne repond pas sur {self.url} ({erreur}).\n"
                "  1. Demarrez-le      : ollama serve\n"
                f"  2. Tirez le modele : ollama pull {self.modele}\n"
                "  3. Ou passez par l'API : collez MISTRAL_API_KEY dans .env"
            ) from erreur
        if "error" in donnees:
            raise ErreurMoteur(
                f"Ollama : {donnees['error']}\n"
                f"  Modele manquant ? ollama pull {self.modele}"
            )
        return Reponse(
            texte=donnees["message"]["content"],
            tokens_entree=donnees.get("prompt_eval_count", 0),
            tokens_sortie=donnees.get("eval_count", 0),
            secondes=time.perf_counter() - debut,
            moteur=self.nom(),
        )


# ---------------------------------------------------------------------------
# 2. API Mistral
# ---------------------------------------------------------------------------
class MoteurMistralAPI:
    def __init__(self, cle: str, modele: str = MODELE_MISTRAL):
        try:
            from mistralai import Mistral
        except ImportError as erreur:  # pragma: no cover
            raise ErreurMoteur("pip install mistralai") from erreur
        self.client = Mistral(api_key=cle)
        self.modele = modele

    def nom(self) -> str:
        return f"mistral-api:{self.modele}"

    def generer(self, messages, temperature=0.0, max_tokens=256) -> Reponse:
        debut = time.perf_counter()
        reponse = self.client.chat.complete(
            model=self.modele,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = reponse.usage
        return Reponse(
            texte=reponse.choices[0].message.content,
            tokens_entree=getattr(usage, "prompt_tokens", 0),
            tokens_sortie=getattr(usage, "completion_tokens", 0),
            secondes=time.perf_counter() - debut,
            moteur=self.nom(),
        )


# ---------------------------------------------------------------------------
# 3. Modele + adaptateur LoRA (TP 4 et TP 6)
# ---------------------------------------------------------------------------
DOSSIER_ADAPTATEUR = "resultats/tp04/adaptateur"


class MoteurAdaptateur:
    """Le modele quantifie **plus** l'adaptateur entraine au TP 4.

    Deux implementations, parce qu'aucun outil ne couvre les deux mondes :
    ``mlx-lm`` sur Apple Silicon, ``peft`` sur NVIDIA et CPU. L'interface, elle,
    est la meme — c'est tout l'interet de cacher ca ici.
    """

    def __init__(self, dossier: str = DOSSIER_ADAPTATEUR):
        from commun import RACINE

        self.dossier = RACINE / dossier
        if not self.dossier.exists():
            raise ErreurMoteur(
                f"Aucun adaptateur dans {self.dossier}.\n"
                "  Entrainez-le     : python tp.py entrainer\n"
                "  Ou telechargez-le : python tp.py adaptateur --telecharger"
            )
        self._charger()

    def _charger(self):
        config = json.loads((self.dossier / "adaptateur.json").read_text(encoding="utf-8"))
        self.cadre = config["cadre"]
        self.modele_base = config["modele_base"]
        if self.cadre == "mlx":
            from mlx_lm import load

            self.modele, self.tokenizer = load(
                self.modele_base, adapter_path=str(self.dossier)
            )
        else:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base = AutoModelForCausalLM.from_pretrained(
                self.modele_base, device_map="auto", torch_dtype=torch.bfloat16
            )
            self.modele = PeftModel.from_pretrained(base, str(self.dossier))
            self.tokenizer = AutoTokenizer.from_pretrained(self.modele_base)

    def nom(self) -> str:
        return f"adaptateur:{self.cadre}"

    def generer(self, messages, temperature=0.0, max_tokens=256) -> Reponse:
        debut = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if self.cadre == "mlx":
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            texte = generate(
                self.modele,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature),
                verbose=False,
            )
            entree = len(self.tokenizer.encode(prompt))
            sortie = len(self.tokenizer.encode(texte))
        else:
            entrees = self.tokenizer(prompt, return_tensors="pt").to(self.modele.device)
            produits = self.modele.generate(
                **entrees,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
            )
            texte = self.tokenizer.decode(
                produits[0][entrees["input_ids"].shape[1]:], skip_special_tokens=True
            )
            entree = int(entrees["input_ids"].shape[1])
            sortie = int(produits.shape[1]) - entree
        return Reponse(texte, entree, sortie, time.perf_counter() - debut, self.nom())


# ---------------------------------------------------------------------------
# 4. Le moteur factice
# ---------------------------------------------------------------------------
# Traductions « spontanees » : celles qu'un modele produit sans qu'on lui impose
# rien. Elles contredisent volontairement le glossaire du client — c'est tout
# l'interet du glossaire.
LEXIQUE = {
    "abonnemanget": "l'abonnement", "abonnemang": "abonnement",
    "förfrågningar": "demandes", "förfrågning": "demande", "förfrågan": "la demande",
    "hastighetsgränsen": "la limite de vitesse", "hastighetsgräns": "limite de vitesse",
    "slutpunkten": "le point final", "slutpunkt": "point final",
    "miljö": "milieu", "miljöer": "milieux", "företag": "entreprise",
    "du": "tu", "kan": "peux", "per": "par", "till": "à", "upp": "jusqu'à",
    "i": "dans", "om": "si", "minut": "minute", "för": "pour", "bjuda": "inviter",
    "in": "", "konto": "compte", "bort": "", "dagar": "jours", "dag": "jour",
    "tillåter": "autorise", "version": "version", "tas": "est", "är": "est",
    "skapa": "créer", "en": "une", "ett": "un", "den": "la", "det": "le",
    "månader": "mois", "månad": "mois", "förnyas": "est renouvelé", "var": "tous les",
    "efter": "après", "plus": "Plus", "pro": "Pro", "bas": "Bas", "från": "depuis",
    "att": "de", "kod": "code", "läsare": "lecteurs", "lösenordet": "le mot de passe",
    "testas": "essayé", "utvecklare": "développeurs", "timmar": "heures",
    "administratörer": "administrateurs", "ägare": "propriétaires",
    "åtkomsttoken": "jeton d'accès", "webhook": "webhook", "webhooks": "webhooks",
    "api-nyckeln": "la clé d'API", "nyckeln": "la clé", "fakturan": "la facture",
    "skickas": "est envoyée", "första": "premier", "dagen": "jour", "varje": "chaque",
    "vecka": "semaine", "veckor": "semaines", "inom": "en moins de", "svarar": "répond",
    "arbetsdagar": "jours ouvrés", "certifikatet": "le certificat", "projekt": "projets",
    "satt": "fixée", "överföring": "transfert", "innan": "avant", "av": "de",
    "och": "et", "nior": "neuf", "garanterade": "garantie",
    "tillgängligheten": "la disponibilité", "tillgänglig": "disponible",
    "underhållsarbeten": "les travaux d'entretien", "meddelas": "sont annoncés",
    "förväg": "à l'avance", "uppgifterna": "les données", "ägarskap": "la propriété",
    "kräver": "exige", "bekräftelse": "confirmation", "båda": "les deux",
    "betyder": "signifie", "felaktigt": "incorrectement", "utformad": "formée",
    "inte": "ne pas", "funktionen": "la fonctionnalité", "här": "cette",
    "supporten": "le support", "mitt": "au milieu", "period": "période",
    "beräknas": "est calculé", "mellanskillnaden": "la différence",
    "proportionellt": "proportionnellement", "ändrar": "modifies",
}

_NOMBRES = re.compile(r"\d+")


def _graine(texte: str) -> int:
    return int(hashlib.sha256(texte.encode("utf-8")).hexdigest()[:8], 16)


class MoteurFactice:
    """Un modele de langue en trente lignes, deterministe, sans reseau.

    **Ce moteur imite les defauts d'un vrai modele ; il ne les reproduit pas.**
    Les chiffres qu'il produit ne mesurent rien : ils servent a faire tourner la
    suite de tests et a lire le code sans connexion. Ce qu'il imite fidelement,
    en revanche, ce sont les trois comportements sur lesquels la formation
    s'appuie :

    * sans glossaire dans le prompt, il emploie la traduction spontanee du terme
      — « abonnement » la ou le client impose « formule » ;
    * sans consigne de style, il tutoie ;
    * avec un voisin de memoire tres proche, il le recopie **y compris son
      chiffre**, une fois sur quatre. C'est exactement la faute que la categorie
      ``piege`` du corpus est faite pour attraper.
    """

    def nom(self) -> str:
        return "factice"

    def generer(self, messages, temperature=0.0, max_tokens=256) -> Reponse:
        systeme = next((m["content"] for m in messages if m["role"] == "system"), "")
        contenu = next((m["content"] for m in messages if m["role"] == "user"), "")
        source = self._segment(contenu)
        voisins = self._voisins(contenu)
        avec_glossaire = "### Glossaire impose" in contenu
        avec_consignes = len(systeme.splitlines()) > 3

        texte = self._traduire(source, voisins, avec_glossaire, avec_consignes)
        entree = max(1, len(contenu.split()) + len(systeme.split()))
        return Reponse(texte, entree, max(1, len(texte.split())), 0.001, "factice")

    # -- analyse du prompt --------------------------------------------------
    @staticmethod
    def _segment(contenu: str) -> str:
        trouve = re.search(r"### Segment a traduire\nsv: (.+)", contenu)
        return trouve.group(1).strip() if trouve else ""

    @staticmethod
    def _voisins(contenu: str) -> list[dict]:
        bloc = re.search(r"### Memoire de traduction\n(.*?)(?=\n### |\Z)", contenu, re.S)
        if not bloc:
            return []
        lignes = bloc.group(1).splitlines()
        voisins = []
        for i in range(0, len(lignes) - 1):
            if lignes[i].startswith("sv: ") and lignes[i + 1].startswith("fr: "):
                voisins.append({"src": lignes[i][4:], "tgt": lignes[i + 1][4:]})
        return voisins

    # -- « generation » -----------------------------------------------------
    def _traduire(self, source, voisins, avec_glossaire, avec_consignes) -> str:
        if voisins:
            meilleur = max(voisins, key=lambda v: SequenceMatcher(None, source, v["src"]).ratio())
            score = SequenceMatcher(None, source, meilleur["src"]).ratio()
            if score >= 0.90:
                texte = meilleur["tgt"]
                if _graine(source) % 4 != 0:
                    texte = self._reporter_chiffres(meilleur["src"], texte, source)
                return self._styliser(texte, avec_glossaire, avec_consignes, source)
        return self._styliser(self._mot_a_mot(source), avec_glossaire, avec_consignes, source)

    @staticmethod
    def _reporter_chiffres(voisin_src, voisin_tgt, source) -> str:
        anciens, nouveaux = _NOMBRES.findall(voisin_src), _NOMBRES.findall(source)
        if len(anciens) != len(nouveaux):
            return voisin_tgt
        morceaux, i = [], 0
        for part in re.split(r"(\d+)", voisin_tgt):
            if part.isdigit() and i < len(anciens):
                morceaux.append(nouveaux[i] if part == anciens[i] else part)
                i += 1
            else:
                morceaux.append(part)
        return "".join(morceaux)

    @staticmethod
    def _mot_a_mot(source: str) -> str:
        mots = []
        for mot in re.findall(r"[\wåäöÅÄÖ'-]+|[.,:;!?]", source):
            if mot.isdigit() or mot in ".,:;!?":
                mots.append(mot)
            else:
                mots.append(LEXIQUE.get(mot.lower(), mot))
        texte = " ".join(m for m in mots if m)
        texte = re.sub(r"\s+([.,:;!?])", r"\1", texte)
        return texte[:1].upper() + texte[1:]

    def _styliser(self, texte, avec_glossaire, avec_consignes, source) -> str:
        from commun.corpus import charger_glossaire

        if avec_glossaire:
            for terme in charger_glossaire():
                for interdit in terme.get("interdits", []):
                    texte = re.sub(
                        rf"\b{re.escape(interdit)}\b", terme["fr"], texte, flags=re.IGNORECASE
                    )
        if avec_consignes:
            texte = re.sub(r"\bTu peux\b", "Vous pouvez", texte)
            texte = re.sub(r"\btu\b", "vous", texte)
        elif _graine(source) % 3 == 0:
            texte = re.sub(r"\bVous pouvez\b", "Tu peux", texte)
            texte = re.sub(r"\bvous\b", "tu", texte)
        return texte


# ---------------------------------------------------------------------------
# La porte d'entree
# ---------------------------------------------------------------------------
def moteur_demande() -> str:
    """Nom du moteur qui sera choisi, sans rien charger. Utilise par ``check``."""
    choix = os.environ.get("TP_MOTEUR", "auto").lower()
    if choix == "factice":
        return "factice"
    if choix in ("ollama", "api", "mistral"):
        return "mistral-api" if choix in ("api", "mistral") else "ollama"
    return "mistral-api" if os.environ.get("MISTRAL_API_KEY") else "ollama"


def obtenir_moteur(role: str = "base") -> Moteur:
    """Rend le moteur a utiliser. ``role`` vaut ``"base"`` ou ``"adapte"``."""
    choix = moteur_demande()
    if choix == "factice":
        return MoteurFactice()
    if role == "adapte":
        try:
            return MoteurAdaptateur()
        except ErreurMoteur:
            if choix == "mistral-api":
                print(
                    "  ! Aucun adaptateur local : on retombe sur le modele de base.\n"
                    "    La comparaison base / adapte n'aura pas de sens.",
                    flush=True,
                )
                return obtenir_moteur("base")
            raise
    if choix == "mistral-api":
        return MoteurMistralAPI(os.environ["MISTRAL_API_KEY"])
    return MoteurOllama()
