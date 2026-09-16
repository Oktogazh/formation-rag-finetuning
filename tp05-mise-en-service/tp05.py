# %% [markdown]
# # TP 5 — Mettre en service : un serveur, quatre assaillants
#
# **Durée** : noyau ≈ 70 min · bonus ≈ 20 min · **Niveau** ★★☆☆☆
#
# **Ce TP ne dépend d'aucun autre.** Le service que vous allez bombarder est
# fourni : il tourne sur un Space HuggingFace (celui du formateur) et, à
# l'identique, sur votre machine. Rien de ce que vous avez écrit aux TP 1 à 4
# n'est nécessaire ici.
#
# ## L'incident, avant de coder
#
# Un service de traduction est en ligne. Il répond correctement à **un** client,
# en une vingtaine de secondes. Son développeur l'a essayé sur sa machine, avec
# un client, et tout allait bien.
#
# **La question du TP est : que se passe-t-il à quatre ?**
#
# Vous allez le mesurer, retrouver vos requêtes dans les traces, lire le code du
# service pour y trouver le défaut — il y en a quatre, et aucun n'est dans le
# modèle —, écrire la pièce qui manque, puis rejouer l'assaut sur une copie
# réparée.
#
# ## Pourquoi un notebook pour un TP de mise en service
#
# **Tout ce TP se fait ici.** Aucun terminal, aucune commande à taper ailleurs :
# les mesures, les deux exercices de code et les comparaisons sont des cellules.
#
# Un serveur, ça boucle : lancé dans une cellule, il figerait le notebook. Le
# service tourne donc **dans un autre processus** — soit sur le Space, soit
# démarré par une cellule qui rend la main tout de suite (`demarrer_en_fond`).
# Dans les deux cas c'est une cellule qui s'en charge, et vous n'avez rien à
# lancer à la main.
#
# **Deux exercices vous font ouvrir un navigateur**, et aucun ne demande de
# taper quoi que ce soit : OBS 0, pour faire travailler le service à la main sur
# sa page, et OBS 2, pour lire ses traces dans LangSmith.
#
# **Une chose que ce TP ne fait pas** : réparer le service en ligne. Le Space du
# formateur est mal configuré **exprès**, et il doit le rester pour la salle
# suivante. Le code réparé est montré ici, en extrait commenté — il n'est pas
# redéployé.

# %% [markdown]
# ## RÉG 0 · Choisir la cible — **à faire maintenant**
#
# Le service du formateur est déjà en ligne, et son adresse est pré-remplie
# ci-dessous :
#
# > **https://oktogazh-formation-rag-finetuning.hf.space**
#
# Trois cibles possibles :
#
# | Cible | Ce que ça donne | Comment |
# |---|---|---|
# | **le Space du formateur** | un vrai service distant, saturé par toute la salle en même temps — la démonstration la plus parlante | l'adresse pré-remplie ci-dessous |
# | **votre machine, sans garde** | le même défaut, reproduit chez vous, sans dépendre du réseau ni du Space | **videz** la variable |
# | **votre machine, avec garde** | le service réparé, pour la comparaison finale | une cellule plus bas s'en charge |
#
# Si le Space est indisponible, **videz `URL_SERVICE`** : la cellule d'OBS 1
# démarrera un service local à sa place, et **rien du TP n'est perdu** — seuls
# OBS 0 (la page) et OBS 2 (les traces) demandent le Space.

# %%
URL_SERVICE = "https://oktogazh-formation-rag-finetuning.hf.space"
DUREE_PALIER = 30.0       # <- secondes par palier ; 15 pour dégrossir, 30 en séance

# %% [markdown]
# ## OBS 0 · Le service à la main, avant de le mesurer
#
# *C'est le seul exercice où vous faites travailler le service autrement que
# par une cellule.* Ouvrez l'adresse ci-dessus dans un onglet : le Space sert
# une **page de traduction**.
#
# **1. Seul.** Collez un segment suédois — la page en propose trois — et
# chronométrez grossièrement. Notez la durée annoncée sous le résultat.
#
# **2. Toute la salle en même temps.** Au signal du formateur, tout le monde
# clique sur « Traduire » en boucle, sans attendre son tour. Regardez ce que
# devient votre propre temps de réponse.
#
# **3. Ce qu'il faut comprendre de la page.** Elle ne traduit rien : c'est un
# *front-end*. Elle envoie `POST /traduire` au *back-end*, qui tourne dans le
# même Space, et c'est lui qui appelle le modèle. L'assaut que vous écrirez en
# CODE 1 s'adresse **directement au back-end** — il ne passe jamais par la page.
# Les deux chemins frappent donc exactement le même service.
#
# > *La durée seul, puis à toute la salle :*
# >
# >
#
# **Ce que la page ne vous dira pas**, et que la suite du TP va chercher : de
# combien exactement, à partir de combien de clients, où le temps est passé, et
# ce qu'il aurait fallu faire. Une impression n'est pas une mesure.

# %% [markdown]
# ## Préparation
#
# La cellule ci-dessous se place à la racine du dépôt et rend le paquet
# `service/` importable. Ce paquet est **fourni** : il contient le serveur, la
# fonction d'appel chronométrée, et une garde de référence. Vous en réécrirez
# deux morceaux dans ce notebook, et vous les comparerez.

# %%
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "commun").is_dir())
sys.path[:0] = [str(RACINE), str(RACINE / "tp05-mise-en-service")]
import os

os.chdir(RACINE)

import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from commun.corpus import charger_evaluation
from service import assaut as socle
from service.reparer import Delai, Sature
from service.serveur_local import demarrer_en_fond

SEGMENTS = charger_evaluation(n=8)
print(f"  {len(SEGMENTS)} segments d'évaluation en boucle pour l'assaut")
print(f"  exemple : {SEGMENTS[0]['src']}")

# %% [markdown]
# ## LIRE 1 · Trois chiffres, et pourquoi pas la moyenne
#
# | | |
# |---|---|
# | **p50** | la latence médiane. La moitié des requêtes sont plus rapides. |
# | **p95** | la latence que 95 % des requêtes ne dépassent pas. **C'est le chiffre qui figure dans un engagement de service**, parce que c'est celui que vos utilisateurs mécontents vivent. |
# | **débit** | requêtes terminées par seconde. S'il ne monte pas quand les clients se multiplient, le service n'en traite qu'une à la fois. |
#
# Une moyenne cache exactement ce qu'on cherche : quand un service sature, la
# moyenne monte doucement pendant que le p95 explose. La cellule ci-dessous le
# montre sur deux services fictifs — **même moyenne, pas du tout le même
# service**.
#
# **Lisez la sortie et répondez** : lequel des deux mettriez-vous en production,
# et qu'est-ce qu'un tableau de bord qui n'afficherait que la moyenne vous
# aurait fait croire ?

# %%
regulier = [1.0] * 95 + [1.2] * 5
irregulier = [0.4] * 95 + [12.4] * 5

for nom, serie in (("régulier", regulier), ("irrégulier", irregulier)):
    ordonnee = sorted(serie)
    print(f"  {nom:12} moyenne {statistics.mean(serie):5.2f}s   "
          f"p50 {statistics.median(serie):5.2f}s   "
          f"p95 {ordonnee[int(len(ordonnee) * 0.95)]:5.2f}s   "
          f"max {max(serie):5.2f}s")

# %% [markdown]
# **Réponse.** Les deux services ont la même moyenne (≈ 1 s) et une expérience
# utilisateur opposée : dans le second, une requête sur vingt prend douze
# secondes. C'est celle-là qu'on remarque, celle-là qui fait l'appel au support,
# et elle est **invisible** sur un tableau de bord qui n'affiche que la moyenne.
# Un engagement de service s'écrit donc toujours sur un centile — p95 ou p99 —,
# jamais sur une moyenne.

# %% [markdown]
# ## LIRE 2 · Le défaut est dans le service, pas dans le modèle
#
# Le service est dans `tp05-mise-en-service/space/app.py` : c'est **exactement**
# le fichier déployé sur le Space. La cellule ci-dessous en affiche le cœur.
#
# **Lisez-le et cherchez ce qui manque.** Il y a quatre défauts, et aucun ne
# concerne la qualité de la traduction :
#
# 1. le point d'entrée est déclaré avec `def` et non `async def`, et il appelle
#    le modèle de façon **bloquante** ;
# 2. **aucune limite de concurrence** : tout ce qui arrive entre ;
# 3. **aucune file d'attente bornée** : personne n'est refusé, tout le monde
#    attend ;
# 4. **aucun délai maximum** : une requête peut traîner indéfiniment.
#
# Le premier mérite une explication, parce qu'il surprend tout le monde : dans
# FastAPI, un point d'entrée `async def` s'exécute **dans la boucle asyncio** —
# s'il bloque, il fige tout le serveur, `/sante` comprise. Un point d'entrée
# `def`, lui, part dans un **pool de threads** : le serveur reste réactif, mais
# les quatre requêtes se disputent alors deux cœurs de processeur, et la latence
# de chacune est **multipliée** au lieu d'être partagée. Aucune des deux
# écritures ne sauve un modèle qui ne sait traiter qu'une chose à la fois : ce
# qui manque, c'est une décision explicite sur **combien on en accepte**.

# %%
extrait = (RACINE / "tp05-mise-en-service" / "space" / "app.py").read_text(encoding="utf-8")
debut = extrait.index("@app.post(\"/traduire\")")
print(extrait[debut:debut + 420])

# %% [markdown]
# ## CODE 1 · L'assaut : N clients en parallèle
#
# Pour mesurer une saturation, il faut la provoquer. `assaillir` lance
# `clients` fils d'exécution qui appellent le service **en boucle** pendant
# `duree` secondes, puis résume les latences.
#
# Deux choses à ne pas manquer :
#
# - **une boucle séquentielle ne mesurerait rien.** Un seul client n'attend
#   jamais derrière personne ; c'est la concurrence qui fait apparaître la file
#   d'attente ;
# - **on mesure pendant une durée, pas un nombre de requêtes.** Sinon les
#   paliers lents durent dix fois plus longtemps que les rapides et la salle
#   attend.
#
# Deux fonctions du socle vous sont fournies et font le reste :
# `socle.appel(url, segment, clients, timeout)` envoie **une** requête
# chronométrée, et `socle.resumer(appels, clients, ecoule)` calcule p50, p95,
# max, débit, erreurs et codes HTTP.

# %%
def assaillir(url: str, clients: int, duree: float, segments: list[dict],
              timeout: float = 120.0) -> dict:
    """``clients`` fils d'exécution bombardent ``url`` pendant ``duree`` secondes.

    Rend ``{"clients", "requetes", "p50", "p95", "max", "debit", "erreurs", "codes"}``.
    """
    appels: list[dict] = []
    verrou = threading.Lock()
    fin = time.perf_counter() + duree

    def client(rang: int) -> None:
        """Un client : il appelle en boucle jusqu'à la fin du palier. Fourni."""
        i = rang
        while time.perf_counter() < fin:
            resultat = socle.appel(url, segments[i % len(segments)], clients, timeout)
            with verrou:
                appels.append(resultat)
            i += clients

    debut = time.perf_counter()
    # <<<CODE 1 ★★ Lancer les clients en parallèle
    # Lancez « clients » exemplaires de la fonction client() ci-dessus, en
    # parallèle, et attendez qu'ils aient tous fini.
    # Indice : with ThreadPoolExecutor(max_workers=clients) as executeur:
    # puis list(executeur.map(client, range(clients)))
    # Test : python tp.py test tp05 -k code1
    raise NotImplementedError(
        "CODE 1 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp05-mise-en-service/README.md"
    )
    # >>>CODE 1
    return socle.resumer(appels, clients, time.perf_counter() - debut)


# %% [markdown]
# **Vérifiez tout de suite, sans modèle et sans réseau.** La cellule ci-dessous
# monte un faux serveur qui dort 0,15 s et ne traite **qu'une requête à la
# fois** : c'est le défaut du Space, en miniature et en accéléré. À un client,
# p50 ≈ 0,15 s ; à quatre, il doit être nettement plus haut, et le débit ne doit
# pas être multiplié par quatre.

# %%
from http.server import BaseHTTPRequestHandler, HTTPServer


class FauxServiceLent(BaseHTTPRequestHandler):
    """Un service mono-fil qui met 0,15 s à répondre. Fourni."""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(0.15)
        corps = json.dumps({"traduction": "La facture est envoyée chaque semaine."}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def log_message(self, *args):
        pass


def essai_a_blanc(fonction) -> None:
    """Fait tourner ``fonction`` contre le faux service. Fourni."""
    httpd = HTTPServer(("127.0.0.1", 0), FauxServiceLent)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}"
    try:
        socle.entete()
        for clients in (1, 4):
            socle.ligne(fonction(url, clients, 2.0, [{"src": "Fakturan skickas varje vecka."}]))
    finally:
        httpd.shutdown()


essai_a_blanc(assaillir)

# %% [markdown]
# ## OBS 1 · La référence, puis la montée en charge
#
# Maintenant le vrai service. La cellule ci-dessous prend la cible réglée en
# RÉG 0 : le Space si vous avez collé une URL, sinon elle démarre le service
# **sur votre machine, sans garde** — le défaut du Space reproduit chez vous.
#
# **Si le formateur a donné l'URL du Space, lancez l'assaut en même temps que
# vos collègues.** C'est le même serveur pour toute la salle, et c'est le but :
# la saturation que vous mesurez est celle que vous provoquez ensemble.

# %%
serveur = None
if URL_SERVICE:
    cible = URL_SERVICE.rstrip("/")
    print(f"  Cible : le service distant {cible}")
else:
    serveur = demarrer_en_fond(port=8000, repare=False)
    cible = serveur.url

# %% [markdown]
# **Le premier palier est la référence** : la latence du service quand personne
# d'autre ne l'utilise, et le seul chiffre que le développeur du service ait
# jamais vu. Les suivants disent ce qu'il devient quand la salle arrive.
#
# **Relevez p50, p95 et le débit à chaque palier.** Compter en minutes : à une
# vingtaine de secondes par traduction, les quatre paliers prennent ≈ 4 × la
# durée réglée en RÉG 0.

# %%
mesures = []
socle.entete()
for clients in (1, 2, 4):
    mesures.append(assaillir(cible, clients, DUREE_PALIER, SEGMENTS))
    socle.ligne(mesures[-1])
socle.verdict(mesures)

# %% [markdown]
# **Ce que vous devez voir.** Le p95 est multiplié par à peu près le nombre de
# clients, et le débit **ne bouge pas**. Un service qui traiterait vraiment les
# requêtes en parallèle ferait l'inverse : p95 stable, débit multiplié.
#
# Et si vous voyez des codes `0` : ce sont des délais dépassés **côté client**.
# C'est un résultat, pas une panne — notez-le, c'est ce que vit un utilisateur
# dont la page ne charge jamais.

# %% [markdown]
# ## OBS 2 · Retrouver ses requêtes dans les traces
#
# *Cet exercice demande le Space du formateur : le service y est tracé dans
# LangSmith, projet `formation-helios`. Sur votre service local, il ne l'est
# pas, et vous pouvez sauter cet exercice sans rien perdre du reste.*
#
# Ouvrez le projet, et cherchez **vos** requêtes : le client d'assaut envoie un
# en-tête `X-Clients`, le service le recopie dans la trace, et c'est comme ça
# qu'on se retrouve à quinze dans le même projet.
#
# **Trois choses à relever :**
#
# 1. la durée d'une trace au palier 1, et la même au palier 4 ;
# 2. **où le temps est passé** : ouvrez une trace lente et regardez la
#    répartition. Le temps est dans le modèle, pas dans le réseau ni dans le
#    code du service — le goulot n'est pas là où on l'imagine d'habitude ;
# 3. combien de traces le projet a reçu pendant votre palier, comparé au nombre
#    de requêtes que votre table annonce. L'écart, s'il y en a un, ce sont les
#    requêtes abandonnées côté client.
#
# > *Ce que vous avez relevé :*
# >
# >

# %% [markdown]
# ## LIRE 3 · Ce qu'un service doit faire quand il sature
#
# Le défaut du Space est un défaut de conception, pas de modèle : **il accepte
# tout ce qui arrive**. Chaque requête entre, occupe un fil d'exécution, attend
# le modèle qui ne sait traiter qu'une chose à la fois, et personne ne répond à
# personne.
#
# Un service qui sature doit faire trois choses, **dans cet ordre** :
#
# 1. **limiter** le nombre de requêtes traitées en même temps. Le modèle n'ira
#    pas plus vite parce qu'on lui en donne plus ;
# 2. **refuser vite** ce qu'il ne pourra pas traiter. Un `503` en cinq
#    millisecondes vaut mieux qu'une réponse en deux minutes : l'appelant peut
#    réessayer, basculer ailleurs, ou afficher un message honnête ;
# 3. **abandonner** ce qui prend trop longtemps, avec un `504`.
#
# Les deux codes ne disent pas la même chose, et un client automatique les
# traite différemment : `503 Service Unavailable` = « reviens plus tard, je n'ai
# même pas commencé » ; `504 Gateway Timeout` = « j'ai commencé et j'ai
# renoncé ». Le premier se réessaie sans risque, le second peut avoir eu un
# effet de bord.
#
# **Refuser proprement n'est pas un aveu d'échec** : c'est ce qui permet aux
# requêtes acceptées de rester rapides. Un service sans garde n'a pas de
# latence, il a une loterie.

# %% [markdown]
# ## CODE 2 · La règle d'admission
#
# `Garde` est la pièce qui manque. Trois réglages, trois compteurs, et une seule
# méthode à écrire.
#
# `_lancer` vous est fournie, et elle mérite une lecture : elle exécute la
# fonction bloquante **hors de la boucle asyncio**
# (`run_in_executor`), avec un `asyncio.wait_for`. Sans ça, le serveur ne
# pourrait même plus répondre à `/sante` pendant qu'il traduit.
#
# **L'ordre compte.** Le refus se décide **avant** d'attendre le sémaphore :
# c'est toute la différence entre refuser en cinq millisecondes et refuser au
# bout de deux minutes d'attente.

# %%
import asyncio


class Garde:
    """Limite la concurrence, borne la file d'attente, coupe au bout du temps.

    ``concurrence_max``  requêtes traitées en même temps. 1 pour un modèle qui
                         ne sait pas traiter de lot.
    ``file_max``         requêtes qui peuvent attendre leur tour. Au-delà, on
                         refuse.
    ``timeout_s``        temps maximum d'une requête.
    """

    def __init__(self, concurrence_max: int = 1, file_max: int = 2, timeout_s: float = 30.0):
        self.concurrence_max = concurrence_max
        self.file_max = file_max
        self.timeout_s = timeout_s
        self.semaphore = asyncio.Semaphore(concurrence_max)
        self.en_attente = 0
        self.refusees = 0
        self.expirees = 0

    async def _lancer(self, fonction, *arguments):
        """Exécute la fonction bloquante hors de la boucle, avec un délai. Fourni."""
        boucle = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                boucle.run_in_executor(None, fonction, *arguments), timeout=self.timeout_s)
        except asyncio.TimeoutError as erreur:
            self.expirees += 1
            raise Delai(f"depassement de {self.timeout_s:.0f} s") from erreur

    async def executer(self, fonction, *arguments):
        """Exécute ``fonction(*arguments)`` sous protection."""
        # <<<CODE 2 ★★ La règle d'admission
        # 1. Si self.en_attente >= self.file_max : incrémentez self.refusees et
        # levez Sature(...) TOUT DE SUITE, sans attendre. Refuser en cinq
        # millisecondes vaut mieux que répondre en deux minutes.
        # 2. Sinon, incrémentez self.en_attente, puis « async with
        # self.semaphore: » ; une fois le jeton obtenu, décrémentez
        # self.en_attente et rendez « await self._lancer(fonction, *arguments) ».
        # Test : python tp.py test tp05 -k code2
        raise NotImplementedError(
            "CODE 2 — à compléter. La consigne est juste au-dessus, "
            "le détail dans tp05-mise-en-service/README.md"
        )
        # >>>CODE 2

    def etat(self) -> dict:
        return {
            "concurrence_max": self.concurrence_max,
            "file_max": self.file_max,
            "timeout_s": self.timeout_s,
            "en_attente": self.en_attente,
            "refusees": self.refusees,
            "expirees": self.expirees,
        }


# %% [markdown]
# **Vérifiez tout de suite.** Six requêtes arrivent ensemble sur une garde qui
# accepte une requête à la fois et deux en attente. Attendu : **trois passent**
# (une traitée, deux en file) et **trois sont refusées immédiatement**.

# %%
async def scenario(garde, lenteur=0.2, requetes=6):
    """Six appels simultanés sur une fonction lente. Fourni."""
    def lent(valeur):
        time.sleep(lenteur)
        return valeur

    debut = time.perf_counter()
    resultats = await asyncio.gather(
        *(garde.executer(lent, i) for i in range(requetes)), return_exceptions=True)
    acceptes = [r for r in resultats if not isinstance(r, Exception)]
    refuses = [r for r in resultats if isinstance(r, Sature)]
    print(f"  {len(acceptes)} acceptées, {len(refuses)} refusées (503), "
          f"en {time.perf_counter() - debut:.2f}s")
    print(f"  {garde.etat()}")


await scenario(Garde(concurrence_max=1, file_max=2, timeout_s=5))

# %% [markdown]
# ## RÉG 1 · Le réglage de la file
#
# Les trois valeurs de la garde sont un arbitrage, pas une vérité. Changez
# `file_max` ci-dessous — essayez `1`, puis `5` — et relisez les deux chiffres :
# combien passent, et **en combien de temps**.
#
# **Ce que vous devez constater** : une file longue ne fait passer personne plus
# vite. Elle transforme des refus immédiats en attentes longues. La bonne
# question n'est pas « combien puis-je faire patienter » mais « au-delà de
# combien de secondes d'attente ma réponse ne vaut-elle plus rien ».

# %%
for file_max in (1, 2, 5):
    print(f"file_max = {file_max}")
    await scenario(Garde(concurrence_max=1, file_max=file_max, timeout_s=5))

# %% [markdown]
# Et le délai maximum, l'autre moitié de la garde : une fonction qui dort une
# demi-seconde sous un `timeout_s` de 50 ms doit lever `Delai` — le `504`.

# %%
async def essai_delai():
    garde = Garde(concurrence_max=1, file_max=4, timeout_s=0.05)
    try:
        await garde.executer(time.sleep, 0.5)
    except Delai as erreur:
        print(f"  Delai levé : {erreur}  →  {garde.etat()}")


await essai_delai()

# %% [markdown]
# ## LIRE 4 · Ce que deviendrait `space/app.py`, réparé
#
# Voici la même application, avec la garde. **Cet extrait n'est pas déployé** :
# le Space doit rester cassé pour la prochaine session, et le redéployer prendrait
# dix minutes de compilation. Lisez-le comme un diff — quatre changements, tous
# en dehors du modèle.

# %%
print('''
from fastapi import FastAPI, HTTPException, Request
from service.reparer import Delai, Garde, Sature          # (1) la garde

garde = Garde(concurrence_max=1, file_max=4, timeout_s=60.0)

@app.post("/traduire")
async def traduire(demande: Demande, requete: Request):    # (2) async def
    clients = requete.headers.get("X-Clients", "?")
    try:
        # (3) l_appel bloquant part dans un executeur, sous semaphore et delai
        return await garde.executer(
            _appeler_modele, demande.segment, demande.voisins, clients)
    except Sature as erreur:                               # (4) refuser vite
        raise HTTPException(status_code=503, detail=str(erreur)) from erreur
    except Delai as erreur:                                #     abandonner
        raise HTTPException(status_code=504, detail=str(erreur)) from erreur

@app.get("/sante")
def sante():
    return {"statut": "ok", "garde": garde.etat()}         # (5) le rendre visible
''')

# %% [markdown]
# Le cinquième point n'est pas un des quatre défauts, mais il vaut le détour :
# **exposer l'état de la garde dans `/sante`**. Sans ça, personne ne sait
# combien de requêtes ont été refusées, et un service qui refuse en silence
# ressemble, dans les journaux, à un service qui va bien.

# %% [markdown]
# ## OBS 3 · Rejouer l'assaut sur le service réparé
#
# La cellule ci-dessous arrête le service sans garde, en démarre un **avec** la
# garde de référence (`concurrence_max=1, file_max=4, timeout_s=60`), et lance
# l'assaut à **8 clients**.
#
# Pourquoi huit et pas quatre : cette garde tient `1 + 4 = 5` requêtes — une en
# traitement, quatre en file. À quatre clients, personne n'est refusé et on ne
# verrait rien. **Le seuil de refus est un chiffre que vous avez réglé
# vous-même**, pas une propriété du service : c'est le premier enseignement de
# cette cellule.
#
# **Attendu, et c'est le résultat qu'on cherche** : des `503` apparaissent dans
# les codes HTTP, et le p95 des requêtes **servies** devient **calculable** —
# au pire `(1 + file_max)` traductions d'attente, et jamais plus que
# `timeout_s`. Le service ne traduit pas plus vite — il n'a changé ni de modèle
# ni de machine — mais il **tient une promesse** : soit une réponse dans un
# délai qu'on sait annoncer, soit un refus immédiat.

# %%
if serveur is not None:
    serveur.arreter()
serveur = demarrer_en_fond(port=8000, repare=True)

socle.entete()
avec_garde = assaillir(serveur.url, 8, DUREE_PALIER, SEGMENTS)
socle.ligne(avec_garde)
print(f"""
  {avec_garde['ok']} traduites, {avec_garde['codes'].get(503, 0)} refusées (503),
  {avec_garde['codes'].get(504, 0)} abandonnées (504), {avec_garde['codes'].get(0, 0)} perdues côté client.
  p95 des traduites : {avec_garde['p95_ok']:.1f}s
  État de la garde  : {serveur.sante()['garde']}""")

# %% [markdown]
# **Lisez d'abord la répartition des codes**, pas le p95 global. Un `200` est une
# traduction, un `503` un refus immédiat, un `504` un abandon après attente, un
# `0` un client qui a renoncé de lui-même. Un service honnête sous charge produit
# des trois premiers — et **aucun du quatrième**.
#
# Et le p95 global n'est plus lisible : il est écrasé par des dizaines de
# milliers de refus à cinq millisecondes. **Le chiffre à tenir est `p95 ok`**,
# celui des requêtes servies. Un service qui refuse tout aurait un p95 global
# parfait, et c'est pour ça qu'on ne met jamais ce chiffre-là dans un
# engagement.
#
# Un dernier enseignement, involontaire et précieux : le nombre de requêtes a
# explosé. Nos clients **réessaient sans attendre** dès qu'ils sont refusés, et
# ils bombardent le service bien plus fort qu'avant la réparation. C'est le
# comportement qu'on trouve dans du vrai code appelant, et c'est ainsi qu'un
# incident mineur devient une panne : **un client qui réessaie doit attendre**,
# de plus en plus longtemps à chaque échec (*exponential backoff*), et de
# préférence en respectant l'en-tête `Retry-After` que le service lui renvoie.
# La garde protège du côté serveur ; la temporisation de réessai est la moitié
# du travail, et elle est du côté client.

# %% [markdown]
# ## BONUS 5 · Le palier de saturation
#
# « Combien d'utilisateurs ce service supporte-t-il ? » La réponse n'est jamais
# un nombre d'utilisateurs : c'est un **nombre d'appels simultanés, pour un
# engagement de latence donné**. C'est cette mesure-là qu'on fournit à un
# client, et elle se trouve par dichotomie.

# %%
def palier_sature(url: str, seuil_p95: float, segments: list[dict],
                  duree: float = 20.0, maximum: int = 16) -> int:
    """Le plus grand nombre de clients dont le p95 reste sous ``seuil_p95``."""
    # <<<BONUS 5 ★★ Recherche du palier de saturation
    # Par dichotomie entre 1 et « maximum » : cherchez le plus grand nombre de
    # clients dont le p95 reste sous seuil_p95. Utilisez assaillir(url, n,
    # duree, segments) à chaque essai et rendez ce nombre (0 si même 1 client
    # dépasse déjà le seuil).
    # Test : python tp.py test tp05 --bonus -k bonus5
    raise NotImplementedError(
        "BONUS 5 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp05-mise-en-service/README.md"
    )
    # >>>BONUS 5


# %% [markdown]
# ## LIRE 5 · Ce qui aurait vraiment réglé le problème
#
# **La garde protège ; elle n'accélère rien.** Elle transforme une dégradation
# invisible en refus explicite, ce qui est un immense progrès — mais le service
# traduit toujours un segment à la fois, en vingt secondes.
#
# Pour en servir davantage, par ordre de coût croissant :
#
# | Levier | Effet | Ce que ça coûte |
# |---|---|---|
# | **une carte graphique** | 20 s → ≈ 1 s par segment | le matériel, ou l'heure de GPU louée |
# | **le *batching* continu** (vLLM, TGI) | plusieurs requêtes dans un même passage du modèle, débit multiplié | un serveur d'inférence à exploiter, et un GPU pour que ça vaille le coup |
# | **plusieurs répliques** derrière un répartiteur | débit multiplié par le nombre de répliques | autant de machines, et un état à ne pas partager |
# | **quantifier davantage** (Q4 → Q3) | un peu de mémoire et de vitesse | de la qualité, à mesurer avant |
#
# Le Space gratuit — 2 vCPU, pas de GPU — n'offre aucun des trois premiers, et
# c'est aussi une leçon : **la première décision d'une mise en service est le
# choix du matériel**, et elle se prend avec des mesures comme celles que vous
# venez de faire, pas avec une intuition.

# %% [markdown]
# ## LIRE 6 · Comment ce Space est déployé
#
# *À lire, pas à faire : le Space appartient au formateur. C'est le mode
# d'emploi pour le jour où vous en déploierez un.*
#
# - **Le dossier** `tp05-mise-en-service/space/` contient tout : `app.py`,
#   `requirements.txt`, un `Dockerfile`, et un `README.md` dont l'en-tête YAML
#   configure le Space (`sdk: docker`, `app_port: 7860`). On pousse ce dossier
#   dans le dépôt git du Space, et HuggingFace construit l'image.
# - **Un seul processus sert les deux étages.** `app.py` déclare l'API
#   (`POST /traduire`, `GET /sante`), puis monte l'interface Gradio sur `/` avec
#   `gr.mount_gradio_app(app, interface, path="/")`. Un Space n'expose qu'**un
#   seul port** : c'est le montage qui permet d'avoir une page *et* une API sans
#   déployer deux services. L'ordre compte — les routes d'API sont déclarées
#   avant le montage, sinon `/` les avalerait.
# - **L'interface est un client de l'API, pas une deuxième implémentation.**
#   Elle fait un `POST /traduire` sur elle-même plutôt que d'appeler la fonction
#   du modèle en direct. C'est trois lignes de plus, et ça garantit qu'une
#   correction faite au back-end vaut pour la page comme pour le notebook.
# - **Le matériel** est *CPU basic*, l'offre gratuite : c'est un choix, il faut
#   que le service soit saturable en trente secondes par quatre stagiaires.
# - **Les secrets** se posent dans `Settings → Variables and secrets`, jamais
#   dans le dépôt : `LANGSMITH_API_KEY` en secret, `LANGSMITH_TRACING`,
#   `LANGSMITH_PROJECT` en variables. Elles sont injectées **à l'exécution**.
# - **Le piège numéro un**, et il vaut pour vos propres déploiements : une clé
#   LangSmith créée dans un espace de travail **européen** n'est pas reconnue
#   par l'endpoint américain, qui est celui par défaut. Les traces n'arrivent
#   jamais et **rien ne le signale**. D'où `LANGSMITH_ENDPOINT` posée
#   explicitement, et la route `/sante` qui affiche l'endpoint retenu : sur un
#   service déployé, ce qu'on ne peut pas interroger, on ne le sait pas.
# - **Un Space gratuit s'endort** après 48 h sans trafic, et le premier appel
#   après réveil paie le rechargement du modèle. Ce n'est pas un défaut du TP :
#   c'est le *cold start*, et il compte dans un engagement de service.

# %% [markdown]
# ## ARB 1 · Votre arbitrage
#
# Vous devez rendre ce service utilisable par une équipe de traducteurs, et vous
# avez un budget d'une journée de travail et zéro euro de matériel.
#
# **Écrivez en trois phrases** ce que vous faites en premier, et ce que vous
# annoncez comme engagement de service : quel p95, pour combien d'appels
# simultanés, et ce qui se passe au-delà.
#
# Appuyez-vous sur **vos** chiffres — ceux de la table d'OBS 1 et d'OBS 3 —, pas
# sur l'intuition. Et dites explicitement ce que vous **refusez** de promettre.
#
# > *Votre réponse :*
# >
# >

# %% [markdown]
# ## Avant de fermer
#
# Arrêtez le service : il occupe un port et, sans garde, un cœur de processeur.

# %%
if serveur is not None:
    serveur.arreter()
    serveur = None
