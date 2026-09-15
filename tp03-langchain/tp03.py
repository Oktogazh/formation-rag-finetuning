# %% [markdown]
# # TP 3 — Orchestrer : chaîne, vérification, correction
#
# **Durée** : noyau ≈ 60 min · **Niveau** ★★★☆☆
#
# On refait le RAG du TP 2 avec LangChain, puis on ajoute ce qu'une chaîne rend
# simple : un contrôle automatique et une boucle de correction. Et on mesure ce
# que ça coûte.
#
# **Ce TP ne dépend d'aucun autre.** La recherche et le prompt augmenté viennent
# de `commun/`, pas de ce que vous avez écrit au TP 2.
#
# ## La question honnête, posée d'emblée
#
# LangChain ne traduit pas mieux. **C'est de la plomberie**, et la question au
# bureau sera : qu'est-ce qu'elle apporte, et à quel prix ?
#
# Ce qu'elle apporte, et que ce TP montre :
#
# - un assemblage **déclaratif** : la chaîne se lit comme un schéma, et chaque
#   morceau se remplace sans toucher aux autres ;
# - la même interface pour Ollama, Mistral ou autre chose. Vous avez déjà cette
#   abstraction dans `commun/moteur.py`, écrite à la main en quarante lignes :
#   comparez, c'est instructif dans les deux sens ;
# - **le traçage gratuit** : une variable réglée en haut du notebook, et chaque
#   étape devient visible dans LangSmith sans une ligne de code en plus. C'est
#   le RÉG 0 juste en dessous, et c'est le meilleur argument du lot.
#
# Ce qu'elle coûte : une dépendance de plus, des abstractions à apprendre, et
# une pile d'appels difficile à lire quand ça casse.

# %% [markdown]
# ## RÉG 0 · Régler votre clé LangSmith — **à faire maintenant, avant tout le reste**
#
# Ce TP est le premier où la chaîne fait plusieurs choses à la suite. Quand elle
# rendra une mauvaise traduction, la question sera : *lequel des quatre maillons
# a fauté ?* Un `print` ne vous le dira pas. **LangSmith** enregistre chaque
# étape — le prompt exact envoyé, les voisins injectés, la réponse brute, la
# durée, les tokens — et vous les montre en arbre.
#
# Et la démonstration du jour tient en une phrase : **vous n'écrirez aucune
# ligne de code pour tracer la chaîne.** Une clé réglée une fois, et LangChain
# trace tout seul. C'est le principal argument commercial d'une couche
# d'orchestration, et il vaut la peine d'être vu une fois.
#
# **Créez votre compte et votre jeton — ≈ 3 minutes :**
#
# 1. <https://smith.langchain.com> → **Sign up** (Google, GitHub ou e-mail). Le
#    palier gratuit *Developer* suffit : 5 000 traces par mois, aucune carte
#    bancaire ;
# 2. ⚠️ l'inscription demande une **région, `US` ou `EU`**, **définitive** pour
#    l'organisation. Prenez **`EU`** si vous hésitez — vos traces contiennent
#    la documentation du client ;
# 3. votre avatar en bas à gauche → **Settings** → **API Keys** → **Create API
#    Key**, type **Personal Access Token**. La clé commence par `lsv2_pt_` et
#    **n'est affichée qu'une fois** : copiez-la tout de suite.
#
# **Réglez-la ci-dessous**, comme on règle n'importe quel paramètre du TP.
# Rien d'autre à ouvrir, rien d'autre à éditer : le notebook n'est **jamais
# versionné** (`*.ipynb` dans `.gitignore` — `python tp.py notebooks` le
# refabrique à chaque fois depuis `tp03.py`), donc coller une clé ici ne finit
# ni dans un commit ni dans une pull request.
#
# Mettez **votre prénom** dans `LANGSMITH_PROJECT` : à quinze dans le même
# projet, vos traces se mélangent et l'exercice OBS 2 ne veut plus rien dire.

# %%
LANGSMITH_API_KEY = ""                      # <- collez votre clé lsv2_pt_… ici
LANGSMITH_PROJECT = "formation-helios-prenom"  # <- et votre prénom ici

# %% [markdown]
# La cellule ci-dessous branche le traçage avec ces deux réglages. **À exécuter
# avant la première traduction** : LangChain décide de tracer au moment de
# l'appel, mais son client se met en cache dès le premier envoi — régler la
# clé trop tard, c'est perdre les premières cellules.
#
# Le message qui s'affiche dit exactement ce qui s'est passé. Pas de clé ?
# **Le TP fonctionne quand même** — vous sautez l'exercice OBS 2, rien de plus.
# Clé refusée par les deux serveurs (US et EU) ? Un copier-coller a laissé un
# espace ou une fin de ligne — c'est l'erreur la plus fréquente.

# %%
import os
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
os.chdir(RACINE)

from commun import tracage

etat = tracage.activer(cle=LANGSMITH_API_KEY, projet=LANGSMITH_PROJECT)

# %% [markdown]
# ## Préparation
#
# `modele_langchain()` choisit le modèle selon la même règle que
# `commun/moteur.py` : l'API Mistral si vous avez une clé, Ollama sinon. Le
# reste de la chaîne ne voit pas la différence, et c'est exactement ce qu'on
# attend d'une couche d'orchestration.

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from commun import atelier, corpus
from commun.augmenter import GlossaireVectoriel, construire
from commun.consignes import CONSIGNE_STYLE
from commun.memoire import charger_memoire
from commun.recherche import chercheur


def modele_langchain(temperature: float = 0.0, max_tokens: int = 256):
    """Le modèle, enveloppé pour LangChain. Fourni."""
    from commun.moteur import MODELE_MISTRAL, MODELE_OLLAMA, URL_OLLAMA, moteur_demande

    choix = moteur_demande()
    if choix == "mistral-api":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=MODELE_MISTRAL, temperature=temperature,
                             max_tokens=max_tokens, api_key=os.environ["MISTRAL_API_KEY"])
    if choix == "factice":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=["La formule Bas autorise 20 requêtes par minute."])
    from langchain_ollama import ChatOllama

    return ChatOllama(model=MODELE_OLLAMA, temperature=temperature, num_predict=max_tokens,
                      num_ctx=8192, base_url=URL_OLLAMA)


memoire = charger_memoire()
glossaire = corpus.charger_glossaire()
index_glossaire = GlossaireVectoriel(glossaire)
segments = atelier.segments(n=20)
chercher = chercheur(memoire, "dense", k=3)
modele = modele_langchain()

print(f"Modèle : {type(modele).__name__} · {len(memoire)} segments de mémoire")

# %% [markdown]
# ## LIRE 1 · Les trois briques de LCEL
#
# LCEL (*LangChain Expression Language*) compose des objets `Runnable` avec
# l'opérateur `|`. Trois briques suffisent pour tout ce TP, et la cellule
# ci-dessous les fait tourner une par une.
#
# **Lisez la sortie et répondez :**
#
# 1. que reçoit `RunnableParallel` en entrée, et que rend-il ?
# 2. `RunnablePassthrough` a l'air de ne rien faire. À quoi sert-il, alors ?
# 3. dans quel ordre les trois branches du `RunnableParallel` s'exécutent-elles,
#    et est-ce que ça change quelque chose ici ?

# %%
demo = RunnableParallel({
    "longueur": RunnableLambda(len),
    "majuscules": RunnableLambda(str.upper),
    "tel quel": RunnablePassthrough(),
})
for cle, valeur in demo.invoke("Fakturan skickas varje vecka.").items():
    print(f"  {cle:12} {valeur}")

# %% [markdown]
# ## CODE 1 · Assembler la chaîne
#
# La chaîne s'invoque avec **une chaîne de caractères** (le segment suédois) et
# rend **une chaîne de caractères** (la traduction). Entre les deux, quatre
# étapes :
#
# 1. un `RunnableParallel` qui prépare trois valeurs à partir du segment : les
#    voisins (`chercher`), les termes du glossaire (`index_glossaire.chercher`),
#    et le segment lui-même ;
# 2. un `RunnableLambda` qui appelle `construire(...)` et rend la liste de
#    messages ;
# 3. le modèle ;
# 4. `StrOutputParser()`, qui extrait le texte du message rendu.
#
# **Le prompt reste celui de `commun/prompts.py`.** On ne le réécrit pas en
# `ChatPromptTemplate` : ce serait un deuxième format, et le TP 4 entraînera le
# modèle sur le premier.
#
# **Analogue visible** : le `RunnableParallel` de la cellule précédente a
# exactement la forme attendue.

# %%
def construire_chaine(chercher, index_glossaire, modele, consignes=CONSIGNE_STYLE):
    """La chaîne du cran 3 : recherche, prompt, modèle, texte."""
    en_messages = RunnableLambda(
        lambda d: construire(d["segment"], d["voisins"], d["glossaire"], consignes))
    # <<<CODE 1 ★★ Assembler la chaîne LCEL
    # Construisez un RunnableParallel avec exactement ces trois clés :
    # "voisins"   -> RunnableLambda(chercher)
    # "glossaire" -> RunnableLambda(index_glossaire.chercher)
    # "segment"   -> RunnablePassthrough()
    # puis composez avec l'opérateur | : preparation | en_messages | modele
    # | StrOutputParser(), et rendez le tout.
    # Test : python tp.py test tp03 -k code1
    raise NotImplementedError(
        "CODE 1 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp03-langchain/README.md"
    )
    # >>>CODE 1


# `with_config` ne change rien à ce que la chaîne calcule : il lui donne un nom
# et des étiquettes. Sans ça, LangSmith affiche « RunnableSequence » trois fois
# et vous ne savez plus laquelle des trois variantes vous regardez.
chaine = construire_chaine(chercher, index_glossaire, modele).with_config(
    run_name="helios-chaine-seule",
    tags=["tp03", "cran3", "chaine-seule"],
    metadata={"recherche": "dense", "k": 3, "modele": type(modele).__name__},
)
print(chaine.invoke(segments[0]["src"]))

# %% [markdown]
# ## OBS 1 · La chaîne fait-elle la même chose que le TP 2 ?
#
# Vous devez retrouver, au bruit près, les chiffres du TP 2 en recherche dense.
# **Si l'écart est grand, quelque chose a changé dans le prompt sans que vous le
# vouliez** — et c'est exactement le genre de dérive que ce TP doit vous
# apprendre à repérer.

# %%
from commun.mesure import Sortie
from commun.prompts import nettoyer_sortie

sans_controle = atelier.mesurer(
    "tp03-chaine",
    segments,
    lambda s: Sortie(texte=nettoyer_sortie(chaine.invoke(s["src"])), appels=1),
    titre="LangChain — chaîne seule",
)

# %% [markdown]
# ## LIRE 2 · Les quatre contrôles de l'agence
#
# La procédure de relecture (`data/corpus/helios-sv/docs/004-procedure-relecture.md`)
# impose quatre contrôles automatiques avant qu'un humain regarde. Ils existent
# déjà dans `commun/verification.py` : ils servaient à **noter**, ils vont
# maintenant **déclencher**.
#
# **Ouvrez `commun/verification.py`, puis répondez :**
#
# 1. lequel des quatre contrôles attrape « abonnement » au lieu de « formule » ?
# 2. lequel attraperait « tous les 14 jours » quand la source dit 180 ?
# 3. lequel ne se déclenche jamais sur ce corpus, et pourquoi ?

# %%
from commun.verification import verifier

source = "Abonnemanget Bas tillåter 20 förfrågningar per minut."
for candidate in ["La formule Bas autorise 20 requêtes par minute.",
                  "La formule Bas autorise 20 demandes par minute.",
                  "La formule Bas autorise 50 requêtes par minute.",
                  "Tu peux utiliser la formule Bas."]:
    anomalies = verifier(source, candidate, glossaire)
    print(f"  {candidate}")
    print(f"      → {[str(a) for a in anomalies] or 'rien à redire'}")

# %% [markdown]
# ## RÉG 1 · La boucle de correction
#
# Si le contrôle trouve un défaut, on redemande au modèle en lui disant lequel.
# Avec un compteur : une boucle d'agent sans compteur est un incident de
# production, pas une audace.
#
# **Changez `MAX_TENTATIVES`** entre 1 (aucune correction) et 3, et regardez
# deux colonnes : `Chiffres` et `Appels`.
#
# Et préparez-vous à un résultat désagréable : **la correction améliore les
# chiffres et fait baisser BLEU**. Un modèle à qui on demande de corriger un
# détail réécrit toute la phrase. Elle reste juste, elle s'éloigne de la
# référence, et BLEU compte des n-grammes.

# %%
MAX_TENTATIVES = 2


# `@traceable` est l'autre moitié du traçage : la chaîne LCEL est instrumentée
# toute seule, mais cette fonction-ci est du Python ordinaire. Sans décorateur,
# LangSmith montrerait deux traces sans lien ; avec, il montre **un** arbre —
# premier appel, contrôle, deuxième appel — et c'est là qu'on lit le coût réel
# d'une boucle d'agent.
@tracage.traceable(name="helios-chaine-verifiee", run_type="chain",
                   tags=["tp03", "cran3", "verification"])
def traduire_et_corriger(segment: dict) -> Sortie:
    traduction = nettoyer_sortie(chaine.invoke(segment["src"]))
    appels = 1
    anomalies = verifier(segment["src"], traduction, glossaire)
    while anomalies and appels < MAX_TENTATIVES:
        remarques = "\n".join(f"- {a}" for a in anomalies)
        messages = construire(segment["src"], chercher(segment["src"]),
                              index_glossaire.chercher(segment["src"]), CONSIGNE_STYLE)
        messages += [
            {"role": "assistant", "content": traduction},
            {"role": "user", "content": f"Le contrôle a relevé :\n{remarques}\n"
                                        f"Rends la traduction corrigée, et rien d'autre."},
        ]
        traduction = nettoyer_sortie(StrOutputParser().invoke(modele.invoke(messages)))
        appels += 1
        anomalies = verifier(segment["src"], traduction, glossaire)
    return Sortie(texte=traduction, appels=appels, anomalies=[str(a) for a in anomalies])


avec_controle = atelier.mesurer(
    "tp03-chaine-verifiee",
    segments,
    traduire_et_corriger,
    titre=f"LangChain — chaîne + vérification (max {MAX_TENTATIVES} tentatives)",
)
atelier.comparer(("chaîne seule", sans_controle), ("+ vérification", avec_controle))

# %% [markdown]
# ## RÉG 2 · Et si on réparait sans appeler le modèle ?
#
# Une anomalie de chiffres se corrige par un remplacement de texte. Zéro appel,
# zéro token, zéro seconde — et plus fiable qu'un second appel, parce qu'un
# modèle qui s'est trompé une fois se trompe souvent deux.
#
# **Passez `REPARER_LOCALEMENT` à `True`**, relancez, et comparez les trois
# lignes. La question à se poser à chaque étape d'une chaîne : *ai-je vraiment
# besoin d'un modèle de langue pour ça ?*

# %%
from commun.verification import reparer_chiffres

REPARER_LOCALEMENT = True


@tracage.traceable(name="helios-reparation-locale", run_type="chain",
                   tags=["tp03", "cran3", "reparation-locale"])
def traduire_et_reparer(segment: dict) -> Sortie:
    traduction = nettoyer_sortie(chaine.invoke(segment["src"]))
    if REPARER_LOCALEMENT:
        traduction = reparer_chiffres(segment["src"], traduction)
    return Sortie(texte=traduction, appels=1,
                  anomalies=[str(a) for a in verifier(segment["src"], traduction, glossaire)])


repare = atelier.mesurer("tp03-chaine-reparee", segments, traduire_et_reparer,
                         titre="LangChain — réparation locale des chiffres",
                         enregistrer=False)
atelier.comparer(("chaîne seule", sans_controle), ("+ vérification", avec_controle),
                 ("+ réparation locale", repare))

# %% [markdown]
# ## OBS 2 · Voir la chaîne tourner
#
# Tout ce qui précède est déjà parti dans LangSmith — vous avez réglé la clé au
# RÉG 0, et **pas une ligne de code de plus n'a été écrite pour ça**. La cellule
# ci-dessous ajoute trois traductions bien étiquetées,
# vide la file d'envoi, compte ce qui est arrivé et vous donne le lien.
#
# **Ouvrez le projet et répondez, en regardant l'arbre :**
#
# 1. dépliez une trace `helios-chaine-seule`. Combien de nœuds enfants ?
#    Retrouvez-y le `RunnableParallel`, et vérifiez que ses trois branches —
#    `voisins`, `glossaire`, `segment` — sont bien **côte à côte et non en
#    file** : c'est le parallélisme du LIRE 1, rendu visible ;
# 2. cliquez sur le nœud du modèle. Vous voyez le **prompt exact** envoyé,
#    voisins compris. Comptez les tokens d'entrée, et comparez-les aux 4 461
#    tokens du mur du TP 1 ;
# 3. filtrez sur le tag `verification`. Ouvrez une trace qui a **deux** appels
#    au modèle : vous lisez la remarque du contrôle, puis la réécriture. C'est
#    le coût de la boucle, en clair ;
# 4. comparez la latence médiane des tags `chaine-seule` et `verification`. Le
#    rapport est-il celui que vous prédisiez à la lecture de la colonne
#    `Appels` ?
#
# **Si le compte est vide**, ne cherchez pas au hasard : remontez au RÉG 0,
# corrigez `LANGSMITH_API_KEY` et relancez sa cellule — elle vous dira lequel
# des trois problèmes vous avez (pas de clé, clé refusée, traçage coupé).
#
# C'est l'outil du TP 5, où il servira à voir un service se faire bombarder.

# %%
for segment in segments[:3]:
    print(f"  {chaine.invoke(segment['src'], config={'tags': ['obs2']})}")

tracage.rapport()

# %% [markdown]
# ## ARB 1 · Votre arbitrage
#
# Votre boucle de correction gagne des points sur les chiffres et en perd sur
# BLEU. **Écrivez en trois phrases** ce que vous livrez au client Helios :
# la chaîne seule, la chaîne vérifiée, ou la chaîne avec réparation locale.
#
# Donnez le chiffre de votre propre table qui justifie le choix, et dites au nom
# de quel indicateur vous tranchez. C'est la même question qu'au TP 1, mais
# cette fois elle coûte des appels.
#
# > *Votre réponse :*
# >
# >
