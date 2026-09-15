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
# - **le traçage gratuit** : une variable d'environnement, et chaque étape
#   devient visible dans LangSmith sans une ligne de code en plus.
#
# Ce qu'elle coûte : une dépendance de plus, des abstractions à apprendre, et
# une pile d'appels difficile à lire quand ça casse.

# %% [markdown]
# ## Préparation
#
# `modele_langchain()` choisit le modèle selon la même règle que
# `commun/moteur.py` : l'API Mistral si vous avez une clé, Ollama sinon. Le
# reste de la chaîne ne voit pas la différence, et c'est exactement ce qu'on
# attend d'une couche d'orchestration.

# %%
import os
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
os.chdir(RACINE)

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


chaine = construire_chaine(chercher, index_glossaire, modele)
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
# Le formateur vous donne une clé LangSmith pendant la séance. Dans `.env` :
#
# ```
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2_...
# LANGSMITH_PROJECT=formation-helios-<votre prénom>
# ```
#
# Si le compte est européen, ajoutez
# `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`. Sans cette ligne les
# traces n'arrivent jamais et **rien ne vous le dit** : c'est le piège numéro un.
#
# Relancez la cellule ci-dessous, ouvrez le projet, et retrouvez : le prompt
# exact, les voisins injectés, la réponse, la durée, les tokens. **Aucune ligne
# de code n'a été ajoutée pour ça.** C'est l'outil du TP 5.

# %%
print("Traçage actif :", os.environ.get("LANGSMITH_TRACING", "non"))
for segment in segments[:3]:
    print(f"  {chaine.invoke(segment['src'])}")

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
