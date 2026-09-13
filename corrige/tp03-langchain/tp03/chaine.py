"""TP 3 — le meme RAG qu'au TP 2, mais assemble avec LangChain.

Pourquoi refaire ce qui marche deja ? Parce que la question qu'on vous posera au
bureau n'est pas « est-ce que LangChain traduit mieux » — il ne traduit pas du
tout, c'est de la plomberie. La question est : **qu'est-ce que la plomberie
apporte, et ce qu'elle coute.**

Ce qu'elle apporte, et que vous verrez dans ce TP :

* un assemblage **declaratif** : la chaine se lit comme un schema, et chaque
  morceau se remplace sans toucher aux autres ;
* la meme interface pour Ollama, Mistral, OpenAI ou n'importe quoi d'autre ;
* **le tracage gratuit** : posez ``LANGSMITH_TRACING=true`` dans votre ``.env``
  et chaque etape devient visible, sans une ligne de code en plus. C'est ce que
  le TP 5 exploitera pour diagnostiquer un incident.

Ce qu'elle coute : une dependance de plus, des abstractions a apprendre, et une
pile d'appels difficile a lire quand ca casse.
"""

from __future__ import annotations

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough


def modele_langchain(temperature: float = 0.0, max_tokens: int = 256):
    """Le modele, choisi selon la meme regle que ``commun/moteur.py``. Fourni.

    Une cle Mistral dans l'environnement ? On prend l'API. Sinon, Ollama en
    local. Le reste de la chaine ne voit pas la difference : c'est exactement ce
    qu'on attend d'une couche d'orchestration.
    """
    from commun.moteur import MODELE_MISTRAL, MODELE_OLLAMA, URL_OLLAMA, moteur_demande

    choix = moteur_demande()
    if choix == "mistral-api":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=MODELE_MISTRAL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.environ["MISTRAL_API_KEY"],
        )
    if choix == "factice":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=["La formule Bas autorise 20 requêtes par minute."])
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=MODELE_OLLAMA,
        temperature=temperature,
        num_predict=max_tokens,
        num_ctx=8192,
        base_url=URL_OLLAMA,
    )


def construire_chaine(chercher, glossaire: list[dict], modele, consignes: str = "") -> Runnable:
    """La chaine du cran 3 : recherche, prompt, modele, texte.

    Elle s'invoque avec **une chaine de caracteres** (le segment suedois) et
    rend **une chaine de caracteres** (la traduction).

    Le prompt reste celui de ``commun/prompts.py``. On ne le reecrit pas en
    ``ChatPromptTemplate`` : ce serait un deuxieme format, et le TP 4 entrainera
    le modele sur le premier.
    """
    from tp02.augmenter import construire, glossaire_pertinent

    # <<<TODO 1 ★★ Assembler la chaine LCEL
    #! Composez, avec l'operateur | :
    #!   1. RunnableParallel qui prepare trois valeurs a partir du segment :
    #!        "voisins"   -> chercher(segment)
    #!        "glossaire" -> glossaire_pertinent(segment, glossaire)
    #!        "segment"   -> le segment lui-meme (RunnablePassthrough())
    #!   2. un RunnableLambda qui appelle construire(segment, voisins, glossaire,
    #!      consignes) et rend la liste de messages ;
    #!   3. le modele ;
    #!   4. StrOutputParser().
    #! Analogue : chaine_de_correction() juste en dessous est ecrite en entier,
    #! dans le meme style.
    #! Test : python tp.py test tp03 -k todo1
    return (
        RunnableParallel(
            {
                "voisins": RunnableLambda(chercher),
                "glossaire": RunnableLambda(lambda src: glossaire_pertinent(src, glossaire)),
                "segment": RunnablePassthrough(),
            }
        )
        | RunnableLambda(
            lambda donnees: construire(
                donnees["segment"], donnees["voisins"], donnees["glossaire"], consignes
            )
        )
        | modele
        | StrOutputParser()
    )
    # >>>TODO 1


def chaine_de_correction(modele) -> Runnable:
    """La chaine qui repare. Fournie — c'est votre modele pour le TODO 1.

    Elle s'invoque avec ``{"src": ..., "traduction": ..., "anomalies": ...}``.
    """
    gabarit = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es relecteur pour une agence de traduction. On te donne un "
                "segment suédois, sa traduction française et la liste des "
                "défauts relevés par le contrôle automatique. Tu rends la "
                "traduction corrigée, et rien d'autre.",
            ),
            (
                "human",
                "sv: {src}\nfr proposée: {traduction}\n\n"
                "Défauts à corriger :\n{anomalies}\n\nfr corrigée:",
            ),
        ]
    )
    return gabarit | modele | StrOutputParser()
