"""TP 6 — le graphe : router chaque segment vers le moyen le moins cher qui suffit.

Les cinq TP precedents ont construit cinq moyens, du moins cher au plus cher :

    reutiliser  0 appel     la memoire suffit, il n'y a qu'a reporter les chiffres
    RAG         1 appel     le modele traduit avec les voisins et le glossaire
    correction  2 appels    le controle a trouve un defaut, on redemande

Aucun n'est bon partout. **Le travail d'ingenierie, c'est de choisir lequel
s'applique a quel segment** — et c'est exactement ce que fait un graphe.

Ce que le tableau final montre, et qui surprend toujours : le chemin le moins
cher est aussi le plus exact sur la moitie du corpus. On ne le savait pas avant
de mesurer, et c'est pour ca qu'on mesure.
"""

from __future__ import annotations

from tp06.etat import EtatTraduction
from tp06.noeuds import MAX_TENTATIVES, SEUIL_REUTILISATION, difference_chiffres_seulement


def router(etat: EtatTraduction) -> str:
    """Apres analyse : reutiliser la memoire, ou passer par le modele ?

    La regle est celle du client (``docs/003-consignes-client.md``) :
    au-dessus de 95 % de correspondance, on reutilise le segment de la memoire
    **apres verification des chiffres**. Si autre chose qu'un nombre a change,
    on ne reutilise pas.
    """
    # <<<TODO 1 ★★ Le routeur
    #! Rendez la chaine "reutiliser" si les DEUX conditions sont remplies :
    #!   - etat["taux_memoire"] >= SEUIL_REUTILISATION ;
    #!   - difference_chiffres_seulement(etat["segment"], etat["voisin"]["src"])
    #!     est vrai.
    #! Sinon, rendez "recuperer".
    #! Attention : les deux conditions, pas une seule. Un segment tres proche
    #! dont un MOT a change ne se reutilise pas — c'est la faute la plus chere
    #! qu'un outil de TAO puisse produire.
    #! Test : python tp.py test tp06 -k todo1
    if etat["taux_memoire"] >= SEUIL_REUTILISATION and difference_chiffres_seulement(
        etat["segment"], etat["voisin"]["src"]
    ):
        return "reutiliser"
    return "recuperer"
    # >>>TODO 1


def apres_verification(etat: EtatTraduction) -> str:
    """Apres controle : corriger, ou livrer ?"""
    # <<<TODO 2 ★★ La sortie de boucle
    #! Rendez "corriger" si etat["anomalies"] n'est pas vide ET que
    #! etat["tentatives"] est strictement inferieur a MAX_TENTATIVES.
    #! Sinon, rendez "livrer".
    #! Sans la seconde condition, un segment que le modele n'arrive pas a
    #! corriger boucle indefiniment. Une boucle d'agent sans compteur est un
    #! incident de production, pas une audace.
    #! Test : python tp.py test tp06 -k todo2
    if etat["anomalies"] and etat["tentatives"] < MAX_TENTATIVES:
        return "corriger"
    return "livrer"
    # >>>TODO 2


def construire_graphe(noeuds: dict, avec_rag: bool = True):
    """Assemble le graphe et le compile.

    Topologie visee :

        analyser ──┬─(reutiliser)──────────────────────────┐
                   └─(recuperer)── traduire ── verifier ──┬┴─ livrer ── FIN
                                       ▲                  │
                                       └──── corriger ◄───┘
    """
    from langgraph.graph import END, StateGraph

    graphe = StateGraph(EtatTraduction)
    for nom, fonction in noeuds.items():
        graphe.add_node(nom, fonction)

    # <<<TODO 3 ★★ Cabler le graphe
    #! Dans l'ordre :
    #!   graphe.set_entry_point("analyser")
    #!   graphe.add_conditional_edges("analyser", router,
    #!         {"reutiliser": "reutiliser", "recuperer": "recuperer"})
    #!   graphe.add_edge("recuperer", "traduire")
    #!   graphe.add_edge("traduire", "verifier")
    #!   graphe.add_conditional_edges("verifier", apres_verification,
    #!         {"corriger": "corriger", "livrer": "livrer"})
    #!   graphe.add_edge("corriger", "verifier")
    #!   graphe.add_edge("reutiliser", "livrer")
    #!   graphe.add_edge("livrer", END)
    #! La boucle corriger -> verifier est ce qui distingue un graphe d'une
    #! chaine : une chaine ne revient jamais en arriere.
    #! Test : python tp.py test tp06 -k todo3
    graphe.set_entry_point("analyser")
    graphe.add_conditional_edges(
        "analyser", router, {"reutiliser": "reutiliser", "recuperer": "recuperer"}
    )
    graphe.add_edge("recuperer", "traduire")
    graphe.add_edge("traduire", "verifier")
    graphe.add_conditional_edges(
        "verifier", apres_verification, {"corriger": "corriger", "livrer": "livrer"}
    )
    graphe.add_edge("corriger", "verifier")
    graphe.add_edge("reutiliser", "livrer")
    graphe.add_edge("livrer", END)
    # >>>TODO 3

    if not avec_rag:
        return _graphe_sans_rag(noeuds)
    return graphe.compile()


def _graphe_sans_rag(noeuds: dict):
    """BONUS — le modele adapte, seul, sans voisins ni glossaire.

    La question du dernier apres-midi : **le fine-tuning remplace-t-il le RAG ?**
    On ne peut y repondre qu'en mesurant le modele adapte **prive de contexte**.
    Ce graphe-la sert exactement a ca : analyser, traduire, verifier, livrer.
    """
    from langgraph.graph import END, StateGraph

    graphe = StateGraph(EtatTraduction)
    for nom in ("analyser", "traduire", "verifier", "corriger", "livrer"):
        graphe.add_node(nom, noeuds[nom])

    # <<<BONUS 6 ★★ Le graphe sans RAG
    #! Meme graphe, mais sans le noeud « recuperer » ni le noeud
    #! « reutiliser » : analyser va directement a traduire, le reste est
    #! identique (verifier, la boucle de correction, livrer, END).
    #! Les voisins et le glossaire restent vides : c'est le but.
    #! Test : python tp.py test tp06 --bonus -k bonus6
    graphe.set_entry_point("analyser")
    graphe.add_edge("analyser", "traduire")
    graphe.add_edge("traduire", "verifier")
    graphe.add_conditional_edges(
        "verifier", apres_verification, {"corriger": "corriger", "livrer": "livrer"}
    )
    graphe.add_edge("corriger", "verifier")
    graphe.add_edge("livrer", END)
    return graphe.compile()
    # >>>BONUS 6
