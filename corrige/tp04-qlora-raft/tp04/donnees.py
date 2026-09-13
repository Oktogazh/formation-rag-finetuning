"""TP 4 — fabriquer le jeu d'entrainement. C'est 80 % du travail de fine-tuning.

On n'entraine pas le modele a traduire : il sait deja. On l'entraine a **se
servir du contexte qu'on lui donne**, dans le format exact ou on le lui donnera
en production. Cette technique porte un nom, **RAFT** (*Retrieval-Augmented
Fine-Tuning*, Zhang et al., 2024), et elle tient en une idee :

    pendant l'entrainement, le contexte contient parfois les bons documents,
    parfois des documents qui n'ont rien a voir.

Un modele entraine uniquement avec le bon contexte apprend a le recopier ; le
jour ou la recherche se trompe, il recopie une betise avec assurance. Un modele
qui a vu des distracteurs pendant l'entrainement apprend a **trier**.

Deux regles que ce fichier applique et qu'il faut connaitre :

1. **Le prompt d'entrainement est exactement le prompt d'inference.** Meme
   gabarit, meme ordre de blocs, meme ponctuation — ``commun/prompts.py``. Un
   ecart ici ne fait pas planter le modele, il le rend simplement moins bon que
   prevu, et c'est indetectable sans mesure.
2. **On n'entraine jamais sur ce qu'on mesure.** Les segments de la memoire qui
   sont des quasi-copies des segments d'evaluation sont retires du jeu
   d'entrainement.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from commun import DOSSIER_RESULTATS
from commun.corpus import charger_evaluation, charger_glossaire
from commun.prompts import construire_messages

DOSSIER_DONNEES = Path(DOSSIER_RESULTATS) / "tp04" / "donnees"
SEUIL_FUITE = 0.98
GRAINE = 13


def separer(memoire: list[dict], part_validation: float = 0.1, graine: int = GRAINE):
    """Partage la memoire en entrainement et validation, **sans fuite**.

    Un segment de memoire trop proche d'un segment d'evaluation est ecarte des
    deux jeux : l'entrainer reviendrait a donner les reponses de l'examen.

    Le seuil est un arbitrage, pas une verite : a 0,98 on ecarte 50 segments sur
    444 ; a 0,95 on en ecarterait 226, soit la moitie de la memoire. On garde
    0,98 et on le dit.

    Rend ``(entrainement, validation, ecartes)``.
    """
    from tp02.recherche import similarite

    sources_eval = [s["src"] for s in charger_evaluation()]
    # <<<TODO 1 ★★ Separer sans fuite
    #! 1. Constituez « retenus » : les segments de la memoire dont la similarite
    #!    maximale avec un des sources_eval est STRICTEMENT inferieure a
    #!    SEUIL_FUITE. Les autres vont dans « ecartes ».
    #!    Indice : max(similarite(s["src"], e) for e in sources_eval)
    #! 2. Melangez « retenus » avec random.Random(graine).shuffle — la graine
    #!    fixe rend le partage reproductible, sinon deux stagiaires ne mesurent
    #!    pas la meme chose.
    #! 3. Coupez : les part_validation derniers pour cent en validation, le
    #!    reste en entrainement.
    #! Test : python tp.py test tp04 -k todo1
    retenus, ecartes = [], []
    for segment in memoire:
        proximite = max(similarite(segment["src"], source) for source in sources_eval)
        (ecartes if proximite >= SEUIL_FUITE else retenus).append(segment)
    random.Random(graine).shuffle(retenus)
    coupe = max(1, int(len(retenus) * part_validation))
    return retenus[coupe:], retenus[:coupe], ecartes
    # >>>TODO 1


def exemple_raft(segment: dict, memoire: list[dict], glossaire: list[dict],
                 k: int = 3, p_oracle: float = 0.8, rng: random.Random | None = None,
                 consignes: str = "") -> dict:
    """Un exemple d'entrainement RAFT, au format ``{"messages": [...]}``.

    Avec une probabilite ``p_oracle``, le contexte contient les ``k`` vrais
    voisins. Sinon il contient ``k`` **distracteurs** : des segments de la
    memoire qui ne sont pas dans les dix plus proches. Dans les deux cas, la
    reponse attendue reste la bonne traduction — c'est comme cela que le modele
    apprend a ignorer un contexte inutile au lieu de le recopier.
    """
    from tp02.recherche import rechercher_lexical

    rng = rng or random.Random(GRAINE)
    from tp02.augmenter import glossaire_pertinent

    autres = [s for s in memoire if s["id"] != segment["id"]]
    termes = glossaire_pertinent(segment["src"], glossaire)

    # <<<TODO 2 ★★★ Construire l'exemple RAFT
    #! 1. proches = rechercher_lexical(segment["src"], autres, k=10)
    #! 2. si rng.random() < p_oracle : voisins = les k premiers de proches ;
    #!    sinon : voisins = rng.sample(<les segments de « autres » qui ne sont
    #!    PAS dans proches>, k)  — ce sont les distracteurs.
    #! 3. messages = construire_messages(segment["src"], voisins, termes, consignes)
    #! 4. rendez {"messages": messages + [{"role": "assistant",
    #!                                     "content": segment["tgt"]}]}
    #! Attention : le segment ne doit jamais figurer dans son propre contexte.
    #! C'est deja garanti par « autres », ne le cassez pas.
    #! Test : python tp.py test tp04 -k todo2
    proches = rechercher_lexical(segment["src"], autres, k=10)
    identifiants_proches = {s["id"] for s in proches}
    if rng.random() < p_oracle:
        voisins = proches[:k]
    else:
        lointains = [s for s in autres if s["id"] not in identifiants_proches]
        voisins = rng.sample(lointains, min(k, len(lointains)))
    messages = construire_messages(segment["src"], voisins, termes, consignes)
    return {"messages": messages + [{"role": "assistant", "content": segment["tgt"]}]}
    # >>>TODO 2


def exemple_sans_contexte(segment: dict, glossaire: list[dict], consignes: str = "") -> dict:
    """BONUS — un exemple sans aucun voisin.

    RAFT recommande d'en glisser quelques-uns : sinon le modele apprend qu'il y
    a **toujours** un bloc « Memoire de traduction », et le jour ou la recherche
    ne rend rien, il est desoriente par un prompt qu'il n'a jamais vu.
    """
    from tp02.augmenter import glossaire_pertinent

    # <<<BONUS 4 ★ Exemple sans contexte
    #! Meme forme que exemple_raft, mais avec une liste de voisins vide.
    #! Le glossaire pertinent, lui, reste present.
    #! Test : python tp.py test tp04 --bonus -k bonus4
    messages = construire_messages(
        segment["src"], [], glossaire_pertinent(segment["src"], glossaire), consignes
    )
    return {"messages": messages + [{"role": "assistant", "content": segment["tgt"]}]}
    # >>>BONUS 4


def construire_jeux(k: int = 3, p_oracle: float = 0.8, part_sans_contexte: float = 0.1) -> dict:
    """Ecrit ``train.jsonl`` et ``valid.jsonl``. Fourni.

    Le format ``{"messages": [...]}`` est celui que ``mlx-lm`` attend, et c'est
    aussi celui que ``trl`` accepte : un seul fichier pour les deux mondes.
    """
    from tp01.prompt import consigne_systeme
    from tp02.memoire import charger_memoire

    memoire = charger_memoire()
    glossaire = charger_glossaire()
    consignes = consigne_systeme()
    entrainement, validation, ecartes = separer(memoire)

    rng = random.Random(GRAINE)
    DOSSIER_DONNEES.mkdir(parents=True, exist_ok=True)

    compte = {"oracle": 0, "distracteurs": 0, "sans_contexte": 0}
    for nom, lot in (("train", entrainement), ("valid", validation)):
        chemin = DOSSIER_DONNEES / f"{nom}.jsonl"
        with chemin.open("w", encoding="utf-8") as f:
            for segment in lot:
                if rng.random() < part_sans_contexte:
                    exemple = exemple_sans_contexte(segment, glossaire, consignes)
                    compte["sans_contexte"] += 1
                else:
                    avant = rng.random()
                    rng_local = random.Random(hash(segment["id"]) & 0xFFFFFFFF)
                    exemple = exemple_raft(
                        segment, memoire, glossaire, k, p_oracle, rng_local, consignes
                    )
                    compte["oracle" if avant < p_oracle else "distracteurs"] += 1
                f.write(json.dumps(exemple, ensure_ascii=False) + "\n")

    longueurs = [
        sum(len(m["content"]) for m in json.loads(l)["messages"])
        for l in (DOSSIER_DONNEES / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    print(f"""
Jeu d'entrainement RAFT
=======================
  Entrainement       {len(entrainement)} exemples
  Validation         {len(validation)} exemples
  Ecartes (fuite)    {len(ecartes)} segments trop proches de l'evaluation (seuil {SEUIL_FUITE})
  Contexte oracle    ~{100 * p_oracle:.0f} % des exemples
  Sans contexte      ~{100 * part_sans_contexte:.0f} % des exemples
  Longueur moyenne   {sum(longueurs) / max(len(longueurs), 1):.0f} caracteres par exemple

  Ecrit dans {DOSSIER_DONNEES}

  Ouvrez train.jsonl et lisez trois lignes. C'est le seul moyen de voir ce que
  le modele va reellement apprendre — et c'est la que se trouvent 80 % des
  erreurs de fine-tuning.""")
    return {"entrainement": len(entrainement), "validation": len(validation),
            "ecartes": len(ecartes), **compte}
