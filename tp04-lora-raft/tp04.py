# %% [markdown]
# # TP 4 — LoRA et RAFT : apprendre au modèle à se servir du contexte
#
# **Durée** : noyau ≈ 90 min, dont ~10 min d'entraînement · **Niveau** ★★★★☆
#
# **Ce TP ne dépend d'aucun autre.** Et il n'a besoin d'aucune carte graphique.
#
# ## Le modèle change, et il faut savoir pourquoi
#
# Fine-tuner les 3 milliards de paramètres de Ministral demande un GPU que
# personne n'a ici. Deux façons de traiter ça : regarder le formateur le faire,
# ou faire l'opération en vrai sur un modèle assez petit pour tenir sur un
# processeur. On choisit la seconde.
#
# Le TP 4 travaille donc sur **SmolLM2-135M-Instruct** : 135 millions de
# paramètres, 270 Mo, une génération en moins d'une seconde sur votre machine.
#
# **Ses traductions sont mauvaises.** Vous allez le voir tout de suite, et c'est
# une bonne nouvelle : l'écart entre « avant » et « après » sera d'autant plus
# lisible. Ce que le TP démontre reste vrai à toutes les tailles :
#
# - un adaptateur LoRA apprend une **forme** — un format de sortie, un
#   vocabulaire imposé, un registre ;
# - il n'apprend **pas de connaissance** ;
# - la taille du modèle qu'on peut fine-tuner est une **décision matérielle**,
#   et c'est la première question à poser quand on vous demande un budget.
#
# Ne comparez pas les chiffres de ce TP à ceux des autres. Comparez « avant » et
# « après », sur le même petit modèle.
#
# ## Les trois mots, séparés
#
# **LoRA.** On gèle le modèle et on entraîne, à côté, deux petites matrices par
# couche. Quelques centaines de milliers de paramètres au lieu de 135 millions.
# Le résultat, l'« adaptateur », pèse quelques mégaoctets et se branche ou se
# débranche.
#
# **Quantization.** Stocker les poids sur 4 bits au lieu de 16. C'est ce qui
# permet d'entraîner un gros modèle sur une petite carte — on n'en a pas besoin
# ici, notre modèle tient déjà.
#
# **RAFT** (*Retrieval-Augmented Fine-Tuning*, Zhang et al., 2024) est la vraie
# idée du TP : pendant l'entraînement, le contexte contient **parfois** les bons
# voisins, parfois des distracteurs. Un modèle entraîné uniquement avec le bon
# contexte apprend à le recopier ; le jour où la recherche se trompe, il recopie
# une bêtise avec assurance.

# %% [markdown]
# ## Préparation
#
# La cellule charge le petit modèle. Comptez quelques secondes, plus le
# téléchargement la première fois (270 Mo).

# %%
import os
import pathlib
import random
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
os.chdir(RACINE)

from commun import atelier, corpus
from commun.augmenter import GlossaireVectoriel
from commun.consignes import CONSIGNE_STYLE
from commun.memoire import charger_memoire
from commun.petit_modele import DOSSIER_ADAPTATEUR, MoteurPetit, entrainer_lora
from commun.prompts import construire_messages
from commun.recherche import rechercher_lexical, similarite

GRAINE = 13
memoire = charger_memoire()
glossaire = corpus.charger_glossaire()
index_glossaire = GlossaireVectoriel(glossaire)
segments = atelier.segments(n=20)

base = MoteurPetit()
print(f"Modèle : {base.nom()} · {len(memoire)} segments de mémoire")

# %% [markdown]
# ## OBS 1 · Ce que le petit modèle sait faire avant l'entraînement
#
# On lui donne le prompt complet — consigne, glossaire, voisins — et on regarde
# ce qu'il rend. Ne vous attendez à rien de bon : à 135 millions de paramètres,
# il ne sait pas traduire le suédois.
#
# **Notez précisément ce qu'il fait de travers.** Recopie-t-il le suédois ?
# Répond-il en anglais ? Continue-t-il le prompt au lieu d'y répondre ? C'est ce
# comportement-là que l'entraînement va corriger, et vous saurez dire lequel.

# %%
for segment in segments[:3]:
    messages = construire_messages(
        segment["src"], voisins=rechercher_lexical(segment["src"], memoire, 3),
        glossaire=index_glossaire.chercher(segment["src"]), consignes=CONSIGNE_STYLE)
    print(f"  sv   {segment['src']}")
    print(f"  réf  {segment['tgt']}")
    print(f"  brut {base.generer(messages).texte.strip()[:110]!r}\n")

# %% [markdown]
# ## Le jeu d'entraînement : séparer sans fuite
#
# **Fourni, mais lisez-le : c'est la règle la plus violée du fine-tuning.**
#
# Un segment de mémoire trop proche d'un segment d'évaluation est écarté des
# deux jeux. L'entraîner reviendrait à donner les réponses de l'examen, et votre
# mesure finale serait flatteuse et fausse.
#
# Le seuil de 0,98 est un arbitrage, et il faut le dire : à 0,98 on écarte une
# cinquantaine de segments sur 444 ; à 0,95 on en écarterait la moitié.

# %%
SEUIL_FUITE = 0.98

sources_eval = [s["src"] for s in corpus.charger_evaluation()]
retenus, ecartes = [], []
for segment in memoire:
    proximite = max(similarite(segment["src"], source) for source in sources_eval)
    (ecartes if proximite >= SEUIL_FUITE else retenus).append(segment)
random.Random(GRAINE).shuffle(retenus)
coupe = max(1, len(retenus) // 10)
validation, entrainement = retenus[:coupe], retenus[coupe:]

print(f"  entraînement {len(entrainement)} · validation {len(validation)} · "
      f"écartés pour fuite {len(ecartes)}")

# %% [markdown]
# ## OBS 2 · Fabriquer un exemple RAFT
#
# **Fourni, et c'est volontaire : c'est le code le plus dense de la formation,**
# celui où se concentrent 80 % des erreurs de fine-tuning. Vous n'allez pas
# l'écrire — vous allez le lire jusqu'à pouvoir répondre à trois questions.
#
# Un exemple RAFT est une conversation à trois messages : système, utilisateur
# (le prompt augmenté), assistant (la bonne traduction). Ce qui change d'un
# exemple à l'autre, c'est **ce qu'il y a dans le contexte** :
#
# - avec une probabilité `p_oracle`, les `k` vrais voisins ;
# - sinon, `k` **distracteurs** : des segments qui ne sont pas dans les dix plus
#   proches.
#
# Dans les deux cas, la réponse attendue reste la bonne traduction. C'est comme
# cela que le modèle apprend à **trier** au lieu de recopier.
#
# **Trois questions, en lisant le corps de la fonction :**
#
# 1. Qu'est-ce qui se passe dans ce code ? Repérez la ligne qui tire à pile ou
#    face entre « vrais voisins » et « distracteurs », et celle qui construit
#    le message `assistant`.
# 2. **De quelle manière est-ce qu'on découpe le jeu d'entraînement** ? Ce n'est
#    pas un découpage en deux fichiers : à chaque exemple, `rng.random() <
#    p_oracle` décide, au tirage, s'il reçoit le bon contexte ou un leurre — le
#    même segment peut tomber d'un côté ou de l'autre selon le tirage.
# 3. Pourquoi le code exclut-il `segment` lui-même de `autres`, et pourquoi les
#    distracteurs sont-ils tirés en dehors de `proches` plutôt que dans
#    n'importe quel segment ?
#
# > *Vos réponses :*
# >
# >

# %%
def exemple_raft(segment: dict, memoire: list[dict], index_glossaire, k: int = 3,
                 p_oracle: float = 0.8, rng: random.Random | None = None) -> dict:
    """Un exemple d'entraînement, au format {"messages": [...]}."""
    rng = rng or random.Random(GRAINE)
    autres = [s for s in memoire if s["id"] != segment["id"]]
    proches = rechercher_lexical(segment["src"], autres, k=10)
    termes = index_glossaire.chercher(segment["src"])
    if rng.random() < p_oracle:
        voisins = proches[:k]
    else:
        identifiants = {s["id"] for s in proches}
        lointains = [s for s in autres if s["id"] not in identifiants]
        voisins = rng.sample(lointains, min(k, len(lointains)))
    messages = construire_messages(segment["src"], voisins=voisins, glossaire=termes,
                                   consignes=CONSIGNE_STYLE)
    return {"messages": messages + [{"role": "assistant", "content": segment["tgt"]}]}


apercu = exemple_raft(entrainement[0], memoire, index_glossaire, rng=random.Random(1))
atelier.montrer_prompt(apercu["messages"])

# %% [markdown]
# ## RÉG 1 · Le dosage des distracteurs
#
# **Changez `P_ORACLE`** — essayez 1.0, puis 0.5 — et relancez la cellule.
# Elle ne réentraîne rien : elle reconstruit le jeu et compte.
#
# À `1.0`, le modèle ne voit jamais de contexte trompeur. Il apprend que le
# contexte est toujours bon, donc qu'il peut le recopier. Le jour où votre
# recherche se trompe, il vous suit dans l'erreur avec assurance.
#
# Le papier RAFT recommande 0,8. C'est un réglage, pas une loi : notez le vôtre.

# %%
P_ORACLE = 0.8
K_VOISINS = 3

rng = random.Random(GRAINE)
jeu = [exemple_raft(s, memoire, index_glossaire, K_VOISINS, P_ORACLE, rng)
       for s in entrainement]
longueurs = [sum(len(m["content"]) for m in e["messages"]) for e in jeu]
print(f"  {len(jeu)} exemples · {sum(longueurs) / len(longueurs):.0f} caractères en moyenne")
print(f"  p_oracle = {P_ORACLE} → environ {100 * (1 - P_ORACLE):.0f} % des exemples "
      f"ont un contexte trompeur")

# %% [markdown]
# ## LIRE 1 · Ce que le modèle va réellement apprendre
#
# **Ouvrez la sortie ci-dessous et lisez-la en entier.** C'est le seul moyen de
# voir ce qu'on donne au modèle, et c'est cinq minutes bien employées.
#
# **Trois questions :**
#
# 1. le message `assistant` contient-il autre chose que la traduction ?
# 2. la traduction du segment apparaît-elle quelque part dans le message
#    `user` ? (si oui, votre jeu est cassé) ;
# 3. quelle proportion du texte est du contexte, et quelle proportion est la
#    réponse à apprendre ?

# %%
exemple = jeu[3]
for message in exemple["messages"]:
    print(f"--- {message['role']} ({len(message['content'])} caractères)")
    print(message["content"][:400])
    print()

# %% [markdown]
# ## CODE 1 · Régler LoRA
#
# Quatre nombres, et ils ont un sens :
#
# | | |
# |---|---|
# | `rang` | largeur des matrices ajoutées. Plus grand = plus de capacité, plus de mémoire, plus de risque de surapprentissage. 4 à 16 sur un jeu de 350 exemples. |
# | `alpha` | facteur d'échelle appliqué à la sortie de l'adaptateur. La convention la plus répandue est `alpha = 2 × rang`. |
# | `dropout` | régularisation. 0,05 sur un petit jeu. |
# | `cibles` | quelles matrices reçoivent un adaptateur. `q_proj` et `v_proj` sont le minimum et suffisent ici. |

# %%
def config_lora(rang: int = 8) -> dict:
    """Les réglages de LoRA."""
    # <<<CODE 1 ★ Compléter la configuration
    # Rendez un dictionnaire avec exactement ces quatre clés :
    # "rang"    -> l'argument rang
    # "alpha"   -> deux fois le rang
    # "dropout" -> 0.05
    # "cibles"  -> ["q_proj", "v_proj"]
    # Test : python tp.py test tp04 -k code1
    raise NotImplementedError(
        "CODE 1 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp04-lora-raft/README.md"
    )
    # >>>CODE 1


print(config_lora())

# %% [markdown]
# ## RÉG 2 · Ce que le rang coûte
#
# **Changez `RANG`** — 2, 8, 32 — et relancez. La cellule ne réentraîne pas :
# elle compte les paramètres qui seraient entraînés.
#
# Vous cherchez l'ordre de grandeur : combien de paramètres pour combien
# d'exemples ? Avec 350 exemples et 2 millions de paramètres entraînables, on
# apprend surtout à réciter le jeu d'entraînement. C'est le surapprentissage, et
# il ne se voit pas dans la perte d'entraînement — seulement en validation.

# %%
RANG = 8

from peft import LoraConfig, get_peft_model

from commun.petit_modele import charger

modele_sonde, _ = charger()
config = config_lora(RANG)
greffe = get_peft_model(modele_sonde, LoraConfig(
    r=config["rang"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
    target_modules=config["cibles"], task_type="CAUSAL_LM"))
entrainables = sum(p.numel() for p in greffe.parameters() if p.requires_grad)
print(f"  rang {RANG} → {entrainables:,} paramètres entraînés "
      f"({entrainables / len(jeu):,.0f} par exemple d'entraînement)")
del greffe, modele_sonde

# %% [markdown]
# ## LIRE 2 · La boucle d'entraînement
#
# **Ouvrez `commun/petit_modele.py` et lisez `entrainer_lora`.** Six étapes,
# quarante lignes, et il n'y en a pas une de plus dans les bibliothèques qui
# font ça pour vous.
#
# **Trois questions :**
#
# 1. à quelle ligne les poids du modèle d'origine sont-ils gelés ?
# 2. `entrees["labels"] = entrees["input_ids"].clone()` — pourquoi la cible
#    est-elle une copie de l'entrée ? Qu'est-ce que le modèle apprend à
#    prédire ?
# 3. que se passerait-il si on oubliait `optimiseur.zero_grad()` ?

# %% [markdown]
# ## OBS 3 · Entraîner
#
# Comptez cinq à quinze minutes selon votre machine. Regardez la perte
# descendre : si elle stagne, le pas d'apprentissage est trop petit ; si elle
# oscille, il est trop grand.
#
# C'est le bon moment pour aller lire `entrainer_lora` pendant que ça tourne.

# %%
resultat = entrainer_lora(jeu, config_lora(RANG), epoques=1, taille_lot=4,
                          sortie=DOSSIER_ADAPTATEUR)

# %% [markdown]
# ## OBS 4 · Avant, après
#
# Mêmes segments, même prompt, même température. La seule chose qui change est
# l'adaptateur branché sur le modèle.
#
# **Regardez dans cet ordre :**
#
# 1. les sorties brutes ci-dessous — le modèle a-t-il appris à produire du
#    français au bon format ?
# 2. la table par catégorie — le gain se concentre-t-il sur `repetition` et
#    `piege` ?
# 3. **la ligne `nouveau`** — si elle progresse autant que les autres, cherchez
#    la fuite dans votre jeu d'entraînement.

# %%
adapte = MoteurPetit(adaptateur=DOSSIER_ADAPTATEUR)


def prompt_complet(src: str):
    return construire_messages(
        src, voisins=rechercher_lexical(src, memoire, 3),
        glossaire=index_glossaire.chercher(src), consignes=CONSIGNE_STYLE)


for segment in segments[:3]:
    print(f"  sv     {segment['src']}")
    print(f"  réf    {segment['tgt']}")
    print(f"  avant  {base.generer(prompt_complet(segment['src'])).texte.strip()[:100]!r}")
    print(f"  après  {adapte.generer(prompt_complet(segment['src'])).texte.strip()[:100]!r}\n")

# %%
mesure_base = atelier.mesurer("tp04-base", segments,
                              atelier.traducteur(base, prompt_complet),
                              titre="Petit modèle — avant entraînement")
mesure_adapte = atelier.mesurer("tp04-adapte", segments,
                                atelier.traducteur(adapte, prompt_complet),
                                titre="Petit modèle — après entraînement")
atelier.comparer(("avant", mesure_base), ("après", mesure_adapte))
atelier.par_categorie(("avant", mesure_base), ("après", mesure_adapte))

# %% [markdown]
# ## ARB 1 · Qu'a appris le modèle ?
#
# **Écrivez en trois phrases** ce que l'adaptateur a appris, et ce qu'il n'a pas
# appris. Appuyez-vous sur deux chiffres de votre propre table : le gain sur
# `repetition`, et le gain sur `nouveau`.
#
# Puis répondez à la question qu'on vous posera au bureau : **si le
# fine-tuning n'apporte pas de connaissance, pourquoi en faire ?**
#
# > *Votre réponse :*
# >
# >
#
# ---
#
# ## Pour aller plus loin, s'il reste du temps
#
# RAFT recommande de glisser quelques exemples **sans aucun contexte**. Sinon le
# modèle apprend qu'il y a toujours un bloc « Mémoire de traduction », et le jour
# où la recherche ne rend rien, il est désorienté par un prompt qu'il n'a jamais
# vu.

# %%
def exemple_sans_contexte(segment: dict, index_glossaire) -> dict:
    """BONUS — un exemple d'entraînement sans aucun voisin."""
    # <<<BONUS 4 ★ Exemple sans contexte
    # Même forme que exemple_raft, mais avec une liste de voisins vide.
    # Le glossaire pertinent, lui, reste présent.
    # Test : python tp.py test tp04 --bonus -k bonus4
    raise NotImplementedError(
        "BONUS 4 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp04-lora-raft/README.md"
    )
    # >>>BONUS 4
