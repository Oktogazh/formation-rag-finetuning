# %% [markdown]
# # TP 1 — Le prompt : ce qu'il donne, et où il s'arrête
#
# **Durée** : noyau ≈ 45 min · **Niveau** ★★☆☆☆
#
# Vous allez traduire des segments de documentation suédoise vers le français
# avec un modèle de 3 milliards de paramètres qui tourne sur votre machine.
# D'abord sans rien lui dire. Puis en lui donnant des consignes. Puis en lui
# donnant tout ce qu'on a — et c'est là que ça casse.
#
# ## Comment lire ce notebook
#
# Chaque exercice porte un type, pour que vous sachiez d'avance si vous allez
# écrire du code :
#
# | | |
# |---|---|
# | **OBS** | lancer, relever un chiffre, le noter. Zéro ligne de code. |
# | **RÉG** | changer *une* valeur, relancer, expliquer l'écart. Une ligne. |
# | **LIRE** | pointer dans le code ou dans une trace où se passe quelque chose. |
# | **CODE** | compléter un `CODE n` numéroté, dix lignes au plus, un test dédié. |
# | **ARB** | trancher et justifier par écrit, à partir de vos propres chiffres. |
#
# **Ce TP ne dépend d'aucun autre.** Vous pouvez le faire seul, le rater, ou le
# reprendre plus tard : les TP suivants ne s'appuient jamais sur ce que vous
# écrivez ici.
#
# ## Ce que vous allez mesurer
#
# | | |
# |---|---|
# | **BLEU** | la métrique de référence de la traduction automatique, au niveau du corpus |
# | **chrF** | la même idée sur les caractères, plus lisible sur des segments courts |
# | **Termino** | part des termes du glossaire client correctement rendus |
# | **Vous** | le vouvoiement, imposé par le guide de style |
# | **Chiffres** | les nombres de la source se retrouvent-ils dans la traduction ? |

# %% [markdown]
# ## Préparation
#
# La cellule ci-dessous fait trois choses, et c'est tout ce dont le TP a besoin :
#
# 1. elle se place à la racine du dépôt, puis importe `commun`, le socle
#    partagé par les six TP — vous ne le modifiez jamais, mais vous pouvez
#    l'ouvrir ;
# 2. elle ouvre un **moteur**, c'est-à-dire le modèle de langue. Selon votre
#    machine ce sera Ollama en local, ou l'API Mistral si le formateur vous a
#    donné une clé. Le reste du notebook ne voit pas la différence ;
# 3. elle charge **20 segments d'évaluation**, équilibrés entre les quatre
#    catégories du corpus. C'est peu, et c'est volontaire : vous allez les
#    traduire une dizaine de fois dans la matinée.

# %%
# Jupyter démarre le noyau dans le dossier du notebook, pas à la racine du dépôt.
# Ces trois lignes remontent jusqu'à la racine et s'y placent : sans elles,
# « import commun » échoue et « !python tp.py » ne trouve rien. Toutes les
# cellules qui suivent en dépendent.
import os
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
os.chdir(RACINE)

from commun import atelier, corpus, materiel
from commun.consignes import CONSIGNE_STYLE, EXEMPLES_MEMOIRE
from commun.moteur import obtenir_moteur
from commun.prompts import construire_messages

moteur = obtenir_moteur()
glossaire = corpus.charger_glossaire()
segments = atelier.segments(n=20)

print(f"Moteur : {moteur.nom()}")
print(f"{len(segments)} segments, {len(glossaire)} termes de glossaire imposés")

# %% [markdown]
# ## OBS 1 · Ce que votre machine sait faire
#
# Relevez votre accélérateur et votre palier, et comparez avec votre voisin.
# Ce n'est pas une coquetterie : c'est ce qui explique pourquoi il finira avant
# vous, ou l'inverse. Notez aussi le temps annoncé par segment — vous allez le
# multiplier par vingt à chaque mesure.

# %%
etat = materiel.diagnostic()
for cle in ("systeme", "accelerateur", "memoire_go", "moteur_retenu"):
    print(f"  {cle:16} {etat[cle]}")
print(f"  palier           {materiel.palier(etat)}")

# %% [markdown]
# ## OBS 2 · Le point zéro
#
# On donne au modèle le segment suédois, et rien d'autre. Pas de consigne, pas
# d'exemple, pas de glossaire.
#
# **Ce que fait la cellule, ligne par ligne :**
#
# - `construire_messages(src)` fabrique le prompt : un message *système* qui dit
#   « tu es traducteur du suédois vers le français », et un message *utilisateur*
#   qui contient le segment. C'est le gabarit unique de la formation, dans
#   `commun/prompts.py` ;
# - `atelier.traducteur(...)` enveloppe ça dans la fonction que la mesure attend :
#   elle appelle le modèle, nettoie la sortie (un modèle bavard répond parfois
#   « Voici la traduction : … ») et compte les tokens et les secondes ;
# - `atelier.mesurer(...)` traduit les 20 segments, calcule les cinq indicateurs,
#   affiche la table par catégorie et enregistre le résultat dans `resultats/`.
#
# Comptez une à deux secondes par segment. Regardez surtout la colonne
# **Termino** et la liste des fautes, en bas.

# %%
nu = atelier.mesurer(
    "tp01-nu",
    segments,
    atelier.traducteur(moteur, lambda src: construire_messages(src)),
    titre="Prompt nu — le modèle ne sait rien de votre client",
)

# %% [markdown]
# ## LIRE 1 · Où est le prompt, exactement ?
#
# Avant d'améliorer quelque chose, il faut l'avoir vu. La cellule affiche le
# prompt tel que le modèle l'a reçu pour le premier segment.
#
# **Trouvez, dans ce qui s'affiche :**
#
# 1. le message système, et ce qu'il dit exactement ;
# 2. l'endroit où le segment suédois est inséré ;
# 3. les deux dernières lettres du prompt. Pourquoi le prompt se termine-t-il
#    par `fr:` et pas par un point ?
#
# La réponse à la question 3 tient en un mot, et c'est le mot le plus important
# du fichier `commun/prompts.py`.

# %%
atelier.montrer_prompt(construire_messages(segments[0]["src"]))

# %% [markdown]
# ## RÉG 1 · La température
#
# La température règle le hasard dans le choix de chaque mot. On traduit **le
# même segment** trois fois, à trois températures.
#
# **Changez une seule valeur** dans la liste ci-dessous, relancez, et regardez.
# Trois choses doivent apparaître, dans cet ordre :
#
# 1. à `0.0`, les trois sorties sont **identiques**. C'est la seule valeur qui
#    permet de comparer deux systèmes, et c'est pourquoi toutes les mesures de
#    la formation sont à zéro ;
# 2. vers `0.7`, la terminologie décroche et le registre part ;
# 3. pour une traduction, la température n'est pas un réglage de créativité,
#    c'est **un taux de défaut**. Traduction, extraction, classification : zéro.
#    Production de variantes qu'un humain va trier : plus haut, et c'est un
#    usage légitime.

# %%
TEMPERATURES = [0.0, 0.7, 1.2]

segment = segments[4]
print(f"sv  {segment['src']}\nréf {segment['tgt']}\n")
for temperature in TEMPERATURES:
    print(f"T = {temperature}")
    for _ in range(3):
        reponse = moteur.generer(construire_messages(segment["src"]), temperature=temperature)
        print("   ", reponse.texte.strip().splitlines()[0][:90])
    print()

# %% [markdown]
# ## CODE 1 · La consigne de style
#
# Le modèle traduit `abonnemang` par « abonnement ». Le client Helios impose
# « formule ». Le modèle n'a aucun moyen de le savoir : **ce n'est pas une faute
# du modèle, c'est une faute du prompt.**
#
# Le client impose quatre choses, et elles sont écrites dans
# `data/corpus/helios-sv/docs/` — allez les lire, c'est la moitié du travail
# d'un traducteur professionnel :
#
# 1. le **vouvoiement**, sans exception (`001-guide-de-style.md`) ;
# 2. un **registre neutre**, ni familier ni commercial ;
# 3. les **noms de produit et de formules ne se traduisent pas** : une formule
#    « Företag » reste « Företag » (`003-consignes-client.md`) ;
# 4. les **chiffres de la source se reportent tels quels**, même quand ils
#    surprennent.
#
# Écrivez cette consigne. Dix lignes au plus, en français, comme vous la diriez
# à un traducteur humain qui débute.
#
# **Analogue visible** : `commun/consignes.py` contient `CONSIGNE_STYLE`, une
# version de référence. Ne la recopiez pas : écrivez la vôtre, puis comparez.
# Vous verrez au prochain exercice laquelle des deux le modèle suit le mieux.

# %%
def consigne_systeme() -> str:
    """La consigne de style envoyée au modèle."""
    # <<<CODE 1 ★ Écrire la consigne de style
    # Rendez une chaîne de caractères qui dit au modèle les quatre règles
    # ci-dessus. Des phrases, pas des mots-clés.
    # Le test vérifie qu'elle parle des quatre sujets et qu'elle dépasse
    # 120 caractères. Il ne juge pas votre style : c'est le modèle qui juge,
    # et vous le verrez dans la table de mesure.
    # Test : python tp.py test tp01 -k code1
    raise NotImplementedError(
        "CODE 1 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp01-prompt/README.md"
    )
    # >>>CODE 1


print(consigne_systeme())

# %% [markdown]
# Lancez le test. Il doit passer du rouge au vert.

# %%
# !python tp.py test tp01 -k code1

# %% [markdown]
# ## OBS 3 · Ce que la consigne change
#
# Même modèle, mêmes segments, même température. La seule chose qui change est
# le texte que vous venez d'écrire, ajouté au message système.
#
# **Préparez-vous à un résultat désagréable.** La conformité terminologique
# monte fortement, et **BLEU baisse**. Ce n'est pas une erreur de votre part :
# c'est la mesure. BLEU compte des n-grammes de mots ; une consigne qui impose
# « formule » ne rapporte qu'un mot, pendant que le modèle, devenu plus prudent
# et plus bavard, en perd d'autres.
#
# Deux indicateurs qui ne disent pas la même chose valent mieux qu'un seul.

# %%
avec_consigne = atelier.mesurer(
    "tp01-consigne",
    segments,
    atelier.traducteur(moteur, lambda src: construire_messages(src, consignes=consigne_systeme())),
    titre="Prompt + votre consigne de style",
)
atelier.comparer(("prompt nu", nu), ("+ consigne", avec_consigne))

# %% [markdown]
# ## RÉG 2 · Combien d'exemples faut-il montrer ?
#
# Décrire le style fonctionne à moitié. **Le montrer** fonctionne mieux : on
# glisse dans le prompt des traductions déjà validées par l'agence. C'est ce
# qu'on appelle le *few-shot*.
#
# `EXEMPLES_MEMOIRE` contient trois segments recopiés tels quels depuis la
# mémoire de traduction. **Changez le nombre d'exemples** (0, 1, 2, 3), relancez,
# et notez où le gain s'arrête.
#
# Regardez la table par catégorie qui suit : le gain se concentre sur
# `repetition` et `fuzzy`, c'est-à-dire là où un voisin ressemble au segment.
# Sur `nouveau`, trois exemples qui n'ont rien à voir n'aident pas.
# **Retenez cette ligne : c'est déjà la thèse du TP 2.**

# %%
NOMBRE_EXEMPLES = 3

avec_exemples = atelier.mesurer(
    "tp01-exemples",
    segments,
    atelier.traducteur(moteur, lambda src: construire_messages(
        src, voisins=EXEMPLES_MEMOIRE[:NOMBRE_EXEMPLES], consignes=consigne_systeme())),
    titre=f"Prompt + consigne + {NOMBRE_EXEMPLES} exemples",
)
atelier.par_categorie(("nu", nu), ("consigne", avec_consigne), ("exemples", avec_exemples))

# %% [markdown]
# ## OBS 4 · Le mur
#
# La suite logique, celle que toutes les équipes tentent : **mettons-y tout.**
# Le glossaire complet, les six documents de consignes, soixante exemples de
# mémoire. Environ 9 000 tokens par segment, contre 300.
#
# La cellule est lente — comptez cinq à dix fois le temps d'une mesure ordinaire,
# c'est précisément ce qu'on veut vous faire sentir. On la lance sur 8 segments.
#
# Trois choses apparaissent :
#
# 1. le prompt est **plus de dix fois plus long**, donc dix fois plus cher, à
#    chaque segment, pour toujours ;
# 2. la latence suit ;
# 3. la qualité, elle, **ne suit pas** — elle stagne. Un modèle de 3 milliards de
#    paramètres traite mal le milieu d'un contexte long, et 57 exemples sur 60
#    ne concernent pas le segment qu'il a sous les yeux.
#
# La question n'est donc pas *combien* mettre dans le prompt. C'est **lequel**.

# %%
from commun.memoire import charger_memoire

memoire = charger_memoire()
consignes_completes = corpus.texte_des_consignes()

mur = atelier.mesurer(
    "tp01-mur",
    segments[:8],
    atelier.traducteur(moteur, lambda src: construire_messages(
        src, voisins=memoire[:60], glossaire=glossaire, consignes=consignes_completes)),
    titre="Tout dans le prompt",
)
atelier.comparer(("3 exemples (8 seg.)", avec_exemples[:8]), ("le mur (8 seg.)", mur))

# %% [markdown]
# ## ARB 1 · Votre arbitrage
#
# Votre consigne fait **monter** la conformité terminologique et **baisser** le
# BLEU. Aucune des deux mesures n'a tort.
#
# **Écrivez ici, en trois phrases :** laquelle des deux vous défendez devant le
# client Helios, et pourquoi. Gardez cette réponse — c'est l'indicateur que vous
# allez suivre pendant trois jours, et on vous la redemandera au TP 6.
#
# > *Votre réponse :*
# >
# >
#
# ---
#
# ## Pour aller plus loin, s'il reste du temps
#
# `BONUS 1` dans la cellule ci-dessous : au lieu de trois exemples figés,
# choisissez-les **en fonction du segment à traduire**. Trois lignes, et vous
# venez d'écrire le TP 2.

# %%
def exemples_du_domaine(domaine: str, memoire: list[dict]) -> list[dict]:
    """BONUS — les trois premiers segments de la mémoire de ce domaine."""
    # <<<BONUS 1 ★ Choisir les exemples en fonction du segment
    # Rendez les trois premiers segments de « memoire » dont la clé
    # « domaine » vaut exactement l'argument reçu.
    # Chaque entrée de la mémoire a les clés : id, src, tgt, domaine, date, statut.
    # Test : python tp.py test tp01 --bonus -k bonus1
    raise NotImplementedError(
        "BONUS 1 — à compléter. La consigne est juste au-dessus, "
        "le détail dans tp01-prompt/README.md"
    )
    # >>>BONUS 1


print([s["src"] for s in exemples_du_domaine("cle_rotation", memoire)])
