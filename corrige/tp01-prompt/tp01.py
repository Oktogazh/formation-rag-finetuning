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
# Le TP s'ouvre par un **chapitre 0** où l'on fait tout à la main, avec
# `transformers` : tokeniser, embedder, générer, détokeniser, et voir le
# préprompt que les bibliothèques posent à votre place. Le reste du TP utilise
# ensuite `commun/`, qui ne fait que raccourcir ces mêmes étapes.
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
# ## Chapitre 0 · Sous le capot
#
# Avant d'utiliser `commun/`, on l'ouvre. Toute cette partie se fait **à la
# main**, avec `transformers`, sur un très petit modèle — `SmolLM2-135M`, 135
# millions de paramètres, 270 Mo, celui que vous fine-tunerez au TP 4. Il tient
# sur un processeur et il répond en une seconde.
#
# **Ses traductions sont mauvaises, et ce n'est pas le sujet.** Ce qu'on regarde
# ici, c'est la mécanique — elle est rigoureusement la même dans un modèle de
# 3 milliards de paramètres, et dans celui que vous appelez par une API.
#
# Quatre étapes, dans cet ordre, et rien d'autre ne se passe :
#
# | | |
# |---|---|
# | **1. Tokenisation** | une suite de caractères devient une liste d'entiers |
# | **2. Embedding** | chaque entier devient une liste de nombres |
# | **3. Génération** | le modèle produit un entier de plus, puis recommence |
# | **4. Détokenisation** | la liste d'entiers redevient une suite de caractères |
#
# Le chargement prend quelques secondes la première fois — le modèle est déjà
# sur votre disque si vous avez suivi les prérequis.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from commun.petit_modele import MODELE_PETIT

tokenizer = AutoTokenizer.from_pretrained(MODELE_PETIT)
petit = AutoModelForCausalLM.from_pretrained(MODELE_PETIT, dtype=torch.float32)
petit.eval()

print(f"Modèle      {MODELE_PETIT}")
print(f"Paramètres  {sum(p.numel() for p in petit.parameters()) / 1e6:.0f} millions")
print(f"Vocabulaire {petit.config.vocab_size} tokens")
print(f"Dimension   {petit.config.hidden_size}   ·   {petit.config.num_hidden_layers} couches")

# %% [markdown]
# ### RÉG 1 · Étape 1 — tokeniser : du texte à des nombres
#
# Un modèle ne voit jamais de lettres. Le **tokenizer** découpe le texte en
# morceaux tirés d'un vocabulaire figé (ici 49 152 entrées) et rend le numéro de
# chaque morceau. C'est une simple table de correspondance, apprise une fois
# pour toutes sur le corpus d'entraînement — il n'y a pas de modèle là-dedans.
#
# **Changez `MON_TEXTE`** ci-dessous et relancez. Essayez, dans l'ordre :
#
# 1. la phrase suédoise fournie ;
# 2. sa traduction française ;
# 3. un mot que vous inventez (`« anticonstitutionnellement »`, votre nom) ;
# 4. un nombre long : `1234567`.
#
# **Trois choses à relever :**
#
# - un token n'est **pas** un mot : `Lösenordet` en coûte cinq, et `ö` vit seul ;
# - le suédois coûte plus cher que le français, qui coûte plus cher que
#   l'anglais — le vocabulaire a été appris surtout sur de l'anglais. Vous payez
#   au token : la même phrase ne coûte pas le même prix selon la langue ;
# - les chiffres partent souvent un par un. Retenez-le : c'est une des raisons
#   pour lesquelles les modèles se trompent sur les nombres, et la catégorie
#   `piege` du corpus est faite pour l'attraper.

# %%
MON_TEXTE = "Lösenordet förnyas var 180:e dag."

identifiants = tokenizer.encode(MON_TEXTE)
morceaux = [tokenizer.decode([i]) for i in identifiants]

print(f"texte    {MON_TEXTE!r}")
print(f"          {len(MON_TEXTE)} caractères  ->  {len(identifiants)} tokens\n")
for morceau, identifiant in zip(morceaux, identifiants):
    print(f"  {identifiant:>6}   {morceau!r}")

# %% [markdown]
# ### OBS 2 · Étape 2 — l'embedding : de nombres à des listes de nombres
#
# Un entier ne se calcule pas : `5568` n'est pas « plus grand » que `275`, c'est
# une étiquette. La première couche du modèle, la **couche d'embedding**, est un
# gros tableau à 49 152 lignes et 576 colonnes ; elle remplace chaque entier par
# la ligne correspondante — un vecteur de 576 nombres à virgule.
#
# C'est la seule opération de la journée qui ressemble à un dictionnaire, et
# c'est là que le sens commence : ces 576 nombres sont appris, et deux mots
# proches y ont des vecteurs proches. **Le TP 2 est entièrement bâti là-dessus.**
#
# Relevez la forme du tableau qui sort : `(1, n, 576)` — une phrase, `n` tokens,
# 576 nombres chacun. Un texte est devenu une matrice.

# %%
couche_embedding = petit.get_input_embeddings()
vecteurs = couche_embedding(torch.tensor([identifiants]))

print(f"tableau d'embedding  {tuple(couche_embedding.weight.shape)}  "
      f"= {couche_embedding.weight.numel() / 1e6:.1f} M paramètres, "
      f"pour la seule entrée\n")
print(f"{len(identifiants)} tokens  ->  {tuple(vecteurs.shape)}\n")
for morceau, vecteur in list(zip(morceaux, vecteurs[0]))[:4]:
    debut = ", ".join(f"{x:+.3f}" for x in vecteur[:6].tolist())
    print(f"  {morceau!r:>12}  [{debut}, …]   ({len(vecteur)} nombres)")

# %% [markdown]
# ### OBS 3 · Étape 3 — un pas de modèle, et un seul
#
# On passe la matrice dans les 30 couches. Ce qui sort n'est **pas du texte** :
# c'est un score par token du vocabulaire — 49 152 nombres, appelés *logits* —
# pour chaque position. Seule la dernière position nous intéresse : elle dit ce
# qui vient **après** le texte qu'on a donné.
#
# `softmax` transforme ces scores en probabilités qui somment à 1. On affiche
# les cinq premières.
#
# **Un modèle de langue ne fait rien d'autre que ça.** Il ne « comprend » pas
# une question, il ne « décide » pas de traduire : il donne une distribution de
# probabilité sur le token suivant. Tout le reste — le dialogue, la traduction,
# les agents du TP 6 — est construit par-dessus cette unique opération.

# %%
with torch.no_grad():
    sortie = petit(torch.tensor([identifiants]))

print(f"logits  {tuple(sortie.logits.shape)}   "
      f"(1 phrase, {len(identifiants)} positions, {petit.config.vocab_size} scores)\n")

probabilites = torch.softmax(sortie.logits[0, -1], dim=-1)
meilleurs = torch.topk(probabilites, 5)
print(f"Après {MON_TEXTE!r}, les cinq tokens les plus probables :\n")
for probabilite, identifiant in zip(meilleurs.values, meilleurs.indices):
    barre = "█" * int(probabilite * 120)
    print(f"  {float(probabilite):6.1%}  {tokenizer.decode([identifiant])!r:>12}  {barre}")

# %% [markdown]
# ### LIRE 1 · La boucle, à la main
#
# Le modèle produit **un** token. Pour en produire dix, on le rappelle dix fois,
# en lui redonnant à chaque fois tout ce qui précède, sa propre production
# comprise. C'est tout ce que fait `.generate()`, et c'est tout ce que fait
# ChatGPT.
#
# La cellule déroule douze pas et affiche le token retenu à chacun. La dernière
# ligne vérifie que `.generate()` sort exactement la même chose : **la fonction
# toute faite n'ajoute aucune magie.**
#
# **Repérez, dans le code ci-dessous :**
#
# 1. la ligne qui choisit le token — `argmax`, c'est-à-dire « le plus probable,
#    toujours ». C'est ça, une température de 0, et vous la retrouverez à l'exercice
#    RÉG 2 ;
# 2. la ligne qui rallonge la liste. C'est la boucle, et elle explique pourquoi
#    un prompt deux fois plus long coûte plus de deux fois plus cher ;
# 3. la condition d'arrêt : le modèle produit un token spécial de fin. Personne
#    ne lui dit quand s'arrêter — il l'a appris.

# %%
courant = list(identifiants)
for pas in range(12):
    with torch.no_grad():
        logits = petit(torch.tensor([courant])).logits[0, -1]
    suivant = int(torch.argmax(logits))              # 1. le plus probable
    print(f"  pas {pas + 1:>2}   id {suivant:>6}   {tokenizer.decode([suivant])!r}")
    courant.append(suivant)                          # 2. la boucle
    if suivant == tokenizer.eos_token_id:            # 3. l'arrêt
        print("  (token de fin)")
        break

# %% [markdown]
# ### Étape 4 — détokeniser
#
# La liste d'entiers redevient du texte. C'est l'opération inverse de l'étape 1,
# et elle est exacte : rien ne se perd.
#
# `skip_special_tokens=True` masque les jetons de structure — vous allez voir
# lesquels dans une minute, et ils comptent.

# %%
produits = courant[len(identifiants):]
print(f"à la main   {tokenizer.decode(produits, skip_special_tokens=True)!r}")

avec_generate = petit.generate(
    torch.tensor([identifiants]), max_new_tokens=12,
    do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(f".generate() {tokenizer.decode(avec_generate[0][len(identifiants):], skip_special_tokens=True)!r}")

# %% [markdown]
# ## Chapitre 0 bis · Le préprompt que personne ne vous montre
#
# Vous venez de donner au modèle un texte **nu**, et il l'a *continué* : c'est sa
# seule compétence native. Un modèle de langue complète du texte. Il ne répond
# pas à des questions, il ne dialogue pas, il ne traduit pas.
#
# Le dialogue est une **mise en scène**, et elle tient dans une chaîne de
# caractères qu'on appelle le **gabarit de conversation** (*chat template*).
# Quand vous écrivez `[{"role": "user", "content": "..."}]`, une fonction met ça
# à plat en un seul texte, avec des jetons de structure, et c'est ce texte-là —
# et rien d'autre — que le modèle reçoit.
#
# La cellule affiche les deux versions. **Regardez ce qui apparaît sans que vous
# l'ayez demandé.**

# %%
question = [{"role": "user", "content": MON_TEXTE}]
gabarit_defaut = tokenizer.apply_chat_template(
    question, tokenize=False, add_generation_prompt=True)

print("CE QUE VOUS ÉCRIVEZ")
print(f"  {question}\n")
print("CE QUE LE MODÈLE REÇOIT")
print("  " + gabarit_defaut.replace("\n", "\n  "))
print(f"\nJetons de structure du modèle : {tokenizer.all_special_tokens}")

# %% [markdown]
# ### OBS 4 · Le préprompt est déjà là, et ce n'est pas le vôtre
#
# Deux choses sont apparues toutes seules dans ce que le modèle reçoit :
#
# 1. des **jetons de structure** — `<|im_start|>`, `<|im_end|>` — qui découpent
#    les tours de parole. Le modèle a appris pendant son entraînement que ce qui
#    suit `<|im_start|>assistant` est à lui de produire. C'est toute la
#    différence entre un modèle « base » et un modèle « instruct » ;
# 2. un **message système par défaut**, écrit par Hugging Face, que vous n'avez
#    jamais tapé : *« You are a helpful AI assistant named SmolLM… »*.
#
# **Ce message est le préprompt.** Il est toujours là, dans chaque produit que
# vous utilisez — ChatGPT, Claude, Copilot en ont un, long de plusieurs milliers
# de tokens, et vous le payez à chaque requête. Quand vous n'en donnez pas, vous
# n'en avez pas zéro : vous prenez celui de quelqu'un d'autre.
#
# Écrire le sien, c'est reprendre cette place. Comparez les deux prompts réels.

# %%
PREPROMPT = (
    "Tu es traducteur technique du suédois vers le français.\n"
    "Tu rends uniquement la traduction française, sans commentaire."
)

avec_preprompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": PREPROMPT}, {"role": "user", "content": MON_TEXTE}],
    tokenize=False, add_generation_prompt=True)

print("SANS PRÉPROMPT (celui de Hugging Face s'installe)")
print("  " + gabarit_defaut.replace("\n", "\n  "))
print("\nAVEC LE VÔTRE")
print("  " + avec_preprompt.replace("\n", "\n  "))
print(f"\ncoût : {len(tokenizer.encode(gabarit_defaut))} tokens  ->  "
      f"{len(tokenizer.encode(avec_preprompt))} tokens, à chaque requête")

# %% [markdown]
# ### OBS 5 · Les trois requêtes, sur le vrai modèle
#
# `SmolLM2-135M` est trop petit pour traduire : il recopie le suédois. La
# mécanique était la même, la compétence ne l'est pas — c'est exactement ce que
# les 3 milliards de paramètres du modèle de la formation achètent.
#
# On repasse donc sur **le moteur ouvert à la préparation**, et on lui envoie le
# même segment de trois façons. Il n'y a aucune autre différence entre les trois
# appels : même modèle, même température, même segment.
#
# **Lisez les trois sorties avant de lire le commentaire ci-dessous.**

# %%
for titre, messages in [
    ("1. Le segment, et rien d'autre",
     [{"role": "user", "content": MON_TEXTE}]),
    ("2. Une instruction en langage courant",
     [{"role": "user", "content": f"Traduis en français : {MON_TEXTE}"}]),
    ("3. Le préprompt de la formation (commun/prompts.py)",
     construire_messages(MON_TEXTE)),
]:
    reponse = moteur.generer(messages)
    print(f"{titre}\n{'-' * len(titre)}")
    print(f"  {reponse.texte.strip()[:400]}\n")

# %% [markdown]
# ### Ce que vous venez de voir
#
# | | |
# |---|---|
# | **1. Sans préprompt** | le modèle **répond en suédois**, en conseiller sécurité. Il n'a jamais eu l'idée de traduire : rien ne le lui demandait. C'est le défaut, pas un accident. |
# | **2. Instruction nue** | il traduit, mais il emballe — « Voici la traduction : », du gras, une offre d'aide. Inexploitable dans une chaîne automatique : il faudrait nettoyer la sortie à chaque fois. |
# | **3. Préprompt de la formation** | la traduction, seule, prête à écrire dans un fichier. |
#
# Le passage de 1 à 3 n'a **rien coûté** : ni calcul, ni entraînement, ni
# données. Quelques dizaines de tokens de consigne. C'est ça, le *prompt
# engineering*, et c'est le sujet de la matinée.
#
# **Et c'est très exactement ce que `commun/` vous cache à partir d'ici :**
#
# | Ce que vous appellerez | Ce que ça fait, et que vous venez de faire à la main |
# |---|---|
# | `construire_messages(src)` | pose le préprompt et met le segment en dernier |
# | `moteur.generer(messages)` | applique le gabarit, tokenise, boucle, détokenise |
# | `prompts.nettoyer_sortie(…)` | enlève le « Voici la traduction : » de la ligne 2 |
# | `atelier.mesurer(…)` | refait tout ça sur 20 segments et compte les points |
#
# Ces quatre fonctions tiennent en 200 lignes, elles sont dans `commun/`, et
# vous pouvez les ouvrir à tout moment. **Rien de ce qui suit n'est une boîte
# noire** — c'est du raccourci, pour qu'on puisse mesurer vingt segments au lieu
# d'en regarder un.

# %% [markdown]
# ## Chapitre 1 · Mesurer, puis améliorer le prompt
#
# ### OBS 6 · Le point zéro
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
# ## LIRE 2 · Où est le prompt, exactement ?
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
# ## RÉG 2 · La température
#
# La température règle le hasard dans le choix de chaque mot. On traduit **le
# même segment** trois fois, à trois températures.
#
# **Changez une seule valeur** dans la liste ci-dessous, relancez, et regardez.
# Trois choses doivent apparaître, dans cet ordre :
#
# 1. à `0.0`, les trois sorties sont **identiques** — à `argmax`, vu au chapitre 0,
#    il n'y a pas de dés à jeter. C'est la seule valeur qui permet de comparer
#    deux systèmes, et c'est pourquoi toutes les mesures de la formation sont à
#    zéro ;
#
# Si les trois sorties d'une même température vous paraissent identiques **au-delà
# de zéro**, le coupable est presque toujours une graine fixe (`seed`) laissée
# dans les options du moteur : elle fige le tirage, et la température ne fait
# plus rien. `commun/moteur.py` ne la pose qu'à température 0, et dit pourquoi.
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
    #> Rendez une chaîne de caractères qui dit au modèle les quatre règles
    #> ci-dessus. Des phrases, pas des mots-clés.
    #> Le test vérifie qu'elle parle des quatre sujets et qu'elle dépasse
    #> 120 caractères. Il ne juge pas votre style : c'est le modèle qui juge,
    #> et vous le verrez dans la table de mesure.
    #> Test : python tp.py test tp01 -k code1
    return (
        "Respecte les règles de l'agence :\n"
        "- Vouvoie toujours le lecteur. Le suédois « du » se traduit par « vous », "
        "jamais par « tu ».\n"
        "- Emploie un registre neutre et professionnel, sans exclamation ni "
        "tournure familière.\n"
        "- Ne traduis pas les noms de produit ni les noms de formules : "
        "« Företag », « Bas », « Pro » et « Plus » restent tels quels.\n"
        "- Reporte les nombres de la source à l'identique, même s'ils te "
        "paraissent surprenants."
    )
    # >>>CODE 1


print(consigne_systeme())

# %% [markdown]
# Lancez le test. Il doit passer du rouge au vert.

# %%
# !python tp.py test tp01 -k code1

# %% [markdown]
# ## OBS 7 · Ce que la consigne change
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
# ## RÉG 3 · Combien d'exemples faut-il montrer ?
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
# ## OBS 8 · Le mur
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
# ## Ce que vous emportez
#
# Vous avez tout fait à la main au chapitre 0 — tokeniser, embedder, générer,
# détokeniser — puis vu que le préprompt, seul, fait passer le modèle d'une
# réponse en suédois à une traduction exploitable. Ensuite vous avez mesuré :
# une consigne de style fait monter la terminologie et **baisser** BLEU, trois
# exemples bien choisis valent mieux qu'une longue description, et soixante
# exemples valent moins que trois.
#
# Le mur que vous venez de toucher n'est pas un problème de taille de prompt.
# C'est un problème de **choix**. Il faut donc un moyen de retrouver, pour
# chaque segment, les quelques exemples qui le concernent — et c'est exactement
# ce qu'on construit au TP 2.
