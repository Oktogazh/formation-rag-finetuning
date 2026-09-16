# TP 5 — Un serveur, quatre assaillants

**Objectif.** Un service de traduction tourne pour de bon — sur un Space
HuggingFace, et à l'identique sur votre machine. Vous allez le bombarder,
regarder la latence s'effondrer, diagnostiquer, écrire la pièce qui manque, et
rejouer l'assaut sur un service protégé.

**Durée.** Noyau ≈ 70 min · bonus ≈ 20 min · difficulté ★★☆☆☆

**Ce TP ne dépend d'aucun autre.** Le service est fourni ; rien de ce que vous
avez écrit aux TP 1 à 4 n'est nécessaire ici.

## Où se fait le TP

**Dans le notebook `tp05.py` → `tp05.ipynb`**, comme les cinq autres :

```bash
python tp.py notebooks tp05
jupyter lab
```

Tout y est : les notions, les mesures, les deux exercices de code et les
comparaisons. Un serveur, ça boucle — il tourne donc **dans un autre
processus**, que le notebook démarre lui-même (`demarrer_en_fond`). Si vous
préférez le voir vivre, lancez-le dans un second terminal, c'est la même chose :

```bash
python tp.py api                  # avec la garde de référence
python tp.py api --sans-garde     # le défaut du Space, reproduit chez vous

python tp.py assaut --url http://localhost:8000 --clients 1,2,4,8 --duree 30
```

## Ce qu'on mesure, et pourquoi ce ne sont pas des moyennes

| | |
|---|---|
| **p50** | la latence médiane. La moitié des requêtes sont plus rapides. |
| **p95** | la latence que 95 % des requêtes ne dépassent pas. **C'est le chiffre qui figure dans un engagement de service**, parce que c'est celui que vos utilisateurs mécontents vivent. |
| **p95 ok** | le même, sur les seules requêtes **servies**. Dès qu'un service refuse, c'est le seul qui veuille encore dire quelque chose. |
| **débit** | requêtes terminées par seconde. S'il ne monte pas quand les clients se multiplient, le service n'en traite qu'une à la fois. |

Une moyenne cache exactement ce qu'on cherche : quand un service sature, la
moyenne monte doucement pendant que le p95 explose.

## Prérequis

Aucun prérequis logiciel en plus des autres TP : `fastapi` et `uvicorn` sont
déjà dans l'environnement conda, et le modèle est celui de toute la formation.

**Tout le TP se fait dans le notebook**, sans terminal. Deux exercices ouvrent
un navigateur, et aucun ne demande de taper une commande :

- **OBS 0** : la page du service, à l'adresse
  `https://oktogazh-formation-rag-finetuning.hf.space` — déjà pré-remplie dans
  la cellule `URL_SERVICE` (RÉG 0). Si le Space est éteint, videz la variable :
  le notebook démarre le même service sur votre machine et **rien n'est perdu**,
  sauf OBS 0 et OBS 2 ;
- **OBS 2** : les traces LangSmith du Space, projet `formation-helios`.

## Le déroulé du notebook

| | Exercice | Ce qu'on y fait |
|---|---|---|
| RÉG 0 | la cible | le Space du formateur (adresse pré-remplie), ou votre machine |
| OBS 0 | le service à la main | la page du Space, seul puis à toute la salle — *le seul exercice hors notebook* |
| LIRE 1 | p50, p95, débit | pourquoi une moyenne ment sur un service saturé |
| LIRE 2 | le défaut | quatre manques dans `space/app.py`, aucun dans le modèle |
| **CODE 1** | `assaillir` | N clients en parallèle, latences mesurées |
| OBS 1 | la montée en charge | p95 × N, débit plat : le service ne traite qu'une requête à la fois |
| OBS 2 | LangSmith | retrouver ses requêtes par l'en-tête `X-Clients` |
| LIRE 3 | limiter, refuser vite, abandonner | `503` et `504` ne disent pas la même chose |
| **CODE 2** | `Garde.executer` | sémaphore, file bornée, délai maximum |
| RÉG 1 | `file_max` | une file longue ne fait passer personne plus vite |
| LIRE 4 | `space/app.py` réparé | l'extrait est **montré**, pas redéployé |
| OBS 3 | l'assaut rejoué | des `503` apparaissent, et c'est le résultat attendu |
| BONUS 5 | `palier_sature` | le nombre d'appels simultanés qu'on peut annoncer |
| LIRE 5 | ce qui règle vraiment | GPU, *batching* continu, répliques |
| LIRE 6 | le Space | déploiement, secrets, le piège de l'endpoint européen |
| ARB 1 | votre arbitrage | ce que vous promettez, et ce que vous refusez de promettre |

```bash
python tp.py test tp05 -k code1
python tp.py test tp05 -k code2
python tp.py test tp05 --bonus -k bonus5
```

## Ce qui est fourni, et pourquoi

Le paquet `service/` n'est **jamais édité par le stagiaire** :

- `service/assaut.py` — l'assaut de référence, celui de `python tp.py assaut`,
  et les fonctions d'affichage que le notebook réutilise ;
- `service/reparer.py` — une garde de référence, dont le serveur local se sert
  pour tourner **dès la première minute du TP**. Vous réécrivez la vôtre dans
  le notebook, et vous comparez ;
- `service/serveur_local.py` — le serveur, avec et sans garde, lançable depuis
  un terminal ou depuis une cellule.

`space/` est le service déployé, **tel quel**. Il est volontairement mal
configuré : voir le bloc « DÉFAUT VOLONTAIRE » en tête de `space/app.py`.
**N'essayez pas de le réparer** — il appartient au formateur, il doit rester
cassé pour la session suivante, et le code réparé est montré dans le notebook
(LIRE 4) plutôt que redéployé.

## Si ça coince

- *Le Space ne répond pas* → un Space gratuit s'endort après 48 h sans trafic ;
  prévenez le formateur. Le TP se fait entièrement en local : laissez
  `URL_SERVICE` vide.
- *Le service local ne démarre pas* → `serveur.journal()` affiche ses dernières
  lignes ; le journal complet est dans `resultats/tp05-serveur-8000.log`.
- *L'assaut rend surtout des codes 0* → ce sont des délais dépassés côté client.
  C'est un résultat, pas une panne : notez-le.
- *Rien dans LangSmith* → le projet est `formation-helios`, et les traces du
  Space n'arrivent que si le formateur a posé les variables décrites dans
  `space/README.md`.
