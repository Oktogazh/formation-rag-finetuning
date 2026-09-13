# TP 5 — Un serveur, quatre assaillants

**Objectif.** Un service de traduction tourne pour de bon, sur un Space
HuggingFace, tracé dans LangSmith. Vous allez le bombarder, regarder la latence
s'effondrer, diagnostiquer, puis réparer **une copie locale** et rejouer
l'assaut dessus.

**Durée.** Noyau ≈ 70 min · bonus ≈ 20 min · difficulté ★★☆☆☆

## L'incident, avant de coder

Le service du formateur répond correctement à **un** client, en une vingtaine de
secondes. La question du TP est : que se passe-t-il à quatre ?

Ce que vous allez mesurer, et pourquoi ce ne sont pas des moyennes :

| | |
|---|---|
| **p50** | la latence médiane. La moitié des requêtes sont plus rapides. |
| **p95** | la latence que 95 % des requêtes ne dépassent pas. **C'est le chiffre qui figure dans un engagement de service**, parce que c'est celui que vos utilisateurs mécontents vivent. |
| **débit** | requêtes terminées par seconde. S'il ne monte pas quand les clients se multiplient, le service n'en traite qu'une à la fois. |

Une moyenne cache exactement ce qu'on cherche : quand un service sature, la
moyenne monte doucement pendant que le p95 explose.

## Prérequis

L'URL du service, donnée par le formateur, dans `.env` :

```
TP_URL_SPACE=https://<compte>-helios-traduction.hf.space
```

Et, pour regarder les traces, les identifiants LangSmith que le formateur donne
dans le chat de la visio. Ils sont révoqués après la session.

Aucun prérequis logiciel : `fastapi` et `uvicorn` sont déjà dans
l'environnement conda.

## Déroulé

**1. Prendre la référence** — TODO 1, dans `tp05/assaut.py`.

```bash
python tp.py test tp05 -k todo1
python tp.py assaut --clients 1 --duree 30
```

Notez le p50. C'est la latence du service quand personne d'autre ne l'utilise,
et c'est le seul chiffre que le développeur du service a jamais vu.

**2. Monter en charge, tous ensemble.**

```bash
python tp.py assaut --clients 1,2,4,8 --duree 30
```

**Lancez-le en même temps que vos collègues** : c'est le même serveur pour toute
la salle, et c'est le but. Regardez le p95 et le débit. Le débit ne monte pas.
Les requêtes ne sont pas traitées en parallèle : elles font la queue, et
personne n'obtient de réponse rapide.

**3. Regarder les traces.**

Ouvrez LangSmith, projet `formation-helios`. Chaque requête y est, avec sa durée
et le nombre de clients déclaré dans l'en-tête `X-Clients` : vous retrouvez les
vôtres. Vous voyez aussi que **le temps est passé dans le modèle**, pas dans le
réseau. Le goulot n'est pas là où on l'imagine d'habitude.

**4. Trouver le défaut.**

Lisez `space/app.py`, en particulier le bloc `DÉFAUT VOLONTAIRE` en tête. Quatre
choses manquent, et aucune n'est dans le modèle :

- le point d'entrée est déclaré avec `def` et appelle le modèle de façon
  bloquante ;
- aucune limite de concurrence : tout ce qui arrive entre ;
- aucune file d'attente bornée : personne n'est refusé, tout le monde attend ;
- aucun délai maximum : une requête peut traîner indéfiniment.

C'est le défaut le plus fréquent en production, et le moins visible en
développement : sur la machine de celui qui l'a écrit, avec un seul client, tout
allait bien.

**5. Réparer** — TODO 2, dans `tp05/reparer.py`.

Un service qui sature doit faire trois choses, dans cet ordre : **limiter** la
concurrence, **refuser vite** ce qu'il ne pourra pas traiter, **abandonner** ce
qui prend trop longtemps.

Refuser proprement n'est pas un aveu d'échec : c'est ce qui permet aux requêtes
acceptées de rester rapides. Un service sans garde n'a pas de latence, il a une
loterie.

```bash
python tp.py test tp05 -k todo2
```

**6. Rejouer l'assaut sur votre service réparé.**

Dans un premier terminal :

```bash
python tp.py api
```

Dans un second :

```bash
python tp.py assaut --url http://localhost:8000 --clients 8 --duree 30
```

Le p95 est borné, et des `503` apparaissent. **C'est le résultat attendu.** Pour
comparer, relancez avec `python tp.py api --sans-garde` : c'est le défaut du
Space, reproduit sur votre machine.

**7. Ce qui aurait vraiment réglé le problème.**

La garde protège ; elle n'accélère rien. Pour servir davantage il faut, par
ordre de coût : un GPU (le modèle passe de 20 s à 1 s), du *batching* continu
(vLLM traite plusieurs requêtes dans un même passage), plusieurs répliques
derrière un répartiteur. Le Space gratuit n'offre aucune des trois, et c'est
aussi une leçon : **la première décision d'une mise en service est le choix du
matériel**, et elle se prend avec des mesures comme celles que vous venez de
faire.

## Les TODO

| # | Fichier | Difficulté | Ce qu'on attend | Test |
|---|---|---|---|---|
| 1 | `tp05/assaut.py::assaillir` | ★★ | N clients en parallèle, latences mesurées | `-k todo1` |
| 2 | `tp05/reparer.py::Garde.executer` | ★★ | sémaphore, file bornée, délai maximum | `-k todo2` |

## Bonus

**BONUS 5** — `palier_sature` : trouver par dichotomie le nombre de clients à
partir duquel le p95 dépasse un seuil. C'est la mesure qu'on fournit à un client
qui demande « combien d'utilisateurs ce service supporte-t-il ». La réponse n'est
jamais un nombre d'utilisateurs : c'est un nombre d'appels simultanés, pour un
engagement de latence donné.

## Si ça coince

- *Le Space ne répond pas* → il se met en pause après 48 h d'inactivité ;
  prévenez le formateur. Le TP se fait entièrement en local :
  `python tp.py api --sans-garde` reproduit le même défaut.
- *`assaut` rend surtout des codes 0* → ce sont des délais dépassés côté client.
  C'est un résultat, pas une panne : notez-le.
- *Rien dans LangSmith* → les identifiants sont donnés en séance, et le projet
  s'appelle `formation-helios`.
- **N'essayez pas de réparer le Space.** Il est mal configuré exprès, et il
  appartient au formateur. Vous réparez votre copie locale.
