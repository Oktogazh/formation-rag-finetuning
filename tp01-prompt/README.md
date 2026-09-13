# TP 1 — Le prompt : ce qu'il donne, et où il s'arrête

**Objectif.** Traduire 20 segments suédois avec un modèle de 3 milliards de
paramètres, mesurer ce que ça vaut, améliorer le prompt à la main, mesurer de
nouveau — puis trouver le point où « en mettre plus dans le prompt » cesse de
payer. C'est ce point-là qui rend les deux jours suivants nécessaires.

**Durée.** Noyau ≈ 45 min · bonus ≈ 15 min · difficulté ★★☆☆☆

## Ce que vous allez mesurer

| Indicateur | Ce qu'il dit |
|---|---|
| **BLEU** | la métrique de référence de la traduction automatique, calculée sur le corpus. C'est le chiffre qu'on compare d'un cran à l'autre. |
| **chrF** | la même idée sur les caractères. Plus lisible que BLEU sur des segments courts. |
| **Termino** | part des termes du glossaire client correctement rendus. **C'est l'indicateur qui fait vivre un client.** |
| **Vous** | le vouvoiement, imposé par le guide de style. |
| **Chiffres** | les nombres de la source se retrouvent-ils dans la traduction ? |

Les 20 segments sont tirés des quatre catégories du corpus, en parts égales :
`repetition` (un voisin existe dans la mémoire), `piege` (un voisin existe mais
le chiffre est inattendu), `fuzzy` (un voisin partiel), `nouveau` (rien).

## Prérequis

Ollama avec `ministral-3:3b`, ou une clé `MISTRAL_API_KEY` dans `.env`.
Rien d'autre. `python tp.py check` vous le dit.

## Déroulé

**1. Voir où vous en êtes.**

```bash
python tp.py check
```

Notez votre accélérateur et votre palier. Comparez avec votre voisin : ce n'est
pas une coquetterie, c'est ce qui explique pourquoi il finira avant vous.

**2. Le point zéro.**

```bash
python tp.py traduire --variante nu --n 20
```

Le modèle reçoit le segment suédois et rien d'autre. Regardez la colonne
`Termino` et la liste des fautes fréquentes en bas : le modèle traduit
`abonnemang` par « abonnement », et le client impose « formule ». Il n'a aucun
moyen de le savoir. **Ce n'est pas une faute du modèle, c'est une faute du
prompt.**

**3. La température, en trois minutes.**

```bash
python tp.py temperature --segment ev-005
```

Le même segment, quatre températures, trois fois chacune. Trois choses à voir,
dans cet ordre :

1. à `T = 0`, les trois sorties sont **identiques**. C'est la seule valeur qui
   permet de comparer deux systèmes, et c'est pour ça que toutes les mesures de
   la formation sont à 0 ;
2. à partir de `T = 0,7`, la terminologie décroche et le registre part ;
3. pour une traduction, la température n'est pas un réglage de créativité :
   **c'est un taux de défaut**. Traduction, extraction, classification → 0.
   Production de variantes qu'un humain va trier → plus haut, et c'est légitime.

**4. Écrire la consigne** — TODO 1, dans `tp01/prompt.py`.

Allez lire `data/corpus/helios-sv/docs/001-guide-de-style.md` et
`003-consignes-client.md`. Écrivez la consigne, puis :

```bash
python tp.py test tp01 -k todo1
python tp.py traduire --variante consigne --n 20
```

`Termino` monte fortement. **Et BLEU baisse.** Ce n'est pas une erreur de votre
part : c'est le résultat mesuré, et c'est la chose la plus instructive de la
matinée. BLEU compte des n-grammes de mots ; une consigne qui impose « formule »
là où la référence dit « formule » mais où le modèle disait « abonnement » ne
rapporte qu'un mot, pendant que le modèle, devenu plus bavard et plus prudent,
en perd d'autres.

**Deux indicateurs qui ne disent pas la même chose valent mieux qu'un seul.**
Et si vous deviez n'en garder qu'un pour ce client, ce serait `Termino` : une
traduction à BLEU 80 qui écrit « abonnement » est refusée en relecture.

**5. Montrer plutôt que décrire** — TODO 2.

Choisissez trois traductions déjà validées dans `data/corpus/helios-sv/tm.jsonl`
et recopiez-les. Puis :

```bash
python tp.py test tp01 -k todo2
python tp.py traduire --variante exemples --n 20
```

BLEU monte nettement, surtout sur `repetition` et `fuzzy`. Sur `nouveau`, rien
ne bouge : trois exemples qui ne ressemblent pas au segment n'aident pas.
**Retenez cette ligne, c'est déjà la thèse du TP 2.**

**6. Le mur.**

```bash
python tp.py mur --n 8
```

On met tout dans le prompt : le glossaire entier, les six documents de
consignes, soixante exemples. Environ 9 000 tokens par segment, contre 600.
Regardez les trois lignes de conclusion. Le prompt est quinze fois plus long,
la latence suit, **et la qualité ne suit pas**.

Deux explications, et elles sont complémentaires : un modèle de 3 milliards de
paramètres traite mal le milieu d'un contexte long, et 57 exemples sur 60 ne
concernent pas le segment qu'il a sous les yeux.

La question n'est donc pas *combien* mettre dans le prompt. C'est **lequel**.
C'est le TP 2.

## Ce que ça donne — mesuré le 13 septembre 2026

`ministral-3:3b` sous Ollama, MacBook Apple Silicon 16 Go, températeur 0,
20 segments (8 pour le mur). À régénérer si vous changez de modèle.

| Variante | BLEU | chrF | Termino | Chiffres | Tokens | s/seg |
|---|---|---|---|---|---|---|
| prompt nu | 45,5 | 67,3 | **14 %** | 95 % | 85 | 0,96 |
| + consigne | 41,3 | 65,6 | 36 % | 90 % | 211 | 0,74 |
| + 3 exemples | 57,3 | 73,4 | 57 % | 85 % | 331 | 0,74 |
| **le mur** | 68,9 | 85,4 | 100 % | 88 % | **4 461** | **3,19** |

Le mur fait mieux, et c'est important de le dire : entasser du contexte **n'est
pas absurde**, ça marche. Ce qui ne marche pas, c'est le rapport : treize fois
plus de tokens et trois fois plus lent, pour un résultat que le TP 2 atteindra
avec 343 tokens. Sur 100 000 segments par mois, c'est cette division par treize
qu'on facture.

Les fautes de terminologie du prompt nu, telles que la commande les affiche :

```
     5 x  « abonnement » au lieu de « formule »
     3 x  « Entreprise » au lieu de « Företag »
     2 x  « point de terminaison » absent
     1 x  « demande » au lieu de « requête »
```

Aucune n'est une faute de traduction. Ce sont des décisions du client que le
modèle n'avait aucun moyen de connaître.

## Les TODO

| # | Fichier | Difficulté | Ce qu'on attend | Test |
|---|---|---|---|---|
| 1 | `tp01/prompt.py::consigne_systeme` | ★ | une consigne qui couvre vouvoiement, registre, noms propres, chiffres | `python tp.py test tp01 -k todo1` |
| 2 | `tp01/prompt.py::exemples_manuels` | ★ | trois paires recopiées depuis `tm.jsonl`, accents compris | `python tp.py test tp01 -k todo2` |

## Bonus

**BONUS 1** — `exemples_cibles(domaine)` : choisir les exemples **en fonction du
segment** au lieu de les figer. Trois lignes, et vous venez d'écrire le TP 2.

```bash
python tp.py test tp01 --bonus -k bonus1
```

## Si ça coince

```bash
python tp.py indice tp01/prompt.py --todo 1   # le diff avec le corrigé
python tp.py rattraper tp01                   # repartir du corrigé, ce TP entier
```

- *Les trois sorties à `T = 0` ne sont pas identiques* → votre moteur ne reçoit
  pas la température. Vérifiez `python tp.py check`.
- *`traduire` met plus de dix secondes par segment* → vous êtes sur processeur.
  Baissez à `--n 8`, et demandez une clé d'API au formateur.
- *Le test todo2 refuse mes exemples* → ils doivent être **recopiés** depuis
  `tm.jsonl`, à l'identique. Les accents comptent.
