---
title: Traduction Helios
emoji: 🌐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Service de traduction Helios — support du TP 5

Traduction suédois → français par **Ministral 3B servi par Ollama** sur
processeur. Ce Space existe pour une seule raison : **être saturé en direct par
une salle de formation**, et montrer ce qu'un service mal configuré fait sous
charge.

Il est volontairement mal configuré. Voir le bloc « DÉFAUT VOLONTAIRE » en tête
de `app.py`.

## Deux étages, un seul port

Un Space n'expose qu'un port. `app.py` déclare d'abord l'API, puis monte
l'interface dessus avec `gr.mount_gradio_app(app, interface, path="/")` —
**l'ordre compte**, sinon la page avalerait les routes de l'API.

| Route | Étage | Qui s'en sert |
|---|---|---|
| `GET /` | **front-end** Gradio | la salle, à la main, en OBS 0 |
| `POST /traduire` | **back-end** — c'est lui qui porte le défaut | l'assaut du notebook, `curl`, et la page ci-dessus |
| `GET /sante` | back-end | dit si le traçage est actif |

L'interface **poste sur `/traduire` par HTTP**, comme n'importe quel client :
elle n'appelle jamais le modèle en direct. C'est ce qui garantit que la page et
le notebook mesurent le même service.

## Déploiement par le formateur

```bash
git clone https://huggingface.co/spaces/Oktogazh/formation-rag-finetuning espace-helios
cp tp05-mise-en-service/space/{app.py,requirements.txt,Dockerfile,demarrer.sh,README.md} espace-helios/
cd espace-helios && git add -A && git commit -m "Service de traduction Helios" && git push
```

Le Space est en **Docker, CPU basic** (gratuit). Le build installe Ollama et les
dépendances ; le modèle (≈ 2 Go) se télécharge au **premier démarrage**, pas au
build — le premier appel après un déploiement attend donc quelques minutes.

Ne poussez **pas** `.env` : il est dans le `.gitignore` du dépôt de TP, et les
secrets se posent dans l'interface du Space (section suivante).

## Variables à poser dans le Space

`Settings → Variables and secrets`. Elles sont injectées comme variables
d'environnement **à l'exécution** — rien à faire de particulier pour le build.

| Type | Nom | Valeur |
|---|---|---|
| Secret | `LANGSMITH_API_KEY` | la clé de service du compte de formation |
| Variable | `LANGSMITH_TRACING` | `true` |
| Variable | `LANGSMITH_PROJECT` | `formation-helios` |
| Variable | `LANGSMITH_ENDPOINT` | `https://eu.api.smith.langchain.com` **si le compte est européen** |

Le dernier point est le piège classique : une clé créée dans un espace de
travail européen renvoie une erreur silencieuse sur l'endpoint américain, qui
est celui par défaut. Les traces n'arrivent jamais et rien ne le signale.

Redémarrez le Space après avoir posé les variables (`Restart this Space`, un
*factory rebuild* est inutile).

## Vérifier

```bash
curl -s https://oktogazh-formation-rag-finetuning.hf.space/sante
```

Attendu : `{"statut":"ok","tracing":true,"projet":"formation-helios",...}`.
La route `/sante` dit si le traçage est actif : c'est le seul moyen de le savoir
sans ouvrir LangSmith.

```bash
curl -s -X POST https://oktogazh-formation-rag-finetuning.hf.space/traduire \
  -H 'Content-Type: application/json' -H 'X-Clients: 1' \
  -d '{"segment":"Abonnemanget Bas tillåter 20 förfrågningar per minut."}'
```

Et ouvrez `https://oktogazh-formation-rag-finetuning.hf.space/` : la page doit
répondre, et la durée s'afficher sous la traduction.

La trace doit apparaître dans LangSmith en moins de dix secondes (l'envoi est
asynchrone et groupé).

## Repli

Si le Space est indisponible le jour de la session, le TP se fait entièrement en
local : il suffit de laisser la cellule `URL_SERVICE` vide dans le notebook
`tp05.ipynb`, qui démarre alors `python tp.py api --sans-garde` de lui-même —
le même défaut, sur la machine du stagiaire. Seul l'exercice OBS 2 (les traces)
demande le Space.

## Ne pas déployer la version réparée

Le notebook montre, en LIRE 4, ce que deviendrait `app.py` avec une garde. C'est
un **extrait à lire**, et il ne doit pas être poussé ici : ce Space est le
support d'un exercice, il doit rester saturable pour la session suivante. Un
`git push` sur ce dépôt relance dix minutes de compilation pour rien.
