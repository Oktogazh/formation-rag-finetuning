---
title: Traduction Helios
emoji: 🇸🇪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Service de traduction Helios — support du TP 5

Traduction suédois → français par Ministral 3B quantifié en 4 bits, servi par
`llama-cpp-python` sur processeur. Ce Space existe pour une seule raison :
**être saturé en direct par une salle de formation**, et montrer ce qu'un
service mal configuré fait sous charge.

Il est volontairement mal configuré. Voir le bloc « DÉFAUT VOLONTAIRE » en tête
de `app.py`.

## Déploiement par le formateur

```bash
# 1. Créer le Space : Docker, CPU basic (gratuit), visibilité publique ou privée
# 2. Pousser le contenu de ce dossier
git clone https://huggingface.co/spaces/<compte>/helios-traduction
cp tp05-mise-en-service/space/* helios-traduction/
cd helios-traduction && git add -A && git commit -m "Service de traduction Helios" && git push
```

Le build prend une dizaine de minutes : `llama-cpp-python` se compile, puis le
GGUF (≈ 2 Go) se télécharge au premier démarrage.

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
curl -s https://<compte>-helios-traduction.hf.space/sante
```

Attendu : `{"statut":"ok","tracing":true,"projet":"formation-helios",...}`.
La route `/sante` dit si le traçage est actif : c'est le seul moyen de le savoir
sans ouvrir LangSmith.

```bash
curl -s -X POST https://<compte>-helios-traduction.hf.space/traduire \
  -H 'Content-Type: application/json' -H 'X-Clients: 1' \
  -d '{"segment":"Abonnemanget Bas tillåter 20 förfrågningar per minut."}'
```

La trace doit apparaître dans LangSmith en moins de dix secondes (l'envoi est
asynchrone et groupé).

## Repli

Si le Space est indisponible le jour de la session, le TP se fait entièrement en
local : `python tp.py api --sans-garde` reproduit le même défaut sur la machine
du stagiaire, et l'assaut fonctionne à l'identique contre `http://localhost:8000`.
