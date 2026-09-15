---
id: doc-hastighetsgranser
titre: Limites de débit et quotas
source: helios/docs/fr
date: 2026-05-12
langue: fr
---

## Limites de débit

La limite de débit est fixée à 60 requêtes par minute pour la formule Bas. La
formule Pro autorise 240 requêtes par minute, et la formule Företag 1200. La
limite se compte par environnement et non par compte.

Chaque réponse contient un en-tête indiquant le débit restant. Si vous dépassez
la limite, le service renvoie le code 429. Attendez le délai indiqué avant de
réessayer.

## Quotas

Vous pouvez créer jusqu'à 10 environnements par formule. Chaque environnement a
son propre jeu de clés, et une clé ne fonctionne jamais en dehors de son
environnement. Le point de terminaison de test ne consomme pas votre quota.

Le point de terminaison de la version 1 sera retiré dans 12 mois. Migrez vos
intégrations avant cette échéance. Nous annonçons chaque changement 30 jours à
l'avance.

## Webhooks

Un webhook est réessayé 5 fois avant d'être abandonné. Vérifiez que l'adresse du
webhook répond en moins de 3 secondes. Une réponse tardive compte comme un
échec.

Les données sont chiffrées au repos et en transit. La liste des sous-traitants
est publiée sur la page de confidentialité.
