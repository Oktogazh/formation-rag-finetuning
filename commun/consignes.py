"""La consigne de style du client Helios, et trois exemples de référence.

Pourquoi ce fichier existe : **chaque TP doit pouvoir être fait seul.** Si le
TP 2 avait besoin de la consigne que vous écrivez au TP 1, rater le TP 1 vous
interdirait le TP 2. Ici, chaque TP écrit sa propre version dans son notebook,
et tous les autres se servent de la référence ci-dessous.

Le contenu vient de ``data/corpus/helios-sv/docs/`` : guide de style,
consignes client, procédure de relecture. Vous pouvez l'ouvrir, la comparer à
la vôtre, et la trouver perfectible. C'est le but.
"""

from __future__ import annotations

CONSIGNE_STYLE = (
    "Respecte les règles de l'agence :\n"
    "- Vouvoie toujours le lecteur. Le suédois « du » se traduit par « vous », "
    "jamais par « tu ».\n"
    "- Emploie un registre neutre et professionnel, sans exclamation ni "
    "tournure familière.\n"
    "- Ne traduis pas les noms de produit ni les noms de formules : "
    "« Företag », « Bas », « Pro » et « Plus » restent tels quels.\n"
    "- Reporte les nombres de la source à l'identique, même s'ils te "
    "paraissent surprenants. Ne les arrondis pas, ne les convertis pas."
)

# Trois segments recopiés tels quels depuis data/corpus/helios-sv/tm.jsonl.
# Trois domaines différents : un quota, un rôle, une version.
EXEMPLES_MEMOIRE = [
    {
        "src": "Abonnemanget Bas tillåter 20 förfrågningar per minut.",
        "tgt": "La formule Bas autorise 20 requêtes par minute.",
    },
    {
        "src": "Du kan bjuda in upp till 10 utvecklare per konto.",
        "tgt": "Vous pouvez inviter jusqu'à 10 développeurs par compte.",
    },
    {
        "src": "Slutpunkten för version 3 tas bort om 180 månader.",
        "tgt": "Le point de terminaison de la version 3 sera retiré dans 180 mois.",
    },
]
