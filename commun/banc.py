"""La table finale : tout ce qui a été mesuré pendant trois jours.

Elle se construit à partir des fichiers de ``resultats/`` — chaque cellule
`atelier.mesurer(...)` de chaque notebook y dépose le sien. Ce tableau est donc
**votre** parcours, pas celui du formateur. Les lignes que vous n'avez pas
mesurées sont marquées comme telles, et ce n'est pas grave : les TP sont
indépendants, on ne fait pas tout.

Comment la lire, dans cet ordre :

1. **BLEU monte-t-il d'un cran à l'autre ?** Si un cran ne fait pas mieux que le
   précédent, il ne mérite pas sa complexité. Vrai pour le RAG comme pour le
   fine-tuning.
2. **La colonne Appels.** Le cran le plus juste n'est pas le plus cher, et c'est
   le résultat le plus utile de ces trois jours.
3. **La ligne « nouveau » des tables par catégorie.** Elle rappelle ce qu'aucune
   technique ne sait faire : inventer ce qui n'est nulle part.
"""

from __future__ import annotations

ORDRE = [
    ("tp01-nu", "0  prompt nu"),
    ("tp01-consigne", "1  + consigne de style"),
    ("tp01-exemples", "1  + exemples de mémoire"),
    ("tp01-mur", "1b tout dans le prompt"),
    ("tp02-rag-lexical", "2  RAG lexical"),
    ("tp02-rag-dense", "2  RAG dense"),
    ("tp03-chaine", "3  chaîne LangChain"),
    ("tp03-chaine-verifiee", "3  + vérification"),
    ("tp04-base", "4  petit modèle, base"),
    ("tp04-adapte", "4  petit modèle, adapté"),
    ("tp06-graphe", "6  graphe routé"),
    ("tp06-sans-rag", "6b sans RAG"),
]


def afficher_banc() -> list[dict]:
    from commun.rapport import lire_mesures

    mesures = {m["cran"]: m for m in lire_mesures()}
    print("\nLe parcours, cran par cran")
    print("=" * 92)
    print(f"{'Cran':26} {'n':>3} {'BLEU':>7} {'chrF':>7} {'Termino':>8} {'Chiffres':>9} "
          f"{'Tokens':>8} {'s/seg':>7} {'Appels':>7}")
    print("-" * 92)
    lignes = []
    for cle, etiquette in ORDRE:
        mesure = mesures.get(cle)
        if not mesure:
            print(f"{etiquette:26} {'—':>3}   non mesuré")
            continue
        s = mesure["synthese"]
        termino = f"{s['termino']:7.0f}%" if s.get("termino") is not None else "      -"
        tokens = f"{s['tokens_entree']:8.0f}" if s["tokens_entree"] else "       -"
        print(f"{etiquette:26} {s['n']:>3} {s['bleu']:7.1f} {s['chrf']:7.1f} {termino:>8} "
              f"{s['chiffres']:8.0f}% {tokens} {s['secondes']:7.2f} {s['appels']:7.2f}")
        lignes.append({"cran": cle, **s})
    print("-" * 92)

    if len(lignes) < 2:
        print("""
  Il n'y a pas encore de quoi comparer. Chaque cellule de mesure dépose sa ligne
  dans resultats/ : faites-en deux, et ce tableau devient lisible.""")
        return lignes

    comparables = {}
    for ligne in lignes:
        comparables.setdefault(ligne["n"], []).append(ligne)
    lot = max(comparables.values(), key=len)
    if len(lot) >= 2:
        meilleur = max(lot, key=lambda l: l["bleu"])
        moins_cher = min((l for l in lot if l["bleu"] >= meilleur["bleu"] - 2),
                         key=lambda l: l["appels"])
        print(f"""
  Sur les lignes mesurées avec le même nombre de segments ({lot[0]['n']}) :
    meilleur BLEU              {meilleur['cran']}  ({meilleur['bleu']:.1f})
    le moins cher à 2 points   {moins_cher['cran']}  ({moins_cher['bleu']:.1f} BLEU,
                               {moins_cher['appels']:.2f} appel par segment)

  C'est cette deuxième ligne qu'on défend devant un client. La première sert à
  savoir jusqu'où on pourrait aller.

  Ne comparez que des lignes de même « n » : une mesure sur 8 segments et une
  sur 80 ne se comparent pas, et c'est la faute la plus fréquente.""")
    return lignes
