"""TP 6 — la table finale. Tout ce qui a ete mesure pendant trois jours. Fourni.

Elle se construit a partir des fichiers de ``resultats/`` : chaque commande de
chaque TP en a depose un. Ce tableau est donc **votre** parcours, pas celui du
formateur. Les lignes que vous n'avez pas mesurees sont marquees comme telles.

Comment la lire, dans cet ordre :

1. **BLEU monte d'un cran a l'autre.** Si un cran ne fait pas mieux que le
   precedent, il ne merite pas sa complexite. C'est vrai pour le RAG comme pour
   le fine-tuning.
2. **La colonne Appels.** Le cran le plus juste n'est pas toujours le plus cher,
   et c'est le resultat le plus utile de la formation.
3. **La ligne « nouveau » des tableaux par categorie.** Elle rappelle ce
   qu'aucune technique ne sait faire : inventer ce qui n'est nulle part.
"""

from __future__ import annotations

ORDRE = [
    ("tp01-nu", "0  prompt nu"),
    ("tp01-consigne", "1  + consigne de style"),
    ("tp01-exemples", "1  + 3 exemples choisis"),
    ("tp01-mur", "1b tout dans le prompt"),
    ("tp02-rag-lexicale-k3", "2  RAG lexical k=3"),
    ("tp02-rag-dense-k3", "2  RAG dense k=3"),
    ("tp03-chaine", "3  LangChain"),
    ("tp03-chaine-verifiee", "3  + verification"),
    ("tp04-adaptateur", "4  + adaptateur LoRA"),
    ("tp06-sans-rag", "4b adaptateur sans RAG"),
    ("tp06-graphe", "6  graphe routé"),
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
              f"{s['chiffres']:8.0f}% {tokens} {s['secondes']:7.2f} "
              f"{s['appels']:7.2f}")
        lignes.append({"cran": cle, **s})
    print("-" * 92)

    if len(lignes) < 2:
        print("""
  Il n'y a pas encore de quoi comparer. Chaque commande de chaque TP depose sa
  mesure dans resultats/ : faites-en deux, et ce tableau devient lisible.""")
        return lignes

    meilleur_bleu = max(lignes, key=lambda l: l["bleu"])
    moins_cher = min((l for l in lignes if l["bleu"] >= meilleur_bleu["bleu"] - 2),
                     key=lambda l: l["appels"])
    print(f"""
  Meilleur BLEU        {meilleur_bleu['cran']}  ({meilleur_bleu['bleu']:.1f})
  Le moins cher a moins de 2 points du meilleur :
                       {moins_cher['cran']}  ({moins_cher['bleu']:.1f} BLEU,
                       {moins_cher['appels']:.2f} appel par segment)

  C'est cette deuxieme ligne qu'on defend devant un client. La premiere sert a
  savoir jusqu'ou on pourrait aller.""")
    return lignes
