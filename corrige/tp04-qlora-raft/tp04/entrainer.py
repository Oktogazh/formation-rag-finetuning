"""TP 4 — quantifier, puis entrainer un adaptateur LoRA. Le « Q » de QLoRA.

Deux idees empilees, et il faut les distinguer :

**Quantization.** Les poids du modele sont stockes en 4 bits au lieu de 16. Le
modele occupe environ quatre fois moins de memoire, ce qui le fait tenir sur une
machine de bureau. On y perd un peu de qualite — le TP le mesure.

**LoRA.** On gele les poids du modele et on entraine, a cote, deux petites
matrices par couche. Au lieu de 3 milliards de parametres, on en entraine
quelques millions. Le resultat, l'« adaptateur », pese quelques dizaines de
megaoctets et se branche ou se debranche.

**QLoRA** = LoRA sur un modele quantifie. C'est ce qui rend l'exercice possible
sur un portable.

Trois chemins materiels, un seul point d'entree :

=====================  =========================================================
Apple Silicon          ``mlx-lm``, modele ``mlx-community/…-4bit``
NVIDIA                 ``transformers`` + ``peft`` + ``bitsandbytes`` (nf4)
CPU seul, ou Windows   **pas d'entrainement.** Recuperez l'adaptateur du
sans carte              formateur : ``python tp.py adaptateur --telecharger``
=====================  =========================================================
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from commun import RACINE
from commun.materiel import accelerateur

MODELE_HF = "mistralai/Ministral-3-3B-Instruct-2512"
MODELE_MLX = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
DOSSIER_ADAPTATEUR = RACINE / "resultats" / "tp04" / "adaptateur"


def config_lora(rang: int = 16) -> dict:
    """Les reglages de LoRA. Quatre nombres, et ils ont un sens.

    ``rang``     largeur des matrices ajoutees. Plus grand = plus de capacite,
                 plus de memoire, plus de risque de sur-apprentissage. 8 a 32
                 pour une tache de style comme la notre.
    ``alpha``    facteur d'echelle applique a la sortie de l'adaptateur. La
                 convention la plus repandue est ``alpha = 2 x rang``.
    ``dropout``  regularisation. 0,05 sur un petit jeu de donnees.
    ``cibles``   quelles matrices du transformeur recoivent un adaptateur. Les
                 projections d'attention (``q, k, v, o``) sont le minimum ; y
                 ajouter les couches du MLP (``gate, up, down``) donne un
                 adaptateur plus expressif et plus lourd.
    """
    # <<<TODO 3 ★ Completer la configuration LoRA
    #! Rendez un dictionnaire avec exactement ces quatre cles :
    #!   "rang"    -> l'argument rang
    #!   "alpha"   -> deux fois le rang (la convention rappelee ci-dessus)
    #!   "dropout" -> 0.05
    #!   "cibles"  -> ["q_proj", "k_proj", "v_proj", "o_proj",
    #!                 "gate_proj", "up_proj", "down_proj"]
    #! Analogue : config_entrainement() juste en dessous, deja remplie.
    #! Test : python tp.py test tp04 -k todo3
    return {
        "rang": rang,
        "alpha": 2 * rang,
        "dropout": 0.05,
        "cibles": ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    }
    # >>>TODO 3


def config_entrainement(epoques: int = 2, exemples: int = 350) -> dict:
    """Les reglages d'entrainement. Fourni — c'est votre modele pour le TODO 3."""
    taille_lot = 4
    pas_par_epoque = max(1, exemples // taille_lot)
    return {
        "epoques": epoques,
        "taille_lot": taille_lot,
        "iterations": pas_par_epoque * epoques,
        "pas_apprentissage": 2e-4,
        "longueur_max": 1024,
        "couches_adaptees": 8,
    }


def entrainer(epoques: int = 2, rang: int = 16) -> int:
    """Entraine l'adaptateur sur la machine disponible, ou explique pourquoi non."""
    import os

    from tp04.donnees import DOSSIER_DONNEES

    if os.environ.get("MISTRAL_API_KEY"):
        print("""
  Vous travaillez avec l'API Mistral. On ne fine-tune pas par l'API dans cette
  formation : le sujet du TP est ce qui se passe dans les poids, et une API le
  cache par construction.

      python tp.py adaptateur --telecharger

  recupere l'adaptateur entraine par le formateur, et « comparer » fonctionnera.""")
        return 2

    if not (DOSSIER_DONNEES / "train.jsonl").exists():
        print("  Le jeu d'entrainement n'existe pas encore : python tp.py donnees")
        return 2

    lora = config_lora(rang)
    nombre = sum(1 for _ in (DOSSIER_DONNEES / "train.jsonl").open(encoding="utf-8"))
    entrainement = config_entrainement(epoques, nombre)
    DOSSIER_ADAPTATEUR.mkdir(parents=True, exist_ok=True)

    materiel = accelerateur()
    print(f"\nEntrainement QLoRA · {materiel} · rang {lora['rang']}, "
          f"alpha {lora['alpha']}, {entrainement['iterations']} iterations")

    if materiel == "mps":
        code = _entrainer_mlx(lora, entrainement)
        cadre, base = "mlx", MODELE_MLX
    elif materiel == "cuda":
        code = _entrainer_peft(lora, entrainement)
        cadre, base = "peft", MODELE_HF
    else:
        print("""
  Aucun accelerateur : entrainer 3 milliards de parametres sur le processeur
  prendrait des heures. Ce n'est pas un defaut de votre machine, c'est la
  realite du fine-tuning.

      python tp.py adaptateur --telecharger

  recupere l'adaptateur du formateur. Vous ferez « comparer » avec, et vous
  aurez le meme tableau que vos voisins.""")
        return 2

    if code == 0:
        (DOSSIER_ADAPTATEUR / "adaptateur.json").write_text(
            json.dumps({"cadre": cadre, "modele_base": base, **lora}, indent=2),
            encoding="utf-8",
        )
        print(f"""
  Adaptateur ecrit dans {DOSSIER_ADAPTATEUR}
  Mesurez ce qu'il a change :  python tp.py comparer""")
    return code


def _entrainer_mlx(lora: dict, entrainement: dict) -> int:
    """Apple Silicon. ``mlx-lm`` lit ses hyperparametres LoRA dans un YAML."""
    from tp04.donnees import DOSSIER_DONNEES

    config = DOSSIER_ADAPTATEUR / "config-mlx.yaml"
    config.write_text(
        "lora_parameters:\n"
        f"  rank: {lora['rang']}\n"
        f"  scale: {lora['alpha'] / lora['rang']:.1f}\n"
        f"  dropout: {lora['dropout']}\n"
        "  keys: [" + ", ".join(f'"self_attn.{c}"' for c in lora["cibles"][:4]) + "]\n",
        encoding="utf-8",
    )
    commande = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODELE_MLX,
        "--train",
        "--data", str(DOSSIER_DONNEES),
        "--fine-tune-type", "lora",
        "--mask-prompt",
        "--num-layers", str(entrainement["couches_adaptees"]),
        "--batch-size", str(entrainement["taille_lot"]),
        "--iters", str(entrainement["iterations"]),
        "--learning-rate", str(entrainement["pas_apprentissage"]),
        "--max-seq-length", str(entrainement["longueur_max"]),
        "--adapter-path", str(DOSSIER_ADAPTATEUR),
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--seed", "13",
        "-c", str(config),
    ]
    print("  " + " ".join(commande) + "\n")
    try:
        return subprocess.call(commande)
    except FileNotFoundError:
        print("  mlx-lm est absent :  pip install mlx-lm")
        return 2


def _entrainer_peft(lora: dict, entrainement: dict) -> int:
    """NVIDIA. Quantization nf4 par bitsandbytes, LoRA par peft, boucle par trl."""
    from tp04.donnees import DOSSIER_DONNEES

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as erreur:
        print(f"""  Il manque une bibliotheque ({erreur.name}). Sur NVIDIA :

      pip install torch --index-url https://download.pytorch.org/whl/cu124
      pip install transformers peft trl bitsandbytes datasets accelerate""")
        return 2

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    jeu = load_dataset("json", data_files={
        "train": str(DOSSIER_DONNEES / "train.jsonl"),
        "validation": str(DOSSIER_DONNEES / "valid.jsonl"),
    })
    tokenizer = AutoTokenizer.from_pretrained(MODELE_HF)
    entraineur = SFTTrainer(
        model=MODELE_HF,
        args=SFTConfig(
            output_dir=str(DOSSIER_ADAPTATEUR),
            num_train_epochs=entrainement["epoques"],
            per_device_train_batch_size=1,
            gradient_accumulation_steps=entrainement["taille_lot"],
            learning_rate=entrainement["pas_apprentissage"],
            max_length=entrainement["longueur_max"],
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            model_init_kwargs={"quantization_config": quantization, "device_map": "auto"},
        ),
        train_dataset=jeu["train"],
        eval_dataset=jeu["validation"],
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=lora["rang"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["cibles"],
            task_type="CAUSAL_LM",
        ),
    )
    entraineur.train()
    entraineur.model.save_pretrained(str(DOSSIER_ADAPTATEUR))
    return 0


def telecharger_adaptateur(depot: str = "Oktogazh/helios-sv-lora") -> int:
    """Recupere l'adaptateur du formateur. Pour les machines qui n'entrainent pas."""
    materiel = accelerateur()
    suffixe = "-mlx" if materiel == "mps" else "-peft"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  pip install huggingface-hub")
        return 2
    DOSSIER_ADAPTATEUR.mkdir(parents=True, exist_ok=True)
    print(f"  Telechargement de {depot}{suffixe} …")
    try:
        snapshot_download(repo_id=depot + suffixe, local_dir=str(DOSSIER_ADAPTATEUR))
    except Exception as erreur:
        print(f"""  Echec : {erreur}

  Ce depot est publie par le formateur avant la session. S'il n'existe pas
  encore, demandez-lui — ou sautez « comparer » et lisez le tableau d'un
  voisin : c'est le meme adaptateur pour toute la salle.""")
        return 2
    print(f"  Adaptateur installe dans {DOSSIER_ADAPTATEUR}\n  Mesurez : python tp.py comparer")
    return 0
