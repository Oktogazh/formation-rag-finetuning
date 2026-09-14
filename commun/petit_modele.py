"""Le petit modèle du TP 4, et sa boucle d'entraînement sur processeur.

**Pourquoi un petit modèle.** Fine-tuner les 3 milliards de paramètres de
Ministral demande une carte graphique que personne dans la salle n'a. Plutôt
que de regarder le formateur le faire, on fait l'opération en vrai sur un modèle
assez petit pour tenir sur un processeur : ``SmolLM2-135M-Instruct``, 135
millions de paramètres, 270 Mo.

**Ce que ça change, et ce que ça ne change pas.** Les traductions produites par
ce modèle sont mauvaises — il est vingt fois plus petit que celui des autres TP.
En revanche, tout ce que le TP 4 veut montrer reste visible, et même plus
visible :

* un adaptateur LoRA apprend une **forme** — le format de sortie, le vocabulaire
  imposé, le registre ;
* il n'apprend **pas de connaissance** : sur un segment dont la réponse n'est
  nulle part, il ne fait pas mieux qu'avant ;
* la boucle d'entraînement, la perte qui descend, le surapprentissage : tout est
  là, en quelques minutes au lieu de quelques heures.

Ne comparez donc pas les chiffres du TP 4 à ceux des autres TP. Comparez
« avant » et « après », sur le même petit modèle.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from commun import RACINE

MODELE_PETIT = os.environ.get("TP_PETIT_MODELE", "HuggingFaceTB/SmolLM2-135M-Instruct")
DOSSIER_ADAPTATEUR = RACINE / "resultats" / "tp04" / "adaptateur"


def _torch():
    try:
        import torch
    except ImportError as erreur:  # pragma: no cover
        raise ImportError(
            "Le TP 4 a besoin de torch et peft :\n"
            "    pip install torch peft\n"
            "Aucun GPU n'est nécessaire, environ 250 Mo."
        ) from erreur
    return torch


def est_telecharge() -> bool:
    """Le petit modèle est-il déjà dans le cache Hugging Face ?

    Il sert **dès le TP 1**, au chapitre 0, pour montrer à la main la
    tokenisation, l'embedding et la boucle de génération. Le télécharger en
    salle coûte dix minutes de session : ``python tp.py check`` le vérifie.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MODELE_PETIT, local_files_only=True)
        return True
    except Exception:
        return False


def charger(adaptateur: str | Path | None = None):
    """Rend ``(modele, tokenizer)``. ``adaptateur`` branche un LoRA entraîné."""
    torch = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    tokenizer = AutoTokenizer.from_pretrained(MODELE_PETIT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    modele = AutoModelForCausalLM.from_pretrained(MODELE_PETIT, dtype=torch.float32)
    if adaptateur:
        from peft import PeftModel

        modele = PeftModel.from_pretrained(modele, str(adaptateur))
    modele.eval()
    return modele, tokenizer


class MoteurPetit:
    """Le petit modèle derrière la même porte que les autres moteurs.

    Il se branche donc dans ``atelier.mesurer`` sans rien changer d'autre :
    c'est ce qui rend la comparaison « base contre adapté » honnête.
    """

    def __init__(self, adaptateur: str | Path | None = None, max_tokens: int = 64):
        self.adaptateur = str(adaptateur) if adaptateur else None
        self.max_tokens = max_tokens
        self.modele, self.tokenizer = charger(adaptateur)

    def nom(self) -> str:
        return f"petit:{MODELE_PETIT.split('/')[-1]}" + (" + adaptateur" if self.adaptateur else "")

    def generer(self, messages, temperature: float = 0.0, max_tokens: int | None = None):
        from commun.moteur import Reponse

        torch = _torch()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        entrees = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        debut = time.perf_counter()
        with torch.no_grad():
            produits = self.modele.generate(
                **entrees,
                max_new_tokens=max_tokens or self.max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        nouveaux = produits[0][entrees["input_ids"].shape[1]:]
        texte = self.tokenizer.decode(nouveaux, skip_special_tokens=True)
        return Reponse(
            texte=texte,
            tokens_entree=int(entrees["input_ids"].shape[1]),
            tokens_sortie=int(nouveaux.shape[0]),
            secondes=time.perf_counter() - debut,
            moteur=self.nom(),
        )


def entrainer_lora(exemples: list[dict], config: dict, *, epoques: int = 1,
                   taille_lot: int = 4, pas_apprentissage: float = 2e-4,
                   longueur_max: int = 512, sortie: Path | None = None) -> dict:
    """Entraîne un adaptateur LoRA sur le processeur. Fourni, et lisible exprès.

    Six étapes, et il n'y en a pas une de plus dans les bibliothèques qui font
    ça pour vous :

    1. charger le modèle et geler ses poids ;
    2. y greffer les matrices LoRA (``get_peft_model``) ;
    3. transformer chaque exemple en texte, avec le gabarit de conversation ;
    4. pour chaque lot : calculer la perte, la rétropropager, avancer d'un pas ;
    5. regarder la perte descendre ;
    6. écrire l'adaptateur sur le disque — quelques mégaoctets, pas des
       gigaoctets, parce qu'on n'a entraîné que les matrices ajoutées.
    """
    torch = _torch()
    from peft import LoraConfig, get_peft_model

    sortie = Path(sortie or DOSSIER_ADAPTATEUR)
    modele, tokenizer = charger()
    modele = get_peft_model(modele, LoraConfig(
        r=config["rang"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
        target_modules=config["cibles"], task_type="CAUSAL_LM"))
    entrainables = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modele.parameters())
    print(f"  Paramètres entraînés : {entrainables:,} sur {total:,} "
          f"({100 * entrainables / total:.2f} %)")

    textes = [tokenizer.apply_chat_template(e["messages"], tokenize=False) for e in exemples]
    lots = [textes[i:i + taille_lot] for i in range(0, len(textes), taille_lot)]
    optimiseur = torch.optim.AdamW(
        [p for p in modele.parameters() if p.requires_grad], lr=pas_apprentissage)

    modele.train()
    journal, debut = [], time.perf_counter()
    for epoque in range(epoques):
        for rang, lot in enumerate(lots):
            entrees = tokenizer(lot, return_tensors="pt", padding=True,
                                truncation=True, max_length=longueur_max)
            entrees["labels"] = entrees["input_ids"].clone()
            perte = modele(**entrees).loss
            perte.backward()
            optimiseur.step()
            optimiseur.zero_grad()
            journal.append(float(perte))
            if rang % 10 == 0:
                ecoule = time.perf_counter() - debut
                faits = epoque * len(lots) + rang + 1
                reste = ecoule / faits * (epoques * len(lots) - faits)
                print(f"    époque {epoque + 1}/{epoques}  pas {rang:3}/{len(lots)}  "
                      f"perte {float(perte):.3f}  reste ~{reste / 60:.1f} min", flush=True)

    sortie.mkdir(parents=True, exist_ok=True)
    modele.save_pretrained(str(sortie))
    (sortie / "entrainement.json").write_text(json.dumps({
        "modele_base": MODELE_PETIT, "config": config, "epoques": epoques,
        "exemples": len(exemples), "perte": journal,
    }, indent=2), encoding="utf-8")
    poids = sum(f.stat().st_size for f in sortie.glob("*.safetensors")) / 1e6
    print(f"\n  Adaptateur écrit dans {sortie} ({poids:.1f} Mo)")
    print(f"  Perte : {journal[0]:.3f} au départ → {sum(journal[-5:]) / 5:.3f} à la fin")
    print(f"  Durée : {(time.perf_counter() - debut) / 60:.1f} min")
    return {"perte": journal, "minutes": (time.perf_counter() - debut) / 60, "sortie": str(sortie)}
