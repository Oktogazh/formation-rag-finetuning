"""TP 5 — la piece qui manque au service : une garde.

Le service du Space a un defaut, et c'est un defaut de conception, pas de
modele : **il accepte tout ce qui arrive**. Chaque requete entre, occupe un fil
d'execution, attend le modele qui ne sait traiter qu'une chose a la fois, et
personne ne repond a personne.

Un service qui sature doit faire trois choses, dans cet ordre :

1. **limiter** le nombre de requetes traitees en meme temps (le modele n'ira pas
   plus vite parce qu'on lui en donne plus) ;
2. **refuser vite** ce qu'il ne pourra pas traiter — un ``503`` en 5
   millisecondes vaut mieux qu'une reponse en 120 secondes ;
3. **abandonner** ce qui prend trop longtemps, avec un ``504``.

Refuser proprement n'est pas un aveu d'echec : c'est ce qui permet aux requetes
acceptees de rester rapides. Un service qui n'a pas de garde n'a pas de latence,
il a une loterie.
"""

from __future__ import annotations

import asyncio


class Sature(RuntimeError):
    """Trop de requetes en attente : on refuse tout de suite (HTTP 503)."""


class Delai(RuntimeError):
    """La requete a pris trop longtemps : on abandonne (HTTP 504)."""


class Garde:
    """Limite la concurrence, borne la file d'attente, coupe au bout du temps.

    ``concurrence_max``  requetes traitees en meme temps. 1 pour un modele qui
                         ne sait pas traiter de lot.
    ``file_max``         requetes qui peuvent attendre leur tour. Au-dela, on
                         refuse.
    ``timeout_s``        temps maximum d'une requete.
    """

    def __init__(self, concurrence_max: int = 1, file_max: int = 2, timeout_s: float = 30.0):
        self.concurrence_max = concurrence_max
        self.file_max = file_max
        self.timeout_s = timeout_s
        self.semaphore = asyncio.Semaphore(concurrence_max)
        self.en_attente = 0
        self.refusees = 0
        self.expirees = 0

    async def _lancer(self, fonction, *arguments):
        """Execute la fonction bloquante hors de la boucle, avec un delai. Fourni.

        ``fonction`` appelle le modele : si on la lancait directement, la boucle
        asyncio serait figee et le serveur ne pourrait meme plus repondre a
        ``/sante`` pendant le calcul.
        """
        boucle = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                boucle.run_in_executor(None, fonction, *arguments), timeout=self.timeout_s)
        except asyncio.TimeoutError as erreur:
            self.expirees += 1
            raise Delai(f"depassement de {self.timeout_s:.0f} s") from erreur

    async def executer(self, fonction, *arguments):
        """Execute ``fonction(*arguments)`` sous protection."""
        # <<<CODE 2 ★★ La regle d'admission
        #> 1. Si self.en_attente >= self.file_max : incrementez self.refusees et
        #>    levez Sature(...) TOUT DE SUITE, sans attendre. Refuser en cinq
        #>    millisecondes vaut mieux que repondre en deux minutes.
        #> 2. Sinon, incrementez self.en_attente, puis « async with
        #>    self.semaphore: » ; une fois le jeton obtenu, decrementez
        #>    self.en_attente et rendez « await self._lancer(fonction, *arguments) ».
        #> Test : python tp.py test tp05 -k code2
        if self.en_attente >= self.file_max:
            self.refusees += 1
            raise Sature(f"{self.en_attente} requetes attendent deja "
                         f"(file_max={self.file_max})")
        self.en_attente += 1
        async with self.semaphore:
            self.en_attente -= 1
            return await self._lancer(fonction, *arguments)
        # >>>CODE 2

    def etat(self) -> dict:
        return {
            "concurrence_max": self.concurrence_max,
            "file_max": self.file_max,
            "timeout_s": self.timeout_s,
            "en_attente": self.en_attente,
            "refusees": self.refusees,
            "expirees": self.expirees,
        }
