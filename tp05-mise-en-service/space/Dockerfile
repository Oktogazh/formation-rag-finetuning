# Space HuggingFace — service de traduction Helios, volontairement mal configure.
#
# Materiel : CPU basic (2 vCPU, 16 Go), l'offre gratuite. C'est un choix :
# l'exercice du TP 5 consiste a faire saturer ce service, et il faut donc qu'il
# soit saturable en trente secondes par quatre stagiaires.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 utilisateur
USER utilisateur
ENV PATH=/home/utilisateur/.local/bin:$PATH \
    HOME=/home/utilisateur \
    HF_HOME=/home/utilisateur/.cache/huggingface

WORKDIR /home/utilisateur/app
COPY --chown=utilisateur requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=utilisateur app.py .

EXPOSE 7860
CMD ["python", "app.py"]
