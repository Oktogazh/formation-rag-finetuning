#!/bin/sh
# Demarre Ollama en arriere-plan, tire le modele, puis lance le service.
set -e

ollama serve &

for i in $(seq 1 60); do
    curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
done

ollama pull "${OLLAMA_MODELE:-ministral-3:3b}"

exec python app.py
