# Image for the routing runs in deploy/jobs/: the mmorch command with the 'live' extra, plus the prompts.
# Build it from the repository root:
#   docker build -f deploy/Dockerfile -t <registry>/mmorch:2.0.0 .
# WORKDIR /app holds data/ and acts as the repository root (--root), so `mmorch route` reads
# /app/data/prompts.jsonl.gz and writes to /app/results/live/. Credentials come from the environment
# (LLM_API_BASE, LLM_API_KEY), never from the image.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir '.[live]'

COPY data/ data/

ENTRYPOINT ["mmorch"]
