FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS deps

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app \
    && chown app:app /app
WORKDIR /app
COPY --chown=app:app pyproject.toml ./
USER app
RUN uv sync --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

# Lightweight application image used by API, fast worker, notification worker, Beat, and Flower.
# Keeping TeX out of this image makes ordinary application rebuilds much faster and smaller.
FROM deps AS app-base
COPY --chown=app:app . .
RUN uv sync --no-dev && mkdir -p /app/runs

FROM app-base AS api
USER app
CMD ["uvicorn", "job_hunt.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

# TeX is isolated in a stable base layer. Source-only rebuilds of the document worker reuse this
# expensive layer unless the TeX package list or an earlier Dockerfile stage changes.
FROM deps AS worker-deps
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra tex-gyre \
    && rm -rf /var/lib/apt/lists/*

FROM worker-deps AS worker
COPY --chown=app:app . .
USER app
RUN uv sync --no-dev && mkdir -p /app/runs
CMD ["celery", "-A", "job_hunt.worker.celery_app", "worker", "--loglevel=INFO"]
