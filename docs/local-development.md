# Local development

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose for the optional full stack
- A TeX distribution for PDF smoke tests (optional for the offline demo)

## No-account demo

```bash
uv sync --extra dev
uv run python examples/demo.py --output runs/demo
cat runs/demo/decisions.json
```

This creates decisions from fictional postings and one fictional applicant.
It uses domain models and stable job identity from the real pipeline. It does not
call external services, spend credits or contact recruiters.

## Run tests and static analysis

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

The PDF integration tests skip when `pdflatex` is absent.

## Optional multi-service deployment

```bash
cp .env.example .env
# Set your own credentials, endpoints, table/option IDs and secrets.
docker compose config
docker compose build api
```

The illustrative `config/users.toml` contains nonfunctional sample Baserow
table identifiers. The `config/seeds/` files are **examples**, not a ready-made
Baserow workspace. Import/adapt the seed schema to your own Baserow installation,
configure the matching Prompts and Search Criteria tables, and provide provider
credentials before attempting a live worker run.

For full service operation you must configure your own Redis, Baserow, external
discovery provider, LiteLLM deployments, and Telegram credentials.

See `docs/public-private-integration.md` for the planned private overlay pattern.
