# Security guidelines

- Use local environment variables or a secret manager for credentials.
- Keep `.env`, private CVs, outputs, and deployment artifacts untracked.
- Do not paste real user data into GitHub Issues or CI logs.
- Review changes to `config/fillout` carefully: synthetic field identifiers
  will not match anyone's real Fillout form without explicit configuration.
- Never allow the document renderer to execute arbitrary model-supplied
  LaTeX commands. Keep shell escape disabled when running `pdflatex`.
- URL ingestion must continue to use the existing SSRF-aware fetch helper.
- Set strong Redis, database, Telegram and LiteLLM credentials when deploying.
- Restrict operators, service ports, webhooks and credentials to least privilege.

## Pre-publication checks

Review all newly tracked text for credentials, personal names, email addresses,
phone numbers, production domains, internal IP addresses, and access identifiers.
Use a dedicated secret scanner such as Gitleaks, then manually inspect
any remaining live configuration identifiers. A passing scanner cannot
guarantee the absence of private information.
