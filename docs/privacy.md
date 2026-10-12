# Privacy and public repository boundary

This repository is a clean, independent public snapshot. It does **not** include
the Git history of the private development repository.

- The Atlas and Nova applicant profiles are newly written fictional examples.
- The CV and cover-letter templates use fictional names, addresses, and email
  domains. None were copied from a real applicant template.
- Tenant configuration uses example Baserow IDs and no valid credentials.
- Fillout widget identifiers are demonstration identifiers and require
  replacement with identifiers exported from your own form.
- Prompt seeds show workflow structure but are examples, not user-specific
  production settings.
- Screenshots, logs, generated PDFs, ZIPs and runtime caches should never
  contain actual users or job-tracking data.

## Before contributing

Do not commit real master CVs, `.env` files, private keys, API responses, inbox
contents, uploaded resumes, candidate email addresses or deployment tokens.
Use `example.org` addresses and synthetic JSON payloads.

A file's absence from the current branch does not remove it from Git history.
Do not import the private repository as a Git remote or merge its commit history
into this public repository. Use a reviewed source snapshot and new commits.

## Future data injection

A future private production wrapper can install the public package at a pinned
version and mount local-only `tenants/` and runtime configuration into the
application container. Implement, test, and deploy this interface separately;
this public snapshot does **not** change the existing private deployment.
