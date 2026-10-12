# Shared Fillout job-entry form

`job-hunt.form.json` is the single application-owned Fillout form template for every tenant.

Users should import this file into their own Fillout account and keep the form fields, option values, webhook body keys, and widget IDs unchanged. The application owns that contract internally, so there is no tenant-level `fillout_form_id` or `fillout_field_ids` configuration.

The repository template intentionally contains obvious placeholders for the only two tenant-specific Fillout values:

- REST webhook URL: `https://example.com/webhooks/fillout/YOUR_TENANT`
- `Authorization` header: `Bearer REPLACE_WITH_TENANT_FILLOUT_SECRET`

After importing the form:

1. Change the REST webhook URL to your deployed API origin plus `/webhooks/fillout/<tenant>`, where `<tenant>` is the key from `config/users.toml`.
2. Replace the Authorization placeholder with `Bearer <tenant Fillout secret>`. The secret must match the environment variable referenced by that tenant's `fillout_secret_env` entry in `config/users.toml`.

Nothing else in the form should be tenant-specific. In particular, do not add a form ID to the webhook body and do not store a Fillout form ID or widget IDs in Baserow. The endpoint path identifies the tenant and that tenant's bearer secret authenticates the request.

If an older tenant Configuration table still contains `fillout_form_id` or `fillout_field_ids`, both rows can be deleted. The runtime no longer reads either value.

The shared form currently sends these semantic keys: `entryType`, `linkedinJobId`, `indeedJobId`, `companyName`, `jobTitle`, `jobDescription`, `jobUrl`, `location`, `contractType`, `publishedAt`, `externalJobId`, and `pageContent`.
