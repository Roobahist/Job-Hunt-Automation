from __future__ import annotations

import json
from pathlib import Path


def test_shared_fillout_template_is_tenant_neutral() -> None:
    template = json.loads(Path("config/fillout/job-hunt.form.json").read_text())
    integrations = template["template"]["integrations"]
    assert len(integrations) == 1

    integration = next(iter(integrations.values()))["writeQuery"]["config"]
    assert integration["url"]["logic"]["value"] == "https://example.com/webhooks/fillout/YOUR_TENANT"
    assert integration["simpleURL"]["url"] == "https://example.com/webhooks/fillout/YOUR_TENANT"

    headers = {item["key"]: item["value"] for item in integration["headers"]}
    assert headers["Authorization"]["logic"]["stringLiteral"] == ("Bearer REPLACE_WITH_TENANT_FILLOUT_SECRET")

    body_keys = [item["key"] for item in integration["body"]]
    assert "formId" not in body_keys
    assert body_keys == [
        "entryType",
        "linkedinJobId",
        "indeedJobId",
        "companyName",
        "jobTitle",
        "jobDescription",
        "jobUrl",
        "location",
        "contractType",
        "publishedAt",
        "externalJobId",
        "pageContent",
    ]
