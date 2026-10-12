from __future__ import annotations

from typing import Final

# Apify actor references are part of the application contract. Each integration below
# depends on that actor's specific input and output schema, so tenants must not be able
# to replace these references through Baserow configuration.
LINKEDIN_ROOBAHIST_ACTOR: Final = "munificent_geastrum/apify-linkedin"
LINKEDIN_LEGACY_SEARCH_ACTOR: Final = "cheap_scraper/linkedin-job-scraper"
LINKEDIN_COREGENT_SEARCH_ACTOR: Final = "coregent/linkedin-jobs-scraper"
LINKEDIN_SINGLE_JOB_ACTOR: Final = "apt_marble/linkedin-job-listings-scraper-by-url"
INDEED_SINGLE_JOB_ACTOR: Final = "kaix/indeed-scraper"
