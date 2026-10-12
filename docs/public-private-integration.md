# Planned public-core / private-data integration

**Status: design only. The private production repository is unchanged.**

The long-term goal is one maintainable implementation, with the reusable
engine published here and all real applicant data isolated in a private
configuration/deployment repository.

```text
Public (versioned): job_hunt package, adapters, workers, tests, demo
                           |
                           v
Private wrapper: pinned public version + deployment compose + secrets
                           |
                           v
Private mounts: tenants/<person>/master_cv.json, templates, config registry
                           |
                           v
Running application with private applicant data
```

## Proposed migration

1. Release tagged, tested public versions with a stable package/import contract.
2. Add an explicit tenant asset-root configuration option to avoid assumptions
   about the working directory and to support local-only read-only mounts.
3. Convert the private repository into a thin deployment/configuration layer,
   keeping its personal templates, CV JSON, secrets and generated output private.
4. Pin the public package by immutable tag or commit, and integrate through a
   reproducible build (package dependency, submodule, or image).
5. Test document rendering, tenant lookup, queue state and deployments with
   **synthetic** data before changing production.
6. Migrate deliberately with rollback steps. Never automatically overwrite
   private CVs or credentials during an upstream update.

## Important boundaries

A git subtree merge from private into public would disclose historical data and
must never be used. Future syncing should export only reviewed public code from
private, or preferably move all reusable changes into the public repository first.
Use explicit dependency versions rather than bidirectional file synchronization.

The current public code still loads assets from the configured project root,
as in the private codebase. The cleaner external tenant root is a **future**
interface, not an implemented feature.
