# Security Policy

## Reporting a Vulnerability

- Please email me with details and steps to reproduce.
- We aim to acknowledge within 72 hours and provide updates as we triage.
- Do not include secrets or PII in reports.

## Supported Versions

This repository tracks `main`. We generally patch the latest release.

## Best Practices

- Do not commit API keys or credentials. Use `api_keys.json` locally and keep it out of VCS.
- Prefer HTTPS in production and configure firewalls appropriately when exposing the API.
