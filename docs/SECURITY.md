# Security Policy

## Reporting a Vulnerability

- Please email me with details and steps to reproduce.
- We aim to acknowledge within 72 hours and provide updates as we triage.
- Do not include secrets or PII in reports.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.x.x   | ✅ Current |
| 2.x.x   | ⚠️ Security fixes only |
| < 2.0   | ❌ No longer supported |

## Best Practices

- Do not commit API keys or credentials. Use `api_keys.json` locally and keep it out of VCS.
- Prefer HTTPS in production and configure firewalls appropriately when exposing the API.
- Keep vLLM and PyTorch dependencies updated for security patches.
- Review `docs/API_AUTHENTICATION.md` for proper API key management.
