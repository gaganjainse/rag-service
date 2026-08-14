# Security Policy

## Reporting a Vulnerability

If you find a security vulnerability in this project, please report it
**privately** to `gagan.jain.se@gmail.com`. Do not open a public issue.

Please include:

- the affected repository and version/commit,
- steps to reproduce,
- a description of the potential impact.

## Response

I treat security reports as high priority and aim to acknowledge within
48 hours and ship a fix for confirmed vulnerabilities promptly.

## Scope

This policy covers the code and deployment configuration in this repository.
Dependency vulnerabilities are tracked via Dependabot alerts and fixed on a
rolling basis.

## Known dependency advisory

**chromadb** (`>=0.5`) — [CVE-2026-45829 / PYSEC-2026-311](https://osv.dev/vulnerability/PYSEC-2026-311),
a pre-auth code-injection in chromadb's HTTP API (EPSS 0.124, actively exploited).
This service uses `chromadb.PersistentClient` (embedded, no listening socket), so the
vulnerable HTTP endpoint is never exposed. A regression test
(`tests/test_chroma_embedded.py`) enforces this. Pin a fixed version the day one
ships (Dependabot will surface it).
