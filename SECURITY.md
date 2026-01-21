# Security Policy

This document outlines security practices and deployment guidelines for Integradio.

## Supported Versions

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| 0.3.x   | :white_check_mark: | Current release with security hardening |
| < 0.3.0 | :x:                | Upgrade recommended |

## Security Requirements

### Gradio Version

**Minimum required: Gradio 5.0.0+**

Older versions have known critical vulnerabilities:

| CVE | Severity | Description | Affected Versions |
|-----|----------|-------------|-------------------|
| CVE-2024-1561 | High | Arbitrary file read via component_server API | 3.47 - 4.12 |
| CVE-2024-8021 | Medium | Open redirect vulnerability | <= 4.37.2 |
| CVE-2023-51449 | High | Path traversal in file endpoint | 4.0 - 4.10 |
| CVE-2023-34239 | High | Arbitrary file read + URL proxying | < 3.34.0 |

Gradio 5.0.0 was audited by Trail of Bits and all reported issues were fixed.

### Verify Your Gradio Version

```python
from integradio.events import validate_gradio_version

is_secure, message = validate_gradio_version()
print(message)
```

## Deployment Checklist

### Before Deploying to Production

- [ ] **Gradio version >= 5.0.0** - Run `pip show gradio` to verify
- [ ] **Never use `share=True`** - This exposes your app to the internet via Gradio's servers
- [ ] **Use HTTPS** - Deploy behind a reverse proxy (nginx, Caddy) with TLS
- [ ] **Set strong secret keys** - Use `EventSigner.generate_key()` for 256-bit keys
- [ ] **Bind to localhost** - Use `server_name="127.0.0.1"` and proxy external traffic
- [ ] **Enable security headers** - Use the provided middleware
- [ ] **Run security scans** - `bandit -r integradio/` and `pip-audit`

### Secure Launch Configuration

```python
import gradio as gr
from integradio.events import get_secure_gradio_config

demo = gr.Blocks()
# ... build your app ...

# Use secure defaults
demo.launch(**get_secure_gradio_config())
```

Or manually configure:

```python
demo.launch(
    share=False,              # NEVER True in production
    server_name="127.0.0.1",  # Bind to localhost only
    show_error=False,         # Don't expose stack traces
    show_api=False,           # Disable API docs
    max_file_size="50mb",     # Limit upload sizes
)
```

### Security Headers Middleware

For FastAPI/Starlette deployments:

```python
from fastapi import FastAPI
from integradio.events import create_security_headers_middleware, SecurityHeadersConfig

app = FastAPI()

# Add security headers
config = SecurityHeadersConfig(
    enable_hsts=True,  # Only if behind HTTPS
    enable_csp=True,
)
app.add_middleware(create_security_headers_middleware(config))
```

## Security Features

### File Upload Protection

All file uploads are validated for:
- **Path traversal** - `../` sequences are stripped
- **Null byte injection** - `\x00` removed to prevent extension bypass
- **Blocked extensions** - `.exe`, `.bat`, `.php`, `.sh`, etc. are rejected
- **Double extensions** - `file.jpg.exe` is caught
- **Size limits** - 100MB default maximum

### WebSocket Security

- **HMAC-SHA256 signing** - Message integrity verification
- **Constant-time comparison** - Prevents timing attacks
- **Rate limiting** - Token bucket algorithm per-client
- **Nonce tracking** - Replay attack prevention
- **Origin validation** - CORS-like protection
- **Connection limits** - Per-IP and global limits

### Secret Key Requirements

```python
from integradio.events import EventSigner

# Generate a secure key (recommended)
secret = EventSigner.generate_key()  # 256-bit key

# Use in your app
signer = EventSigner(secret)

# Weak keys are REJECTED by default
signer = EventSigner("weak")  # Raises ValueError

# Only for testing (not production!)
signer = EventSigner("test", allow_weak_key=True)  # Warning issued
```

## Running Security Scans

### Static Analysis with Bandit

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run bandit
bandit -r integradio/ -ll

# Generate report
bandit -r integradio/ -f json -o security-report.json
```

### Dependency Audit

```bash
# Check for vulnerable dependencies
pip-audit

# With fix suggestions
pip-audit --fix --dry-run
```

### Recommended CI/CD Integration

```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run Bandit
        run: bandit -r integradio/ -ll

      - name: Run pip-audit
        run: pip-audit
```

## Reporting Vulnerabilities

If you discover a security vulnerability, please:

1. **Do NOT** open a public GitHub issue
2. Email the maintainers directly (see pyproject.toml for contacts)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We aim to respond within 48 hours and release patches within 7 days for critical issues.

## Security Updates

Subscribe to security advisories:
- Watch this repository for releases
- Check [Gradio Security Advisories](https://github.com/gradio-app/gradio/security/advisories)
- Monitor [Python Security](https://python-security.readthedocs.io/)

## References

- [Trail of Bits Gradio 5 Audit](https://blog.trailofbits.com/2024/10/10/auditing-gradio-5-hugging-faces-ml-gui-framework/)
- [OWASP Security Headers](https://owasp.org/www-project-secure-headers/)
- [Python Security Best Practices](https://snyk.io/blog/python-security-best-practices-cheat-sheet/)
- [Gradio Security Blog](https://huggingface.co/blog/gradio-5-security)
