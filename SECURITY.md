# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

This project is an educational ML demonstration kit that uses only synthetic data and no network connections. It does not handle real user data or credentials.

If you discover a security concern, please open a GitHub Issue labeled **security** or contact the maintainer directly via the email in the git commit history.

## Scope

- No secrets, API keys, tokens, or passwords are present in this codebase.
- All datasets are synthetically generated at runtime — no external data files are stored.
- The `.tracking/` directory and virtual environments are excluded from version control via `.gitignore`.

## Security Best Practices for Contributors

- Do not add API keys, tokens, or credentials to any file.
- Do not commit virtual environment directories.
- Use `pip install -r requirements.txt` to reproduce the clean dependency set.
