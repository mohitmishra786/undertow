# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's security advisories:
https://github.com/mohitmishra786/undertow/security/advisories/new

Please do not open public issues for security problems. You can expect an
acknowledgment within a week. Once a fix ships, the advisory is published
with credit to the reporter unless you prefer otherwise.

## Supported versions

The project is pre-1.0; only the latest release on `main` receives fixes.

## What counts

Things worth reporting include memory-safety issues in the kernels or FFI
surface, path traversal or resource exhaustion through model files
(checkpoints are treated as untrusted input by design: headers are
size-validated before any allocation), and anything exploitable in the
HTTP server. Model *outputs* being wrong or unpleasant is a quality bug,
not a security one; open a regular issue for those.

## Automated posture

Every pull request runs CodeQL (Rust and workflow analysis), cargo-deny
(advisories, licenses, bans, sources), dependency review, and the full
cross-platform test suite. cargo-audit runs daily against the RUSTSEC
database, and OpenSSF Scorecard tracks repository hygiene. Miri checks the
kernel crate for undefined behavior on every change.
