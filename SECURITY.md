# Security Policy

CodeLeWM is a research artifact that parses and indexes untrusted source
code. The project treats secret leakage, checkpoint deserialization,
license contamination, and non-execution violations as first-class
correctness failures. The contracts that back this commitment are
documented in `docs/spec/06-security.md`.

## Supported Versions

While the project is pre-1.0, only the `main` branch receives security
fixes. Patched releases are tagged as `0.x.y`. Once a `1.x` line is
released, security fixes will be backported to the most recent `1.x`
release.

| Branch / version line | Receives security fixes |
| --------------------- | ----------------------- |
| `main`                | Yes                     |
| Pre-1.0 tagged releases | No                    |

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for security-sensitive
reports.

- Use [GitHub Security Advisories](https://github.com/AbdelStark/CodeLeWM/security/advisories)
  on this repository to submit a private report. The maintainers are
  notified automatically.
- If you cannot use Security Advisories, email the repository owner
  (see `pyproject.toml` for the maintainer contact) and include
  `[CodeLeWM-SECURITY]` in the subject line.

Include in your report:

- a summary of the impact;
- the smallest reproducer you can share (an input file or command
  line is ideal);
- the affected commit, tag, or branch;
- any mitigations you already applied locally.

## Triage Timeline

The maintainers aim to:

- acknowledge a report within five business days;
- share a working remediation plan within fifteen business days;
- ship a fix or a documented workaround within thirty business days
  of triage for issues rated `high` or `critical`.

Coordinated disclosure is welcomed. Public disclosure dates are
agreed with the reporter before any release.

## In-Scope Vulnerabilities

Examples of issues that are in scope:

- secret-pattern leakage through manifests, reports, logs, or
  CLI output;
- bypass of `codelewm.security.require_trusted_checkpoint` or
  `codelewm.security.parse_python_source_text`;
- license-policy bypass that promotes a non-permissive row into a
  public artifact;
- non-execution policy violation (any code path that imports,
  evaluates, or runs untrusted project code);
- path traversal in manifest or artifact handling;
- deserialization of untrusted serialized objects from a checkpoint
  manifest mismatch.

## Out of Scope

The following are not security vulnerabilities for this project:

- denial of service through user-controlled large inputs without
  bypass of an explicit length limit;
- crashes from malformed configs that produce structured
  `*_error` reports as designed;
- non-public reports about hypothetical concerns without a
  reproducer.

## Acknowledgement

CodeLeWM credits reporters in `CHANGELOG.md` under the relevant
release once a fix is published, unless the reporter requests
anonymity.
