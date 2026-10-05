# Open SWE Runtime Agent Guidelines

This repository owns execution runtime only.

It may implement:
- semantic execution
- diagnosis
- bounded repair
- model transport
- tool execution inside authorized scope

It does NOT own:
- Nexus Completion truth
- Evidence Trust truth
- CapabilityPlanner authority
- Workforce admission
- Candidate acceptance
- merge/release authority
- production claim authority

Ordinary bounded engineering does not require Task Cards.

Escalate only on:
- public execution contract break
- security/isolation weakening
- authority expansion
- destructive external effect
- release/production boundary


## Nexus Core issue-bound completion evidence

- This repository is enrolled in the standalone `nexus-certify` Golden Path through `.nexus-core/config.toml`.
- For mutation work tracked by a repository-local GitHub Issue, run `nexus-certify issue-init --issue <N>` before relying on Issue-bound completion evidence, and run `nexus-certify issue-check --issue <N>` before claiming engineering completion.
- This binding is Evidence Trust + Completion only. It does not select the execution lane, route, worker/model, Candidate acceptance, merge, release, deployment, or production authority.
- DIRECT work remains transport-neutral. A Core mutation session is not required solely because repository files are being changed.
