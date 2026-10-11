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

## Handoff

`HANDOFF.md` at the repository root is the shared progress log for every agent (Claude, Codex, others) and every device. It records state only; it grants no authority and never overrides this file, Issues, or Task Cards.

- **Read first.** At session start read `HANDOFF.md`. If its recorded HEAD is behind, run `git log <recorded-HEAD>..HEAD` and correct stale entries before acting on them.
- **Update at milestones.** Update it when a meaningful step finishes or before the session ends — not every turn. Overwrite stale state; do not append history.
- **Format.** The header line holds date, author agent, and current short HEAD. `▶ 下一步` has exactly one item. Every todo carries an Issue number (`#N`); items without one go under `待確認`. Record why, blockers, and next — not changed-file lists (git already has those).
- **Size cap: 150 lines.** Move finished or expired content to `docs/archive/HANDOFF-YYYY-MM.md`.
- **Read back.** After writing, re-read `HANDOFF.md` to confirm the change actually landed.
- **No secrets.** Never put tokens, passwords, credentials, or internal hostnames in it.
