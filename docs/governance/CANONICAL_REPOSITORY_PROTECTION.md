# CANONICAL REPOSITORY PROTECTION

- Baseline: `CANONICAL_REPOSITORY_PROTECTION_BASELINE_V1`
- Scope: `James3014/nexus-core`, `James3014/nexus-learning`, `James3014/nexus-open-swe-runtime`
- Target Branch: `refs/heads/main`
- Mechanism: GitHub Repository Ruleset (ID: `22367922`, Name: `"Canonical Main Protection v1"`)
- Enforcement: `active`

## Policy Baseline Specification

| Rule | Specification | Configuration |
|---|---|---|
| Target | `refs/heads/main` | `include: ["~DEFAULT_BRANCH", "refs/heads/main"]` |
| Direct Push | Blocked | Enforcement active |
| Pull Request | Required | `required_approving_review_count: 0`<br>`require_last_push_approval: false`<br>`dismiss_stale_reviews_on_push: false`<br>`require_code_owner_review: false` |
| Conversations | Required | `required_review_thread_resolution: true` |
| Force Push | Blocked | `non_fast_forward` enabled |
| Branch Deletion | Blocked | `deletion` enabled |
| Status Checks | Strict | `strict_required_status_checks_policy: true`<br>`required_status_checks: [{"context": "test"}]` |
| Bypass Actors | None | Empty bypass actors list |

## Repository Required Checks

- Repository: `James3014/nexus-open-swe-runtime`
- Required Status Check Context:
  - `test` (GitHub Actions workflow `ci.yml` job `test`)

## Server-Side Verification

Ruleset verified via GitHub REST API:
```bash
gh api repos/James3014/nexus-open-swe-runtime/rulesets/22367922
gh api repos/James3014/nexus-open-swe-runtime/branches/main --jq '{protected: .protected}'
```
Expected output:
```json
{
  "protected": true
}
```
