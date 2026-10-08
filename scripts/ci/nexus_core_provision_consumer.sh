#!/usr/bin/env bash
# Trusted Nexus Core helper: provision the pinned Nexus-new consumer checkout.
# Usage: nexus_core_provision_consumer.sh <40-hex sha>
# Prints the checkout path on stdout (all diagnostics go to stderr).
# Idempotent; fails closed unless HEAD == sha and the tree is clean.
set -euo pipefail

sha="${1:-}"
case "$sha" in
  *[!0-9a-f]*|"") echo "usage: $0 <40-hex sha>" >&2; exit 2 ;;
esac
[ "${#sha}" -eq 40 ] || { echo "usage: $0 <40-hex sha>" >&2; exit 2; }

root="${HOME:?HOME must be set}/nexus-new-consumer"
g() { git -c safe.directory='*' -C "$root" "$@"; }

if [ -d "$root/.git" ] && [ "$(g rev-parse HEAD 2>/dev/null || true)" = "$sha" ] \
   && [ -z "$(g status --short 2>/dev/null)" ]; then
  printf '%s\n' "$root"
  exit 0
fi

rm -rf "$root"
mkdir -p "$root"
git -c safe.directory='*' init -q "$root" >&2
g remote add origin https://github.com/James3014/Nexus-new.git
g fetch -q --no-tags --depth=1 origin "$sha" >&2
g checkout -q --detach FETCH_HEAD >&2
[ "$(g rev-parse HEAD)" = "$sha" ] || { echo "CONSUMER_REVISION_MISMATCH" >&2; exit 1; }
[ -z "$(g status --short)" ] || { echo "CONSUMER_DIRTY" >&2; exit 1; }
printf '%s\n' "$root"
