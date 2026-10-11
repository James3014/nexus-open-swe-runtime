# Shared by handoff-inject.sh / handoff-check.sh.
# Line 1: fingerprint of HEAD + all non-handoff changes. Line 2: HANDOFF.md content hash.
fingerprint() {
  {
    git rev-parse HEAD 2>/dev/null
    git diff HEAD -- . ':(exclude)HANDOFF.md' ':(exclude)docs/archive' 2>/dev/null
    git ls-files --others --exclude-standard -- . ':(exclude)HANDOFF.md' ':(exclude)docs/archive' 2>/dev/null
  } | git hash-object --stdin
  if [ -f HANDOFF.md ]; then git hash-object HANDOFF.md; else echo missing; fi
}
