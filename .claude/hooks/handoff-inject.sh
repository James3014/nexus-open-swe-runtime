#!/usr/bin/env bash
# SessionStart: inject HANDOFF.md, flag drift since its recorded HEAD, and
# snapshot the starting state so handoff-check.sh can tell what this session changed.
set -u
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$root" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0
input="$(cat)"
sid="$(printf '%s' "$input" | sed -n 's/.*"session_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
state="$(git rev-parse --git-dir)/handoff-hook"
mkdir -p "$state"
. "$root/.claude/hooks/handoff-lib.sh"
fingerprint > "$state/${sid:-default}"

if [ ! -f HANDOFF.md ]; then
  echo "[handoff] HANDOFF.md 不存在。請依 AGENTS.md 的「Handoff」規則建立。"
  exit 0
fi
echo "===== HANDOFF.md（專案交接檔，依 AGENTS.md「Handoff」規則維護）====="
cat HANDOFF.md
echo "===== end HANDOFF.md ====="
lines=$(wc -l < HANDOFF.md | tr -d ' ')
[ "$lines" -gt 150 ] && echo "[handoff] HANDOFF.md 有 $lines 行，超過 150 行上限，請把過期內容搬到 docs/archive/。"
rec="$(sed -n 's/.*HEAD[[:space:]]*`\{0,1\}\([0-9a-f]\{7,40\}\).*/\1/p' HANDOFF.md | head -1)"
if [ -n "$rec" ] && git cat-file -e "$rec^{commit}" 2>/dev/null; then
  n=$(git rev-list --count "$rec..HEAD" -- . ':(exclude)HANDOFF.md' ':(exclude)docs/archive')
  if [ "$n" -gt 0 ]; then
    echo "[handoff] 交接檔記錄於 $rec，之後還有 $n 個 commit，內容可能已過期："
    git log --oneline -n 15 "$rec..HEAD" -- . ':(exclude)HANDOFF.md' ':(exclude)docs/archive'
  fi
fi
exit 0
