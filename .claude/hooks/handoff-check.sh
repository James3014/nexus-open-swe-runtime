#!/usr/bin/env bash
# Stop: if this session changed code but not HANDOFF.md (or HANDOFF.md is over
# the size cap), block once and ask for an update + read-back. Never blocks twice in a row.
set -u
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$root" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0
input="$(cat)"
printf '%s' "$input" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true' && exit 0
sid="$(printf '%s' "$input" | sed -n 's/.*"session_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
start="$(git rev-parse --git-dir)/handoff-hook/${sid:-default}"
[ -f "$start" ] || exit 0
. "$root/.claude/hooks/handoff-lib.sh"
now="$(fingerprint)"
code_start="$(sed -n 1p "$start")"; handoff_start="$(sed -n 2p "$start")"
code_now="$(printf '%s\n' "$now" | sed -n 1p)"; handoff_now="$(printf '%s\n' "$now" | sed -n 2p)"

if [ "$code_now" != "$code_start" ] && [ "$handoff_now" = "$handoff_start" ]; then
  echo "[handoff] 本次 session 有程式碼或 commit 變動，但 HANDOFF.md 尚未更新。請依 AGENTS.md「Handoff」規則更新（含日期、作者、目前 HEAD），寫完後重新讀取檔案確認內容確實寫入。若這輪變動不影響進度，在 HANDOFF.md 更新日期與 HEAD 即可。" >&2
  exit 2
fi
if [ -f HANDOFF.md ]; then
  lines=$(wc -l < HANDOFF.md | tr -d ' ')
  if [ "$lines" -gt 150 ]; then
    echo "[handoff] HANDOFF.md 有 $lines 行，超過 150 行上限。請把已完成或過期的內容搬到 docs/archive/HANDOFF-YYYY-MM.md，然後讀回確認。" >&2
    exit 2
  fi
fi
exit 0
