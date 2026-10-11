# HANDOFF

更新：2026-10-11 · by Claude · HEAD `546a59e`
<!-- 規則見 AGENTS.md「Handoff」。上限 150 行；過期內容移到 docs/archive/HANDOFF-YYYY-MM.md。 -->

## ▶ 下一步（只放一件，必須帶 #Issue）
#104 把寫死在七處的 Nexus-new consumer SHA `2999d62e` 收斂成單一來源並加一致性檢查。依據：PR #116 已涵蓋最新的 invariant 缺口 #112/#113，剩下未處理項目中 #104 是最高優先（P1）；main 上 `.nexus-core/config.toml`、`.github/workflows/ci.yml`、`tests/test_cross_repo_acceptance.py` 仍各自寫死，下次 re-pin 必然漏改。

## 進行中
- PR #116（Refs #112、#113）repair 寫入綁定 effect authorization 的 write_paths，並拒絕 symlink 路徑元件：mergeable=clean，test 與 Nexus Core issue run/completion 最新一輪皆綠（Core issue 綁 #112）。等 owner 審查合併；合併後跑 `nexus-certify issue-check` 再關閉 #112/#113。
- PR #76（draft，對應 #83）Wave 1C recovery/resume oracle 補強：最後活動 2026-10-06，Nexus Core issue completion 失敗、部分 job cancelled，需 rebase 到目前 main 後重跑。

## 待確認（沒有 Issue 號）
- 無

## 卡住 / 注意
- #105 [P2] cli.py 過大、死模組、未宣告依賴，尚未開工。
- #73 為 `[HYBRID_CANARY]` 研究探針（no implementation），不是工程待辦；`NEXUS-HYBRID-REPLICATION-*` 留言屬 RESEARCH_OBSERVATION_ONLY。
- 本 repo 只擁有 execution runtime；修 #112 類問題時不得放寬 isolation 或擴大 authority（見 AGENTS.md escalate 條件）。
