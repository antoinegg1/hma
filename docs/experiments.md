# Experiment and analysis map

All commands below use `hma-repro`. The source of truth is
[`experiments.json`](../src/hma/repro/assets/experiments.json). Defaults are a
**new-run protocol**, not a replay of heterogeneous historical cohorts. Six HMA
pairs use the manuscript's stated three-repeat protocol. NTA/ablations use one
repeat; one repeat for each NTA configuration is an explicit fresh-run choice.
Every native task has 21,600 seconds; HMA exploration ends at 20,700, then reserves
900 seconds for final review. The paper specifies that reserve; this implementation
limits the model call to 600 seconds and leaves time for shutdown/finalization.
Appendices F.1/F.2 report 21,700-second terminal windows in the historical reverse
order and cap-1/3 archives. Fresh runs enforce the nominal 21,600-second budget.

Native and external 16-task runs share the Appendix C.1 resource defaults:
one NVIDIA A10, 30 vCPUs, and 220 GiB RAM per task. Table 5 lists other papers'
original baseline hardware; those entries do not configure our reruns. The
64 GiB shared-memory and separate 2-CPU/16-GiB evaluator limits are implementation
settings not specified in the manuscript. Actual hardware is recorded and frozen
for resume; report runs with a changed GPU expectation as hardware variants.

The [75-node Swarm workflow](swarm.md) partitions this same matrix by benchmark
task. Each node executes all selected configurations/repeats for its assigned
task sequentially. The frozen cluster identity binds every shard to one planned
cohort; collection restores the complete plan and preserves per-node environments,
failures and denominators. It does not merge independently sampled campaigns.

| Selection | Configurations | Tasks × repeats | New task-runs |
| --- | --- | --- | ---: |
| `--suite goal` | Opus, GPT, GLM, DS4, DS4.1, Kimi | 6 × 75 × 3 | 1,350 |
| `--suite hma` | Opus→GPT, GPT→Opus, GPT→DS4.1, GPT→Kimi, GPT→GLM, DS4.1→Kimi | 6 × 75 × 3 | 1,350 |
| `--suite nta` | GPT→Kimi, Kimi→GPT, GLM→DS4, DS4.1→GLM, Opus→GPT | 5 × 75 × 1 | 375 |
| `--suite ablation` | GPT→Kimi cap 1/3; Kimi→GPT cap 5 | 3 × 75 × 1 | 225 |
| `--suite cases` | GLM→Kimi cap 5, Pawpularity | 1 × 1 × 1 | 1 |
| `--suite harness16` | ML-Master 2.0, MLEvolve no-prior, ScienceFlow × DS4/DS4.1 | 6 × 16 × 1 | 96 |
| `--suite paper` | Union of the above | | **3,397** |

Model IDs are `claude-opus-5`, `gpt-5.6-sol`, `glm-5.3`,
`deepseek-v4-flash`, `deepseek-v4.1-flash`, and `kimi-k3`; native reasoning effort
is `max`. Codex/Claude Code/Kimi Code/DeepSeek Harness are pinned in
`sources.lock.json` and Docker/runtime dependency locks.

Goal returns after its single native goal session; HMA/NTA use non-goal sessions.
NTA starts a fresh session after natural termination and keeps the latest accepted
candidate at its endpoint, with no HMA review reserve. This explicitly freezes the
fresh-run NTA endpoint policy. If natural completion coincides with reaching the cap, natural completion takes
precedence (Appendix A). The actor watchdog uses a distinct quota-exit code so
forced cap closures remain distinguishable from natural completion. This corrects
the old controller's cap-first polling label. No cap is communicated to the model. All arms use
the same task description/public data and blind submission service.

## External comparison

ML-Master and ScienceFlow receive 86,400 research seconds; MLEvolve receives
43,200. ML-Master/MLEvolve have up to 900 postprocessing seconds, ScienceFlow
zero. Total windows are 88,200 / 44,280 / 88,200 seconds respectively, including
startup, finalization and slack. Table 1 supplies the 24/12/24-hour research
budgets; postprocessing and total windows are adapter implementation settings.
The accepted candidate is selected by the frozen
upstream export rule (`unique` or `last_final`); if missing, use the most recently
modified eligible `runs/**/submission.csv` before the deadline. The fallback is
score-independent and tries exactly one candidate. No accepted candidate is a
counted miss. A source/validator preflight failure is a failure, not success.

No original cold-start competition prior is injected. The MLEvolve memory encoder
and audit of core search mechanisms remain enabled. Generic pretrained weights
used during learning are allowed. The selected 16 tasks are in
[`tasks-16.json`](../src/hma/repro/assets/tasks-16.json). Native DS4/DS4.1 references
reuse the corresponding 16 tasks from all three goal repeats; do not run an extra
cohort and silently pick the better result.

## Paper output → computation → generated files

Run `hma-repro report --run-root runs/paper --output outputs/rerun` once the
campaign is complete. Tables are CSV; figures are PDF/SVG/PNG. Raw normalized
records and derived numerical tables are retained alongside the plots.

| Paper item | Required new evidence / calculation | Output |
| --- | --- | --- |
| Fig. 1 | Method schematic, no separate experiment | Protocol in README/AGENTS |
| Table 1 | Six external arms + native DS4/DS4.1 16-task subset, means/SE and differences | `table01-harness.csv` |
| Table 2 | Goal/HMA final candidates; fixed 22/38/15/75 task denominators; repeat means/SE, pair-reference error propagation | `table02-main.csv` |
| Tables 3/4 | Configuration and exact native model/harness mapping | `table03-configurations.csv`, `table04-models.csv` |
| Table 5 | Original papers' configuration metadata only; no new run or historical scores | `table05-published-configurations.csv` |
| Fig. 2 / Table 6 | All goal repeats; response-weighted output tokens, trailing 30 minutes at one-minute checkpoints; early [0,2h), late [3h,5h) | `fig02.*`, `table06-early-late.csv` |
| Fig. 3 / Appendix B | Repeat 0 of six goal + five NTA; observed prefix gain and empirical KL lower bounds for k=1…50 | `fig03.*`, `prefix-retention.csv` |
| Fig. 4 / Appendix E | Latest accepted medal trajectory; review-returned endpoint; raw response counts including review; task-level comparisons | `fig04a.*`, `fig04b.*`, `trajectories.csv`, `matched-curves.csv`, `matched-early-late.csv`, `task-comparisons.csv` |
| Table 7 | All external 16-task outcomes and native repeat counts | `table07-harness-tasks.csv`, `runs.csv` |
| Fig. 6 / Tables 8/9 | GPT/Kimi starting order and cap 1/3/5; repeat 0; partner first response and right-censoring at six hours | `fig06a.*`, `fig06b.*`, `table08-participation.csv`, `table09-scheduling.csv` |
| Table 10 | Repeat 0 of six main HMA arms (450 planned tasks); initial/later closures; zero-submission natural exits; natural rate excludes deadline-censored options | `table10-option-closures.csv` |
| Table 11 | Same exploration histories: historical any-medal, before review, returned candidate, gains/losses/missed, retained fraction | `table11-review.csv` |
| Fig. 5 | `hma-gpt-opus`, INGV (`mbh_07`), repeat 0; per-experiment response intervals | `fig05.*`, `fig05-scores.csv`, `fig05-experiment-responses.csv` |
| Fig. 7 | `hma-gpt-ds41`, QUEST (`mbm_10`), repeat 0 | `fig07.*`, `fig07-scores.csv`, `fig07-windows.csv` |
| Fig. 8 | `case-glm-kimi`, Pawpularity (`mbm_27`), repeat 0 | `fig08.*`, `fig08-scores.csv`, `fig08-windows.csv` |
| Fig. 9 | `hma-ds41-kimi`, Smartphone (`mbh_12`), repeat 0 | `fig09.*`, `fig09-scores.csv`, `fig09-windows.csv` |

`table02-main.csv` also includes the other selected arms for convenient inspection;
filter its `experiment` column to the paper's main groups. It does not import the
three published external 75-task baseline scores. Tables 1/7 are our own new
16-task reruns, distinct from those published comparison rows.

## Analysis definitions and limits

- Report metrics are percentages. A task's quality is `1 - (rank - 1) / N`,
  where a score below all leaderboard entries has rank `N + 1` and quality zero.
  Failure/no-submission has zero task metrics. A failure with an earlier valid
  standing candidate retains that candidate but still appears as failed coverage.
- Calculate task-suite metrics per repeat, then mean and sample standard error
  across repeats. SE for a single repeat is blank. Pair-reference SE is
  `sqrt(SE_A² + SE_B²) / 2`. Do not round before computing differences.
- Prefix analysis tracks improvements in the running best quality within each
  option relative to its entry best. Sum option gains within a task before taking
  prefix/full ratios; exclude zero-total-gain tasks from that ratio, report their
  defined count, then average tasks. Horizontal fraction averages `min(k,l)/l`
  over nonempty options. Configuration means are equally weighted. KL bounds
  use M=1100 simultaneous comparisons and delta=0.05. These are offline empirical
  quantities, not prospective guarantees from an agent's observed validation.
- Matched Fig. 4 weights: each HMA arm 1/6; goal GPT 5/12; Opus, DS4.1, Kimi
  each 1/6; GLM 1/12. Early/late ratios are computed from response-weighted
  configuration means before applying the same configuration weights.
- Native usage parses provider output tokens, not text length or input tokens.
  Claude message IDs and Codex cumulative snapshots are deduplicated; Kimi step
  UUIDs and DeepSeek turn/step IDs identify responses. Subagent/coordinator logs
  are excluded. Missing usage is exposed instead of estimated.
- Run timestamps are elapsed wall time from controller start within the declared
  execution window, including handoff overhead. Interrupted runs are never resumed
  inside the same task. Native event bounds provide diagnostic option time shares;
  the denominator remains the fixed task window, so shares need not sum to one.
- Case figures preserve all accepted exploration scores, including regressions,
  and exclude final review. They show the best of new constituent goal endpoints
  when available, plus gold (Smartphone: bronze) threshold. INGV uses a log axis
  with linear inset; others use disjoint first/last 20-response windows only when
  an option has at least 40 responses. Short options display their counts. Raw
  scores/windows accompany every case. New runs need not exhibit the manuscript's
  same option count, score improvements, or qualitative cooperation.
- There is no automatic reconstruction of the paper's historical leaderboard-top
  annotation or prose case interpretation. Inspect the new native trajectories
  and shared workspace to substantiate any new qualitative claim.
- `coverage.json` identifies selected vs full-paper plans and missing evidence.
  Strict reporting refuses incomplete evidence; `--allow-partial` labels plots.
  Empty figures are skipped. A report from a selected plan is not full coverage.
