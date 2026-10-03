# HMA 实验重跑库修改计划

本计划将现有 HMA supplementary package 扩展为可以从零重新运行论文自有实验的 GitHub 仓库。使用者提供 README 列明的模型和数据访问权限，在符合要求的机器上执行文档命令，即可完成安装、数据准备、实验、评分和新结果图表生成；维护代码的 agent 按根目录 AGENTS.md 实现并验证同一协议。

按用户最新范围，本项目只准备重新运行实验所需的代码与文档，不交付旧论文归档下载、历史结果重算或原文数字逐项匹配。本文件保留最初的分阶段修改计划；相应入口、配置、程序与文档现已在本地实现。实际文件名、命令和已验证范围以 README.md、docs/experiments.md 和 docs/verification.md 为准。Docker、真实数据/API smoke 与全量实验尚未执行。准备重跑库不等于已经执行昂贵的全量实验。

## 1 范围和完成标准

覆盖六模型 goal、六组主实验 HMA、五组 NTA、起始顺序与 cap 消融、三种外部 harness 的 16-task 对照、final review/option 统计及四个案例任务。

Table 2 从其他论文引用的三行结果只保留出处，并标明 published results；它们不是本项目重新执行的实验。不复现外部论文自己的整套 75-task 原始实验。Figure 1 为概念图，Appendix A 为理论证明，也不需要运行入口。

交付标准：

- 每个自有实验都有明确配置、运行命令、预算、评分规则和输出。
- 所有输入由安装、公开上游下载、数据准备或本次实验产生；不依赖历史 session、BeeGFS、私有镜像或未提供的本机文件。
- 原始事件、候选、最终结果和新图表形成完整链路；分析程序不读取或硬编码论文报告的实验结果。
- README 命令经过相应层级的验证，不需要编辑 Python 或阅读内部源码才能运行。
- 明确分别记录离线测试、真实端到端 smoke 和全量重跑的验证状态。未执行全量实验不冒充“已复现全部指标”，也不把旧归档缺失作为准备代码的阻碍。
- 新实验保留失败任务和完整分母，允许指标与论文有随机波动。

## 2 第一阶段 冻结新的运行协议与配置

拟新增 `repro/experiments.yaml`、`repro/sources.lock.json` 和配置 schema。

`experiments.yaml` 定义 workflow、模型顺序、task suite、重复次数、seed、预算、review、失败与恢复政策，以及对应图表。`sources.lock.json` 固定上游代码、harness/CLI、镜像、依赖和数据处理版本。每次新运行自动生成 resolved config 和 run manifest，无需提前取得旧 run 清单。

新的标准运行矩阵建议采用：

| 实验 | 配置与任务 | 重复次数 |
| --- | --- | --- |
| 单模型 goal | 六模型 × 75 tasks | 每模型 3 次 |
| 主实验 HMA | 六组模型顺序 × 75 tasks，k=5 | 每组 3 次，遵循论文声明的重复协议 |
| NTA prefix/顺序分析 | 五组模型顺序 × 75 tasks | 每组 1 次；可配置增加 |
| cap 消融 | GPT→Kimi，k=1 和 k=3，各75 tasks；k=5复用主实验 | 各1次 |
| 反向顺序消融 | Kimi→GPT，k=5，75 tasks | 1次 |
| 外部 harness 对照 | 三 harness × 两 DeepSeek 模型 ×16 tasks | 每格1次；native reference复用goal的三个repeat |
| 额外案例 | GLM→Kimi，Pawpularity | 1次；其他三个案例取新主实验预先指定的repeat |

这些是新实验的运行次数，不声称对应旧论文每个结果实际已有的 repeats。主表分析使用全部主实验 repeats；option/review 分析和图中单轮对照默认选择事先固定的 repeat 0，避免根据新分数挑选运行。配置与报告均显示所用 repeats。

需要在实现时明确的协议问题：

- Goal 按论文的一次 goal session 执行，不把多次新建 session 的循环当作同一基线。
- HMA 使用21,600秒总预算、900秒review reserve，探索在20,700秒结束；accepted submission计数触发cap。
- NTA 明确无cap、自然结束后交替、六小时预算及最终候选策略。review是否适用必须在NTA配置中明确，依据论文与原实现确定，不能无说明地继承HMA默认值。
- 新的cap/顺序消融统一采用声明的六小时有效预算；历史21,700秒terminal window仅作为已知历史差异说明。若需要额外收尾窗口，单独记录，禁止延长模型探索或候选接收预算。
- corrected answer keys属于评分输入的正确性问题，仍需确认修正方法并提供可重复的数据处理代码；这不要求找回历史模型轨迹。如未确认，不声称评分与论文完全等价。

验收：配置检查器能生成完整实验矩阵、task-run数量和预算估算，并检查所有实验的模型、任务、重复和分析入口均已定义。

## 3 第二阶段 安装 环境和数据

主要复用 `KaggleBench/src/flowbench/mle_native/` 的 catalog、prepare_data、prepare_worker、integrity、task_staging 和 grader_smoke。抽取必要依赖，不整体搬运集群运维代码。

拟新增或修改：

- `src/hma/benchmark/catalog.py`：75-task清单，22/38/15 splits、slug、metric direction。
- `src/hma/benchmark/prepare.py`：下载、数据准备、checksum、公共/私有分区、prompt生成。
- `src/hma/benchmark/preflight.py`：数据、评分器、镜像、权限、GPU及provider检查。
- `docker/`：从可获得且固定版本的基础镜像构建agent/evaluator，包含完整依赖。
- `configs/local.example.yaml`：集中配置数据根目录、输出目录、凭据环境变量名、设备和并发。
- `pyproject.toml`、依赖锁和CLI入口：统一安装与运行。

MLE-bench本地catalog锁定commit为 `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`，在此基础上显式记录必要的数据修正。现有部分agent镜像默认Python 3.11，HMA要求>=3.12，必须明确controller、hmz和训练环境使用的解释器。

支持逐task准备、续传和校验，不强制先下载全量数据。README写明Kaggle访问/竞赛规则接受步骤、磁盘需求，以及宿主机、actor、evaluator资源配额。版本化的运行配置不能静默换成其他模型；服务不可用时给出明确失败原因。

验收：干净环境可构建镜像，准备一个真实任务并完成grader smoke；所有75 tasks有明确准备路径和校验规则。全量运行前检查对应数据manifest。

## 4 第三阶段 三种workflow与campaign runner

### 4.1 Goal

参考 `KaggleBench/src/flowbench/flows/goal/{codex,claude,kimi,dsh}/run.py`，接入六种单模型的一次goal执行、原生response记录和最终候选处理。

现有 `flowverse/flows/goal` 循环启动新session，不能直接用作论文的单session基线。基础设施恢复与算法重启分别记录；不得隐式增加运行时间或根据分数重跑。

### 4.2 HMA

保留现有共享workspace、fresh HOME、accepted-submission cap、blind evaluator和final review机制。提供六组主配置：

- Opus→GPT；GPT→Opus。
- GPT→DeepSeek-V4.1-Flash；GPT→Kimi。
- GPT→GLM；DeepSeek-V4.1-Flash→Kimi。

补齐Kimi proxy注入、GLM的Claude Code配置、DeepSeek Harness配置和跨backend日志收集。当前 `session_evidence()` 仅识别Claude/Codex，应按实际backend扩展，正确诊断review无nomination的状态。

### 4.3 NTA

参考 `antoinegg1/flowverse/flows/flame_chase/` 和KaggleBench flame staging，实现显式 `termination: natural`：

- GPT→Kimi；Kimi→GPT。
- GLM→DeepSeek-V4-Flash。
- DeepSeek-V4.1-Flash→GLM。
- Opus→GPT。

当前cap字段要求>=1，evaluator用limit=0表示review-only；需新增明确的无cap语义，不能直接设置0或任意极大整数冒充NTA。

### 4.4 消融与案例

增加GPT→Kimi的k=1/3、Kimi→GPT的k=5和GLM→Kimi的Pawpularity配置。支持直接重跑单个案例任务；其他案例可复用本次主实验结果。Final review消融固定探索轨迹，比较review前后的选择，不把review时间重新分配给探索。

### 4.5 批量执行和恢复

新增 `src/hma/campaign/`，支持单机顺序运行、多GPU并发、任务筛选、预算估算、状态查询和campaign恢复；多机调度可选。

每个task-run记录配置、版本、尝试次数、active runtime、session/option边界、接受的候选、review和终止原因。完成的task-run不重复执行。

现有supervisor不支持旧root内直接恢复。campaign resume只调度尚未启动的任务，或按预先冻结的基础设施故障规则创建明确的新attempt；不清零已用预算、不隐式重复抽样，不按得分选择attempt。中断且不可恢复的运行要明确显示状态。

验收：每种workflow的session边界、终止、候选和预算可检查；相同配置能在新目录运行；恢复不会改变已经完成的正式结果。

## 5 第四阶段 三种外部harness的16-task对照

复用 `KaggleBench/experiments/harness-repro-16task-20260918/` 的adapter、source locks、Dockerfiles和grading机制，整合到 `baselines/harness16/`。现有三套as-run和三套fixed adapter已通过各自lock校验。

新运行采用明确锁定的修正版adapter，不要求重新部署历史缺陷；在来源说明中记录修正内容。保留upstream原样，适配逻辑独立存放。

需要补齐：

- EvoMaster、MLEvolve、ScienceFlow的上游下载、固定commit和文件校验。
- 三harness×两DeepSeek模型×16 tasks的完整96-cell配置；不能沿用只列8 tasks的部署示例。
- 冷启动知识关闭设置、单机执行入口、输入准备、候选导出及评分。
- research budget和总执行窗口的独立配置，明确24/12小时与约24.5/12.3小时的含义。
- 预先固定无final、无submission、磁盘耗尽、污染checkpoint和基础设施失败处理；新运行不能套用旧结果的逐task剔除名单。
- 每格都产生一条结果，即使失败；所有组分母固定为16。
- native对照从本次goal的对应16 tasks、三个repeat生成。

验收：新运行结果可生成Table 1/7结构的完整对照表，并保留失败与剔除原因；不要求新奖牌总数等于旧论文。

## 6 第五阶段 记录新实验并生成全部分析

拟新增 `src/hma/analysis/`，所有分析只读取本次运行自动导出的规范化记录：

- runs：task/config/repeat/attempt、版本、状态、预算和最终候选。
- responses：response ID、session/option、实际model、parent/main标记、输出token、wall/active timestamp、review标记。
- options：起止时间、accepted count、closure reason、实际actor。
- submissions：ID、artifact hash、作者、接收时间、有效性、离线score、medal、rank/leaderboard size。
- reviews：standing、nomination、returned和review outcome。

原生日志和候选保存在用户的run目录；提供本次运行的导出与校验工具即可，不建设旧论文归档下载系统。workspace/patch证据用于检查新案例中的实际复用行为。

| 对应输出 | 新运行上的计算口径 |
| --- | --- |
| Table 2 | 每repeat先计算suite指标，再求mean/SE；22/38/15/75分母；pair mean与误差传播；先算差值再舍入 |
| Fig 2、Table 6 | provider output tokens/response；1分钟网格、30分钟trailing window；0–2h和3–5h使用原始事件统计；处理空窗口、去重及累计usage |
| Fig 3、Appendix B | 实际option的prefix gain；task内聚合后配置等权；零增益任务规则；k=1..50，M=1100、delta=.05的KL下置信界；option总数由新实验决定 |
| Fig 4、Appendix E | latest accepted轨迹允许回退，终点采用review结果；HMA各1/6；goal权重GPT5/12、Opus/DS4.1/Kimi各1/6、GLM1/12 |
| Fig 6、Table 8/9 | 真实执行k=1/3/5；session时间份额、partner首响应与右删失；剔除辅助subagent；检查新运行实际起始模型与配置一致 |
| Table 10 | 六组主实验预先指定repeat的450 runs；初始/后续option及cap/natural/deadline分类；包含零提交自然退出，deadline不入自然退出率分母 |
| Table 11 | 同一探索轨迹的before/after/historical比较；review重复receipt去重；历史奖牌数和恢复数从新结果计算 |
| Fig 5、7–9 | 四个预先指定任务/run的全部accepted score，保留回退；INGV按实验响应区间；其他案例disjoint first/last20，少于40不构造重叠窗口；排除review |

具体案例为INGV GPT–Opus、QUEST GPT–DeepSeek、Pawpularity GLM–Kimi、Smartphone DeepSeek–Kimi。新运行可能不出现原文相同的协作过程、选项数量或收益，图表与说明必须如实反映本次运行。

旧 `plot_goal75_performance.py` 仅可参考结构，不能复用其中截图数字化的Lite轨迹。分析程序不得补入旧论文曲线、奖牌数或案例分数。

验收：对新结果先生成可核查CSV/JSON，再生成PDF/SVG/PNG和表格；记录task/repeat覆盖率。缺日志或未完成任务不能被静默删除；未完成suite的图表必须标注partial。完整实验报告要求输入完整。

## 7 第六阶段 README和AGENTS.md

使用标准名称 `AGENTS.md`；如需兼容 `agent.md`，只提供指向它的说明。根目录AGENTS.md面向维护本仓库的开发agent，不挂载到被测MLE agent workspace，避免污染实验prompt。

README按执行顺序提供：

1. 覆盖的自有实验，以及只引用不重跑的外部论文结果。
2. OS、GPU、磁盘、Python、Docker、数据与模型权限要求。
3. 固定版本安装、完整镜像构建和provider配置。
4. doctor、配置预检与任务/预算预览。
5. 一个真实任务的端到端smoke。
6. 全量数据准备、按组或全量实验启动。
7. 状态、恢复、失败处理、重新评分和报告生成。
8. 逐图/逐表的配置与输出对照、资源估算和已验证范围。

拟议命令接口：

```sh
python -m hma.repro.cli doctor --config configs/local.json
python -m hma.repro.cli plan --suite paper --config configs/local.json
python -m hma.repro.cli data prepare --task leaf-classification --config configs/local.json
python -m hma.repro.cli smoke --task leaf-classification --config configs/local.json
python -m hma.repro.cli data prepare --suite paper --config configs/local.json
python -m hma.repro.cli run --suite paper --config configs/local.json --run-root runs/paper
python -m hma.repro.cli status --run-root runs/paper
python -m hma.repro.cli run --suite paper --config configs/local.json --run-root runs/paper --resume
python -m hma.repro.cli grade --run-root runs/paper
python -m hma.repro.cli report --run-root runs/paper --output outputs/rerun
```

所有模型、任务、重复、预算和路径由配置管理。安装、镜像构建、凭据配置及按组运行的实际命令在实现后加入README。以上保留原拟议接口，实际实现使用 configs/local.json；请执行 README 中已更新的命令。

AGENTS.md规定：

- 代码来源、模块职责、修改前需要读取的实验schema和配置。
- 预算、模型映射、prompt、context reset、cap计数、final selection、blind scoring的不变条件。
- 不能用best-so-far替代返回候选，不能从分母删除失败任务。
- 代码、镜像、依赖和数据处理变化必须同步更新锁和文档。
- 运行记录schema以及分析程序的数据完整性要求。
- 协议测试、分析测试、Docker integration、真实smoke的具体命令和验收条件。
- 恢复与重跑规则，以及真实验证和全量运行的状态记录要求。

## 8 第七阶段 验证与交付

验证分层进行，重点是证明重跑链路可用，而非匹配旧指标。

1. **协议测试**：accepted/invalid submission、cap1/3/5、natural termination、NTA无cap、review fallback、候选作者、deadline、provider failure、fresh session及campaign恢复。
2. **分析测试**：用小型可手算fixture验证task/repeat对齐、token去重、active-time、latest选择、SE/权重、零增益prefix、review去重、固定分母和剔除处理；不要求旧论文归档。
3. **Docker和真实任务smoke**：真实public/private数据隔离、GPU、grader、四种native backend，以及三个external harness的启动、提交、评分和报告；记录哪些组合真正通过，不能以模拟替代实际通过声明。
4. **干净环境文档验收**：fresh clone按README完成安装→数据→一次真实run→评分→图表；完整suite通过计划生成、配置和依赖预检。全量长实验由使用者执行，代码交付不以先跑完所有长实验为前提。

没有凭据/GPU时执行无凭据测试并明确留下真实smoke的未验证状态。CI运行协议、分析和适用的Docker测试；真实provider/GPU验证单独记录，不在普通CI中自动消耗完整实验预算。

最终交付包括：完整代码、全部实验配置、依赖和来源锁、README、AGENTS.md、测试及验证记录。无须提供历史实验归档或旧论文结果文件。

## 9 实施顺序

按七个可独立审阅的变更推进：

1. 新运行协议、实验矩阵、配置schema和版本锁。
2. 环境、数据与真实评分闭环。
3. goal/NTA/HMA、消融/案例配置和campaign runner。
4. 三外部harness的16-task执行与评测。
5. 新实验日志导出、全部分析和图表程序。
6. README、AGENTS.md与统一可执行命令。
7. 干净环境和真实smoke验收，记录剩余未验证项。

README和AGENTS.md骨架从第一步开始维护，各阶段同步更新。可以立即着手配置、环境和代码整合；不等待历史归档。实现时仍需落实NTA的终端选择语义、数据修正方法和外部服务版本等真正影响新实验有效性的细节。
