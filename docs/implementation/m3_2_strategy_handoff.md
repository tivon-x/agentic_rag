# M3.2 固定检索策略收口：执行交接

## 1. Task Goal

在不改写 M3/M3.1 历史失败结论的前提下，将 M3 的最终目标从“找到全面胜过
B1 的固定 pipeline”调整为“冻结一个可复现、可解释、可供 M4 对照的 fixed
retrieval baseline”。

本任务只完成 M3.2 策略收口。完成后允许 M4 进入设计与实施，但本任务不得
自动实现 M4。

## 2. Problem Definition

M3.1 已完成 24 个新候选的冻结开发实验，但没有候选通过原 promotion gate：

- 最均衡的轻量候选 `r1_01_quote_mixed_minmax`：
  - Recall@10：`0.729167`
  - MRR@10：`0.435499`
  - nDCG@10：`0.483281`
  - 相对 B1：`24 win / 16 tie / 8 loss`
  - p95/B1：`0.680618`
- 最高质量候选 `r3_07_title_section_quote`：
  - Recall@10：`0.739583`
  - MRR@10：`0.477885`
  - nDCG@10：`0.511840`
  - 相对 B1：`27 win / 14 tie / 7 loss`
  - p95/B1：`9.677832`

实验说明：

1. Dense + BM25 的混合召回有效。
2. `mixed_v1` + min-max fusion 是当前质量、延迟和复杂度最均衡的方向。
3. TinyBERT 硬覆盖融合排序会把部分已召回 gold 排出 Top 10。
4. MiniLM blended rerank 能提高总体排序指标，但 CPU 延迟不可接受。
5. 完整 metadata prefix、RRF 替换 min-max、通道权重偏置和统一 reranker
   都没有解决逐题尾部退化。
6. 剩余困难题更适合由 M4 的证据充分性判断和有界补检处理，而不是继续增加
   M3 全局候选。

因此，M3.2 不再启动新一轮参数搜索。它只冻结一个轻量策略，运行尚未打开的
holdout 一次，确定 M4 的 fixed baseline。

## 3. Scope

### 3.1 In scope

- 保留 M3、M3.1 的历史结论和产物，不覆盖、不改名为通过。
- 在路线文档中增加 M3.2，并修订 M4 进入条件。
- 将 M3.1 的 `r1_01_quote_mixed_minmax` 冻结为唯一 M3.2 候选：
  `v2_fixed_hybrid`。
- 使用现有 B1 作为唯一正式对照。
- 在任何 holdout 运行前冻结配置、代码状态和 SHA-256。
- 对 B1 和 `v2_fixed_hybrid` 先运行新 48 题 holdout，再复跑旧 48 题。
- 不根据 holdout 结果继续调参。
- 根据冻结门槛决定默认 fixed pipeline。
- 无论候选是否替换 B1，都冻结一个明确的 M4 fixed baseline。
- 输出逐题结果、失败分类、延迟、配置、manifest、默认选择和 M4 baseline
  contract。
- 生成独立 commit，不 push。

### 3.2 Out of scope

- 不新增第 25 个或任何新的 M3.1 候选。
- 不再搜索 metadata 字段、reranker model、rank blend 或 dense/sparse 权重。
- 不把 M3.1 改写为通过。
- 不使用 holdout 调整 gold、阈值、候选配置或 query。
- 不实现 adaptive graph、run worker、checkpoint、SSE、claim validation。
- 不实现 M4 UI。
- 不引入新 embedding model、生成模型、重型本地 reranker 或外部服务。
- 不修改 parser、chunker、passage ID 或冻结 corpus。
- 不自动 push。

## 4. Current Context

新 session 开始后必须读取：

1. `AGENTS.md`
2. `docs/research/v2_upgrade_plan.md` 的 M3.1、M4 和评测门槛
3. `docs/implementation/m3_acceptance.md`
4. `docs/implementation/m3_1_acceptance.md`
5. `docs/implementation/m3_1_per_question.md`
6. `docs/implementation/m3_1_experiment_handoff.md`
7. 本文件

当前事实：

- 分支：`codex/v2-core`
- M3.1 实现 commit：
  `d4251726509fb33b22a90184a86d3b8b4f62ab2e`
- 当前默认：`v1_flat_rerank`（B1）
- M3.1：`core_passed=false`
- M3.1：`m4_entry_ready=false`
- M3.1 formal holdout runs：`0`
- M3.1 metadata prefix leaks：`0`
- M3.1 active index changed：`false`
- parser artifact SHA-256：
  `98e8adf680c578c21d2fffe5b97f3f85d24b768b827fe81aa8ddfc280af242d9`
- old dev SHA-256：
  `e1da7d23d352cd17a1601f56280a5c9820ff81002a36dc5ad786cb3a8f90c936`
- frozen holdout SHA-256：
  `47e2a70de438468150e22ca07c5f57aaf8630d601b3b645ffcf7a2d3f0dfea78`
- 冻结 corpus：25 篇论文
- embedding：
  - provider：OpenAI-compatible
  - model：`qwen3.7-text-embedding`
  - dimension：1024
  - input mode：raw
  - max input chars：6000

开始前必须执行：

```text
git status --short
git rev-parse HEAD
git branch --show-current
```

现有未跟踪 handoff 文件属于用户输入，不得覆盖或误提交。发现其他 dirty 文件时
必须先判断所有权，不得 reset、checkout 或删除用户改动。

## 5. Required Design Amendment

实现前先修改 `docs/research/v2_upgrade_plan.md`，增加 M3.2：

- M3.1 的原 promotion gate 和失败结论保持不变。
- M3.2 是“固定策略收口”，不是“M3.1 第二次尝试”。
- `v2_fixed_hybrid` 是新的策略候选，不得叫已通过的 B2.1。
- M3.2 候选通过时，默认 fixed pipeline 切换为 `v2_fixed_hybrid`。
- M3.2 候选失败时，默认继续保持 B1。
- 两种结果都必须生成唯一、冻结、可复现的 M4 fixed baseline contract。
- M4 进入条件改为：
  1. M3.2 策略收口流程完成；
  2. fixed baseline 已冻结并可复现；
  3. holdout 只运行一次且结果完整保留；
  4. 用户明确批准 Enhanced。
- `m3_1_core_passed` 与 `m4_entry_ready` 不再是同一个布尔含义。
- 不得把 M3.2 的“允许进入 M4”表述成 M3.1 通过。

本次用户已明确认可“M3 的真实目标是确立好的检索策略，M4 可以继续”，但本
任务仍只修改进入规则和 baseline contract，不实施 M4。

## 6. Frozen Candidate

唯一候选正式命名：

```text
experiment key: s1
pipeline name: v2_fixed_hybrid
source experiment: r1_01_quote_mixed_minmax
```

冻结配置：

```yaml
name: v2_fixed_hybrid
use_metadata_prefix: false
tokenizer: mixed_v1
use_sparse: true
use_dense: true
fusion_method: minmax
use_rerank: false
neighbor_window: 0
sparse_top_k: 40
dense_top_k: 40
rrf_k: 60
rerank_top_n: 30
final_top_k: 8
max_context_passages: 12
context_token_budget: 8000
dense_use_metadata_prefix: false
sparse_use_metadata_prefix: false
dense_rrf_weight: 1.0
sparse_rrf_weight: 1.0
boost_policy: current
```

虽然 `use_rerank=false`，所有无效 rerank 字段仍必须由统一配置 schema 给出稳定
默认值并进入 config hash；不得形成第二套 pipeline 实现。

选择该候选的原因：

- 比 B1 更高的 Recall@10、MRR@10、nDCG@10。
- p95 明显低于 B1。
- 不依赖重型 CPU reranker。
- 架构简单，适合作为 M4 每轮检索的基础工具。
- 它的 8 条退化可以直接转化为 M4 困难查询和证据不足案例。

## 7. Evaluation Protocol

### 7.1 Pre-run freeze

任何正式 holdout 运行前必须：

1. 完成所有 runner、配置、测试和报告代码。
2. 验证 parser artifact、old dev、holdout SHA。
3. 验证 holdout 仍没有正式质量产物。
4. 记录：
   - base commit
   - dirty 状态
   - working-tree patch
   - patch SHA-256
   - config SHA-256
   - parser/dataset SHA-256
   - embedding contract
   - B1/S1 pipeline config hash
5. 快照 active production index。
6. 冻结后不得再修改会影响评测的代码或配置。

### 7.2 External service authorization

新 session 不得继承旧 session 的外部调用授权。

在调用 embedding API 或下载任何模型前，必须向用户明确说明：

- 服务地址。
- embedding model 和 dimension。
- 将外发 corpus retrieval text、holdout query、old dev query 或 answer smoke。
- 是否存在模型下载。

获得明确授权后才可运行。M3.2 的候选不使用 reranker，因此正常情况下不需要
下载 FlashRank 模型。

### 7.3 Formal run order

正式评测只能运行一次，顺序固定：

1. new holdout：B1、S1。
2. old dev regression：B1、S1。

每个 dataset：

- 1 次 warmup，不计入正式延迟。
- 5 个正式轮次。
- 每轮使用固定 seed 打乱问题顺序。
- B1/S1 使用相同问题顺序。
- 分别记录 query embedding、recall、fusion、rerank、expansion、packing、
  end-to-end 延迟。
- 记录逐题 rank、Top 10 passage IDs、context passage IDs 和 stage trace。

run directory 必须使用独占锁。正式 report 已存在时拒绝再次运行。

## 8. Frozen Strategy Gate

M3.2 使用新的“策略候选非劣且更高效”门槛。它是事先声明的新目标，不回写
M3.1 promotion gate。

S1 必须在 new holdout 和 old dev 上分别满足：

- Recall@10 不低于 B1。
- MRR@10 不低于 B1。
- nDCG@10 不低于 B1。
- 相对 B1 至少 10 win。
- 相对 B1最多 8 loss。
- 四个 retrieval 子集各自的 Recall@10 命中数下降不超过 1 条。
- p95 end-to-end retrieval latency 不高于 B1。
- Context Passage Recall 不低于 B1。
- metadata prefix leak 为 0。
- answer smoke 的引用、页码和 context packing 不低于 B1。
- 所有改善和退化必须跨多个 fold 记录，不能只报告 aggregate。

不得使用统一综合分。

### 8.1 Candidate pass

两个 dataset 都通过：

- `strategy_candidate_passed=true`
- 默认 fixed pipeline 切换为 `v2_fixed_hybrid`
- `m4_fixed_baseline=v2_fixed_hybrid`
- active index 只能在全部 gate 通过后切换

### 8.2 Candidate fail

任一 dataset 失败：

- `strategy_candidate_passed=false`
- 默认保持 `v1_flat_rerank`
- `m4_fixed_baseline=v1_flat_rerank`
- active index 不变
- 不修改 S1、不重跑 holdout

### 8.3 Milestone closure

只要以下流程条件全部成立，M3.2 即完成：

- 候选和门槛在 holdout 前冻结。
- holdout 只运行一次。
- 结果、失败原因和逐题差异完整保留。
- 默认策略按上述规则确定。
- 唯一 M4 fixed baseline contract 已生成。
- 测试和静态检查通过。

此时：

- `m3_strategy_closed=true`
- `m4_entry_ready=true`
- `m3_1_core_passed` 仍保持原值 `false`

这三个字段必须分开，禁止用一个 `core_passed` 混淆历史结论、策略收口和 M4
进入资格。

## 9. M4 Bridge Output

必须生成机器可读的：

```text
artifacts/evals/v2_m3_2/m4_fixed_baseline.json
```

至少包含：

- selected pipeline name
- pipeline config 和 config hash
- index contract 和 manifest SHA
- parser artifact SHA
- embedding contract
- old dev/holdout SHA
- code commit/patch SHA
- active index version
- quality metrics
- latency metrics
- candidate pass/fail
- selection reason

还必须在验收文档中把 old dev 的 S1 退化题分类为 M4 输入，至少覆盖：

- dense/sparse Top 结果不一致。
- Top 分数差距小。
- 表格或数字定位。
- 缩写。
- 跨章节。
- 跨论文。
- 多约束问题。
- 首轮 context 未覆盖全部 requirement。

这里只输出案例和可观测信号，不实现 router threshold。M4 必须在自己的冻结
route/answer 数据上验证这些信号，不能直接把 gold 标签写成运行时规则。

## 10. Implementation Requirements

优先复用 M3.1 代码：

- `indexing/retrieval_pipeline.py`
- `evals/v2_runner.py`
- `evals/m3_1_runner.py`
- `evals/m3_1_experiments.py`
- `evals/run_lock.py`
- `evals/build_report.py`

要求：

- 不复制一套完整 runner。
- 若 M3.1 文件名阻碍复用，应把通用 final evaluation、freeze 和 gate 逻辑
  重构为中性模块，再让 M3.1/M3.2 调用。
- pipeline config 的每个字段继续进入 config hash、contract、manifest 和
  trace。
- 内容表示相同的 index 必须复用内容寻址产物。
- index contract 不兼容立即失败。
- embedding 只允许对同一内容批次的明确 transient backend error 重试。
- 正式实验不得静默降级模型、维度、tokenizer、input mode 或 pipeline。
- quote/context 只能使用 source-faithful `quote_text`，不得泄漏 metadata
  prefix。
- active production index 在 gate 完成前不得修改。

预计涉及 8 至 12 个文件。如果超过 12 个文件，先检查是否在提前实现 M4。

## 11. Tests

至少覆盖：

- `v2_fixed_hybrid` 配置与 `r1_01_quote_mixed_minmax` 完全一致。
- S1 是唯一 candidate。
- holdout SHA、old dev SHA、parser SHA 校验。
- pre-run freeze 后 code/config drift 立即失败。
- formal holdout report 已存在时拒绝重跑。
- 运行顺序固定为 holdout 后 old dev。
- warmup 不进入正式延迟样本。
- B1/S1 使用相同 shuffled order。
- 新策略 gate 的每个子条件。
- candidate pass 选择 S1。
- candidate fail 保持 B1。
- 两种结果都生成唯一 M4 baseline contract。
- `m3_1_core_passed`、`m3_strategy_closed`、`m4_entry_ready` 相互独立。
- metadata prefix leak 为 0。
- quote/context 不含 prefix。
- active pointer 在 gate 前不变。
- run lock 阻止并发写入。
- report 包含逐题 win/tie/loss、子集、stage timing 和失败原因。

## 12. Manual Checks

- 至少检查 12 个 B1/S1 rank 变化，必须包含 win、tie、loss。
- 表格、缩写、跨章节、跨论文各检查至少 3 题。
- 检查至少 5 条 S1 trace，确认没有 reranker 阶段参与最终排序。
- 检查引用和 answer preview 不显示 metadata prefix。
- 检查 holdout formal run count 恰好为 1。
- 检查 holdout 后没有配置、gold 或 threshold 修改。
- 检查 selected pipeline 与 M4 baseline contract 一致。
- 检查 active index 只在 S1 两个 dataset 全部通过后才可能改变。

## 13. Validation Commands

根据最终重构补充精确测试文件，但至少执行：

```text
uv run --extra dev python -m pytest \
  tests/test_bm25_index.py \
  tests/test_retriever.py \
  tests/test_retrieval_pipeline.py \
  tests/test_evals.py \
  tests/test_m3_1_experiments.py \
  tests/test_m3_2_strategy.py -q

uv run python -m evals.parser_eval \
  --dataset evals/datasets/parser_v2.json

uv run python -m evals.runner \
  --config evals/configs/v2_m3_2_strategy.yaml

uv run python -m evals.build_report \
  --runs artifacts/evals/v2_m3_2

uv run --extra dev ruff check indexing core evals tests
uv run --extra dev python -m pytest -q

npm --prefix web run lint
npm --prefix web run build
```

正式实验命令只能执行一次。若因外部服务失败中断，必须使用内容寻址 checkpoint
恢复，不得删除已完成结果后从头运行。

## 14. Deliverables

- `docs/research/v2_upgrade_plan.md` 的 M3.2/M4 进入条件修订。
- `v2_fixed_hybrid` pipeline registry 和 alias。
- `evals/configs/v2_m3_2_strategy.yaml`。
- M3.2 freeze manifest。
- new holdout 和 old dev 的正式 B1/S1 结果。
- `artifacts/evals/v2_m3_2/m4_fixed_baseline.json`。
- `artifacts/evals/v2_m3_2/core_report.json`。
- `artifacts/evals/v2_m3_2/core_report.md`。
- `docs/implementation/m3_2_strategy_acceptance.md`。
- `docs/implementation/m3_2_strategy_per_question.md`。
- M4 困难查询分类和 trace 证据。
- 完整测试和验证记录。
- 一个独立 commit，不 push。

不得提交：

- embedding cache
- FAISS/BM25 index
- model cache
- API 日志
- secrets
- 用户提供的 handoff 源文件，除非用户明确要求

## 15. Acceptance Criteria

本任务通过指“M3 策略收口完成”，不是指 S1 必须胜出。

必须全部满足：

- M3/M3.1 历史结论未被覆盖。
- S1 配置、门槛和代码状态在 holdout 前冻结。
- holdout formal run count 为 1。
- 没有根据 holdout 调参。
- S1 按冻结 gate 得到 pass 或 fail。
- pass 时默认选择 S1；fail 时默认保持 B1。
- 唯一 M4 fixed baseline contract 与默认选择一致。
- `m3_strategy_closed=true`。
- `m4_entry_ready=true`。
- `m3_1_core_passed=false` 保持不变。
- metadata prefix leak 为 0。
- active index 修改符合 gate。
- parser gate、指定测试、完整后端、Ruff、前端 lint/build 全部通过。
- 产物包含 dataset/config/code/manifest SHA 和逐题证据。
- 独立 commit 已创建，未 push。
- 完成后停止，不实施 M4。

## 16. Immediate Next Steps

新 session 按以下顺序执行：

1. 读取第 4 节列出的上下文，检查 git 状态和 M3.1 产物。
2. 修订路线文档，明确 M3.1 失败、M3.2 收口、M4 解耦。
3. 复用/重构 M3.1 runner，注册并测试唯一 S1。
4. 完成所有代码、配置和测试后冻结 run。
5. 向用户重新申请 embedding 外部调用授权。
6. 先运行 holdout，再运行 old dev，不得重跑。
7. 按冻结 gate 选择 S1 或 B1，生成 M4 baseline contract。
8. 生成验收、逐题和困难查询分类。
9. 完成所有验证，创建独立 commit，不 push。
10. 明确报告 selected fixed baseline、S1 是否通过、active index 是否改变、
    M4 是否具备进入条件，然后停止。
