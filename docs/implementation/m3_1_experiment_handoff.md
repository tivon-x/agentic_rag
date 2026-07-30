# M3.1 固定检索性能优化实验：执行交接

## 1. Task Goal

在不引入 M4 Enhanced 能力的前提下，开展一轮有预算、可归因、可复现的
fixed retrieval 优化实验，找到稳定优于 B1 的 `B2.1 /
v2_fixed_optimized`。只有旧 48 题回归集和新 48 题冻结 holdout 都通过
原 M3 发布门槛，才将 Core 改为通过、切换默认 fixed pipeline，并把 M4
标记为具备进入条件。

本任务完成后停止。即使 M3.1 通过，也不得自动执行 M4；必须等待用户再次
明确批准 Enhanced。

## 2. Problem Definition

M3 已实现固定检索和可复现评测，但原 B2 未通过发布门槛。修正统一 top-10
口径后，当前结果为：

| Pipeline | Recall@10 | MRR@10 | nDCG@10 | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.697917 | 0.397528 | 0.444475 | 290.1309 |
| B1 | 0.656250 | 0.342642 | 0.402658 | 490.0619 |
| B2 | 0.572917 | 0.336194 | 0.376877 | 442.0734 |
| B3 | 0.572917 | 0.336194 | 0.376877 | 474.2672 |

关键逐题比较：

- B0 相对 B1：17 胜 / 16 平 / 15 负。
- B2 去 rerank 相对 B1：17 胜 / 18 平 / 13 负。
- B2 去 metadata 相对 B1：6 胜 / 35 平 / 7 负。
- 正式 B2 相对 B1：12 胜 / 18 平 / 18 负。

当前主要矛盾不是缺少改善题，而是排名不稳定，无法满足“退化不超过 4 条”。
当前 FlashRank 会用 rerank 顺序完全覆盖融合顺序；人工 trace 已确认多个 gold
在 dense/fused 阶段位于前列，之后被 TinyBERT 排出 top-10。完整 metadata
prefix 也在当前构造下产生净负贡献。

原 M3 结果和修复不能覆盖或改写：

- 原 M3 实现 commit：
  `5e6d7d68da9b59f164533a42b2af3c51b4a88307`
- M3 评测修复 commit：
  `199492dd5adcdeb0e8abce76509e4a4001fbdd0d`
- 原验收：
  `docs/implementation/m3_acceptance.md`
- 原逐题结果：
  `docs/implementation/m3_per_question.md`

M3.1 必须作为新的实验和验收记录，不得回头把原 B2 改写成“已通过”。

## 3. Scope

### 3.1 In scope

- 将现有 48 题改作 M3.1 开发/诊断集。
- 在同一 25 篇冻结 corpus 上新增 48 条 retrieval holdout，四类各 12 条。
- 配置化 metadata 字段、启发式 boost、rerank 输入表示、reranker model、
  rerank/fusion rank blend 和 dense/sparse RRF 权重。
- 以 B0/B1 为锚点，最多评估 24 个新配置。
- 使用开发集选出唯一 finalist。
- finalist 与 B1 在旧 48 题和新 48 题上从冻结配置重新评测。
- 修正延迟测量，单独记录 query embedding、recall/fusion、rerank、
  expansion、packing 和 end-to-end retrieval。
- 生成 M3.1 验收、逐题结果、消融、坏例、延迟、默认决策和回滚说明。
- 若全部门槛通过，切换默认 fixed pipeline 到 B2.1，并标记
  `m4_entry_ready=true`。

### 3.2 Out of scope

- 不实现 query routing、query rewrite、multi-query、query decomposition。
- 不实现第二轮检索、自纠错循环、claim validation 或 refusal judge。
- 不实现 run worker、checkpoint、run queue、详细 Agent trace。
- 不修改 parser artifact、passage 边界、chunker 或 gold 来获得更高分。
- 不更换 embedding provider/model/dimension/input mode。
- 不引入远程 reranker 服务。
- 不引入 Qwen3/BGE 0.6B 等重型本地 reranker 作为正式候选。本机没有
  NVIDIA GPU，总内存约 16 GB，当前可用内存约 5.3 GB，难以满足 p95
  约束。
- 不用统一综合分掩盖子集退化。
- 不在最终 holdout 失败后继续针对同一 holdout 调参。
- 不自动执行 M4。

## 4. Current Context

开始前必须读取：

1. `AGENTS.md`
2. `docs/research/v2_upgrade_plan.md` 第 6、10、11 节
3. `docs/research/phase1_research_report.md`
4. `docs/implementation/m1_acceptance.md`
5. `docs/implementation/m2_acceptance.md`
6. `docs/implementation/m3_acceptance.md`
7. `docs/implementation/m3_per_question.md`
8. 本文件

开始前必须确认：

- 当前基线至少包含 commit `199492d`。
- M2 parser gate 仍通过。
- 冻结 parser artifact SHA-256 仍为
  `98e8adf680c578c21d2fffe5b97f3f85d24b768b827fe81aa8ddfc280af242d9`。
- 工作区已有用户改动必须保留；不得覆盖无关 dirty files。
- 当前默认仍为 `B1 / v1_flat_rerank`。

当前 fixed pipeline 入口：

- `indexing/retrieval_pipeline.py`
- `indexing/retriever.py`
- `indexing/bm25_index.py`
- `evals/v2_runner.py`
- `evals/build_report.py`
- `evals/configs/v2_b1.yaml`
- `evals/configs/v2_b2.yaml`
- `evals/configs/v2_b3.yaml`

## 5. Required Design Amendment

实现前先在 `docs/research/v2_upgrade_plan.md` 增加 M3.1 修订说明：

- 原 B2/B3 是历史冻结实验，结论保持失败。
- 新候选正式命名为 `B2.1 / v2_fixed_optimized`。
- B2.1 由冻结开发实验选择有效组件，不要求为了技术叙事强行包含完整
  metadata prefix 或 rerank。
- `retrieval_text` 与 `quote_text` 的分离仍是硬约束。
- RRF 仍固定 `k=60`；可以配置 dense/sparse 的通道权重。
- M3.1 使用旧 48 题作开发/回归集，新增 48 题作最终 holdout。
- B2.1 必须在两个数据集上都通过原发布门槛。
- B3 后置 neighbor expansion 不改变 rerank top-10，因此不再用 top-10
  排名证明 expansion 收益。M3.1 不默认启用 B3；后续应使用跨章节
  Context Recall、token 增幅和正式 answer test 决策。
- M4 仍要求 M3.1 门槛全部通过且用户再次批准 Enhanced。

## 6. Inputs and Outputs

### 6.1 Frozen inputs

- Parser artifact：25 篇、8,315 passages。
- Embedding：
  - provider：`openai`
  - model：`qwen3.7-text-embedding`
  - dimension：1024
  - batch size：20
  - input mode：raw
  - `check_embedding_ctx_length=false`
  - max input chars：6000
- Sparse/dense top-k：每通道 40。
- RRF：`k=60`。
- Metric top-k：10。
- Final context seed：8。
- Max context passages：12。
- Context budget：8,000 tokens。

### 6.2 Dataset roles

- `evals/datasets/retrieval_v2_core.jsonl`
  - 保持文件和 SHA 不变。
  - M3.1 中改作开发/历史回归集。
- 新建 `evals/datasets/retrieval_v2_core_holdout.jsonl`
  - 48 条，四类各 12 条。
  - 与开发集不能只是同义改写。
  - gold 必须指向冻结 parser artifact 中真实存在的 passage/paper/section IDs。
  - 在任何候选实验前冻结 SHA-256。
  - finalist 确定前不得运行 holdout 质量评测。

### 6.3 Outputs

- `docs/research/v2_upgrade_plan.md` 的 M3.1 修订。
- `evals/datasets/retrieval_v2_core_holdout.jsonl`。
- M3.1 dev/final YAML 配置。
- 配置驱动的实验 registry/overrides，不复制 24 套 pipeline 代码。
- `artifacts/evals/v2_m3_1/` 下的 dev、final、manifest、patch 和报告。
- `docs/implementation/m3_1_acceptance.md`。
- `docs/implementation/m3_1_per_question.md`。
- 一个独立 commit，不推送。

## 7. Pipeline and Evaluation Changes

### 7.1 Configuration contract

扩展 `RetrievalPipelineConfig`，至少显式记录：

- `metadata_prefix_fields`
- dense/sparse 是否使用 metadata prefix
- rerank 输入：`quote`、`title_section_quote` 或 `retrieval`
- reranker model
- rerank merge mode：`replace` 或 `weighted_rrf`
- fusion/rerank rank weights
- dense/sparse RRF weights
- boost policy：`current` 或 `off`

所有字段必须进入：

- config hash
- index/retrieval contract
- run report
- manifest
- 阶段 trace

索引表示未变化的实验必须复用同一内容寻址索引；metadata 字段、tokenizer
或 embedding 输入变化时必须重建。不得复用 contract 不匹配的 FAISS/BM25。

开发配置通过 YAML 中的 experiment overrides 生成，不把 24 个实验全部写成
生产 aliases。只有最终胜出的 B2.1 才加入正式 registry 和 alias。

### 7.2 Rank blending

当前 reranker `replace` 行为保留为对照。新增 `weighted_rrf`：

- 使用融合名次和 rerank 名次做第二次 rank-level fusion。
- RRF 常数仍为 60。
- 测试 fusion/rerank 权重：
  - `0.75 / 0.25`
  - `0.50 / 0.50`
  - `0.25 / 0.75`
- 不实现“前 N 名禁止下降”等硬编码保护。
- 最终排序必须确定性处理缺失 rank 和同分 tie。

### 7.3 Reranker candidates

只使用当前 FlashRank 运行时可支持的 CPU 模型：

- `ms-marco-TinyBERT-L-2-v2`
- `ms-marco-MiniLM-L-12-v2`
- `ms-marco-MultiBERT-L-12`
- `rank-T5-flan`

模型下载必须写入受控 cache，不提交模型文件。下载或调用任何外部服务前，
新 session 必须获得用户对目标服务和数据外发的明确授权；不得把其他
session 的授权当作永久授权。

## 8. Experiment Matrix

B0/B1 始终作为锚点运行，但不计入 24 个新候选预算。

### Round 1：召回表示，6 个配置

所有配置均不使用 rerank：

1. quote-only + mixed BM25 + min-max。
2. quote-only + mixed BM25 + RRF。
3. section prefix + mixed BM25 + RRF。
4. title + section prefix + mixed BM25 + RRF。
5. full metadata prefix + mixed BM25 + RRF。
6. Round 1 当前最优配置，但关闭全部启发式 boost。

Round 1 排序规则：

1. 先排除 Recall@10 低于 B1 的配置。
2. losses 少者优先。
3. wins 多者优先。
4. Recall@10 高者优先。
5. 最差子集 delta 高者优先。
6. nDCG@10 高者优先。
7. p95 低者优先。

保留前两名进入 Round 2。不得使用一个综合分。

### Round 2：reranker，8 个配置

Round 1 前两名分别测试四个 reranker：

1. TinyBERT，rerank 输入 quote-only。
2. MiniLM-L12，rerank 输入 quote-only。
3. MultiBERT-L12，rerank 输入 quote-only。
4. rank-T5-flan，rerank 输入 quote-only。

本轮保持 `replace`，用于测量纯 reranker 能力。按 Round 1 的同一排序规则
保留前两名。

### Round 3：稳定化和权重，10 个配置

1. Round 2 第一名分别测试三个 fusion/rerank 权重。
2. Round 2 第二名分别测试三个 fusion/rerank 权重。
3. 当前最佳 blended 配置改用 `title_section_quote` rerank 输入。
4. 当前最佳 blended 配置改用完整 `retrieval` rerank 输入。
5. 当前最佳 blended 配置使用 dense/sparse RRF 权重 `1.25 / 0.75`。
6. 当前最佳 blended 配置使用 dense/sparse RRF 权重 `0.75 / 1.25`。

总计 24 个新候选。不得在看到 holdout 结果后增加第 25 个配置。

## 9. Dev Promotion Gate

开发集只有同时满足以下条件的配置才可成为 finalist：

- Recall@10 至少比同轮 B1 高 0.02。
- 相对 B1 至少 10 胜。
- 相对 B1 最多 3 负。
- MRR@10 不低于 B1。
- nDCG@10 不低于 B1。
- 四个子集任一子集最多下降 1 条。
- p95 不超过 B1 的 1.35 倍。
- 按四类分层构造四折后，不得只在一个 fold 或一个类别产生全部收益。

若没有配置通过：

- 不打开 holdout。
- 默认保持 B1。
- `m4_entry_ready=false`。
- 写出失败原因和 Pareto frontier 后停止。

## 10. Final Frozen Evaluation

finalist 选择后：

1. 将候选冻结为 `B2.1 / v2_fixed_optimized`。
2. 冻结 YAML、config SHA、代码 commit/working-tree patch SHA。
3. 对 B1 和 B2.1 使用同一 parser、embedding、top-k 和 test set 重建索引。
4. 先运行新 holdout，一次性生成正式质量结果。
5. 再在旧 48 题上生成回归结果。
6. 不因结果失败修改 gold、阈值或候选配置。

两个数据集都必须分别满足：

- B2.1 Recall@10 不低于 B1。
- B2.1 相对 B1 至少 8 条改善 gold rank。
- B2.1 相对 B1 退化不超过 4 条。
- 四个子集没有任何一个出现 Recall@10 下降 2 条以上。
- B2.1 p95 不超过 B1 的 1.5 倍。

同时报告：

- Recall@5/10
- MRR@10
- nDCG@10
- paper Recall@10
- section Recall@10
- context passage recall
- 逐题 W/T/L
- 四个目标子集
- p50/p95
- 每阶段延迟
- 坏例
- paired bootstrap 95% 区间，仅描述不确定性

### 10.1 Latency protocol

- 每个 pipeline 先预热一次，不计入正式延迟。
- 48 条查询以固定随机顺序运行 5 轮。
- 不并发运行两个写同一目录的评测任务。
- end-to-end retrieval 包含 query embedding 和 rerank，不包含 index build。
- 同时报告冷启动首轮和 warm p50/p95。
- ablation 与正式 pipeline 使用相同重复次数。

## 11. B3 Decision

M3.1 默认不启用 neighbor expansion。B2.1 通过即可让 fixed Core 通过，
不要求 B3 成为默认。

如果附带运行 B3.1，只能作为非发布诊断，并报告：

- 跨章节 Context Recall 改善题数。
- 其他三类 Context Recall 退化题数。
- context token 增幅。
- packing drop 数。
- expansion 和 packing 延迟。

B3.1 不得用未变化的 rerank top-10 排名宣称 expansion 收益。

## 12. Failure Handling and Reproducibility

- 同一 `run_dir` 增加独占锁；已有活跃进程时立即失败，禁止重复后台进程
  交错删除或覆盖索引。
- 每个 run 记录 base commit、dirty 状态、working-tree patch 和 SHA-256。
- output 只能位于仓库 `artifacts/` 下。
- embedding/index contract 不一致立即失败。
- reranker 缺失或模型加载失败时，正式实验不得静默降级。
- 外部 embedding API 失败可重试同一内容寻址批次，但不得换模型、维度或
  input mode。
- active production index 在最终门槛通过前不得改变。
- 所有真实索引和模型 cache 保持 ignored，不提交。

## 13. Tests and Manual Checks

### 13.1 Automated tests

至少覆盖：

- config override 生成确定性 pipeline hash。
- index contract 对 metadata/tokenizer 变化敏感。
- rerank 输入不泄漏到 quote/context。
- weighted RRF 对缺失 rank、tie 和同一 passage 去重确定。
- boost policy 进入 config 和 manifest。
- contract 不匹配立即失败。
- reranker 不可用时正式实验失败。
- run directory lock 阻止并发覆盖。
- old dev 与 new holdout SHA 分别记录。
- candidate selection 不使用统一综合分。
- holdout 只接受一个 frozen finalist。
- latency warmup 不进入正式样本。
- metadata prefix 泄漏计数保持 0。

### 13.2 Manual checks

- 至少检查 12 个 B1/B2.1 rank 变化，其中必须包含 win 和 loss。
- 表格、缩写、跨章节、中文术语各检查至少 3 个坏例。
- 检查至少 5 个 blended rerank 案例，确认不是硬编码保留前 N。
- 检查引用和回答预览不显示 metadata prefix。
- 检查新 holdout 问题不是旧问题的简单同义改写。
- 检查 active index pointer 在最终通过前未变化。

## 14. Verification Commands

实现后至少运行：

```text
uv run --extra dev python -m pytest tests/test_bm25_index.py tests/test_retriever.py tests/test_retrieval_pipeline.py tests/test_evals.py -q

uv run python -m evals.runner --config evals/configs/v2_m3_1_dev.yaml
uv run python -m evals.select_candidate --run artifacts/evals/v2_m3_1/dev --max-finalists 1

uv run python -m evals.runner --config evals/configs/v2_m3_1_final.yaml
uv run python -m evals.build_report --runs artifacts/evals/v2_m3_1

uv run --extra dev ruff check indexing core evals tests
uv run --extra dev python -m pytest -q

npm --prefix web run lint
npm --prefix web run build
```

如果仓库实际 frontend package manager 与上述命令不一致，以当前
`web/package.json` 和 M3 已验证命令为准，但必须在验收报告中记录真实命令。

## 15. Acceptance Criteria

### 15.1 M3.1 pass

只有以下条件全部成立才通过：

- dev promotion gate 通过。
- 唯一 finalist 在新 holdout 通过全部正式门槛。
- 同一 finalist 在旧 48 题通过全部正式门槛。
- metadata prefix leak 为 0。
- latency protocol 完整。
- 指定测试、完整后端、Ruff、前端 lint/build 全部通过。
- manifest、dataset SHA、config SHA、代码 commit/patch 均可追溯。
- 默认 pipeline 只在上述条件成立后切换到 B2.1。

通过后：

- `core_passed=true`
- `default_pipeline=b2_1`
- `m4_entry_ready=true`
- 停止，不执行 M4。

### 15.2 M3.1 fail

任一条件失败：

- 默认保持 B1。
- `core_passed=false`
- `m4_entry_ready=false`
- 不降低门槛。
- 不在同一 holdout 上继续调参。
- 报告最接近门槛的配置、Pareto frontier、主要坏例和建议停止原因。

## 16. Deliverables

- M3.1 方案修订。
- 新 48 条冻结 holdout 及 SHA。
- 配置驱动实验 runner 和选择器。
- rank blending、rerank input、metadata/boost/RRF weight 配置。
- dev 与 final 完整产物。
- `docs/implementation/m3_1_acceptance.md`。
- `docs/implementation/m3_1_per_question.md`。
- 独立 commit；不得 push。
- 最终明确报告：
  - Core 是否通过。
  - 默认选择 B1 还是 B2.1。
  - M4 是否具备进入条件。
  - 是否修改 active index。

## 17. Immediate Next Steps for the New Session

1. 读取本文件和第 4 节列出的项目文档，检查 commit、工作区和 M2/parser
   进入条件。
2. 先落地 V2 方案的 M3.1 修订、holdout 数据合同、实验配置 schema 和
   run directory lock。
3. 在任何真实 embedding 或模型下载前，向用户确认目标服务、数据外发和
   模型下载授权。
4. 完成开发集 24 配置实验；未通过 dev promotion gate 就停止。
5. 只有唯一 finalist 冻结后才运行 holdout，并按第 15 节决定默认和
   M4 进入状态。
