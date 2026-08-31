---
project_status_schema: 1
status_updated: "2026-08-31"
completed_through: M7
next_planned_goal: M8
implementation_authorized: false
production_pipeline: v1_flat_rerank
production_pipeline_config_hash: ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17
answer_strategy: fixed
terminated_goals:
  - M4.2
acceptance_evidence:
  - docs/implementation/m1_acceptance.md
  - docs/implementation/m2_acceptance.md
  - docs/implementation/m3_acceptance.md
  - docs/implementation/m3_1_acceptance.md
  - docs/implementation/m3_2_strategy_acceptance.md
  - docs/implementation/m4_1_acceptance.md
  - docs/implementation/m4_1_1_retrieval_quality_acceptance.md
  - docs/implementation/m4_1_2_adaptive_eval_acceptance.md
  - docs/implementation/m5_fixed_product_acceptance.md
  - docs/implementation/m5_1_chat_experience_acceptance.md
  - docs/implementation/m5_1_web_ui_fix_acceptance.md
  - docs/implementation/m6a_kite_data_acceptance.md
  - docs/implementation/m6b_kite_b1_acceptance.md
  - docs/implementation/m6c_kite_pipeline_acceptance.md
  - docs/implementation/m6d_evaluation_presentation_acceptance.md
  - docs/implementation/m7_project_design_and_interview_guide_acceptance.md
---

# Agentic RAG V2 升级方案

> 当前里程碑、授权和生产默认值只读取文件顶部 YAML 状态区块，不从正文或历史文档推断。

文件顶部的 `project_status_schema` 区块是里程碑状态的唯一可变来源。代码、配置或冻结 artifact 与该区块冲突时，停止实施并先修复事实漂移；正文和历史验收报告不得覆盖该区块的当前状态。

## 1. 当前执行状态

本节是状态区块的人类可读摘要，不具有独立状态优先级。后文只描述仍然有效的架构、约束和待执行工作；历史实验的完整过程以对应验收报告和冻结 artifact 为准。

| 里程碑 | 状态 | 当前结论 |
|---|---|---|
| M1 | 已完成 | SQLite 迁移、持久索引任务、lease 恢复和不可变索引版本已交付。 |
| M2 | 已完成 | 论文目录、PyMuPDF4LLM 解析、页码证据、Library、Paper 和 Search 已交付。 |
| M3 / M3.1 / M3.2 | 已完成 | 固定策略已收口。`v1_flat_rerank` 是冻结 B1，复杂固定候选未晋级。 |
| M4.1 / M4.1.1 / M4.1.2 | 已完成，未通过 | Adaptive 两次复验均未证明净收益，`ANSWER_STRATEGY=fixed` 保持默认。 |
| M4.2 | 终止 | 不为未通过的 Adaptive 增加持久 run、checkpoint、worker 或产品入口。 |
| M5 / M5.1 | 已完成 | 证据优先的 Next.js 产品、结构化 Chat evidence、会话回看和 UI 修复已交付。 |
| M6 | M6A 至 M6D 已完成 | B0 至 B3 在同一冻结 snapshot 和 clean evaluation commit 上完成 15 题正式比较；B2/B3 未满足 promotion gate，生产默认继续 `v1_flat_rerank`，README 与 `/evaluation` 已同步冻结结果。 |
| M7 | 已完成 | 项目设计与面试指南已落盘；只修改文档和事实漂移防护，不新增运行代码。 |
| M8 | 暂缓，不实施 | 当前不做部署；如恢复，需要重新规划、明确平台并单独授权。 |

开始任何新 Goal 时必须重新记录分支、HEAD、工作区和依赖状态；活动分支与提交快照不写入本计划。

### 1.1 已冻结的产品默认值

| 项目 | 值 |
|---|---|
| Production Pipeline | `v1_flat_rerank` |
| Pipeline config hash | `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17` |
| Baseline contract | `artifacts/evals/v2_m3_2/m4_fixed_baseline.json` |
| Answer strategy | `fixed` |
| Parser | `pymupdf4llm`，失败时显式 legacy fallback |
| Index write mode | `versioned` |
| Embedding input mode | `raw` |
| Embedding max input chars | `6000` |

M6 可以评测 B0、B1、B2 和 B3，但在新的生产决策获批前，不得改变上述默认值。

### 1.2 历史事实入口

- M3.2 固定策略：`docs/implementation/m3_2_strategy_acceptance.md`
- M4.1.1 复验：`docs/implementation/m4_1_1_retrieval_quality_acceptance.md`
- M4.1.2 复验：`docs/implementation/m4_1_2_adaptive_eval_acceptance.md`
- M5 产品验收：`docs/implementation/m5_fixed_product_acceptance.md`
- M5.1 Chat 验收：`docs/implementation/m5_1_chat_experience_acceptance.md`
- M5.1 UI 修复验收：`docs/implementation/m5_1_web_ui_fix_acceptance.md`

这些文件是历史证据，不是待重复执行的任务。冻结数据、报告和失败结论不得覆盖。

## 2. 产品定位

Agentic RAG V2 是面向个人论文库的 evidence-first academic RAG assistant。用户可以导入论文、检索论文和章节、围绕论文提问，并从答案证据回到原 PDF 页。

产品需要回答两类问题：

1. 用户能否稳定导入、搜索、提问并核对证据。
2. 当前 Production Pipeline 为什么被选择，复杂候选为什么晋级或失败。

M1 至 M5.1 已经完成第一类闭环。M6 补齐第二类闭环，把公开端到端 Benchmark、内部诊断集、Pipeline 决策和产品默认值连起来。

### 2.1 目标

- local-first、单用户、本地文件和 SQLite，不依赖外部数据库或任务服务。
- 数字版 PDF 是主要输入，Markdown 和 TXT 保持兼容。
- 主要结论绑定论文、章节、页码和 source-faithful quote。
- Benchmark 和产品复用同一 parser、index、retriever、context packing 和 fixed answer 实现。
- 公开 Benchmark 评价最终回答，内部数据解释检索成功或失败的原因。
- 所有正式实验冻结数据、配置、模型、索引和代码版本，保留逐题结果与失败案例。

### 2.2 非目标

- 不做多人协作、权限、云同步、浏览器插件或移动端原生应用。
- 不做 PDF 编辑、批注同步、参考文献管理器或论文写作工具。
- 不承诺扫描件、复杂跨页表格和公式语义与数字版 PDF 同等准确。
- 不引入 Docling、OCR 服务、GraphRAG、RAPTOR、dsRAG、RSE、CCH、HyDE、CRAG、Self-RAG、知识图谱或新向量数据库。
- 不继续优化 Adaptive，不增加多轮检索或复杂 Agent loop。
- 不做 pipeline auto-search、prompt optimizer、RAGAS 全量集成或实时实验后台。
- 不向普通用户暴露 B0、B1、B2、B3、S1、Adaptive、BM25、RRF 或 reranker 开关。

## 3. 当前系统架构

```text
Next.js
  | REST + SSE
FastAPI
  |-- Library / Search / Paper / Chat
  |-- SQLite catalog and chat sessions
  |-- index worker with leases
  `-- fixed LangGraph answer path

Paper files
  | parser + deterministic normalization
  v
Paper / Section / Passage
  | quote_text + retrieval_text
  v
Immutable index version
  | FAISS + BM25 + manifest
  v
Retrieval pipeline
  | fusion + optional rerank + context packing
  v
Grounded answer + structured evidence
```

M6 在现有系统外增加只读评测路径，不增加新的线上服务：

```text
                         Evaluation
                             |
             +---------------+---------------+
             |                               |
             v                               v
      KITE AI Papers                 Internal Retrieval Set
       Public E2E Eval                 Frozen Diagnostic
             |                               |
       Answer score                     Recall@K
       Latency                          MRR / nDCG
       Context and usage                Context Passage Recall
             |                               |
             +---------------+---------------+
                             |
                             v
                    Production decision
                             |
                             v
                  Fixed FastAPI + Next.js
```

### 3.1 数据和标识边界

- `paper_id = sha256(raw_file_bytes)`。
- `paper_version_id` 随 parser 或 normalization 版本变化。
- `section_id` 和 `passage_id` 使用稳定、可复现的输入生成。
- `quote_text` 只保存 source-faithful 原文，用于上下文和引用。
- `retrieval_text` 可以包含 metadata prefix，只用于召回和 rerank。
- `index_versions` 不可变，SQLite `app_state` 是 active index 的权威来源。
- Query-time embedding contract 必须与 index manifest 一致，不兼容时失败，不静默重建或降级。

### 3.2 固定检索路径

`indexing/retrieval_pipeline.py` 是固定 Pipeline 的唯一注册表。M6 不创建第二套 Pipeline 实现。

| 名称 | 用途 |
|---|---|
| B0 | Hybrid，无 rerank。 |
| B1 | `v1_flat_rerank`，当前 Production Pipeline。 |
| B2 | Metadata prefix、RRF 和 rerank。 |
| B3 | B2 加 neighbor expansion。 |

S1 和 Adaptive 保留为历史实验资产，不参加第一轮 KITE 正式比较。

## 4. 两层评测各自负责什么

### 4.1 KITE AI Papers

KITE AI Papers 是公开端到端 Benchmark，用于评价完整系统给出的最终答案。

官方资源：

- Repository：<https://github.com/D-Star-AI/KITE>
- Queries：<https://github.com/D-Star-AI/KITE/tree/main/queries>
- AI Papers query file：<https://github.com/D-Star-AI/KITE/blob/main/queries/ai_papers.json>
- AI Papers corpus：<https://github.com/D-Star-AI/KITE/tree/main/knowledge-base-content/ai-papers>
- Official grader：<https://github.com/D-Star-AI/KITE/blob/main/eval/grade_responses.py>

截至本计划修订日，准备阶段固定以下 upstream 快照：

| 项目 | 值 |
|---|---|
| KITE branch | `main` |
| KITE commit | `85e71ad63db9ea410eccbb0158f94e7d72462b99` |
| AI Papers query SHA-256 | `6f242828e2e96b34e152af16afabf981f938eec5f3d11522c205ef635cae57d3` |
| AI Papers questions | 15 |
| Empty rubrics | 6 |
| AI Papers PDF paths | 134 |

正式实施时必须重新验证远程 commit 可访问，然后仍使用本表固定的 commit，不跟随之后的 `main`。如果决定升级 KITE 快照，需要新 Goal 和新报告，不能覆盖本轮结果。

KITE AI Papers 用于记录：

- 0 至 10 的 KITE-compatible answer score。
- 每题答案和逐题分数。
- p50、p95 和逐题 latency。
- context tokens 和 LLM calls。
- provider 明确返回时的 input/output token usage。
- provider 价格配置存在时的成本。
- evidence 数量、引用位置和运行错误。

KITE AI Papers 不提供 gold passage、section 或 paper IDs，因此不用于计算 Recall@K、MRR 或 nDCG。

### 4.2 Internal Retrieval Dataset

现有内部数据继续作为 diagnostic benchmark：

- `evals/datasets/retrieval_v2_core.jsonl`
- `evals/datasets/retrieval_v2_core_holdout.jsonl`
- `artifacts/evals/v2_core/`
- `artifacts/evals/v2_m3_1/`
- `artifacts/evals/v2_m3_2/`

它保留四类问题：

- `exact_term_definition`
- `method_section_location`
- `experiment_number_table`
- `cross_paper_or_section`

它用于解释：

- Recall@K、MRR 和 nDCG 的变化。
- paper、section 和 passage 命中情况。
- Context Passage Recall。
- 冗余、延迟和具体坏例。

M6 不删除、改写或重新标注内部数据。旧 holdout 已经打开，不得再次用于调参，也不为了 KITE 决策重跑后覆盖原报告。生产决策引用已有冻结结果。

### 4.3 两层结果如何合并

KITE 回答系统最终是否能答好，Internal Retrieval Dataset 解释检索为什么成功或失败。两者不能互相替代：

- KITE 分数提高，不代表 passage recall 一定提高。
- 检索指标提高，不代表最终答案一定提高。
- 新候选只有同时满足公开 E2E 收益、延迟和证据完整性边界，才可以进入生产决策。
- KITE AI Papers 只有 15 题，小差异不能写成统计显著提升。

## 5. KITE 数据契约

### 5.1 数据获取

KITE PDF 使用 Git LFS。准备命令必须检测 LFS pointer，不能把下面的文本当 PDF 交给 parser：

```text
version https://git-lfs.github.com/spec/v1
oid sha256:<object-id>
size <bytes>
```

推荐获取方式：

```bash
git clone https://github.com/D-Star-AI/KITE.git <external-cache>/KITE
git -C <external-cache>/KITE checkout 85e71ad63db9ea410eccbb0158f94e7d72462b99
git -C <external-cache>/KITE lfs pull
```

KITE checkout 放在仓库外的用户指定目录。项目不提交 PDF、LFS object、派生全文、索引或模型缓存。

准备阶段必须校验：

- query JSON 可以解析。
- dataset hash 与 manifest 一致。
- 正好读取 15 个 case。
- 每个 case 有非空 `query` 和 `gt_answer`。
- `rubric` key 存在，但字符串可以为空。
- 134 个 PDF 路径都解析到真实文件。
- 每个 PDF 以 `%PDF-` 开头。
- 任一 LFS pointer、缺失文件或空 PDF 都明确失败。

### 5.2 Case 模型

M6 只实现 KITE AI Papers，不为未来数据集建立 Protocol、factory 或 internal adapter。

最小 case 模型：

```python
@dataclass(frozen=True, slots=True)
class KiteCase:
    id: str
    query: str
    reference_answer: str
    rubric: str
    source_index: int
```

`rubric` 保留原始字符串，允许为空。`id` 按冻结文件顺序生成 `ai-papers-001` 至 `ai-papers-015`。原始顺序、query、answer 和 rubric 都不得根据实验结果修改。

### 5.3 冻结 manifest

准备阶段输出独立 manifest，至少记录：

```text
schema_version
benchmark_name
upstream_repository
upstream_commit
query_path
query_sha256
case_count
empty_rubric_count
corpus_root
corpus_file_count
corpus_file_sha256
parser_name
parser_version
normalization_version
created_at
```

`corpus_root` 在提交报告前必须转成逻辑名称或相对标识，不能泄露本机绝对路径。

### 5.4 数据许可边界

KITE 仓库使用 MIT License，但仓库许可证不自动证明其中论文 PDF 可以由本项目再次公开分发。M6 默认只提交来源、commit、hash、报告和必要的聚合结果，不提交或公开托管 KITE PDF。公开部署是否使用 KITE corpus 由 M8 单独确认。

## 6. E2E 运行与评分契约

### 6.1 复用产品实现

Benchmark 必须经过产品使用的 fixed answer path：

```text
KiteCase.query
    |
    v
selected RetrievalPipelineConfig
    |
    v
existing index / retriever / context packing
    |
    v
fixed LangGraph answer path
    |
    v
answer + structured evidence
```

禁止维护 benchmark-only retrieval 或 answer implementation。M6 可以为 runner 注入 Pipeline config 和对应 index manifest，但不能复制 B0 至 B3 的参数。

优先复用：

- `indexing/retrieval_pipeline.py`
- `indexing/retriever.py`
- `indexing/indexer.py`
- `indexing/paper_ingestion.py`
- `evals/v2_corpus.py`
- `evals/v2_runner.py`
- `core/factory.py`
- `agent/graph.py`
- `agent/schemas.py`
- `llms/llm.py`

`evals/m4_1_1_runner.py` 中的 structured model invocation、hash 校验和报告思路可以复用，但它的 Adaptive 比较、claim gold 和 `run_fixed_b1()` 不能成为 KITE runner 的主流程。

### 6.2 每题记录

每个 case 至少保存：

```text
case_id
source_index
query
reference_answer
rubric
pipeline_name
pipeline_config_hash
answer
evidence
latency_ms
context_tokens
llm_calls
input_tokens
output_tokens
judge_model
judge_prompt_version
kite_score
judge_error
run_error
```

`input_tokens` 和 `output_tokens` 可以为 `null`。只有 provider response metadata 给出真实 usage 时才填入。没有价格配置时不计算货币成本。

### 6.3 KITE-compatible judge

正式 KITE score 采用官方兼容语义：judge 接收 query、ground-truth answer、rubric 和 candidate answer，只返回 0 至 10 的整数。

约束固定如下：

- prompt 单独版本化并写入报告。
- `temperature=0`。
- generation model 和 judge model 分开配置。
- 所有 Pipeline 使用同一个 judge model 和 prompt。
- 空 rubric 原样传入，不能自动生成 rubric。
- score 超出 0 至 10、非整数或无法解析时视为 judge failure。
- 网络、限流和无效输出最多重试一次。
- 两次都失败时 `kite_score=null`，记录错误，不能按 0 分吞掉。

官方历史结果使用 `gpt-4o-2024-08-06`。如果本项目使用其他 judge，只能称为 KITE-protocol score，并用于本批实验内的 Pipeline 横向比较，不能与 KITE README 中的绝对分数直接比较。

### 6.4 项目诊断评分

结构化 `reason`、`missing_points` 和 `incorrect_points` 不属于官方 KITE score。需要这些字段时，作为可选 diagnostic judge 单独运行和保存：

```json
{
  "reason": "...",
  "missing_points": [],
  "incorrect_points": []
}
```

第一轮 M6A 和 M6B 不实现 diagnostic judge。M6C 只有在逐题失败分析不足以支持决策时才能增加，且不能改变 KITE score。

### 6.5 运行配置

正式配置使用现有 YAML 风格：

```yaml
schema_version: 1

benchmark:
  name: kite-ai-papers
  upstream_commit: 85e71ad63db9ea410eccbb0158f94e7d72462b99
  subset: full

pipeline:
  name: b1

generation:
  strategy: fixed

judge:
  task_type: kite_judge
  prompt_version: kite-official-compatible-v1
  temperature: 0

runtime:
  concurrency: 1

output:
  dir: artifacts/evals/kite/b1
```

`kite_judge` 通过现有 task-model router 解析。具体模型因当前会话授权和供应商可用性而延后到 M6B 开始时确定，决策人为用户，执行者负责在任何 smoke 或正式运行前把解析后的模型名冻结到配置和报告。没有模型授权时只能完成确定性 prepare、adapter 和测试。

入口沿用现有独立 runner 风格：

```bash
uv run python -m evals.kite_runner prepare --config evals/configs/kite_b1.yaml
uv run python -m evals.kite_runner run --config evals/configs/kite_b1.yaml
uv run python -m evals.kite_runner report --runs artifacts/evals/kite
```

本轮不修改 `main.py`，也不增加嵌套 CLI 框架。

### 6.6 报告元数据

每份正式报告必须能独立解释运行条件：

```text
benchmark name
upstream repository and commit
dataset SHA and case count
corpus manifest SHA
parser artifact SHA
index manifest SHA
pipeline config and config hash
embedding contract
reranker contract
generation model and task-model mapping
judge model and prompt version
code commit SHA
working tree state
hardware and runtime summary
started_at and completed_at
```

如果工作区包含未提交代码，正式 runner 默认拒绝执行。确需运行时必须保存 diff hash，并在报告中标记 `working_tree_clean=false`，不能把结果写成可复现正式基线。

## 7. Production Pipeline 决策

### 7.1 第一轮候选

只比较 B0、B1、B2 和 B3。四条 Pipeline 使用同一份 KITE corpus、同一批 15 个问题、同一 generation model、同一 judge 和同一运行协议。

不运行：

- Adaptive 或 multi-round retrieval。
- S1 或历史调参候选。
- `b2_no_dense`、`b2_no_sparse`、`b2_no_metadata`、`b2_no_rerank` 等消融。
- 新 embedding、reranker、parser 或 chunking 技术。

如果 B0 至 B3 的结果无法解释某个明确差异，后续消融需要新 Goal，不能在正式实验中临时追加。

### 7.2 晋级门槛

B1 是默认对照。B0、B2 或 B3 只有全部满足以下条件，才能成为 promotion candidate：

- 15 个 case 全部生成成功并取得有效 judge score。
- 平均 KITE-protocol score 至少比 B1 高 `0.5/10`。
- 逐题比较至少 4 win，最多 2 loss；相同分数记 tie。
- p95 latency 不超过 B1 的 1.5 倍。
- 平均 context tokens 不超过 B1 的 1.5 倍。
- evidence 全部来自检索结果，quote 保持 source-faithful。
- 人工检查候选相对 B1 的所有 win 和 loss，没有引用完整性退化或明显错误答案换取的分数收益。
- 已有内部冻结报告中的已知退化被明确列入决策，不被 KITE 总分掩盖。

通过门槛只表示可以讨论晋级，不自动修改 `RETRIEVAL_PIPELINE`。用户必须单独批准 production switch。没有候选通过时继续使用 B1，这是正常结果。

### 7.3 决策报告

`docs/production_pipeline_decision.md` 至少包含：

```text
Candidate
KITE result
Pairwise wins / ties / losses
Existing internal diagnostic result
Latency and context cost
Known failure modes
Evidence audit
Decision
Reason
```

不得只按平均分选 winner，也不得为了项目叙事强行晋级更复杂的 Pipeline。

## 8. M6 实施计划

M6 拆成四个独立可合并的 Goal。每个 Goal 完成后停止，等待下一次授权。M6A 至 M6C 不改普通用户 UI，M6D 不运行新的正式实验。

### M6A：KITE 数据与可复现准备

**目标**

接入固定 KITE snapshot，完成 query adapter、LFS/PDF 校验、corpus manifest 和确定性测试，不调用 embedding、generation 或 judge 服务。

**主要文件**

```text
evals/kite.py
evals/kite_runner.py
evals/configs/kite_b1.yaml
tests/test_kite_eval.py
docs/implementation/m6a_kite_data_acceptance.md
```

如果现有 `evals/v2_corpus.py` 能直接承担 manifest 或 parser artifact 工作，优先复用，不创建新的辅助模块。

**测试边界**

- 15 个 case 和 6 个空 rubric 能稳定读取。
- 缺少 `query`、`gt_answer` 或 `rubric` key 时失败。
- 空 `rubric` 合法。
- 非法 JSON、重复 case ID 和 hash 不一致时失败。
- PDF 文件头、LFS pointer、缺失文件和空文件有明确错误。
- manifest 不包含本机绝对路径或密钥。

**验收命令**

```bash
uv run --extra dev python -m pytest tests/test_kite_eval.py -q
uv run --extra dev ruff check evals/kite.py evals/kite_runner.py tests/test_kite_eval.py
uv run python -m evals.kite_runner prepare --config evals/configs/kite_b1.yaml
git diff --check
```

**完成条件**

- KITE 数据和 corpus 能确定性校验。
- manifest 固定 upstream commit、query hash、case count 和 corpus hash。
- 没有外部模型调用，没有修改 Production Pipeline。

**回滚**

移除 KITE adapter、配置、测试和 manifest 即可。它不修改数据库、active index 或产品路径。

### M6B：B1 公开 E2E 基线

**进入条件**

- M6A 验收通过。
- 当前会话明确授权所需的 embedding、generation 和 judge 调用。
- generation model、完整 task-model mapping、judge model 和 prompt 已冻结。
- KITE corpus manifest 和 B1 config hash 未变化。

**目标**

复用产品 fixed path，先运行冻结 smoke，再运行 15 题 B1 正式基线。

Smoke case 固定为：

```text
ai-papers-001
ai-papers-005
ai-papers-010
ai-papers-015
```

Smoke 只验证运行链路，不用于调 prompt、修改数据或发布分数。正式 B1 使用全部 15 题。

**主要文件**

```text
evals/kite_runner.py
evals/configs/kite_b1.yaml
evals/configs/kite_b1_smoke.yaml
tests/test_kite_eval.py
artifacts/evals/kite/b1/
docs/implementation/m6b_kite_b1_acceptance.md
```

**测试边界**

- runner 使用 registry 中的 B1 和对应 index contract。
- generation 与 judge 配置分离。
- judge score 范围、非整数、超时、一次重试和最终失败。
- generation、retrieval 或 judge 失败保留逐题错误，不写伪分数。
- usage 缺失时保留 `null`，不估算。
- evidence 不显示 metadata-prefixed `retrieval_text`。

**验收命令**

```bash
uv run --extra dev python -m pytest tests/test_kite_eval.py tests/test_retrieval_pipeline.py tests/test_agent_grounded_answer.py -q
uv run --extra dev ruff check evals agent core indexing tests
uv run python -m evals.kite_runner run --config evals/configs/kite_b1_smoke.yaml
uv run python -m evals.kite_runner run --config evals/configs/kite_b1.yaml
git diff --check
```

**完成条件**

- 15 题 B1 全部完成或明确报告失败 case。
- 只有 15 题全部有效时才生成正式 B1 聚合分数。
- 报告包含逐题答案、证据、分数、延迟和完整 provenance。
- `RETRIEVAL_PIPELINE` 和产品代码没有变化。

**回滚**

删除本 Goal 的 runner 扩展、配置和生成 artifacts。B1 产品路径不受影响。

### M6C：固定候选比较与生产决策

**进入条件**

- M6B B1 正式基线有效。
- B0、B2、B3 的 index contract 可以在相同 corpus snapshot 上构建。
- 当前会话再次授权本 Goal 的外部模型调用。

**目标**

运行 B0、B2 和 B3，生成统一汇总、逐题差异、失败分析和 production decision。不得修改数据、judge prompt 或 B1 结果。

**主要文件**

```text
evals/configs/kite_b0.yaml
evals/configs/kite_b2.yaml
evals/configs/kite_b3.yaml
evals/kite_runner.py
artifacts/evals/kite/b0/
artifacts/evals/kite/b2/
artifacts/evals/kite/b3/
artifacts/evals/kite/summary.json
docs/kite_benchmark_report.md
docs/production_pipeline_decision.md
docs/implementation/m6c_kite_pipeline_acceptance.md
```

如果 `evals/build_report.py` 能自然承载 KITE 汇总，则扩展它；否则把小型汇总函数留在 `evals/kite_runner.py`。不为目录美观单独创建 report framework。

**人工检查**

- 检查每个候选相对 B1 的全部 win 和 loss。
- 核对引用的 paper、section、page 和 quote。
- 检查分数提高但 evidence 变差、延迟明显增加或回答更冗长的案例。
- 对照 M3.2 已有诊断，说明 B2/B3 的历史失败是否在 KITE 上再次出现。

**验收命令**

```bash
uv run --extra dev python -m pytest tests/test_kite_eval.py tests/test_retrieval_pipeline.py tests/test_retriever.py -q
uv run --extra dev ruff check evals indexing tests
uv run python -m evals.kite_runner run --config evals/configs/kite_b0.yaml
uv run python -m evals.kite_runner run --config evals/configs/kite_b2.yaml
uv run python -m evals.kite_runner run --config evals/configs/kite_b3.yaml
uv run python -m evals.kite_runner report --runs artifacts/evals/kite
git diff --check
```

**完成条件**

- 四条 Pipeline 使用同一冻结协议完成。
- 生成 summary、正式报告和生产决策。
- 未通过门槛时默认继续 B1。
- 通过门槛时只记录 promotion candidate，不自动切换产品默认值。

**回滚**

删除候选 artifacts 和报告即可。任何已批准的生产默认值变更都必须使用独立 commit，并通过 `git revert` 回滚。

### M6D：README 与只读评测展示

**进入条件**

- M6C 报告和生产决策冻结。
- 用户单独授权展示工作。

**目标**

把真实 KITE 结果、内部诊断和失败案例接入 README 与只读 `/evaluation` 页面。页面只读取提交的清洗报告，不触发模型调用或索引操作。

**主要文件**

```text
README.md
docs/research/m6_evaluation_lab_implementation_plan.md
web/src/app/(editorial)/evaluation/page.tsx
web/src/lib/types.ts
web/src/lib/api.ts
web/scripts/ui-contracts.mjs
docs/implementation/m6d_evaluation_presentation_acceptance.md
```

实际文件以当前 Next.js 路由和已安装文档为准。修改 Web 前必须读取 `web/AGENTS.md` 和当前 `web/node_modules/next/dist/docs/` 中相关文档。

**页面边界**

- 展示 KITE snapshot、模型、Pipeline、分数、延迟和逐题坏例。
- 展示 M3/M4 失败结论，但不提供参数编辑或 Pipeline 切换。
- 不显示 API key、prompt 全文、本地路径或未脱敏输入。
- 普通用户的 Library、Search、Paper 和 Chat 不受影响。

**验收命令**

```bash
uv run --extra dev python -m pytest -q
uv run --extra dev ruff check .
npm --prefix web run test:contracts
npm --prefix web run lint
npm --prefix web run build
git diff --check
```

**人工检查**

- 桌面和 375px 视口检查真实页面。
- 页面数值逐项对应冻结 JSON 和 Markdown 报告。
- 所有失败结论、模型名和数据版本没有被营销化改写。
- `/evaluation` 移除后，产品主路径仍完整可用。

**回滚**

移除只读入口和 README 结果段，不影响 Chat、Search、索引或实验 artifacts。

## 9. M7 项目设计与面试指南

M7 在 M6D 完成后单独授权。它只编写文档，不新增运行代码。

当前状态：已完成。本轮只实施 M7 文档，不新增运行代码。

指南以当前代码、M1 至 M6 验收报告和冻结 artifacts 为事实来源，覆盖：

- 用户问题和产品边界。
- parser、稳定标识和 evidence contract。
- B1 检索路径和 `quote_text` / `retrieval_text` 分离。
- KITE E2E 与内部 diagnostic 的职责。
- B2/B3/S1/Adaptive 的真实结果和失败决策。
- 产品、评测、测试、回滚和部署边界。
- 为什么不向普通用户暴露 Pipeline 开关。

所有简历和面试数字只能引用正式报告。未知结果保留为空或使用非量化描述，不能预填收益。

交付物：[`docs/m7_project_design_and_interview_guide.md`](../m7_project_design_and_interview_guide.md)，验收报告：[`docs/implementation/m7_project_design_and_interview_guide_acceptance.md`](../implementation/m7_project_design_and_interview_guide_acceptance.md)。

## 10. M8 部署边界

当前决定：M8 暂缓，不实施部署。除非用户重新授权，否则不新增部署代码、平台配置或云端资源。

部署需要明确平台、存储、密钥、网络和数据许可，因此不作为 M6 的完成条件，也不在未选平台时写通用部署脚手架。

M8 开始前必须决定：

- 部署平台和区域。
- FastAPI 与 Next.js 的进程模型。
- SQLite、uploads 和 immutable indexes 的持久磁盘。
- embedding、generation 和 judge 是否允许在部署环境访问。
- demo corpus 的来源和分发许可。
- 日志、备份、健康检查和恢复流程。

默认公开部署只使用可合法分发的 demo corpus，不打包 KITE PDF。KITE index 是否公开托管需要单独确认。

## 11. 配置与外部依赖

### 11.1 运行配置

所有产品配置继续通过 `AppSettings`。M6 的 dataset path、KITE root、实验输出目录和 judge 配置属于 eval config，不进入普通产品 Settings，也不调用 `os.getenv()` 绕过 `core/settings.py`。

### 11.2 依赖

M6 默认不新增 Python 依赖。使用现有 stdlib、PyYAML、LangChain、parser、FAISS、BM25 和 FlashRank。

外部条件：

- Git 和 Git LFS，用于获取 KITE corpus。
- KITE GitHub 仓库可访问。
- 当前项目已经支持的 embedding 和 LLM provider。
- 正式模型运行需要当前会话明确授权。

不得把 API key、完整环境变量、prompt 中的秘密或本地绝对路径写入 artifact、报告或提交。

## 12. 测试与报告规则

### 12.1 自动测试

所有外部模型调用在测试中使用 fake 或 monkeypatch，不发真实请求。每个新错误边界至少有一个回归测试。

KITE 测试必须覆盖：

- adapter、case ID、空 rubric 和非法字段。
- query/corpus hash。
- Git LFS pointer 和 PDF 文件头。
- judge score、无效输出、超时、一次重试和失败。
- Pipeline contract、generation error、judge error 和报告聚合。
- source-faithful evidence。

### 12.2 正式运行完整性

- 数据、prompt、阈值和 grader 在看到正式结果后不得修改。
- Smoke 不能用于调正式数据或 prompt。
- 正式失败保留原始报告，不能覆盖后重跑伪装成首次运行。
- 报告保存逐题 wins、ties、losses、错误、延迟和坏例。
- 任何非干净工作区结果都标记为非正式。
- 不能把确定性测试、fake 模型或 UI smoke 描述为完整 E2E Benchmark。

### 12.3 报告措辞

允许：

- “在固定的 15 题 KITE AI Papers snapshot 上，B2 相对 B1 的平均分变化为 X。”
- “该结果用于本项目 Pipeline 横向比较，样本较小，不声称统计显著。”
- “候选未通过延迟、证据或逐题门槛，因此继续使用 B1。”

禁止：

- “KITE 证明本系统全面领先。”
- “更复杂 Pipeline 一定更好。”
- “内部 Recall 提升等于答案质量提升。”
- 使用不同 judge 时直接对比 KITE README 的历史绝对分数。

## 13. 采用、保留和拒绝

### 13.1 采用

- KITE AI Papers 作为公开 E2E Benchmark。
- 内部 Retrieval Dataset 作为冻结 diagnostic benchmark。
- 现有 Pipeline Registry、config hash 和 index contract。
- 现有 fixed product path、结构化 evidence 和 Next.js 产品。
- B0 至 B3 的有限正式比较。
- 逐题差异、失败分析和保守生产门槛。

### 13.2 保留但不继续开发

- S1 固定策略候选。
- Adaptive graph、route、answer 和 claim validation 实验。
- M3/M4 冻结报告和坏例。

这些内容可以在只读评测页和项目指南中展示，不能成为普通用户配置。

### 13.3 拒绝

| 方案 | 原因 | 当前替代 |
|---|---|---|
| 通用 Benchmark Protocol | 当前只有一个公开 E2E 数据集，属于提前抽象 | 最小 `KiteCase` 和 runner |
| Internal Adapter 重写 | 现有 runner 和冻结报告已经有效 | 保留现状并引用冻结结果 |
| RAGAS 全量依赖 | 增加框架面，不能解决本轮 KITE 契约问题 | 官方兼容 judge + 项目可选诊断 |
| Adaptive 复验 | 两次复验未证明净收益 | 固定 B0 至 B3 比较 |
| 自动 Pipeline 搜索 | 样本小，容易过拟合 | 预先冻结有限候选和门槛 |
| 新 parser / embedding / reranker | 会同时改变多个变量 | 保持当前 contract |
| 实时 evaluation dashboard | 增加运行和安全边界 | 提交后的静态只读页面 |
| 直接公开 KITE PDF | 再分发权未确认 | 外部获取、hash 和私有缓存 |

## 14. 实施授权边界

- 本文件只合并和批准开发方向，不代表 M6A 已获实施授权。
- M6A、M6B、M6C、M6D 按顺序单独授权，完成一个后停止。
- M6B 和 M6C 的真实 embedding、generation 和 judge 调用需要当前会话明确授权，不能继承旧会话许可。
- M6C 不能自动修改 Production Pipeline。
- M6D 不能在 M6C 报告冻结前开始。
- M7 和 M8 分别授权，不与 M6 捆绑。
- 每个 Goal 开始前记录分支、HEAD 和工作区，保护用户已有未提交文件。
- 每个 Goal 结束后提交独立验收报告和坏例，不自动继续下一阶段。

## 15. V2 完成口径

V2 当前已经是可用的 evidence-first 论文 RAG 产品。M6 完成后，项目再增加可公开复现的工程决策证据：

```text
KITE AI Papers
      +
Internal Retrieval Diagnostics
      |
      v
Fixed Pipeline comparison
      |
      v
Production decision
      |
      v
FastAPI + Next.js evidence-first product
```

M6D 完成时，必须同时满足：

- KITE snapshot、query 和 corpus 有固定 commit 与 hash。
- B1 有完整 15 题 E2E 基线。
- B0、B1、B2、B3 使用同一协议完成比较。
- Production Pipeline 有明确、可审计的保留或晋级决策。
- README 和只读页面只展示真实冻结结果。
- 普通用户仍只看到论文库、搜索、Chat 和证据，不看到技术 Pipeline 开关。
- 全部失败案例和未通过候选被如实保留。

公开叙事固定为：

> Agentic RAG 是一个面向学术论文的 evidence-first RAG engineering project。系统完成论文解析、混合检索、重排、上下文组装、grounded answer 和 citation 链路；KITE AI Papers 评价端到端回答质量，内部 Retrieval Dataset 解释检索差异，冻结实验共同决定固定 Production Pipeline，并通过 FastAPI 与 Next.js 提供论文检索和问答产品。

这段叙事只能在对应结果真实完成后使用。M6 尚未完成时，README 和简历不得把 KITE 集成、KITE 分数或生产决策写成已交付能力。
