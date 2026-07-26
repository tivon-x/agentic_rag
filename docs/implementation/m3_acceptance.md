# M3 验收报告：固定 V2 检索与精简评测

- **日期：** 2026-07-26
- **分支：** `codex/v2-core`
- **结论：** M3 实现与可复现实验已完成；Core 未通过发布门槛。
- **默认 pipeline：** 保持 `B1 / v1_flat_rerank`。
- **M4：** 不具备进入条件，不得自动执行。

## 1. 进入条件与范围

开始前已读取 V2 升级方案第 6、10、11 节、phase1 研究报告和 M1/M2
验收报告。`docs/implementation/m2_acceptance.md` 的结论为通过，并重新执行：

```text
uv run python -m evals.parser_eval --dataset evals/datasets/parser_v2.json
```

结果：16 篇、48 个重点页，10 个 parser gate 全部通过；冻结 parser gold
SHA-256 为
`aca531e7032da25ce0ba955bb815a6fe3e9281fd232bb3267a694c569ab54d29`。

本次只实现固定检索和 Core 评测，没有增加 query routing、multi-query、
自纠错循环、claim validation、run worker 或详细 Agent trace。

## 2. 实现结果

- `retrieval_text` 与 `quote_text` 分离。metadata prefix 只进入 FAISS/BM25
  检索表示；context、离线回答和引用始终读取 `quote_text`。
- 新增中英 mixed tokenizer：NFKC 归一化、英文/缩写/连字符/数字保留，
  中文使用 jieba search mode；索引和查询 tokenizer 不一致立即失败。
- 新增固定 registry：B0、B1、B2、B3 和五个 B2 单因素消融均由同一配置
  模型生成，不在 runner 内拼接临时参数。
- B0：quote-only dense + whitespace BM25 + min-max，无 rerank。
- B1：当前 quote-only flat_rerank。
- B2：metadata-prefixed dense + mixed BM25 + RRF(k=60) + FlashRank。
- B3：B2 后先保留 8 个 core seed，再追加同 section 的邻居，最多 12 条；
  邻居不再挤掉 core rank。
- recall、fusion、rerank、expansion 和 context packing 全部写入逐题
  `stage_results`，各阶段延迟写入 `stage_timings_ms`。
- embedding manifest 与 retrieval contract 在索引加载和查询前强校验。
  provider 返回的 batch 上限 20 已纳入 AppSettings 和 manifest。
- 正式默认仍是 `v1_flat_rerank`；registry 中存在 B2/B3 不代表启用。

## 3. 冻结实验合同

| 项目 | 冻结值 |
| --- | --- |
| Parser artifact | 25 篇、8,315 passages |
| Parser artifact SHA-256 | `98e8adf680c578c21d2fffe5b97f3f85d24b768b827fe81aa8ddfc280af242d9` |
| Retrieval dataset | 48 条，四类各 12 条 |
| Retrieval dataset SHA-256 | `e1da7d23d352cd17a1601f56280a5c9820ff81002a36dc5ad786cb3a8f90c936` |
| Answer smoke | 8 条，仅检查检索回答和 prefix 泄漏 |
| Answer smoke SHA-256 | `fc62a4e39cae6eaa329a07121cc2213d4ccdcca2979cda3ae22c9db0bd314122` |
| Embedding provider/model | `openai` / `qwen3.7-text-embedding` |
| Dimension / batch | 1024 / 20 |
| Input mode | `raw`, `check_embedding_ctx_length=false`, max 6000 chars |
| Reranker | FlashRank `ms-marco-TinyBERT-L-2-v2` |
| Recall / rerank / final top-k | 40 per source / 30 / 8 |
| Context | 最多 12 passages / 8000 tokens |
| B1 config SHA-256 | `f978f0898162c33318d0f94b9b322b0c4d439e5cecb1aba47c5933d89b0ea8c2` |
| B2 config SHA-256 | `7d4510af75155fdb4e36a0c58e13217d69aec76f527d5d3e33579837be4c31d0` |
| B3 config SHA-256 | `10579788f06b29a4bf0136e6ecba1664f54afc3c5852074b64bd755623f989a9` |
| Evaluated base commit | `03b2c5dc343b3a7459377adfc4c20e2fad22aece` + recorded dirty tree |

B0、B1、B2、B3 均从同一 parser artifact 独立重建真实 FAISS/BM25。
没有使用 fake embedding 或历史索引。四个不改变索引表示的 B2 消融
（去 sparse、去 dense、min-max、去 rerank）复用本轮刚生成的 B2 表示，
各自写独立 manifest；去 metadata 使用单独重建的 quote-only/mixed 索引。

报告和索引保存在 `artifacts/evals/v2_core`，该目录不提交。最终交付 commit
在完整评测之后创建；run report 保留评测时的 base commit 和 dirty 标志，
使“评测发生在提交前”可审计。

## 4. Core 指标

| Pipeline | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Paper Recall@10 | Section Recall@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.479167 | 0.656250 | 0.395213 | 0.430818 | 0.958333 | 0.770833 | 238.2695 | 329.6515 |
| B1 | 0.489583 | 0.614583 | 0.340327 | 0.388849 | 0.968750 | 0.781250 | 344.2906 | 458.8358 |
| B2 | 0.479167 | 0.510417 | 0.329712 | 0.358562 | 0.937500 | 0.770833 | 367.7745 | 432.2989 |
| B3 | 0.479167 | 0.520833 | 0.331796 | 0.362254 | 0.937500 | 0.770833 | 389.9468 | 474.1078 |

没有计算或使用统一综合分。

### 子集 Recall@10

| Pipeline | 术语/定义 | 方法/章节 | 数值/表格 | 跨论文/章节 |
| --- | ---: | ---: | ---: | ---: |
| B1 | 0.666667 (8/12 全命中) | 0.750000 (9/12) | 0.500000 (6/12) | 0.541667 (3/12 全命中) |
| B2 | 0.500000 (6/12) | 0.583333 (7/12) | 0.500000 (6/12) | 0.458333 (2/12 全命中) |
| B3 | 0.500000 (6/12) | 0.583333 (7/12) | 0.500000 (6/12) | 0.500000 (2/12 全命中) |

## 5. 发布门槛

B1→B2 逐题首个 gold rank：**10 胜 / 20 平 / 18 负**。

| B2 gate | 结果 | 证据 |
| --- | --- | --- |
| Recall@10 不低于 B1 | 失败 | 0.510417 < 0.614583 |
| 至少 8 条改善 | 通过 | 10 |
| 退化不超过 4 条 | 失败 | 18 |
| 任一子集不下降 2 条以上 | 失败 | 术语、方法均少 2 条全命中 |
| p95 不超过 B1 1.5 倍 | 通过 | 432.2989 / 458.8358 = 0.942165 |

B2 gate 失败，因此 Core 未通过。

B2→B3 为 **1 胜 / 47 平 / 0 负**。唯一改善是
`cross-12-coffee-complexity`，B3 通过 section neighbor 在 rank 10 补入
第二段 gold。跨章节改善只有 1 条，低于 3 条；B3 不通过，且 B2 gate
本身也未通过。

## 6. B2 单因素消融

| Variant | Recall@10 | Δ Recall@10 vs B2 | MRR@10 | nDCG@10 | p95 ms | 相对 B2 W/T/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 去 metadata prefix | 0.604167 | +0.093750 | 0.332316 | 0.381331 | 429.6465 | 16/21/11 |
| 去 sparse | 0.500000 | -0.010417 | 0.312004 | 0.344371 | 402.9356 | 6/38/4 |
| 去 dense | 0.489583 | -0.020834 | 0.326240 | 0.348923 | 205.0424 | 4/39/5 |
| RRF→min-max | 0.510417 | 0.000000 | 0.333011 | 0.365403 | 426.4266 | 4/41/3 |
| 去 rerank | 0.635417 | +0.125000 | 0.401587 | 0.435952 | 324.6915 | 18/20/10 |

结论只适用于本冻结合同：

- sparse 和 dense 各自有小幅净贡献。
- RRF 与 min-max 的 Recall@10 持平，min-max 的 rank 指标略高。
- metadata prefix 在当前构造与模型下出现净退化，不能宣称收益。
- 最大问题是当前 FlashRank：去 rerank 的 Recall@10、MRR、nDCG 和延迟
  均优于正式 B2。不能把该消融直接改名为 B2；它只说明下一轮需要重新
  冻结 reranker/表示方案后再评测。

## 7. 人工 rank 变化检查

至少检查了以下 10 个 B1/B2 变化案例：

| Case | B1→B2 | 人工检查 |
| --- | --- | --- |
| term-02-multi-head | 4→未命中 | gold 进入 dense top-5，但最终 rerank 未进入 top-8 |
| term-12-rnn-dropout | 5→1 | mixed BM25 对中英混合精确短语有效 |
| method-01-transformer-encoder | 6→2 | title/section prefix 有效 |
| method-03-resnet-bottleneck | 4→未命中 | gold 被 rerank 挤出 |
| method-08-deepspeech-bn | 未命中→1 | 章节号、标题和缩写共同改善 |
| number-01-transformer-bleu | 2→4 | 仍命中，但表格/摘要候选次序退化 |
| number-06-nmt-bleu | 未命中→1 | mixed sparse 精确匹配 RNNsearch-50/34.16 |
| cross-02-resnet-identity | 3→1 | prefix 改善其中一篇，仍只覆盖一半 gold |
| cross-05-rn-rmc | 未命中→2 | 命中 Relation Network gold，RMC gold 仍缺失 |
| cross-12-coffee-complexity | 3→未命中 | 两段 gold 都被正式 B2 rerank 排除 |

### 四类坏例检查

阶段 rank 是 gold 在 dense / sparse / fused / rerank 中的最优位置；
`—` 表示该阶段未进入候选。最终只打包 top-8。

| 检查类 | Case | Dense | Sparse | Fused | Rerank | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 表格 | number-02-alexnet-error | 13 | 9 | 9 | 10 | 接近 top-8，仍被截断 |
| 表格 | number-03-resnet-error | 2 | 12 | 3 | 14 | rerank 明确退化 |
| 表格 | number-07-relation-clevr | 16 | — | 34 | 40 | recall 与 rerank 都弱 |
| 缩写 | term-02-multi-head | 5 | 29 | 14 | 10 | mixed sparse 未带来前列命中 |
| 缩写 | term-10-mpnn | 1 | 2 | 1 | 14 | rerank 明确退化 |
| 缩写 | method-06-rmc | 4 | 16 | 6 | 21 | rerank 明确退化 |
| 跨章节 | cross-07-gpipe-method-result | 9 | 2 | 5 | 3 | 只覆盖 1/2 gold |
| 跨章节 | cross-08-scaling-allocation | 13 | 32 | 19 | 3 | 只覆盖 1/2 gold |
| 跨章节 | cross-10-mdl-intro-selection | 14 | — | 33 | 52 | 第二章节未召回 |
| 中文术语 | term-01-transformer | 3 | — | 24 | 30 | dense 命中后被融合/重排压低 |
| 中文术语 | term-04-positional-encoding | 2 | 3 | 2 | 19 | rerank 明确退化 |
| 中文术语 | term-07-pointer-network | 4 | — | 19 | 30 | sparse 缺失且 rerank 退化 |

逐题结果见 `docs/implementation/m3_per_question.md`。完整坏例、每个候选及
各阶段分数保存在 `artifacts/evals/v2_core/core_report.json` 和三个
`report.json`。

## 8. Answer smoke 与引用

8 条 answer smoke 不能代替正式 answer test。B0、B1、B2、B3 的
`metadata_prefix_leak_count` 均为 **0**。人工检查回答预览和 context，
未发现 `[TITLE]`、`[AUTHORS]`、`[YEAR]`、`[SECTION]`、`[BLOCK]`
进入引用正文。

## 9. 验证

```text
uv run --extra dev python -m pytest tests/test_bm25_index.py tests/test_retriever.py tests/test_retrieval_pipeline.py tests/test_evals.py -q
# 44 passed

uv run python -m evals.runner --config evals/configs/v2_b1.yaml
uv run python -m evals.runner --config evals/configs/v2_b2.yaml
uv run python -m evals.runner --config evals/configs/v2_b3.yaml
uv run python -m evals.build_report --runs artifacts/evals/v2_core
# core_passed=false, default_pipeline=b1, m4_entry_ready=false

uv run --extra dev ruff check indexing core evals tests
# passed

uv run --extra dev python -m pytest -q
# 222 passed

cd web && npm run lint
# passed

cd web && npm run build
# passed
```

## 10. 默认与回滚

- 默认保持 `RETRIEVAL_PIPELINE=v1_flat_rerank`（别名 B1）。
- 不激活 B2/B3，不修改 SQLite active index pointer。
- M3 引入 retrieval contract 后，旧 manifest 没有该字段会明确要求重建，
  不会静默加载不兼容索引。
- 代码回滚：回退本次独立 M3 commit。
- 配置回滚：设置 `RETRIEVAL_PIPELINE=v1_flat_rerank`，使用相同 embedding
  合同重新构建 B1 index；需要回到已有 ready version 时使用
  `python main.py activate-index <version>`，前提是该 version 的 embedding
  与 retrieval contract 完全匹配。
- 本结论不批准 M4。若要继续 Enhanced，必须由用户再次批准，并先针对
  metadata prefix 与 reranker 的负贡献提出新的冻结方案。
