# M4.1.1 检索质量复验协议

## 定位

M4.1.1 是用户批准的独立复验，不覆盖 M4.1 的失败验收，也不会自动使 M4.2 可进入。它只检验：冻结 B1 上的逐需求证据缺口判断和一次定向补检，是否改善最终主要事实的可支持性。

## 不可变边界

- 每次事实检索验证并使用 `v1_flat_rerank`，config hash 为 `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17`。
- 不修改 dense、BM25、fusion、reranker、top-k、context packing、embedding 或 active index。
- 不重跑或引用 M3.2 holdout；已打开的 M4.1 `v1` 数据集不作为本次最终集。
- route/answer 数据、gold、可接受 evidence 集、评分语义和阈值见 `m4_1_1_dataset_manifest.json`；其 SHA-256 是冻结标识。
- 不增加 migration、worker、checkpoint、SSE 或前端模式；`ANSWER_STRATEGY` 默认仍为 `fixed`。

## 可比执行

fixed 和 adaptive 使用同一问题、history、scope、B1 index、模型与最终 `AdaptiveAnswer` schema。正式复验的规划、充分性判断、回答和 claim 评分统一固定为 `qwen3.7-max-2026-06-08`；该替换只改善证据判断与回答质量，不改变 B1 检索、embedding 或索引。

- fixed：原问题做一轮 B1，基于该轮可验证 evidence 生成带 claim IDs 的回答。
- adaptive：最多拆 3 个 requirements；首轮 B1 后逐项判断。仅当有缺失项时，再做一次只覆盖缺失项的 B1 补检。
- 最多 2 轮、4 次 retrieval、12 条 evidence、12,000 context tokens。重复 query+scope、evidence ID 无变化、coverage 无提升、预算耗尽、取消或错误立即停止。

## 评分

每个 final major claim 的 `evidence_ids` 与冻结 `claim_specs` 比对；只有引用匹配该 claim 的可接受 evidence 集才计为 supported。requirement coverage 也从最终 claim 计算，而不是从“检索到过的所有 passage”计算。确定性验证仍检查 evidence ID、页码、quote 和 index version；quote 的语义支持由结构化模型判断，并单独报告其误判，不能把它宣称为确定性证明。

## 质量门槛

- adaptive 比 fixed 至少多覆盖 5 个 case，退化不超过 2 个。
- citation correctness、citation completeness、major-fact support-rate 不低于 fixed；unsupported major claim count 不高于 fixed。
- successful termination rate 为 100%，总轮数不超过 2，每题 tool calls 不超过 4，重复 query+scope 为 0。
- 延迟、token、轮数和工具调用全部记录，但 latency 不作为质量晋级门槛。

## M4.2 边界

现有 `docs/research/v2_upgrade_plan.md` 仍把原 M4.1 p95 写为 M4.2 前置硬条件。因此本复验即使通过，也只证明检索质量；是否修订升级计划并允许 M4.2，必须由用户另行批准。
