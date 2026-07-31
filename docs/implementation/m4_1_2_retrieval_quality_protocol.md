# M4.1.2 Adaptive 场景对齐复验协议

M4.1.2 只检验冻结 B1 上「首轮部分覆盖、一次定向补检可补齐」的策略质量。它不改写 M4.1.1，也不授予 M4.2 进入资格。

## 冻结和可比性

- B1 固定为 `v1_flat_rerank`，config hash 为 `ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17`。
- authoring 仅执行一次只读 B1 snapshot；没有调用 Adaptive，active index 未变。
- route 48 条，四类各 12；answer 24 条，含 12 adaptive-eligible、8 fixed-eligible、4 evidence-insufficient。
- fixed/adaptive 对每条 answer case 使用完全相同的 query、history、scope、B1 index 和 task models。fixed 只作完整原问题的一轮 B1。

## 有界执行

运行时先拆分最多 3 项 requirements，但第一轮永远以完整原问题检索。仅 assessor 对实际 returned evidence 标出未覆盖 requirement 时，才能生成一次仅面向这些 requirement 的不同 query；不能读取 case ID、标签、gold 或 snapshot。

上限是 2 rounds、4 tool calls、12 evidence、12,000 context tokens。重复 query+scope、evidence IDs 无变化、coverage 无提升、预算耗尽、取消、模型错误或检索错误立即停止；第二轮后只回答有限事实或 refuse。

## 评分

确定性层校验 evidence 是否来自本 run，以及 ID、quote、paper、section、page、index version 是否完整。语义层由冻结结构化 grader 在 claim/quote 上判断直接支持；布尔值和文字理由冲突时只记录 `grader_inconsistent`，不自动修复。gold evidence 仅用于独立覆盖审计，主分使用「确定性有效 + 语义支持 + requirement 映射」，避免等价可定位 evidence 被 exact-ID 规则系统性判零。

自动结果固定后，随机盲审不少于 `ceil(24*20%)=5` 条 answer case；只报告 false positive、false negative 和 inconsistent，不改变正式分数。

## 门槛

- route 每类 recall ≥ 0.75，macro F1 ≥ 0.80；
- adaptive-eligible 至少 5 条 coverage 改善、退化不超过 2；
- fixed-eligible 报告误触发率，citation/support 不低于 fixed；
- 总体 citation correctness/completeness/major-fact support 不低于 fixed，unsupported major claims 不更高；
- successful termination=100%，平均 rounds≤1.5，tool calls≤4，duplicate query+scope=0；延迟只记录。
