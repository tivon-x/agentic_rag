# M7 项目设计与面试指南验收报告

状态：正式验收通过（2026-08-31）。

## 范围

M7 只交付文档，不新增运行代码，不改变生产 Pipeline、answer strategy、index、数据库 schema 或外部模型配置。

## 交付

- [`docs/m7_project_design_and_interview_guide.md`](../m7_project_design_and_interview_guide.md)：项目定位、架构、parser、稳定 ID、evidence contract、B1 路径、KITE 与内部诊断职责、B2/B3/S1/Adaptive 真实结果、产品边界、测试、回滚、部署边界和面试问答。
- [`docs/research/v2_upgrade_plan.md`](../research/v2_upgrade_plan.md)：将 M7 标记为已完成，M8 标记为暂缓、不实施，并登记本验收报告。
- [`README.md`](../../README.md)：删除已过期的“M7 尚未实施”状态，并增加 M7 指南入口。
- [`tests/test_project_status.py`](../../tests/test_project_status.py)：移除固定的 M6D/M7 状态断言，改为校验状态字段类型和 acceptance evidence 存在，避免下一次里程碑推进时测试复制旧事实。

## 事实与边界检查

- 生产默认仍是 `b1 / v1_flat_rerank`，`ANSWER_STRATEGY=fixed`。
- KITE 只引用冻结的 15 题、134 个 PDF、upstream commit、query/corpus hash 和正式 B0 至 B3 报告。
- B2、B3、S1 和 Adaptive 的失败或未晋级原因均链接到对应验收报告，没有把诊断结果写成线上收益。
- 未知结果保留为非量化表述，没有预填个人简历数字、成本收益或统计显著性。
- 没有把 OCR、Web search、公式语义解析或 M8 部署写成已实现能力。
- 没有修改用户已有的未跟踪文件 `docs/KITE Benchmark 驱动的完整 RAG Engineering Project 执行方案.md` 和 `web/output/`。

## 验证

```text
uv run pytest tests/test_project_status.py -q
通过

git diff --check
通过

文档链接检查
通过，所有仓库内相对链接目标存在

中文标点检查
通过，M7 指南和验收报告无破折号、半角标点或中英混排问题
```

本 Goal 没有运行外部模型、embedding、judge、FastAPI 或 Next.js build，因为 M7 不改变运行代码。M8 已标记暂缓、不实施；如恢复需单独规划和授权。

## 坏例与处理

- README 和状态测试原先会继续写出 M7 未实施或把 M6D 当作最新完成目标，已同步修复，避免事实漂移。
- B2/B3 的平均分更高但未通过晋级门槛，指南保留逐题 loss 和 production decision，不写成“更复杂就更好”。
- Adaptive 的安全预算通过但质量失败，指南明确区分安全边界和质量结论。
- KITE 的本项目 judge 分数不与上游绝对分数比较，避免跨 judge 误读。

## 回滚

M7 没有运行时代码和数据迁移。若需回退文档，只回退本 Goal 的文档、状态和 drift guard 改动，不触碰用户已有未跟踪文件；恢复时保留 M1 至 M6 验收报告和冻结 artifacts。
