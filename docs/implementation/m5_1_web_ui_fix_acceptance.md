# M5.1 Web UI 修复验收报告

日期：2026-08-16  
分支：`codex/v2-core`  
范围：`docs/implementation/m5_1_web_ui_review_findings.md` 中 UI-01 至 UI-09  
结论：`Approve`，UI-01 至 UI-09 已关闭

## 实际修改

- UI-01：`agent/nodes/aggregate_answers.py`、`api/routers/chat.py`、`agent/states.py`、`web/src/components/ChatExperience.tsx`；失败回答不再作为 assistant 成功回答追加或持久化，保留用户问题并可重试。
- UI-07：`agent/tools.py` 的既有检索元数据进入 `retrievalEvidence`/`evidenceGroups`，`api/routers/chat.py` 只从检索证据绑定 `paper_id`；`web/src/lib/evidence.ts` 统一论文页链接。
- UI-02/03/04/06：`web/src/components/ChatExperience.tsx`、`web/src/app/globals.css`、`web/src/components/ui/input.tsx`、`web/src/components/ui/button.tsx`。
- UI-05：`web/src/lib/errors.ts` 集中错误映射；聊天、论文库、搜索和论文详情使用中文恢复文案。
- UI-08：`web/src/lib/assistant-display.ts` 和 Chat 展示层拆分正文/补充信息并做无依赖公式纯文本降级。
- UI-09：Chat 长等待定时器、阶段说明和清理逻辑。
- 回归：`tests/test_agent_grounded_answer.py`、`tests/test_chat_evidence.py`，以及无依赖的 `web/scripts/ui-contracts.mjs`（`npm --prefix web run test:contracts`）。
- 其他当前工作树中的 UI-01 相关既有修改（Qwen 适配、路由消息边界和测试）按 Goal 要求保留，未 reset、checkout 或 stash。

## API、SSE 和持久化兼容性

- `ChatMessage.role`、`content` 保持不变，`evidence` 仍为可选字段；旧会话缺少 evidence 时继续可读。
- `paper_id` 只接受目录检索元数据；模型无法创造目录 ID。缺少目录 ID 的证据保留为不可用状态，不猜测标题或文件名。
- 正常 SSE 顺序仍为 `progress → evidence → answer.final`；生成、检索或保存失败只发 `stream-error`，不发 `answer.final`，不写 assistant 成功消息。
- SQLite JSON 会话结构和数据库 schema 未变；两轮回答的证据仍按 assistant message 分组。
- 未改变固定检索策略、检索参数、Graph 拓扑、索引格式、模型供应商、模型调用策略或 SSE 连接模型。

## 自动化验证

| 命令 | 结果 |
|---|---|
| `uv run --extra dev python -m pytest -q` | 295 passed，3 warnings |
| `uv run --extra dev ruff check .` | 通过 |
| `npm --prefix web run test:contracts` | 通过 |
| `npm --prefix web run lint` | 通过 |
| `npm --prefix web run build` | 通过；Next.js 16.2.0，全部路由构建成功 |
| `git diff --check` | 通过 |

contract checks 覆盖错误映射、未知错误兜底和标点、回答分段、置信度、常见行内公式、焦点/inert、长等待定时器、2px 焦点环、16px 输入字号和 `active:scale-[0.96]`。

## 真实浏览器验收

- 地址：`http://localhost:3000`；真实 FastAPI + Next.js 代理，demo 目录包含 25 篇论文和《Attention Is All You Need》。外部模型调用使用当前会话授权。
- `1280×720`：两轮真实问题分别得到回答 01（4 条证据）和回答 02（2 条证据）；桌面证据面板按轮次切换，刷新和离开后重开会话仍恢复正文、证据数量和链接。
- 逐条展开并打开两轮共 6 条证据链接；论文标题均为《Attention Is All You Need》，链接中的同一 `paper_id` 与 P.4/P.5/P.9 页码一致，论文详情页 PDF 页码同步。
- `375×812`：移动证据列表、回答说明折叠区、会话抽屉键盘流；打开后焦点在关闭按钮，Tab/Shift+Tab 循环，Escape 恢复“会话”触发按钮。
- `320×700`：聊天 textarea 与搜索 input 计算字号均为 16px，`scrollWidth === clientWidth`。
- `1024×720`：证据模态层打开后焦点进入关闭按钮，Tab/Shift+Tab 不离开，Escape 恢复证据触发按钮并关闭遮罩。
- 输入框键盘聚焦时真实计算轮廓为 2px solid 墨蓝。
- 回答持续超过 10 秒时出现“正在检索论文并整理证据，可能需要 1 至 2 分钟”，回答完成后消失。
- 通过浏览器内确定性 EventSource 模拟生成失败、检索失败、保存失败：均保留问题、显示中文恢复文案和“重试回答”；对应 API 会话只持久化 user message。

## 独立只读审查

审查员：独立 `luna_worker`（不得修改文件），范围覆盖 UI-01 至 UI-09、API/旧会话兼容、paper_id 来源、错误状态、证据轮次、键盘焦点、移动适配和定时器。审查输出按 `.agents/skills/interface-review` 与 `.agents/skills/better-interface` 的 change-scoped 格式完成。初审发现三项修复后问题：未知中文错误透传、桌面悬空 `aria-controls`、证据模态焦点陷阱漏掉原生 `summary`；另补齐移动证据缺 `paper_id` 的不可用文案。全部已修复并复验。最终只读 follow-up verdict 为 `Approve`，未发现 UI-01 至 UI-09 仍需处理的范围内问题。

## 遗留项与回滚

- 本 Goal 范围内不保留 UI-01 至 UI-09 遗留项；未覆盖屏幕阅读器真实辅助技术组合测试，属于环境限制，代码保留语义 `role`、`aria-live`、`aria-modal`、`details/summary` 和 focus 管理。
- 本次尚未推送远端或创建 PR。最终交付使用一个独立 commit；需要回滚时执行 `git revert <本次 commit>`，不使用 `git reset --hard`。
