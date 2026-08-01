# M5 固定 RAG Web 应用实施计划

> 状态：待用户授权。本文是 `docs/research/v2_upgrade_plan.md` 中 M5 的详细执行手册。
>
> 执行模型：`gpt-5.6-luna`，reasoning effort=`max`。

## 1. 目标与完成定义

M5 将现有的论文库、搜索、PDF 阅读和 Chat 收成一个可日常使用、可在面试中演示的固定 RAG Web 应用。产品只使用冻结的 B1 `v1_flat_rerank`，不向用户暴露未经验证的检索策略。

完成时必须同时满足：

1. 全站落实根目录 `DESIGN.md` 的证据导向视觉系统。
2. 每条 Chat assistant 消息可以携带自己的结构化 evidence，刷新和重新进入会话后不丢失。
3. evidence rail 能显示来源、论文名、章节、页码、摘录和论文跳转链接。
4. Search 到 PDF 阅读再到 Chat 的信息层级和视觉语言一致。
5. 不改变 B1 检索结果、active index、数据库 schema、模型调用方式和普通用户的策略选择。
6. 通过自动验证、真实浏览器视觉检查和独立 review，修复所有可修复问题后再提交。

## 2. 已有基础与修改边界

### 2.1 现有基础

| 领域 | 当前入口 | M5 如何复用 |
| --- | --- | --- |
| Chat 会话 | `api/routers/chat.py`、`api/models/chat.py`、SQLite `chat_sessions.messages` | 继续使用 JSON 会话历史，不做 migration。 |
| Chat SSE | `GET /api/chat/stream` 与 `web/src/hooks/useSSEStream.ts` | 保持 progress、evidence、answer.final、error 四类事件。 |
| 结构化证据 | `agent/schemas.py` 的 `EvidenceItem`、`GroundedAnswer`，`agent/tools.py` | 让 paper_id 和 paper_title 随最终 evidence 进入 API。 |
| 检索与 PDF | `api/routers/search.py`、`web/src/app/search/page.tsx`、`web/src/app/papers/[id]/page.tsx` | 不改检索，复用 `/papers/{id}?page={page}` 的页面定位。 |
| 前端基础 | `web/src/app/globals.css`、`layout.tsx`、`components/ui/*` | 用既有 Tailwind 和 CSS 变量，不新增依赖。 |

### 2.2 明确不做

- 不实施 Adaptive、M4.2、持久 run、checkpoint、队列、技术调试工作台或检索策略下拉框。
- 不新建数据库表、迁移、API Key、外部服务和前端依赖。
- 不重新评测、不调用真实模型作为验收步骤、不改 M3/M4 结果。
- 不承诺 PDF bbox 高亮。页码跳转已足够可信。
- 不把 `citations_markdown` 当作新前端的主数据源。

## 3. 必须先完成的读取与门禁

执行前依次读取：

1. 根目录 `AGENTS.md`、`web/AGENTS.md`、`DESIGN.md`。
2. `docs/research/v2_upgrade_plan.md` 的 M5、M6、实施授权边界。
3. `docs/research/phase2_goal_prompts.md` 的 Goal 5。
4. `api/models/chat.py`、`api/routers/chat.py`、`api/db/database.py`、`core/rag_answer.py`。
5. `agent/schemas.py`、`agent/tools.py`、`api/db/papers.py` 中 passage metadata 的构造。
6. `web/src/app/layout.tsx`、`globals.css`、首页、Library、Search、Paper、Chat、`CitationAccordion.tsx`、`useSSEStream.ts`、`lib/api.ts`、`lib/types.ts`。
7. `tests/test_api.py`、`tests/test_streaming.py` 与所有直接受影响测试。
8. 修改 Next.js 前读取当前安装版本 `web/node_modules/next/dist/docs/` 中 App Router、Linking and Navigating、Server and Client Components 的文档。

开始前执行并记录：

```bash
git status --short
git log -1 --oneline
```

工作区有非本 Goal 文件时不得覆盖它们。不存在 B1 active index 时，界面仍可展示空状态和错误状态，但不得为了演示创建索引。

## 4. 冻结的接口契约

### 4.1 `ChatEvidence`

在 `api/models/chat.py` 和 `web/src/lib/types.ts` 定义一致的可选 evidence 项。字段如下：

| 字段 | 类型 | 来源与规则 |
| --- | --- | --- |
| `node_id` | string | 稳定 passage/node ID。 |
| `paper_id` | string 或 null | catalogued paper 才有，用于路由。 |
| `paper_title` | string 或 null | passage metadata 的 `paper_title`。 |
| `source` | string | 文件名或来源标签，始终提供兜底值。 |
| `section_path` | string[] | 保留层级，不在 API 中拼接。 |
| `page` | integer 或 null | 正整数页码才接受。 |
| `quote` | string | 仅来源原文，不由前端补写。 |
| `score` | number 或 null | 可展示但不是主要视觉信息。 |
| `relevance` | string 或 null | 模型已有解释才展示。 |

`ChatMessage` 只增加 `evidence: ChatEvidence[] | None`。缺失字段必须兼容旧会话。用户消息和系统消息不写 evidence。对 HTTP response 使用 exclude-none，保持旧客户端得到的用户消息 JSON 不出现无意义的 `evidence: null`。

### 4.2 SSE 与持久化顺序

```text
POST /api/chat
  -> 保存 user message
GET /api/chat/stream
  -> progress
  -> 取得 fixed graph 或 offline retriever 的最终 answer 与 evidence
  -> 在同一次会话消息 upsert 中保存 assistant content + evidence
  -> evidence 事件
  -> 唯一一次 answer.final
GET /api/chat/{session_id}
  -> 返回同一份已保存的 assistant evidence
```

- graph 路径从 `groundedAnswer.evidence` 规范化 `ChatEvidence`。
- offline 路径从 returned documents 的 metadata 构造相同字段，最多取与当前答案有关的有限条目，quote 截断规则与现有 evidence 规则一致。
- `evidence` SSE payload 增加 `evidence: ChatEvidence[]`。保留 `citations_markdown` 为可选兼容字段，不让新 UI 依赖它。
- `answer.final` 仍只发一次。允许携带同一 evidence，便于前端在事件顺序竞争时绑定正确回答。
- 网络、图、检索或保存失败时不写半条 assistant message，也不发送伪造 evidence。

### 4.3 论文跳转

仅当 `paper_id` 存在时生成链接：

```text
/papers/{encodeURIComponent(paper_id)}?page={page}
```

`page` 缺失时只跳论文详情页。不要链接 source 文件路径，也不要暴露本地绝对路径。

## 5. 分阶段实施

每一阶段可以独立提交和回滚。任何阶段完成后都不自动开始 M6。

### 阶段 A：全站设计基线

**修改范围**

- `web/src/app/globals.css`
- `web/src/app/layout.tsx`
- `web/src/components/ui/button.tsx`
- `web/src/components/ui/card.tsx`
- `web/src/components/ui/input.tsx`
- `web/src/components/ui/textarea.tsx`
- `web/src/components/FileUpload.tsx`
- `web/src/app/page.tsx`、`library/page.tsx`、`search/page.tsx`、`papers/[id]/page.tsx`

**实施要求**

1. 将 `DESIGN.md` token 写入现有全局 CSS：纸面背景、墨黑文本、细规则线、墨蓝 evidence 色、警告与失败色。
2. 把 masthead 改为紧凑出版物式导航。保留跳到正文链接和现有路由。
3. 页面标题使用指定 serif fallback，中文正文使用系统 sans，技术 metadata 使用 monospace。中文正文行高保持 1.7 至 1.8。
4. 页面用规则线、留白和阅读列组织内容。去掉大圆角、厚阴影、渐变、玻璃效果和无意义的卡片网格。
5. Button、输入框、textarea、select、details 都有明显 focus-visible 状态。只给需要的属性设置 140 至 220ms transition，并响应 `prefers-reduced-motion`。
6. 首页突出“结论可回到论文原页”。Library 是目录和导入台，Search 是证据结果列表，Paper 是目录、PDF、书目信息三列阅读面板。不能仅靠换色完成改造。

**阶段验收**

- 所有现有页面路由正常打开。
- 1440px 与 375px 截图中无横向溢出、不可读低对比文字、被裁切的主按钮或不可操作的导航。

### 阶段 B：结构化 evidence 数据链路

**修改范围**

- `agent/schemas.py`
- `agent/tools.py`
- `api/models/chat.py`
- `api/routers/chat.py`
- `web/src/lib/types.ts`
- 必要时 `web/src/lib/api.ts`、`web/src/hooks/useSSEStream.ts`
- `tests/test_api.py`、`tests/test_streaming.py` 及新增的窄测试文件

**实施要求**

1. `EvidenceItem` 增加可选 `paper_id`、`paper_title`，只从 retriever metadata 传递，不重建或猜测标题。
2. graph 与 offline 两条路径各自转换到 `ChatEvidence`。转换函数集中在 chat router 私有 helper，避免前端解析原始 LangGraph state。
3. assistant 消息写入必须包含 content 和完整 evidence。刷新 API 读取同一 JSON 内容。
4. 旧会话没有 evidence 时仍能通过 Pydantic 验证并正常显示。
5. 保持 `/api/chat` request、`/api/chat/stream` 事件名、B1 graph 调用和 `ANSWER_STRATEGY=fixed` 行为不变。

**必须测试**

- 新会话创建后的 JSON 与旧结构兼容。
- graph 路径的 evidence event 含结构化字段，session reload 返回相同 evidence。
- 连续两轮回答不会把第二轮 evidence 绑定到第一轮。
- 无 evidence、无 `paper_id`、无页码、空 quote、retriever 错误和 graph 无最终答案的行为明确。
- SSE 不输出 routing、query plan、prompt、secret 或模型中间内容。

### 阶段 C：Chat 会话回看与 evidence rail

**修改范围**

- `web/src/app/chat/page.tsx`
- `web/src/components/CitationAccordion.tsx`，可重命名但应保持单一职责
- 相关前端类型和样式

**实施要求**

1. evidence 绑定到每条 assistant message，不能只显示全局“最后一段 citations”。
2. Chat 主列显示会话和输入框，桌面右侧为 evidence rail。窄屏时 evidence rail 在会话后自然流动，不能依赖悬浮抽屉。
3. 每个 evidence 条目显示编号、论文名或 source、章节、`P.{page}`、可展开的 quote 和论文链接。使用 disclosure，不能默认铺开长 quote。
4. 当前会话无 evidence 时显示诚实空状态，并提示用户可使用 Search 查看命中片段。
5. 新建会话清除本地 session、输入、错误和临时 SSE evidence。hydrate session 只使用服务器持久化的 messages。
6. 提交期间禁用重复发送，但不得阻塞浏览已有消息。错误通过 text 显示，不把错误当 assistant answer 保存。

**手工流程**

1. 提问一次，确认 progress、最终回答和 evidence rail。
2. 提问第二次，确认两条回答各自 evidence 独立。
3. 刷新页面并从 URL session 进入，确认两轮 evidence 都在。
4. 打开含 `paper_id` 与 page 的链接，确认 URL 与 PDF 页码一致。
5. 使用没有索引、无 evidence 和 stream-error 场景，确认不出现假引用。

### 阶段 D：整体验收与性能检查

1. 不引入新的网络请求来渲染 evidence。页面刷新只读取现有 chat session。
2. 不在浏览器保存完整原文、prompt、retrieval debug 或无界 history 副本。
3. 对证据列表使用稳定 key，quote 只在 details 展开时显示完整内容。
4. 检查 Link、按钮、textarea、details 的键盘 Tab 顺序，检查 skip link。
5. 用真实浏览器检查首页、Library、Search、Paper、Chat 的 1440px 与 375px 截图。截图放在 `output/playwright/`，不提交临时浏览器目录。

## 6. 验证与交付顺序

先跑受影响测试，再跑全量命令：

```bash
uv run --extra dev python -m pytest tests/test_api.py tests/test_streaming.py -q
uv run --extra dev ruff check .
uv run --extra dev python -m pytest -q
npm --prefix web run lint
npm --prefix web run build
git diff --check
```

所有命令通过后，再做第 5 节的浏览器检查。随后创建 `docs/implementation/m5_fixed_product_acceptance.md`，至少记录：commit 前 HEAD、改动文件、API 兼容、数据契约、两轮会话回看、视觉检查、命令输出、坏例、回滚步骤。

## 7. 独立代码审查与修复闭环

完成实现和全量验证后，再调用一个独立 review subagent。请求固定为：

```text
模型：gpt-5.6-luna
reasoning effort：max
角色：只读 reviewer，不直接修改工作区
范围：本 Goal diff、Chat API 兼容、JSON 会话持久化、SSE 时序、evidence 真实性、旧会话兼容、前端状态竞争、XSS/路径泄露、可访问性、渲染性能、回归风险
输出：按严重度列出文件和行号，给出可验证的修复建议；没有问题也必须说明审查过的风险面。
```

执行者处理 review 结果的顺序：

1. P0/P1 立即修复，之后重新运行受影响测试、lint、build 和对应手工流程。
2. P2 只要能在 M5 边界内修复，就必须修复并验证。
3. 仅“报告不成立”或“超出 M5 边界”的问题可以不改，必须把证据和判断写入验收报告。
4. 修复涉及 API、SSE 或 evidence 绑定时，重新调用 reviewer 审查修复 diff。仅 CSS 文案修复时，自检并记录即可。
5. 验收报告记录 reviewer 模型、effort、问题、修复 commit 内容和未修复理由。

当前工具环境若无法调用 `gpt-5.6-luna`，停止在 review 门前说明原因，不可伪称已按指定模型审查。用户明确替代模型后才继续。

## 8. 回滚

- 后端回滚：保留 assistant `content`，忽略 `evidence` 可选字段。旧 session JSON 不需要迁移或清洗。
- 前端回滚：恢复先前 chat 展示和样式，不改变 `/api/chat`、Search、Library、Paper 或 active index。
- 不删会话数据、不修改索引版本、不运行 schema migration。

## 9. 最终提交

仅在验收报告、review 修复和所有验证完成后执行：

```bash
git add DESIGN.md docs/research/m5_fixed_product_implementation_plan.md docs/implementation/m5_fixed_product_acceptance.md <实际代码和测试文件>
git commit -m "feat: polish fixed RAG web experience"
```

不推送，不创建 PR。提交后停止，等待用户决定是否开始 M6。
