# M5.1 Web UI 审查问题

## 文档状态

- 审查时间：2026-08-09 至 2026-08-10，2026-08-15 复验
- 审查对象：`codex/v2-core` 的 Web UI
- 审查方式：源码检查和 `http://localhost:3000` 真实浏览器验证
- 视口：1280 x 720、375 x 812、320 x 700
- 历史初审结论：`Block`；M5.1 Goal 最终结论见文末。

本文记录审查时确认的问题。后续代码可能已经修改，关闭问题前需要按文末步骤重新验证。

## 2026-08-15 复验结论

本轮在当前工作树和已运行的 `http://localhost:3000/chat` 上完成两轮真实问答。首轮回答约 100 秒完成并带 4 条结构化证据，第二轮约 110 秒完成并带 1 条结构化证据。刷新后，两轮问题、回答和证据数量均能恢复，桌面证据面板也能在“回答 01”和“回答 02”之间正确切换。

当前结论仍为 `Block`。聊天生成和会话保存已经可用，但所有 5 条证据都缺少 `paper_id`，界面只能显示“该来源暂无论文目录链接”，无法从回答进入论文原页核对证据。UI-02、UI-03、UI-04 和 UI-06 仍然存在。本轮还确认了回答正文格式和长等待反馈问题。

复验范围：

- 桌面端：1280 x 720。
- 移动端：375 x 812、320 x 700。
- 两轮问题：多头注意力相比单头注意力的作用；点积注意力除以根号 `d_k` 的原因。
- 状态：新会话、生成中、成功回答、桌面证据面板、移动证据列表、完整摘录展开、刷新恢复、移动会话抽屉、输入框焦点。
- 未触发：生成失败、检索失败、保存失败和断网状态。

## 已确认问题

### UI-01：回答生成失败被保存为正式回答

- 严重程度：HIGH
- 状态：代码修复已落地，正常链路复验通过；失败链路仍需专项复验
- 分类：本轮工作树修复
- 位置：`agent/nodes/aggregate_answers.py:36`、`web/src/components/ChatExperience.tsx:131`

连续两次真实提问都在 4 至 6 秒后结束，但页面把 `No answers were generated.` 显示为“回答 01”，同时标记“无结构化证据”。界面没有进入错误状态，也没有提供重试入口。刷新页面后，这条内容仍作为正式回答恢复。

搜索页对相同主题能返回 20 条证据，因此该现象不能归因于论文库为空。失败结果被包装为成功回答会误导用户，是当前发布阻断项。

修复要求：

- 后端使用明确、可识别的失败状态，不要用普通 `AIMessage` 表示生成失败。
- 前端收到失败状态后显示中文错误和恢复操作，不得追加或持久化为成功回答。
- 保留本轮问题，并提供“重试回答”。
- 错误状态不能显示回答序号和“无结构化证据”。

验收标准：

- 模拟未生成答案时，页面进入错误状态并可重试。
- 会话刷新或重新打开后，不出现伪造的成功回答。
- 正常回答仍能保存正文和结构化证据。

2026-08-15 复验：两轮正常回答均成功保存，刷新后没有出现 `No answers were generated.` 伪回答。`api/routers/chat.py` 已增加失败识别并返回中文错误，前端保留本轮问题并提供“重试回答”。本轮没有人为触发模型失败，因此失败状态的运行时表现仍未验证。

### UI-02：移动会话抽屉和证据层没有完整的模态焦点管理

- 严重程度：MEDIUM
- 状态：2026-08-15 复验仍存在
- 分类：Pre-existing
- 位置：`web/src/components/ChatExperience.tsx:433`、`web/src/components/ChatExperience.tsx:458`

两个弹层都声明了 `aria-modal="true"`，但打开移动会话抽屉后，焦点仍停在抽屉外的“会话”按钮。背景没有设为 `inert`，Tab 也没有被限制在弹层内。

修复要求：

- 打开弹层后把焦点移到“关闭”按钮或第一个可操作控件。
- 弹层打开期间让背景内容不可聚焦、不可操作。
- Tab 和 Shift+Tab 在弹层内循环。
- Escape、关闭按钮和遮罩关闭后，将焦点恢复到原触发按钮。

验收标准：只用键盘可以完整打开、操作和关闭两个弹层，焦点不会离开当前弹层。

### UI-03：聊天输入框覆盖了全局可见焦点环

- 严重程度：MEDIUM
- 状态：2026-08-15 复验仍存在
- 分类：Pre-existing
- 位置：`web/src/app/globals.css:738`

项目在全局为表单控件设置了 2 px `:focus-visible` 轮廓，但聊天输入框随后使用 `textarea:focus { outline: 0; }` 将其覆盖，键盘聚焦时只剩 1 px 边框变色。

修复要求：删除 `outline: 0`，继续使用全局焦点样式，或为聊天输入框提供等效的 `:focus-visible` 轮廓。

验收标准：键盘聚焦聊天输入框时，在浅色背景和边框旁都能清楚看到至少 2 px 的焦点指示。

### UI-04：移动端输入文字小于 16 px

- 严重程度：MEDIUM
- 状态：2026-08-15 复验仍存在
- 分类：Pre-existing
- 位置：`web/src/components/ui/input.tsx:12`、`web/src/app/globals.css:1050`

375 px 视口实测，搜索框文字为 14 px，聊天输入框为 13.12 px。iOS Safari 聚焦小于 16 px 的输入框时可能自动放大页面。

修复要求：移动端输入框和文本域使用至少 16 px，较大断点可以恢复当前紧凑字号。

验收标准：375 px 下搜索框和聊天输入框的计算字号均不小于 16 px，聚焦后布局不缩放、不产生横向滚动。

### UI-05：后端英文错误直接暴露给中文用户

- 严重程度：MEDIUM
- 状态：本轮未触发错误态，源码状态未变
- 分类：Pre-existing
- 位置：`web/src/components/ChatExperience.tsx:156`、`web/src/app/(editorial)/papers/[id]/page.tsx:60`、`web/src/app/(editorial)/papers/[id]/page.tsx:520`

实测出现过以下文案：

- `No index loaded. Run python main.py index <path> first or use the UI to index documents.。`
- `Paper not found.`

第一条还出现了英文句号和中文句号连续使用。普通用户不应该通过内部 CLI 命令恢复 Web 操作。

修复要求：

- 将已知后端错误映射为中文说明。
- 提供与当前界面一致的恢复操作，比如“前往论文库导入并解析文件”。
- 未识别错误使用稳定的中文兜底文案，技术细节留在日志中。
- 统一标点，避免拼接产生双标点。

验收标准：聊天、搜索和论文详情的常见失败状态均使用中文，并给出可以直接执行的下一步。

### UI-06：共享按钮按压反馈过重

- 严重程度：LOW
- 状态：2026-08-15 源码确认仍存在
- 分类：Pre-existing
- 位置：`web/src/components/ui/button.tsx:17`

共享按钮统一使用 `active:scale-95`。建议调整为 `active:scale-[0.96]`，保留触感并减轻跳动。

### UI-07：结构化证据无法打开论文原页

- 严重程度：HIGH
- 状态：待修复并复验
- 分类：Pre-existing，本轮成功回答后首次完成运行时确认
- 位置：`api/routers/chat.py:458`、`web/src/lib/evidence.ts:4`、`web/src/components/ChatExperience.tsx:701`

两轮回答共返回 5 条结构化证据，论文标题和页码都能显示，但接口中的每条证据都没有 `paper_id`。桌面端统一显示“该来源暂无论文目录链接”，移动端证据项也没有可点击入口。

这会阻断 RAG 聊天最关键的核对动作。用户看到了摘录，却无法从回答直接进入论文对应页确认上下文。

修复要求：

- 从检索元数据到 `ChatEvidence` 的整条链路保留 `paper_id`，不要依赖模型重新生成该字段。
- 桌面和移动证据项都使用 `/papers/{paper_id}?page={page}` 打开对应论文页。
- 只有论文确实不在目录时才显示不可用状态，并说明原因。

验收标准：本轮 5 条证据都能打开《Attention Is All You Need》的对应页码，刷新会话后链接仍然有效。

### UI-08：回答正文暴露英文内部字段，数学公式按源码显示

- 严重程度：MEDIUM
- 状态：待修复并复验
- 分类：Pre-existing
- 位置：`core/rag_answer.py:93`、`core/rag_answer.py:98`、`core/rag_answer.py:103`、`web/src/components/ChatExperience.tsx:743`

两轮回答都直接显示 `Reasoning summary`、`Confidence` 和 `Limitations`。中文界面中混入英文内部字段，且 `$d_k = d_v = d_{model}/h$`、`$\\sqrt{d_k}$` 以 Markdown 源码显示，没有渲染成公式或可读文本。

修复要求：

- 回答正文只展示面向用户的答案。
- 如果保留推理摘要、置信度和限制，使用中文标签并放入独立、可折叠的补充信息区。
- 公式使用项目支持的数学渲染方案，或在没有公式渲染时转换为可读纯文本，如 `sqrt(d_k)`。

验收标准：中文回答中不出现未本地化的内部字段，常见行内公式不显示 `$`、`\\sqrt` 等源码符号。

### UI-09：长时间生成只有“回答中”，缺少阶段说明

- 严重程度：MEDIUM
- 状态：待修复并复验
- 分类：Pre-existing
- 位置：`web/src/components/ChatExperience.tsx:413`

本轮两次回答分别等待约 100 秒和 110 秒。等待期间页面只把按钮改成“回答中”，没有说明正在检索还是正在生成，也没有告诉用户该等待通常可能持续 1 至 2 分钟。用户很难判断页面仍在工作还是已经卡住。

修复要求：超过 10 秒后显示稳定、真实的阶段说明，比如“正在检索论文并整理证据，可能需要 1 至 2 分钟”。不要使用伪造进度条。

验收标准：长回答等待期间始终有可理解的状态说明，回答完成或失败后状态立即清除。

## 已通过的界面路径

- 论文库展示 25 篇论文，25 篇均可检索，0 篇待处理。
- 搜索 `scaled dot-product attention` 返回 20 条证据。
- 搜索首条结果正确定位到《Attention Is All You Need》第 4 页。
- 论文详情 iframe 和“直接打开 PDF 原页”都能打开对应页码。
- 首页、论文库、搜索、聊天在 320 px 和 375 px 下没有横向溢出。
- 会话刷新和重新打开能够恢复已有消息。
- 两轮成功问答刷新后，回答和证据数量仍按轮次分组。
- 桌面证据面板能在“回答 01”和“回答 02”之间正确切换。
- 证据完整摘录可以展开和收起。
- 移动会话抽屉支持 Escape 和遮罩关闭，但焦点管理仍需按 UI-02 修复。

## 尚未完成的验证

以下状态仍未完成验证：

- 生成失败、检索失败和保存失败时，错误提示、重试和会话恢复是否正确。
- 修复 UI-07 后，聊天证据链接是否跳到正确论文与页码。
- 屏幕阅读器对动态回答、错误状态和证据面板的播报。
- UI-05 所列常见后端错误的运行时中文映射。

## 工程门禁

2026-08-15 复验时，`npm --prefix web run lint` 仍报告 4 个错误，涉及论文详情页和聊天页的 React Hook 用法。`npm --prefix web run build` 通过。lint 错误不属于界面设计 finding，但合并前必须修复并重新运行：

```bash
npm --prefix web run lint
npm --prefix web run build
```

## 复验清单

1. 补测一次生成失败，确认问题保留、错误为中文、可重试且不会保存伪回答。
2. 修复 UI-07 后逐条打开证据，核对论文 ID、页码和摘录上下文。
3. 修复 UI-08 后，在桌面和移动端检查中文补充信息和公式显示。
4. 修复 UI-09 后，让回答持续超过 10 秒，确认等待说明出现并在完成后清除。
5. 刷新并重新打开包含两轮回答的会话，确认正文、证据和链接没有丢失或串组。
6. 在 1280 px、375 px 和 320 px 下重复聊天主流程。
7. 只用键盘检查会话抽屉、证据层、输入框、证据 disclosure 和链接。
8. 运行前端 lint 和 build，确认全部通过。

## 2026-08-16 M5.1 Goal 修复复验

本节保留前文历史发现，并记录本 Goal 的最终状态。复验基于当前工作树，未改变固定检索策略、Graph 拓扑、索引格式、数据库 schema、模型供应商或 SSE 连接模型。

| 问题 | 最终状态 | 复验证据 |
|---|---|---|
| UI-01 | `Closed` | `tests/test_agent_grounded_answer.py` 和 `tests/test_chat_evidence.py` 覆盖无答案、检索失败、回答生成失败、保存失败；295 个后端测试通过。真实浏览器用确定性 EventSource 错误模拟三类失败，均保留用户问题、显示中文恢复文案和“重试回答”，`GET /api/chat/{session_id}` 只保留 user message。真实两轮回答仍保存正文和结构化证据。 |
| UI-02 | `Closed` | 375×812 移动会话抽屉和 1024×720 证据模态层均现场验证：打开焦点进入弹层，背景 inert、滚动锁定，Tab/Shift+Tab 不离开，Escape/关闭恢复到触发按钮。 |
| UI-03 | `Closed` | 聊天 textarea 使用 `focus-visible`，真实浏览器计算焦点轮廓为 2px solid 墨蓝；未再覆盖全局焦点样式。 |
| UI-04 | `Closed` | 375×812 和 320×700 真实浏览器中，聊天 textarea、搜索 input 计算字号均为 16px；`scrollWidth === clientWidth`，无横向溢出。 |
| UI-05 | `Closed` | `web/src/lib/errors.ts` 集中映射已知错误并隐藏未知技术细节；contract checks 覆盖 no-index、未知错误和标点；浏览器生成/检索/保存失败均为中文并给出下一步。 |
| UI-06 | `Closed` | 共享 Button 改为 `active:scale-[0.96]`，保留 disabled 和 `motion-reduce` 状态；前端 lint/build 通过。 |
| UI-07 | `Closed` | `api/routers/chat.py` 从 `evidenceGroups`/`retrievalEvidence` 的检索元数据绑定 `paper_id`，模型字段不能伪造 ID；两轮真实回答证据链接逐条打开，均指向 `Attention Is All You Need` 的正确论文 ID 和 P.4/P.5/P.9，刷新、离开后重开会话仍保持回答 01/02 证据归属。 |
| UI-08 | `Closed` | 展示层拆分正文、回答依据、置信度、局限性，补充信息置于可访问 details；Evidence 仍由结构化证据组件展示。contract checks 覆盖 `d_k`、`d_model`、`d_k = d_v = d_model/h`、`sqrt(d_k)`，真实回答未显示 `$`/`\\sqrt` 原始符号。 |
| UI-09 | `Closed` | 真实回答超过 10 秒后显示“正在检索论文并整理证据，可能需要 1 至 2 分钟”，完成、失败、重试、切换会话和卸载路径均清理定时器和状态；contract checks 覆盖 `setTimeout`/`clearTimeout`。 |

### 自动化门禁

- `uv run --extra dev python -m pytest -q`：295 passed，3 warnings。
- `uv run --extra dev ruff check .`：通过。
- `npm --prefix web run test:contracts`：通过。
- `npm --prefix web run lint`：通过。
- `npm --prefix web run build`：通过，Next.js 16.2.0 全部路由构建成功。
- `git diff --check`：通过。

### 浏览器范围

- 服务：`http://localhost:3000`，真实 FastAPI + Next.js 代理，demo 目录 25 篇论文，包含《Attention Is All You Need》。
- 视口：1280×720 完成两轮真实问答、证据面板、刷新和重新打开；375×812 完成移动证据列表、字号、会话抽屉键盘流；320×700 完成聊天/搜索输入字号和无横向溢出；1024×720 完成证据模态键盘流。
- 逐条展开并打开两轮共 6 条证据链接，核对论文标题、paper_id、页码和论文详情页 PDF 页码。
- 外部模型调用获当前会话授权；失败态使用浏览器内确定性 SSE 模拟，不改变持久数据。

### 独立复核

实现完成后由独立 `luna_worker` 只读审查 UI-01 至 UI-09、API/持久化兼容、paper_id 来源、错误态、轮次绑定、焦点、移动适配和定时器。审查结果、范围内问题处理和最终 verdict 记录在 `docs/implementation/m5_1_web_ui_fix_acceptance.md`。

### 最终独立只读复审（2026-08-16）

- 初审 verdict：`Needs changes`。发现三个修复后问题：未知中文技术错误会原文透传、桌面证据触发器存在悬空 `aria-controls`、证据模态焦点陷阱漏掉原生 `summary`；移动证据缺少 `paper_id` 的不可用文案也一并补齐。
- 处理结果：`web/src/lib/errors.ts` 统一未知错误兜底并补充 no-active-index 映射；`ChatExperience.tsx` 仅在目标实际存在时设置 `aria-controls`，移动/桌面证据均显示不可用状态，焦点选择器加入 `summary`。
- 复验：1024×720 现场键盘序列为“关闭 → 查看完整原文摘录 → 两个原页链接 → 关闭”，Shift+Tab 可反向闭环；contract、lint、build 均通过。
- 最终 verdict：`Approve`。未发现 UI-01 至 UI-09 仍需处理的范围内问题。
