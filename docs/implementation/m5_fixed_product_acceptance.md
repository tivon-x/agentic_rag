# M5 固定产品验收报告

日期：2026-08-02
分支：`codex/v2-core`
范围：Goal 5「M5 证据导向的固定 RAG Web 应用」

## 结论

Goal 5 的实现和验证门槛已完成。Chat 继续使用原有 POST + SSE 连接模型；每个 assistant message 现在可携带自己的结构化 evidence，并通过 SSE 与会话读取接口返回同一组字段。全站页面已按根目录 `DESIGN.md` 重构为暖白纸面、墨黑正文、细规则和墨蓝证据标记的中文学术编辑部风格。

## API 兼容与证据契约

- `ChatMessage` 保留原有 `role`、`content`，新增可选 `evidence`；旧会话中没有该字段时仍可正常读取。
- `ChatEvidence` 字段为 `node_id`、`paper_id`、`paper_title`、`source`、`section_path`、`page`、`quote`、`score`、`relevance`。来源路径在 API 输出时只保留文件名，避免泄露本机绝对路径。
- `GET /api/chat/{session_id}` 返回每轮 assistant 自己的 evidence；`POST /api/chat` 的用户消息格式和 session 行为保持不变。
- SSE 顺序为 `progress` → `evidence` → `answer.final`。`evidence` 和 `answer.final` 使用同一组结构化字段，完成后才写入会话；保存失败只发送错误，不发送未保存的完成回答。
- offline retrieval 和 graph 路径均使用同一套 evidence 归一化逻辑；缺少 `paper_id` 或页码时保留证据，并在 UI 显示无链接或页码未知状态。
- 未新增数据库迁移，继续复用 SQLite JSON 会话存储；未改变检索策略、graph 拓扑、索引、数据库 schema、模型调用、SSE 连接模型或普通用户设置。

## 会话回看与证据完整性

新增 `tests/test_chat_evidence.py` 覆盖：

- 连续两轮回答：两条 assistant message 的 paper、页码和 quote 不串轮；重新 `GET` 会话后仍保持归属。
- 旧会话兼容：没有 evidence 的历史 user message 可正常读取。
- 无 evidence：SSE 明确返回 `evidence: []`，会话保存空列表，前端显示空状态而不伪造引用。
- 缺少 paper/page：证据仍可渲染；缺少稳定 `node_id` 或空 quote 的坏 evidence 被丢弃。
- quote 使用 source-faithful 文本，沿用现有 400 字符上限；PDF 跳转由 `/papers/{paper_id}?page={page}` 生成，paper id 使用 URL 编码。

## 全站视觉与可访问性检查

已检查 `/`、`/library`、`/search`、`/papers/demo` 和 `/chat`：

- 首页、论文库、搜索、论文阅读和 Chat 使用统一 masthead、字体、暖白纸面、细分隔线、规则列表和矩形表单；没有后台模板、渐变玻璃或同质化圆角卡片墙。
- Chat evidence rail 按 assistant 回答分组，每张卡显示论文名/来源、章节、页码、quote、关联说明和论文原页链接；没有 evidence 时显示明确空状态。
- 全局提供 skip link、focus-visible 规则、表单标签和错误提示；文件上传控件保持键盘可聚焦并有 focus-within 提示。
- 真实 Chromium 检查了 1440px 桌面和 375px 移动视口；移动端 evidence rail 顺序落在正文之后，未发现横向溢出。
- 截图：[`m5-chat-desktop.png`](../../output/playwright/m5-chat-desktop.png)、[`m5-chat-mobile.png`](../../output/playwright/m5-chat-mobile.png)。

## 手工验收记录

由于当前 checkout 没有激活的本地索引，手工两轮使用了仅用于验收的临时内存 FakeGraph；它没有调用外部模型或写入项目数据，真实 FastAPI + Next proxy、SSE、SQLite JSON 会话和 UI 均走产品路径。临时脚本、临时数据和服务已清理。

实际操作并确认：

1. 在 Chat 中提交两轮问题；页面显示 4 条消息和两组 evidence，分别标为「回答 01」和「回答 02」。
2. 刷新同一 session；两轮回答、各自 evidence、页码 P.4/P.5 和证据卡片均恢复。
3. 离开 Chat 后重新打开原 session URL；历史内容和 evidence rail 仍恢复。
4. 逐一展开两张 evidence 卡片；完整 quote 可见。
5. 打开第一张证据的论文链接，浏览器地址确认跳转到 `/papers/demo-paper?page=4`。
6. 在 375px 视口检查导航、Chat、消息和移动布局；真实 Chromium 的 `scrollWidth` 与 `clientWidth` 均为 360，无横向溢出。

## 验证结果

最终实现和 review 修复后执行：

| 命令 | 结果 |
|---|---|
| `uv run --extra dev python -m pytest -q` | 278 passed，3 warnings |
| `uv run --extra dev ruff check .` | All checks passed |
| `npm --prefix web run lint` | passed |
| `npm --prefix web run build` | passed；Next.js 16.2.0，所有 App routes 构建成功 |

review 后的前端修复已重新执行 lint/build；review 前后的后端全量 pytest/Ruff 均通过。最终真实 Chromium 重启 dev server 后 console 为 0 errors、0 warnings。

## 独立 review

- 模型：`gpt-5.6-luna`
- reasoning effort：`max`
- reviewer：独立只读 review subagent `Volta`
- 初始 review：未发现 P0；指出 3 个可修复 P2：上传控件不可键盘聚焦、evidence rail 回答编号偏移、深色提问标签对比度不足；均已修复并重新执行受影响验证。
- review 还报告了其执行环境的 uv cache/npm PowerShell 权限问题。该问题不属于代码缺陷；实现者随后使用可用的 `uv` 提权验证和 `cmd.exe` 前端命令重新执行，最终 pytest、Ruff、lint、build 全部通过。
- reviewer 未发现 API 兼容、ChatEvidence schema、SSE 时序、SQLite 持久化、旧会话兼容、两轮 evidence 归属、source 脱敏、PDF 跳转或 Goal 5 边界违规问题。

## 坏例与处理

- 没有激活索引或检索失败：返回 `stream-error`，不保存未完成 assistant answer。
- 没有 evidence：保存空列表并显示明确空状态，不把 Markdown 引用当成结构化 evidence。
- 历史 assistant 没有 evidence：读取兼容，前端按无 evidence 状态渲染。
- evidence 缺少 paper id/page：保留论文名、章节和 quote；显示「页码未知」或无论文目录链接。
- evidence 缺少稳定 node 或 quote 为空：归一化阶段丢弃，避免生成无法回溯的证据卡片。
- 会话保存失败：发送错误事件，不发送完成事件，避免 UI 显示未持久化回答。

## 回滚方式

本 Goal 使用一个独立 commit，未推送。需要回滚时执行：

```bash
git revert <本次 M5 commit>
```

不使用 `git reset --hard`，保留工作区和历史可恢复性。

## 实际修改文件

- `agent/schemas.py`
- `agent/tools.py`
- `api/models/chat.py`
- `api/routers/chat.py`
- `tests/test_chat_evidence.py`
- `web/next.config.ts`
- `web/src/app/chat/page.tsx`
- `web/src/app/globals.css`
- `web/src/app/layout.tsx`
- `web/src/app/library/page.tsx`
- `web/src/app/page.tsx`
- `web/src/app/papers/[id]/page.tsx`
- `web/src/app/search/page.tsx`
- `web/src/components/CitationAccordion.tsx`
- `web/src/components/FileUpload.tsx`
- `web/src/components/ui/button.tsx`
- `web/src/components/ui/card.tsx`
- `web/src/components/ui/input.tsx`
- `web/src/components/ui/textarea.tsx`
- `web/src/lib/evidence.ts`
- `web/src/lib/types.ts`
- `output/playwright/m5-chat-desktop.png`
- `output/playwright/m5-chat-mobile.png`
- `docs/implementation/m5_fixed_product_acceptance.md`
