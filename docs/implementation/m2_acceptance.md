# M2 验收报告：论文目录与页码证据

- **日期：** 2026-07-26
- **分支：** `codex/v2-core`
- **结论：** 通过。允许交付 M2，禁止自动进入 M3。

## 1. 范围结论

本次只实施 M2：

- 建立项目自有 parser protocol，默认使用 PyMuPDF4LLM 0.3.4，经 deterministic structure normalizer 生成页、章节和 block。
- 每篇 PDF 同时运行 legacy parser 做质量对照；primary 失败、超时或低于质量门槛时明确降级。
- 建立 `papers`、`paper_versions`、`sections`、`passages` 与 parsed artifact 闭环。
- 文件字节 SHA-256 是 `paper_id`；不同字节的 PDF 不自动合并。
- `paper_version_id`、`section_id`、`passage_id` 均为确定性 ID。
- `quote_text` 保持 parser 原文；`retrieval_text` 加可信元数据前缀。
- 完整的元数据前缀和 passage 在调用 embedding provider 前受
  `EMBEDDING_MAX_INPUT_CHARS=6000` 硬上限保护。
- Search API 返回论文、章节、页码、quote、vector/BM25/fusion/boost/final/rank。
- PDF API 支持单段 Range、suffix Range、206、416 和 `#page=N` 跳转。
- `/kb` 迁移到 `/library`；新增 `/papers/[id]` 和 `/search`。
- 未修改默认检索融合算法，未开始 Adaptive Agent、run checkpoint、trace、Compare 或 Workspace。
- 未安装 Docling，未承诺 OCR、公式语义解析、bbox 高亮或不同 PDF 修订版自动合并。

## 2. Parser 对照与质量门槛

Gold 文件：`evals/datasets/parser_v2.json`

- dev：4 篇、12 个重点页面。
- test：12 篇、36 个重点页面。
- 合计：16 篇、48 个重点页面。
- 覆盖：双栏、表格、公式、长文、低文本、错误 metadata。
- test 标注在首次 test 运行前冻结；之后未根据 test 结果修改。

最终命令：

```text
uv run python -m evals.parser_eval --dataset evals/datasets/parser_v2.json
```

| 指标 | 门槛 | 最终结果 | 结论 |
|---|---:|---:|---|
| 页码准确率 | 1.0000 | 1.0000 | 通过 |
| 字符召回中位数，相对 legacy | >= 1.0000 | 1.0391 | 通过 |
| 章节边界 F1 | >= 0.8000 | 0.9412 | 通过 |
| 表格边界 F1 | >= 0.7500 | 0.8571 | 通过 |
| 标题准确率 | >= 0.9000 | 1.0000 | 通过 |
| p95 延迟比，相对 legacy | <= 15.0000 | 10.2564 | 通过 |

质量检查还验证：

- page coverage >= 95%。
- 页码单调、无重复、在 PDF 页数范围内。
- primary 字符数不少于 legacy 的 60%，否则使用 legacy。
- 非空页比例过低且平均文本少于 200 字符时标记 `needs_ocr`。
- parser 在独立子进程运行，普通文档硬超时 180 秒，100 页以上长文 600 秒。
- 修复了真实大 PDF 暴露的 multiprocessing Queue 回压死锁：父进程先限时读取结果，再回收子进程。

## 3. 数据与元数据闭环

- `papers` 表示上传文件实体，保留原始文件路径、字节哈希、解析状态、错误和用户元数据。
- `paper_versions` 表示 parser + parser version + normalization version 的解析版本。
- `sections` 保存层级、顺序、`page_start`、`page_end`、heading path。
- `passages` 分开保存 `quote_text` 和 `retrieval_text`。
- parsed artifact 原子写入 `PARSED_ARTIFACT_ROOT/<paper_id>/<paper_version_id>.json`。
- 标题、作者、年份、venue、DOI、arXiv ID 保存 value、source、confidence。
- 元数据按 PDF metadata、首屏启发式、文件名降级。可疑 PDF title 不覆盖可信首屏标题。
- 用户 PATCH 使用 `If-Match` 乐观并发控制；只把实际修改字段标记为 `source=user`、`confidence=1.0`。
- 修改元数据后先在事务中重建 passage 前缀并检查 6000 字符上限，再创建完整索引重建任务。

人工对照：

```text
修改前 quote_sha256:
d402e741995af87b36cd36e8299a6720ce551390e40145b4551da3681785763f

修改后 quote_sha256:
d402e741995af87b36cd36e8299a6720ce551390e40145b4551da3681785763f

修改前 retrieval_sha256:
b1b622c362f82ca5d80c5f61455306cda1ae91f88e938c95c48342677a7ba20b

修改后 retrieval_sha256:
086b3c8ab119547054f7d5321ad81e9380ea5b6d5c00533a0b959a7259793265

修改后前缀包含:
[TITLE] Attention Is All You Need (Verified)

人工数据集最大 retrieval_text:
4328 字符
```

结论：修改 title 后新 prefix 生效，quote 原文未变化，输入未越过 6000 字符硬上限。

## 4. API 与产品页面

### API

- `GET /api/papers`
- `GET /api/papers/{paper_id}`
- `PATCH /api/papers/{paper_id}`，要求 `If-Match`
- `GET /api/papers/{paper_id}/file`
- `GET /api/search`

### 页面

- `/library`：上传、任务轮询、论文筛选、解析状态、降级/失败原因、元数据来源与置信度。
- `/papers/[id]`：章节目录、页码控制、内嵌 PDF、原页新窗口跳转、元数据校正、parser 记录。
- `/search`：page evidence、quote、评分阶段、Paper Detail 跳转和 PDF 原页跳转。
- `/kb`：服务端重定向到 `/library`。

## 5. 人工检查

在隔离数据目录、`OFFLINE_MODE=1` 下通过真实浏览器完成：

| 检查项 | 样本 | 结果 |
|---|---|---|
| 双栏、公式、错误 metadata | Attention Is All You Need | 首屏标题胜出，解析完成 |
| 表格、双栏 | GPipe | 解析完成 |
| 低文本 | Coffee Automaton | 解析完成 |
| 长文、低文本、错误 metadata | Minimum Description Length，80 页 | 解析完成 |
| Search 到正确论文页 | `multi-head attention` | 打开 Attention 详情 `?page=5` |
| 用户修改 title | Attention | prefix 更新，quote 哈希不变 |
| needs_ocr | 3 页无文本 PDF | Library 显示“需要 OCR”和 `needs_ocr` |
| legacy fallback | `PAPER_PARSER=legacy` + Pointer Networks | 显示“降级完成”和 `configured_legacy_parser` |
| parser failure | 损坏 PDF | 显示“解析失败”和具体打开失败原因 |
| 移动端 | 375 × 812 | Search 页面无横向溢出，导航和表单可用 |

截图：

- `output/playwright/m2-library-desktop.png`
- `output/playwright/m2-paper-page-5.png`
- `output/playwright/m2-needs-ocr.png`
- `output/playwright/m2-legacy-fallback.png`
- `output/playwright/m2-parser-failure.png`
- `output/playwright/m2-search-mobile.png`

## 6. 已知失败案例与边界

Gold 中仍存在可接受的局部结构误判：

- AlexNet 第 4 页表格边界漏检。
- Message Passing 第 1 页表格边界误报、第 6 页漏检。
- NMT Alignment 第 8 页存在章节误报和表格漏检。
- ResNet 第 8 页存在章节误报。

这些案例没有通过修改 test 标注规避，最终总体 F1 仍超过门槛。产品明确展示 parser 状态和 fallback 原因。

本阶段不处理：

- 扫描件 OCR。
- 公式语义恢复。
- bbox 级高亮。
- 同一论文不同 PDF 修订版自动合并。
- Docling。

## 7. 自动验证

```text
uv run --extra dev python -m pytest \
  tests/test_pdf_parser.py \
  tests/test_parser_quality.py \
  tests/test_metadata.py \
  tests/test_paper_api.py \
  tests/test_search_api.py -q
```

结果：`16 passed`，3 条第三方 SWIG deprecation warning。

```text
uv run python -m evals.parser_eval --dataset evals/datasets/parser_v2.json
```

结果：`passed: true`，所有 6 个 gate 通过。

```text
uv run --extra dev ruff check indexing api tests
```

结果：通过。

```text
pnpm --dir web lint
pnpm --dir web build
```

结果：通过。Next.js 16.2.0 生产构建包含 `/library`、`/papers/[id]`、`/search`。

```text
uv run --extra dev python -m pytest -q
```

结果：`205 passed`，3 条第三方 SWIG deprecation warning。

并发迁移恢复测试额外连续运行 10 次，全部通过。

## 8. 回滚方法

Parser 级回滚：

1. 设置 `PAPER_PARSER=legacy`。
2. 重启 API。
3. 保留上一 active index，不删除 `data/indexes/<version>`。

索引级回滚：

```text
uv run python main.py activate-index <previous-ready-version>
```

数据库级恢复：

1. 停止 API 和 index worker。
2. 保存当前 `sessions.db`，不要直接覆盖唯一副本。
3. 使用迁移前自动生成的
   `sessions.db.backup-v3-<UTC timestamp>` 恢复 `sessions.db`。
4. 启动 API，确认 SQLite active pointer 与 `active.json` 已重新一致。

代码级回滚：对 M2 独立提交执行 `git revert <m2-commit>`，不改写历史。

如果后续环境中 parser quality gate 失败，保持 `PAPER_PARSER=legacy` 和上一 active index，并停止进入 M3。

## 9. 修改文件

配置与说明：

- `.env.example`
- `.gitignore`
- `AGENTS.md`
- `web/DESIGN.md`
- `docs/implementation/m2_acceptance.md`

Parser、结构与 passage：

- `indexing/paper_ingestion.py`
- `indexing/passages.py`
- `indexing/parsers/__init__.py`
- `indexing/parsers/paper_parser.py`
- `indexing/parsers/pymupdf4llm_parser.py`
- `indexing/parsers/legacy_paper_parser.py`
- `indexing/parsers/structure_normalizer.py`
- `indexing/parsers/parser_quality.py`
- `indexing/parsers/metadata.py`

索引与检索：

- `indexing/indexer.py`
- `indexing/index_versions.py`
- `indexing/retriever.py`

数据库、worker 与 API：

- `api/db/database.py`
- `api/db/migrations.py`
- `api/db/papers.py`
- `api/services/index_worker.py`
- `api/models/papers.py`
- `api/models/search.py`
- `api/routers/indexing.py`
- `api/routers/papers.py`
- `api/routers/search.py`
- `api/main.py`
- `core/settings.py`

Gold、eval 与测试：

- `evals/datasets/parser_v2.json`
- `evals/parser_eval.py`
- `tests/test_pdf_parser.py`
- `tests/test_parser_quality.py`
- `tests/test_metadata.py`
- `tests/test_paper_api.py`
- `tests/test_search_api.py`

前端：

- `web/src/app/globals.css`
- `web/src/app/layout.tsx`
- `web/src/app/page.tsx`
- `web/src/app/not-found.tsx`
- `web/src/app/kb/page.tsx`
- `web/src/app/library/page.tsx`
- `web/src/app/papers/[id]/page.tsx`
- `web/src/app/papers/[id]/loading.tsx`
- `web/src/app/search/page.tsx`
- `web/src/components/FileUpload.tsx`
- `web/src/components/ui/button.tsx`
- `web/src/components/ui/card.tsx`
- `web/src/components/ui/input.tsx`
- `web/src/components/ui/textarea.tsx`
- `web/src/lib/api.ts`
- `web/src/lib/i18n.ts`
- `web/src/lib/types.ts`

验收截图：

- `output/playwright/m2-library-desktop.png`
- `output/playwright/m2-paper-page-5.png`
- `output/playwright/m2-needs-ocr.png`
- `output/playwright/m2-legacy-fallback.png`
- `output/playwright/m2-parser-failure.png`
- `output/playwright/m2-search-mobile.png`
