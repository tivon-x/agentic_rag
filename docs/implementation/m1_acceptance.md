# Agentic RAG V2 M1 验收记录

> 验收日期：2026-07-25
> 分支：`codex/v2-core`
> 范围：M1“运行与索引可靠性”，未开始 M2 至 M6

## 1. 基线

- pre-V2 baseline：`5983aca`
- M1 实施起点：`324e7c0`
- 实际开发起点：`692ca29`，比 `324e7c0` 多一个 embedding input mode 相关文档提交
- 开始时 `git status --short --branch -uall`：工作区干净，分支正确
- 基线后端：155 passed、1 failed
- 基线失败：`tests/test_settings.py::test_load_settings_defaults` 被宿主 `EMBEDDING_MODEL` 污染
- 基线 Ruff、前端 lint、前端 build：通过

## 2. 配置与环境复现

- `load_settings()` 的默认根目录统一为仓库根目录，CLI、FastAPI 和测试不再各自选择不同数据目录。
- `load_settings()` 读取 `.env` 时不再修改进程级 `os.environ`；显式调用 `load_dotenv()` 仍保留原有可选写入行为。
- LLM task model、Agent 运行限额和 embedding 配置都通过 `AppSettings` 下传，不再从 `llms/`、`agent/` 或 `indexing/` 直接读取环境变量。
- 开发依赖入口统一为 `uv sync --extra dev`。
- 验收解释器为 Python 3.12.11。
- `.gitignore` 的 `lib/` 已收窄为 `/lib/`，此前被错误忽略但被前端引用的 `web/src/lib/` 已纳入版本控制。

Embedding 索引契约包含：

- provider
- model
- dimension
- input mode
- `check_embedding_ctx_length`
- max input chars

默认 `EMBEDDING_INPUT_MODE=raw`、`EMBEDDING_CHECK_CONTEXT_LENGTH=false`、`EMBEDDING_MAX_INPUT_CHARS=6000`。应用在 provider 调用前拒绝超长 document/query。manifest 不包含 API Key、base URL 或其他凭据。

## 3. 数据库迁移

- 当前 schema version：2
- 版本表：`schema_migrations(version, applied_at)`
- 迁移策略：整数版本、只前进、每个版本在事务中执行
- 每步迁移拿到 SQLite 写锁后重新读取版本，多个 API 首次并发启动不会重复执行 DDL
- 高于当前代码支持范围的未来 schema version 会拒绝启动，不会按旧结构继续运行
- 旧的 `pending` job 在迁移时转换为 `queued`
- SQLite 启用 foreign keys 和 WAL
- 已存在且需要迁移的 `sessions.db` 会先通过 SQLite backup API 创建一致性副本
- 备份命名：`sessions.db.backup-v<from_version>-<UTC timestamp>`
- 新增表：
  - `index_job_items`
  - `idempotency_records`
  - `index_versions`
  - `app_state`
  - `worker_leases`
- `indexing_jobs` 新增 request、attempt、lease、heartbeat、progress、active version 和 target version 字段
- 迁移失败不自动降级，也不删除新增表；恢复使用备份文件

## 4. Job 状态机与 worker

状态统一为：

```text
queued -> running -> completed
                  -> failed
                  -> cancelled
```

- 一个 FastAPI app 只创建一个 `IndexWorker` task。
- 跨进程通过 `worker_leases(name='index')` 保证同一时间最多一个全局 index worker。
- worker 通过 SQLite 事务领取 job，并写入 owner、lease expiry 和 heartbeat。
- 启动和每轮扫描过期 `running` job。
- lease 过期且未达到上限时重新排队；达到上限后转为 `failed`。
- 默认最多 3 次 attempt。
- retry 只把 `failed` job 重置为 `queued`；重复 retry 返回当前 job，不复制任务。
- worker 只接受 `UPLOAD_ROOT` 内已存在的普通文件。

## 5. 上传安全与幂等

- `POST /api/index/files` 强制要求 `Idempotency-Key`。
- 同 key、同 request hash 返回原 job。
- 同 key、不同 request hash 返回 409。
- 一次批量上传只创建一个 job，多文件记录在 `index_job_items`。
- 文件按 1 MiB 分块写入安全 staging 目录，不整文件读入内存。
- 校验空文件名、路径分隔符、绝对路径、控制字符、Windows 非法字符、设备保留名、后缀和大小。
- 默认单文件上限为 50 MiB，可通过 `UPLOAD_MAX_BYTES` 配置。
- staging、最终 job 目录和 worker 读取路径都执行 resolve 后的 `UPLOAD_ROOT` 边界检查。
- 任一失败只清理本请求创建且验证过的隔离目录。

## 6. 不可变索引版本

- 默认 `INDEX_WRITE_MODE=versioned`。
- 新索引写入 `INDEX_ROOT/.building-<version>`。
- 构建会复制当前 active version，再串行追加本 job 文件；旧版本不被修改。
- 校验 FAISS、BM25 和 manifest 后，staging 原子重命名为最终 UUID 目录。
- manifest 记录 embedding 完整契约、index mode、hierarchical 参数和 Git code version。
- 校验失败的 partial version 保存到 `INDEX_ROOT/failed/<version>/failure.json`。
- active version 同时记录在 SQLite `app_state` 和原子替换的 `active.json`；SQLite 是 API 运行时的优先来源。
- worker job 完成、旧 active 降为 ready、新版本激活和 `app_state` 更新在同一 SQLite 事务内完成；租约丢失时事务拒绝激活。
- model、dimension、input mode、provider、context-length check 或 max input chars 不兼容时拒绝加载并要求重建。
- 激活前实际反序列化并交叉校验 FAISS dimension/vector metadata 和 BM25 bundle；损坏文件不会切换 active。
- 已存在但无法反序列化的 FAISS 不再静默退化为空索引。
- 无 active pointer 时保留旧索引只读适配器。
- `INDEX_WRITE_MODE=legacy` 保留旧读写回滚路径。

## 7. SSE

`/api/chat/stream` 只允许：

- `progress`
- `evidence`
- 一次 `answer.final`
- 失败时 `stream-error`

后端不再订阅或转发任意 `on_chat_model_stream`，因此路由、规划、改写、子 Agent 和中间生成 token 不会进入 SSE。前端已同步新事件契约。

## 8. 自动验证

```text
uv run --extra dev python --version
Python 3.12.11

uv run --extra dev python -m pytest -q
179 passed, 3 warnings in 17.00s

uv run --extra dev ruff check .
All checks passed!

pnpm --dir web lint
通过

pnpm --dir web build
Next.js 16.2.0，编译、TypeScript、静态页面生成全部通过
```

3 个 warning 均来自 FAISS/SWIG 第三方类型缺少 `__module__`，不是 M1 回归。

## 9. 故障注入与人工检查

真实进程故障注入使用独立 `C:\tmp` 数据目录，完成后已删除：

1. 启动带 15 秒索引延迟的 API。
2. 一次上传两份论文文本。
3. 使用相同 `Idempotency-Key` 重复上传，返回同一个 job。
4. job 到达 `running` 时检查 active pointer，结果为不存在。
5. 强制终止 API 进程。
6. 同时启动两个 API 实例，共用同一个 SQLite 和 index root。
7. lease 过期后 job 被一个 worker 恢复并完成。

实际结果：

```json
{
  "uploaded_files": 2,
  "duplicate_key_same_job": true,
  "status_before_kill": "running",
  "pointer_during_running": false,
  "status_after_restart": "completed",
  "attempt_count": 2,
  "job_count": 1,
  "worker_lease_rows": 1,
  "active_version_rows": 1,
  "pointer_after_validation": true
}
```

其他确认：

- model、dimension、input mode 分别变化时，旧 manifest 均拒绝加载。
- 恶意 `../escape.txt` 返回 400，`UPLOAD_ROOT` 外没有文件。
- 超过大小上限返回 413，staging 文件被清理。
- 缺少 `Idempotency-Key` 返回 422。
- 相同 key、不同文件返回 409。
- SSE 测试注入了 routing/plan 私有字段，响应中没有这些内容，也没有 `token`/`citations` 旧事件。
- active version 可以先切换到新版本，再通过相同激活入口切回旧 ready version；SQLite 和 `active.json` 保持一致。

## 10. 已知坏例

- embedding contract 不兼容：拒绝查询，提示重建索引。
- active manifest 缺失、身份不匹配或状态不是 ready：拒绝激活和加载。
- FAISS 文件存在但损坏：拒绝加载，提示重建或回滚。
- 新版本缺少 FAISS/BM25/manifest：校验失败，旧 active 不变。
- 文件无可索引内容：本 attempt 失败，达到上限后 job 为 failed。
- 同一幂等 key 对应不同内容：409，不创建第二个 job。
- 文件名路径穿越、Windows 设备名或非法字符：400。

## 11. 回滚方法

### 回滚到旧 ready index version

```bash
python main.py activate-index <previous-version-id>
```

该命令先校验 manifest 和 embedding contract，再同步 SQLite active state 和 `active.json`。

### 回滚到 legacy 索引

```text
INDEX_WRITE_MODE=legacy
```

重启 API/CLI 后读写原 `INDEX_DIR`、`FAISS_DIR` 和 `BM25_PATH`。

### 恢复数据库

1. 停止 API。
2. 保留当前损坏数据库用于诊断。
3. 将最近的 `sessions.db.backup-v*-*` 复制回 `APP_DB_PATH`。
4. 启动 API，让 forward migration 重新执行。

代码回滚不删除 v2 表、index version、上传原文件或备份。

## 12. 实际修改文件

配置、入口和文档：

- `.env.example`
- `.gitignore`
- `README.md`
- `main.py`
- `docs/implementation/m1_acceptance.md`

Settings、LLM 和 Agent 构建边界：

- `core/settings.py`
- `core/factory.py`
- `core/config.py`（删除）
- `llms/llm.py`
- `agent/graph.py`
- `agent/research_search_agent.py`

数据库、API 和 worker：

- `api/main.py`
- `api/db/database.py`
- `api/db/models.py`
- `api/db/migrations.py`
- `api/models/chat.py`
- `api/models/indexing.py`
- `api/routers/chat.py`
- `api/routers/indexing.py`
- `api/services/graph_cache.py`
- `api/services/index_worker.py`

索引：

- `indexing/embeddings.py`
- `indexing/index_versions.py`
- `indexing/vectorstore.py`

前端：

- `web/src/app/chat/page.tsx`
- `web/src/app/kb/page.tsx`
- `web/src/hooks/useSSEStream.ts`
- `web/src/lib/api.ts`
- `web/src/lib/i18n.ts`
- `web/src/lib/types.ts`
- `web/src/lib/utils.ts`

测试：

- `tests/test_api.py`
- `tests/test_embeddings.py`
- `tests/test_settings.py`
- `tests/test_index_job_recovery.py`
- `tests/test_index_version.py`
- `tests/test_migrations.py`
- `tests/test_streaming.py`
- `tests/test_upload_security.py`

## 13. 剩余风险与 M2 进入条件

- SQLite/global lease 设计只面向 local-first 单机共享文件系统，不承诺多主机部署。
- legacy index 没有 manifest，无法证明历史 embedding 契约；只能通过显式 `INDEX_WRITE_MODE=legacy` 使用。
- 当前 `Indexer.index()` 是同步工作，强制进程终止依靠 lease 恢复；Core 不提供运行中单文件的硬取消。
- FAISS pickle 仍只允许加载本机可信索引目录。

M1 的必做项和验收项均已通过。从 M1 门槛看，M2 具备进入条件，但必须等待用户明确授权；本次交付不实施 M2。
