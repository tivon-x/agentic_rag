# M4.2 持久 Run 与恢复交接

## 1. 进入条件

只有以下条件全部满足才开始：

- `docs/implementation/m4_1_acceptance.md` 存在
- `m4_1_quality_passed=true`
- `m4_2_entry_ready=true`
- M4.1 commit、数据集 hash、baseline contract 和逐题结果可复现
- 用户明确批准 M4.2

任一条件不满足立即停止。M4.2 不负责补救 M4.1 的质量问题，也不能调整 route、
证据充分性、补检预算、grader 或冻结数据集。

开始前运行：

```powershell
git status --short --branch -uall
git rev-parse HEAD
```

读取：

1. `AGENTS.md`
2. `web/AGENTS.md`
3. `docs/research/v2_upgrade_plan.md`
4. `docs/research/phase2_goal_prompts.md` 的 Goal 4B
5. `docs/implementation/m1_acceptance.md`
6. `docs/implementation/m3_2_strategy_acceptance.md`
7. `docs/implementation/m4_1_acceptance.md`
8. 当前 `api/db/` migration 和 database 实现
9. `api/services/index_worker.py`
10. 当前 chat route、session repository、SSE 和前端 chat 实现
11. 当前安装版本的 LangGraph 和 Next.js 文档

保护所有已有未提交文件。路径冲突时停止，不覆盖用户改动。

## 2. 目标和边界

M4.2 把 M4.1 已通过的可选 Adaptive 链路接入持久 run，使进程终止、lease 过期和
SSE 断线后可以继续，同时保证消息、证据、claim 和事件不重复。

本 Goal 不做：

- 不重新评测或调优 M4.1 策略
- 不修改冻结 B1 contract
- 不实现完整 debug trace 页面
- 不实现 Compare、Workspace、artifact、备份产品功能或 Docker
- 不新增 Redis、Celery、PostgreSQL 或外部任务服务
- 不把 SQLite 描述为通用高并发生产方案

SQLite 的承诺范围是单机、单用户、单 run worker 的可恢复演示。未来多实例部署使用
Postgres checkpointer 和相应的任务调度方案。

## 3. 组件关系

```text
Chat UI
  |
  v
Runs API
  |
  v
SQLite runs + run_events
  |
  v
single run worker
  |
  v
LangGraph + AsyncSqliteSaver
  |
  v
frozen B1 retriever + evidence repository
  |
  v
finalization transaction
  |
  v
SSE replay from run_events
```

只有 run worker 执行 graph。API 创建、查询、取消 run，不在请求线程内执行同一 run。
SSE 只读 `run_events`，不直接消费进程内 graph stream。

## 4. 数据库

使用现有 forward-only migration 机制。修改已有数据库前创建可恢复备份。

新增或完成以下表：

### 4.1 chat_messages

至少包含：

- `id`
- `session_id`
- `run_id`
- `role`
- `content_json`
- `ordinal`
- `created_at`

同一 session 的 `ordinal` 唯一。Core 旧 JSON history 保留兼容读取一个版本周期，
新 run 只写 `chat_messages`。

### 4.2 runs

至少包含：

- `id`
- `session_id`
- `status`
- `query`
- `history_snapshot_json`
- `initial_state_json`
- `index_version`
- `baseline_config_hash`
- `answer_strategy`
- `budget_json`
- `worker_id`
- `lease_expires_at`
- `heartbeat_at`
- `attempt_count`
- `cancel_requested_at`
- `error_code`
- `error_message`
- `final_answer_json`
- `created_at`
- `started_at`
- `completed_at`
- `updated_at`

状态：

```text
queued -> running -> completed
queued -> cancel_requested -> cancelled
running -> cancel_requested -> cancelled
queued|running -> failed
```

同一 session 同时只允许一个 `queued` 或 `running` run，由数据库部分唯一索引保证。
第二个创建请求返回 409。

### 4.3 run_events

至少包含：

- `run_id`
- `seq`
- `event_type`
- `payload_json`
- `idempotency_key`
- `created_at`

`(run_id, seq)` 和 `(run_id, idempotency_key)` 唯一。

允许的用户事件：

- `run.queued`
- `run.started`
- `plan.ready`
- `retrieval.completed`
- `evidence.accepted`
- `validation.completed`
- `answer.final`
- `run.failed`
- `run.cancelled`

不写 prompt、API Key、完整环境变量或 provisional answer。

### 4.4 retrieval_candidates

唯一键至少覆盖：

`(run_id, round, plan_item_id, passage_id, stage)`

保存当前 index version、各阶段 rank/score、query ID 和接受状态。恢复时使用 upsert。

### 4.5 evidence_items

唯一键至少覆盖：

`(run_id, round, evidence_id)`

保存稳定 passage、paper、section、page、quote、index version 和 claim 关联。最终
artifact 不依赖 checkpoint 数据库。

## 5. 创建 Run 的事务

`POST /api/runs` 在一个事务中：

1. 校验 session 没有 active run。
2. 读取当前 chat messages，生成不可变 `history_snapshot_json`。
3. 读取当前 active index version。
4. 读取 M4 fixed baseline config hash。
5. 固化 answer strategy 和预算。
6. 写入 `runs` queued 记录和 `run.queued` 事件。

创建完成后返回 `202`、`run_id`、状态和 events URL。不能在此请求中启动第二套后台
task。

## 6. Compact GraphState 和 Checkpoint

`thread_id=run_id`。session history 来自 `history_snapshot_json`，checkpoint 不承担
跨 run 记忆。

GraphState 只保存：

- run、session、query 和 scope ID
- history summary
- strategy
- 最多 3 个 plan item
- round
- candidate IDs
- 最多 12 个 evidence IDs
- coverage
- 固定预算
- termination reason
- final result

禁止保存：

- LangChain `Document`
- 完整候选和 quote
- 完整会话消息
- Pydantic model 实例
- LLM client
- repository 或数据库连接

新增并锁定与当前 LangGraph 兼容的 `langgraph-checkpoint-sqlite`。先做最小导入、
setup、checkpoint、进程重建和 resume 测试，再接入业务 graph。

构造要求：

- 独立 `CHECKPOINT_DB_PATH`
- `AsyncSqliteSaver`
- `JsonPlusSerializer(pickle_fallback=False)`
- `LANGGRAPH_STRICT_MSGPACK=true`

checkpoint 在节点边界保存。恢复可能重新执行整个节点，所以节点数据库副作用必须
使用唯一键、upsert 或 read-before-write 保持幂等。

## 7. Run Worker

复用 `api/services/index_worker.py` 的模式，不建立平行线程池或外部调度器。

规则：

- FastAPI 启动时生成唯一 `worker_id`
- 单 worker 每秒扫描 queued 和 lease 过期的 running run
- 使用 `BEGIN IMMEDIATE` 领取
- lease 30 秒
- heartbeat 10 秒
- 最大尝试 2 次
- shutdown 停止领取新 run，并安全结束 heartbeat

首次执行：

- 用保存的 initial state
- config 中设置 `thread_id=run_id`

恢复执行：

- 有 checkpoint 时用同一 config 恢复
- 没有 checkpoint 时用保存的 initial state 幂等重启

每个节点边界检查 `cancel_requested`。取消后写 `run.cancelled`，不能继续生成
`answer.final`。

## 8. 完成事务

只有以下内容在同一事务写成功，run 才能变为 completed：

- final answer
- claims
- citations
- evidence items
- assistant chat message
- `answer.final` 事件
- completed 状态和时间

事务失败时 run 保持可恢复状态，不能出现 completed 但没有 assistant message，或
消息存在但 evidence 缺失。

## 9. API 和 SSE

实现：

- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events`
- `POST /api/runs/{run_id}/cancel`

SSE：

- event ID 使用 `run_events.seq`
- 支持 `Last-Event-ID`
- 重连后只返回 seq 更大的事件
- completed、failed、cancelled 后发送终止事件并关闭
- 慢客户端不能阻塞 worker
- 不直接转发 graph token stream

M5 之前不实现 `/api/runs/{run_id}/debug` 的完整用户界面。

## 10. 前端

修改 `/chat`：

- 创建 run
- 展示 queued、running 和取消状态
- 展示已确认 evidence
- 断线时带 `Last-Event-ID` 重连
- 只展示一次 final answer
- 允许取消
- 刷新页面后按 run ID 恢复状态

不展示 prompt、完整候选、内部异常堆栈或 provisional answer。技术调试模式留给 M5。

修改前读取仓库 `web/AGENTS.md` 和当前安装版本 Next.js 文档。

## 11. 配置和依赖

所有配置进入 `core/settings.py`：

```text
ANSWER_STRATEGY=fixed
CHECKPOINT_DB_PATH=data/checkpoints.db
RUN_WORKER_LEASE_SECONDS=30
RUN_WORKER_HEARTBEAT_SECONDS=10
RUN_MAX_ATTEMPTS=2
LANGGRAPH_STRICT_MSGPACK=true
```

未知 answer strategy 或不安全 serializer 配置必须启动失败。

依赖：

- 新增并锁定 `langgraph-checkpoint-sqlite`
- 不新增外部服务或新的 API Key
- 继续使用现有 LLM、embedding、SQLite 和 FAISS 配置

## 12. 自动测试

至少覆盖：

- migration backup 和 forward-only 升级
- 同 session active run 唯一约束
- create run history snapshot 和 baseline hash
- worker 原子领取
- heartbeat 和 lease 过期
- 两 worker 竞争
- 最大尝试次数
- checkpoint 存在和不存在两种恢复
- 节点重执行不重复 candidate、evidence、claim、event 和 message
- finalization 全事务成功与回滚
- queued、retrieval、generation 边界取消
- SSE 首次连接、断线、Last-Event-ID、终态关闭
- fixed 完整绕过 worker 和 checkpoint
- checkpoint 清理后 completed result 仍可读取
- serializer 拒绝非 JSON 状态

执行：

```powershell
uv run --extra dev python -m pytest tests/test_run_repository.py tests/test_run_recovery.py tests/test_run_streaming.py tests/test_agent_budget.py -q
uv run --extra dev ruff check agent api core tests
uv run --extra dev python -m pytest -q
npm --prefix web run lint
npm --prefix web run build
```

## 13. 故障注入

必须手工执行并记录：

1. 第一轮检索完成后终止 API，重启后继续同一 run。
2. finalization 事务写入前终止 API，重启后只写一次 final answer。
3. 让 worker lease 过期，启动第二 worker 竞争。
4. SSE 断线，使用最后 event ID 重连。
5. 同一 session 并发创建两个 run。
6. 在 queued、retrieval 和 generation 节点边界分别取消。

验收结果：

- 最多一个 worker 持有 lease
- run 能恢复或使用保存输入幂等重启
- assistant message、claims、evidence、events 和工具副作用不重复
- 第二个并发 run 返回 409
- 取消后没有 `answer.final`
- SSE 重连不重复旧事件

## 14. 回滚

- 设置 `ANSWER_STRATEGY=fixed`
- 保留 `/api/chat` fixed 链路一个版本周期
- migration 不回退，新表保留不用
- checkpoint 数据库可以删除
- 删除 checkpoint 不影响 completed answer、chat message 和 evidence
- M4.2 失败不能修改 M4.1 质量结论

## 15. 交付

创建 `docs/implementation/m4_2_acceptance.md`，记录：

- migration 和备份
- run 状态机
- GraphState 字段和序列化大小
- checkpoint 依赖版本
- worker lease、heartbeat 和恢复协议
- API 和 SSE 契约
- 故障注入结果
- 自动测试、Ruff、前端 lint/build
- SQLite 适用边界
- 已知坏例
- 回滚方式
- 实际修改文件

只有全部验收通过后创建一个独立 commit，不推送。完成后停止，不得自动进入 M5。
