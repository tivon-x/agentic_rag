# 发布前硬化验收报告

状态：正式验收通过（2026-08-31）。

## 范围

本次工作处理合并前审查发现的边界问题，不启动 M8，也不改变生产 Pipeline、answer strategy 或 M8 部署范围。

## 修复

- API CLI 仅允许 loopback 监听；M8 部署仍保持暂缓。
- 上传增加文件数、单请求总大小、磁盘占用和持久索引队列上限；索引版本和失败版本按保留数清理。
- Chat 限制单条内容和会话历史大小；会话追加使用 SQLite 原子事务与 compare-and-swap，重复回答请求返回冲突。
- Agent 预算改为调用状态字段，语料库配置更新会失效 graph cache。
- 空文本上传明确失败；带 `paper_id` 的搜索先过滤再应用 limit。

## 验证

```text
uv run --extra dev pytest -q
346 passed, 3 warnings

uv run --extra dev ruff check .
通过

git diff --check
通过
```

测试使用 Fake 模型或临时目录，没有发起外部模型、embedding 或 Benchmark 请求。

## 边界

- 支持的 API 启动入口拒绝非 loopback 主机；真正的多用户鉴权和公网部署仍属于暂缓的 M8。
- 版本保留策略只删除非 active 的旧 ready 版本和过期失败版本，不影响当前 active index。
