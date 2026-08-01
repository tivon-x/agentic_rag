# M6 评测实验室实施计划

> 状态：待用户授权，且必须在 M5 验收完成后执行。
>
> 执行模型：`gpt-5.6-luna`，reasoning effort=`max`。

## 1. 目标与完成定义

M6 增加一个只读评测实验室，让访客理解产品为什么固定使用 B1 `v1_flat_rerank`。页面展示真实的策略选择、冻结数据、指标、失败类型和最终决策，不提供检索策略开关，也不执行评测或模型调用。

完成时必须同时满足：

1. 实验室数据仅来自已提交的评测资产和验收报告，每个展示结论可追溯到一个来源文件和字段或章节。
2. 页面清楚说明 S1 和 Adaptive 未获晋级的原因，固定 B1 是产品决策，不是“可选弱模式”。
3. 页面不包含 API Key、prompt、绝对路径、原始用户问题、完整 gold evidence、模型凭据或可运行评测入口。
4. 页面延续 `DESIGN.md` 的证据导向阅读体验，不做成通用数据看板。
5. 没有新增 API、数据库、后台任务、模型调用、索引写入或外部服务。
6. 通过来源核对、全量验证、浏览器检查和独立 review，修复所有可修复问题后再提交。

## 2. 前置条件与真实来源

### 2.1 必须存在

- `docs/implementation/m5_fixed_product_acceptance.md`，且 M5 已验收通过。
- `docs/implementation/m3_2_strategy_acceptance.md`。
- `artifacts/evals/v2_m3_2/m4_fixed_baseline.json`。
- `docs/implementation/m4_1_1_retrieval_quality_acceptance.md`。
- `docs/implementation/m4_1_2_adaptive_eval_acceptance.md`。
- 与上述报告关联的已提交 summary、逐题结果或配置文件。

开始前逐一检查存在性与 git 状态。任意来源缺失、结论冲突或不能复现数值时停止，请用户决定采用哪一份证据，不能按记忆补数。

### 2.2 已知必须如实呈现的结论

- B1 `v1_flat_rerank` 是冻结产品默认值。
- S1 未通过 M3.2 holdout Context Passage Recall gate，报告值为 0.854167，对照 B1 为 0.875。
- M4.1.1 与 M4.1.2 都没有证明 bounded Adaptive 的净收益，因此不做产品入口，M4.2 终止。
- M4.1.2 需要展示其报告中的关键现象，例如首轮部分覆盖后的补检增益不足、路由误触发或回归，数值必须从验收报告读取后再写入数据模块。

不能把“实验失败”写成“策略不够先进”，也不能把固定 B1 写成线上质量保证。

## 3. 选择的实现方式

使用构建时导入的、人工审核过的静态数据模块，不创建 API：

```text
评测报告与 artifacts
  -> 人工逐项核对、脱敏、填写静态展示模型
  -> web/src/data/evaluation-lab.ts
  -> /evaluation 静态页面与展示组件
```

这个方式优于在运行时读取 Markdown、JSON 或 API，因为 Next.js 客户端不能安全读取仓库文件，运行时解析会增加路径泄露和格式漂移风险。数据模块是经过审计的产品文案，不是第二套评测源。

唯一接近的替代方案是为实验室增加只读后端 API。拒绝它，因为现有目标只需要静态、冻结内容，API 不增加可信度，却增加接口、缓存和脱敏风险。

最脆弱的假设是验收报告包含足够的公开指标和坏例摘要。如果某个报告只含敏感原始样本，则该实验只展示决策和聚合指标，不能硬凑逐题样例。

## 4. 文件与数据接口

### 4.1 计划新增或修改

| 文件 | 责任 |
| --- | --- |
| `web/src/data/evaluation-lab.ts` | 唯一的前端展示数据，含来源 ID、指标、决策和脱敏坏例。 |
| `web/src/app/evaluation/page.tsx` | 静态实验室页面。 |
| `web/src/components/evaluation/*` | 只包含时间线、指标比较、决策记录、坏例 disclosure 等展示组件。 |
| `web/src/app/layout.tsx` 与 i18n 文案文件 | 增加“评测”入口，保持现有导航可访问性。 |
| `web/src/app/globals.css` | 仅在现有 token 与 evidence rail 不能覆盖时增加局部样式。 |
| `docs/implementation/m6_evaluation_lab_acceptance.md` | 来源审计、验证与 review 结论。 |

如果项目当前 i18n 结构不适合新导航，复用其既有模式，不创建第二个文本配置体系。

### 4.2 静态展示模型

`evaluation-lab.ts` 只导出显式常量和 TypeScript 类型。每个实验至少具备：

| 字段 | 规则 |
| --- | --- |
| `id` | `b1_baseline`、`s1_candidate`、`m4_1_1`、`m4_1_2` 等稳定展示 ID。 |
| `stage` | `M3.2`、`M4.1.1`、`M4.1.2`。 |
| `purpose` | 一句话说明验证的问题。 |
| `datasetVersion` | 只展示冻结版本或 SHA 缩写，不展示本地路径。 |
| `metrics` | label、value、对照、unit、sourceRef。无可靠数值时不填。 |
| `decision` | `promoted`、`not_promoted`、`terminated`。 |
| `decisionReason` | 与验收报告一致的简短原因。 |
| `badCases` | 已脱敏的 failure category、现象、影响，不放原题、quote 或 gold ID。 |
| `sourceRef` | 报告文件名和章节名，不含绝对路径。 |

实现一个构建时校验函数，检查 ID 唯一、所有指标有 sourceRef、失败实验有 decisionReason、bad case 不含 URL、绝对路径、`sk-`、`Bearer `、prompt 字段或原始问题字段。校验失败直接让 build 失败。

## 5. 页面信息架构与 UI 要求

### 5.1 路由与导航

- 路由固定为 `/evaluation`，导航名为“评测”。
- 不使用 query 参数切换策略或触发运行。
- 路由是静态内容，首次加载不请求后端。

### 5.2 页面顺序

1. **结论封面**：标题“为什么产品固定使用 B1”，说明实验室是冻结结果的只读说明。
2. **决策时间线**：B1 baseline、S1 gate、M4.1.1、M4.1.2，以结论先行的纵向阅读顺序展示。
3. **关键门槛**：用紧凑比较行展示 S1 Context Passage Recall 0.854167 与 B1 0.875，并明确 S1 未晋级。
4. **Adaptive 复验**：拆开 route、coverage、regression、termination 等报告已有维度，只显示有来源的指标。
5. **坏例档案**：脱敏的可展开条目，说明失败类别、观察到的现象、产品决策，不提供原题或完整证据。
6. **产品边界**：明确用户产品只用 fixed B1，实验室不提供策略切换、实时评测和质量承诺。
7. **来源与方法**：列出报告文件名、冻结数据版本和“如何阅读这些结果”。

### 5.3 视觉和交互

- 必须使用 UI skill 并落实 `DESIGN.md`，视觉标志继续是 evidence rail 和带页码感的编号条目。
- 用编辑部式决策记录替代 KPI 卡片墙。指标比较使用规则线和表格行，不用饼图、渐变图或装饰性仪表盘。
- 墨蓝只表示来源、已核验标记和链接，失败使用文字和小方块加颜色双重表达。
- 深层说明使用 `<details>`，键盘可开关，坏例默认收起。
- 375px 时时间线和指标表改为单列，不裁切数值或 sourceRef。
- 所有图形关系可由文本和表格理解。M6 不引入图表库。

## 6. 实施阶段

### 阶段 A：来源审计与静态数据

1. 阅读第 2 节所有来源，建立一张本地审计表，列出展示字段、原始来源、脱敏动作、是否可公开。
2. 对每个数值逐项比对，数值格式不一致时以验收报告正文的最终结论为准，并记录差异。
3. 写入 `web/src/data/evaluation-lab.ts`，不复制整段报告，不复制原题和 prompt。
4. 实现构建时校验，手工故意加入一个路径或敏感前缀，确认校验会失败后再移除测试数据。

阶段 A 结束时，数据可独立被任何简单页面导入，并且来源可审计。

### 阶段 B：页面和组件

1. 创建 `/evaluation` 页面和最少的展示组件，不做通用组件库。
2. 将每个决策和指标从静态模型渲染，禁止在 JSX 中另写数字或结论。
3. 更新导航和页面 metadata。导航的语义、当前页状态和键盘 focus 必须可用。
4. 将 sourceRef 展示为报告名和章节，不做本地文件超链接，避免本地路径泄露。
5. 实验室页面不读取 `window`、不打开 SSE、不请求 `/api`。

阶段 B 结束时，页面可单独部署为静态页面，并忠实展示冻结数据。

### 阶段 C：内容、隐私和视觉验收

1. 按第 5 节顺序核对每个结论，尤其检查 B1、S1 和 Adaptive 的因果关系没有写反。
2. 搜索页面 bundle 和数据模块，确认不存在 `api_key`、`authorization`、`Bearer `、`sk-`、绝对盘符路径、完整 prompt、未脱敏问题或模型凭据。
3. 使用真实浏览器检查 `/evaluation`、首页导航、Chat、Search。检查 1440px 与 375px。
4. 截图放 `output/playwright/`，不提交临时浏览器目录。

## 7. 自动验证与人工验收

```bash
uv run --extra dev ruff check .
uv run --extra dev python -m pytest -q
npm --prefix web run lint
npm --prefix web run build
git diff --check
rg -n -i "api[_-]?key|authorization|bearer |sk-|[A-Za-z]:\\\\Users\\\\|prompt" web/src/data/evaluation-lab.ts web/src/app/evaluation web/src/components/evaluation
```

最后一条搜索有命中时，逐条人工判定。任何秘密、绝对路径、完整 prompt 或未脱敏原题命中都不能提交。

人工验收清单：

- `/evaluation` 刷新后不产生 API 请求或模型调用。
- B1、S1、M4.1.1、M4.1.2 都有来源和正确晋级结论。
- S1 Context Passage Recall 比较为 0.854167 对 0.875，结论为未晋级。
- 所有 bad case 都是脱敏摘要，无法还原原问题或完整 evidence。
- Tab、Enter、Space 可操作导航、链接和 details。
- 1440px 与 375px 没有溢出、遮挡和读不清的低对比文本。

## 8. 独立代码审查与修复闭环

全量验证完成后调用独立 review subagent，不能由实现 agent 自评替代：

```text
模型：gpt-5.6-luna
reasoning effort：max
角色：只读 reviewer，不直接修改工作区
范围：本 Goal diff、静态数据来源一致性、脱敏、运行时只读边界、bundle/页面性能、可访问性、导航、回归风险
输出：按 P0/P1/P2 分级，附文件和行号、复核来源、修复建议；没有问题也说明审查过的风险面。
```

处理规则：

1. P0/P1 必须修复后重跑全部第 7 节命令。
2. 能在 M6 范围内处理的 P2 必须修复并重跑受影响检查。
3. 只有报告不成立或超出 M6 范围的问题可以不改，验收报告必须引用判断依据。
4. 修复触及数据模块、脱敏规则或静态只读边界时，重新调用 reviewer 审查修复 diff。
5. 如果当前工具无法使用 `gpt-5.6-luna`，停止在 review 门前说明原因，不可伪称完成指定 review。等待用户指定替代模型。

## 9. 验收报告、回滚与提交

创建 `docs/implementation/m6_evaluation_lab_acceptance.md`，包含：

- 开始时 HEAD、来源清单和逐项来源审计。
- 静态数据模型及脱敏规则。
- 页面、导航和截图检查。
- 自动命令输出与坏例。
- reviewer 的模型、effort、发现、修复和未修复理由。
- 回滚步骤与实际修改文件。

回滚只移除 `/evaluation`、导航入口、展示组件和静态数据模块。它不影响 Search、fixed Chat、Library、会话数据、评测资产或 active index。

仅在验收报告、review 修复和所有验证完成后执行：

```bash
git add docs/research/m6_evaluation_lab_implementation_plan.md docs/implementation/m6_evaluation_lab_acceptance.md <实际 Web 文件>
git commit -m "feat: add read-only evaluation lab"
```

不推送，不创建 PR。提交后停止，等待用户决定是否开始 M7。
