import Link from "next/link";

import { kiteEvaluation } from "@/lib/kite-evaluation";
import {
  KITE_PAIRWISE_KEYS,
  KITE_PIPELINE_KEYS,
  type KitePipelineKey,
} from "@/lib/types";

const pipelineLabels: Record<KitePipelineKey, string> = {
  b0: "hybrid",
  b1: "fixed rerank",
  b2: "RRF + metadata",
  b3: "neighbor expansion",
};

function pipelineLabel(key: string | null | undefined) {
  if (!key) {
    return "默认路径";
  }
  return `${key.toUpperCase()} / ${pipelineLabels[key as KitePipelineKey] ?? key}`;
}

function formatScore(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "暂无数据";
}

function formatMilliseconds(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value / 1000).toFixed(1)} s`
    : "暂无数据";
}

function formatTokens(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.round(value).toLocaleString("en-US")
    : "暂无数据";
}

function formatCount(validCount: number | null | undefined, caseCount: number | null | undefined) {
  return typeof validCount === "number" && Number.isFinite(validCount) &&
      typeof caseCount === "number" && Number.isFinite(caseCount)
    ? `${validCount}/${caseCount}`
    : "暂无数据";
}

function formatCaseIds(ids: string[] | null | undefined) {
  if (!Array.isArray(ids)) {
    return "暂无数据";
  }
  return ids.length > 0 ? ids.join("、") : "无";
}

function isPipelineKey(value: unknown): value is KitePipelineKey {
  return typeof value === "string" && KITE_PIPELINE_KEYS.includes(value as KitePipelineKey);
}

export const metadata = {
  title: "评测",
  description: "固定 KITE AI Papers 协议下的只读 Pipeline 诊断结果。",
};

export default function EvaluationPage() {
  const decision = kiteEvaluation.production_decision;
  const defaultPipeline = isPipelineKey(decision?.default_pipeline)
    ? decision.default_pipeline
    : null;
  const defaultPipelineLabel = pipelineLabel(defaultPipeline);
  const defaultName = decision?.default_name?.trim() || "暂无名称";
  const baseline = defaultPipeline ? kiteEvaluation.pipelines?.[defaultPipeline] : undefined;
  const pipelineRows = KITE_PIPELINE_KEYS.map((key) => ({
    key,
    pipeline: kiteEvaluation.pipelines?.[key],
  }));
  const pairwiseRows = KITE_PAIRWISE_KEYS.map((key) => ({
    key,
    comparison: kiteEvaluation.pairwise_vs_b1?.[key],
  }));
  const caseReviews = Array.isArray(kiteEvaluation.case_reviews)
    ? kiteEvaluation.case_reviews
    : [];
  const promotionCandidates = Array.isArray(kiteEvaluation.promotion_candidates)
    ? kiteEvaluation.promotion_candidates.filter(isPipelineKey)
    : [];
  const runStatus = kiteEvaluation.formal_run === true
    ? "正式运行"
    : "非正式诊断（formal_run=false）";

  return (
    <main
      id="main-content"
      className="mx-auto flex w-full max-w-[90rem] flex-1 flex-col px-5 py-10 sm:px-8 sm:py-16"
    >
      <header className="grid gap-8 border-b border-[var(--ink)] pb-10 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,0.42fr)] lg:items-end lg:gap-16">
        <div>
          <p className="editorial-kicker">Evaluation desk / read-only</p>
          <h1 className="page-title mt-5 max-w-4xl">哪条检索链路，值得继续作为默认路径？</h1>
          <p className="page-description mt-6">
            这里展示同一套 KITE AI Papers snapshot 下的固定 Pipeline 对照。当前摘要是{runStatus}，只用于工程诊断，不是冻结正式结果，候选也不代表已获 promotion 批准。
          </p>
        </div>
        <aside className="border-t-2 border-[var(--ink-blue)] pt-4">
          <p className="editorial-kicker">Decision</p>
          <p className="mt-4 font-serif text-3xl leading-tight">默认路径：{defaultPipelineLabel}</p>
          <p className="mt-3 text-sm leading-7 text-[var(--muted-ink)]">
            {defaultName}。当前状态：{runStatus}。诊断候选：{promotionCandidates.map(pipelineLabel).join("、") || "暂无数据"}，未获生产批准。评测页只读，不提供参数编辑、索引操作或 Pipeline 切换。
          </p>
        </aside>
      </header>

      <section className="mt-10 grid gap-8 lg:grid-cols-[minmax(0,1.25fr)_minmax(19rem,0.75fr)]" aria-labelledby="snapshot-title">
        <div>
          <div className="flex flex-wrap items-end justify-between gap-4 border-b border-[var(--line)] pb-4">
            <div>
              <p className="editorial-kicker">Diagnostic snapshot</p>
              <h2 id="snapshot-title" className="mt-2 font-serif text-3xl">运行条件</h2>
            </div>
            <a
              className="text-link inline-flex"
              href={kiteEvaluation.upstream_repository}
              target="_blank"
              rel="noreferrer"
            >
              查看上游仓库 <span className="ml-2" aria-hidden="true">↗</span>
            </a>
          </div>
          <dl className="mt-5 grid gap-x-8 gap-y-4 border-b border-[var(--line)] pb-6 sm:grid-cols-2">
            <div>
              <dt className="font-mono text-[0.68rem] uppercase tracking-[0.12em] text-[var(--muted-ink)]">Upstream commit</dt>
              <dd className="mt-1 break-all font-mono text-xs text-[var(--ink)]">{kiteEvaluation.upstream_commit}</dd>
            </div>
            <div>
              <dt className="font-mono text-[0.68rem] uppercase tracking-[0.12em] text-[var(--muted-ink)]">Corpus</dt>
              <dd className="mt-1 text-sm text-[var(--ink)]">{kiteEvaluation.corpus_file_count} 个 PDF</dd>
            </div>
            <div>
              <dt className="font-mono text-[0.68rem] uppercase tracking-[0.12em] text-[var(--muted-ink)]">Query SHA-256</dt>
              <dd className="mt-1 break-all font-mono text-xs text-[var(--ink)]">{kiteEvaluation.query_sha256}</dd>
            </div>
            <div>
              <dt className="font-mono text-[0.68rem] uppercase tracking-[0.12em] text-[var(--muted-ink)]">Corpus SHA-256</dt>
              <dd className="mt-1 break-all font-mono text-xs text-[var(--ink)]">{kiteEvaluation.corpus_file_sha256}</dd>
            </div>
            <div>
              <dt className="font-mono text-[0.68rem] uppercase tracking-[0.12em] text-[var(--muted-ink)]">Models</dt>
              <dd className="mt-1 text-sm text-[var(--ink)]">生成 {kiteEvaluation.generation_model} · 判分 {kiteEvaluation.judge_model}</dd>
            </div>
          </dl>
        </div>

        <aside className="border-t border-[var(--line)] pt-4 lg:border-l lg:border-t-0 lg:pl-6">
          <p className="editorial-kicker">Reading note</p>
          <p className="mt-4 text-sm leading-7 text-[var(--muted-ink)]">
            {defaultPipelineLabel} 是对照线。候选必须同时满足分数、逐题胜负、延迟、上下文成本和引用完整性，才有资格进入生产讨论。当前页面为{runStatus}，候选列表不代表已批准，也不会在评测页自动切换。
          </p>
          <Link className="text-link mt-5 inline-flex" href="/chat">
            回到证据问答 <span className="ml-2" aria-hidden="true">↗</span>
          </Link>
        </aside>
      </section>

      <section className="mt-12" aria-labelledby="pipeline-title">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-[var(--line)] pb-4">
          <div>
            <p className="editorial-kicker">Pipeline comparison</p>
            <h2 id="pipeline-title" className="mt-2 font-serif text-3xl">同一协议，四条路径</h2>
          </div>
          <p className="max-w-md text-sm leading-7 text-[var(--muted-ink)]">
            分数保留四位小数，延迟使用 p95，context tokens 为每题打包上下文的平均值。
          </p>
        </div>
        <div className="mt-5 overflow-x-auto border-y border-[var(--line)]">
          <table className="w-full min-w-[46rem] border-collapse text-left">
            <caption className="sr-only">KITE Pipeline 评测结果</caption>
            <thead>
              <tr className="font-mono text-[0.68rem] uppercase tracking-[0.1em] text-[var(--muted-ink)]">
                <th scope="col" className="py-3 pr-4 font-normal">Pipeline</th>
                <th scope="col" className="py-3 pr-4 font-normal">KITE score</th>
                <th scope="col" className="py-3 pr-4 font-normal">有效题数</th>
                <th scope="col" className="py-3 pr-4 font-normal">p95 latency</th>
                <th scope="col" className="py-3 font-normal">context tokens</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--line)]">
              {pipelineRows.map(({ key, pipeline }) => (
                <tr key={key} className={key === defaultPipeline ? "bg-[var(--paper)]" : undefined}>
                  <th scope="row" className="py-4 pr-4 font-serif text-xl font-normal">
                    <span className="font-mono text-xs text-[var(--ink-blue)]">{pipelineLabel(key)}</span>
                    {key === defaultPipeline ? <span className="ml-3 status-pill align-middle">当前默认</span> : null}
                  </th>
                  <td className="py-4 pr-4 font-mono text-sm tabular-nums">{formatScore(pipeline?.mean_score)}</td>
                  <td className="py-4 pr-4 font-mono text-sm tabular-nums">{formatCount(pipeline?.valid_count, pipeline?.case_count)}</td>
                  <td className="py-4 pr-4 font-mono text-sm tabular-nums">{formatMilliseconds(pipeline?.p95_latency_ms)}</td>
                  <td className="py-4 font-mono text-sm tabular-nums">{formatTokens(pipeline?.mean_context_tokens)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs leading-6 text-[var(--muted-ink)]">
          {defaultPipelineLabel} baseline：{formatScore(baseline?.mean_score)} 分，p95 {formatMilliseconds(baseline?.p95_latency_ms)}，平均 {formatTokens(baseline?.mean_context_tokens)} tokens。
        </p>
        <p className="mt-2 text-xs text-[var(--muted-ink)] sm:hidden">表格可左右滑动查看完整指标。</p>
      </section>

      <section className="mt-12" aria-labelledby="pairwise-title">
        <div className="border-b border-[var(--line)] pb-4">
          <p className="editorial-kicker">Case audit</p>
          <h2 id="pairwise-title" className="mt-2 font-serif text-3xl">逐题差异与坏例</h2>
        </div>
        <div className="divide-y divide-[var(--line)] border-b border-[var(--line)]">
          {pairwiseRows.map(({ key, comparison }) => (
            comparison ? (
              <details key={key} className="group py-4">
                <summary className="flex min-h-10 cursor-pointer list-none items-center justify-between gap-4 font-serif text-xl group-open:text-[var(--ink-blue)] [&::-webkit-details-marker]:hidden">
                  <span>{pipelineLabel(key)}</span>
                  <span className="font-mono text-xs text-[var(--muted-ink)]">{comparison.candidate_wins ?? "暂无数据"} 胜 · {comparison.ties ?? "暂无数据"} 平 · {comparison.candidate_losses ?? "暂无数据"} 负</span>
                </summary>
                <div className="mt-4 grid gap-4 text-sm leading-7 text-[var(--muted-ink)] sm:grid-cols-3">
                  <p><strong className="text-[var(--ink)]">胜出</strong><br />{formatCaseIds(comparison.win_case_ids)}</p>
                  <p><strong className="text-[var(--ink)]">持平</strong><br />{formatCaseIds(comparison.tie_case_ids)}</p>
                  <p><strong className="text-[var(--ink)]">落后</strong><br />{formatCaseIds(comparison.loss_case_ids)}</p>
                </div>
              </details>
            ) : (
              <p key={key} className="py-4 text-sm leading-7 text-[var(--muted-ink)]">
                {pipelineLabel(key)}：暂无逐题对照数据。
              </p>
            )
          ))}
        </div>
        <div className="mt-8 grid gap-8 lg:grid-cols-2">
          {caseReviews.length > 0 ? (
            caseReviews.map((review) => (
              <article key={review.case_id} className="border-t-2 border-[var(--signal-amber)] pt-4">
                <p className="font-mono text-xs text-[var(--signal-amber)]">{review.case_id} · {review.severity}</p>
                <p className="mt-2 font-serif text-xl">{review.summary}</p>
                <p className="mt-2 text-sm text-[var(--muted-ink)]">候选得分：{review.score ?? "无有效分数"}</p>
              </article>
            ))
          ) : (
            <p className="border-t border-[var(--line)] pt-4 text-sm leading-7 text-[var(--muted-ink)]">
              当前诊断摘要没有单独标注坏例。请以各 Pipeline 的逐题 JSON 报告为准，评测页面不会重新运行问题或模型。
            </p>
          )}
        </div>
      </section>

      <section className="mt-12 border-t-2 border-[var(--ink-blue)] pt-5" aria-labelledby="history-title">
        <p className="editorial-kicker">Existing evidence</p>
        <h2 id="history-title" className="mt-2 font-serif text-3xl">历史结论仍然有效</h2>
        <div className="mt-5 grid gap-5 border-b border-[var(--line)] pb-8 text-sm leading-7 text-[var(--muted-ink)] sm:grid-cols-2">
          <p><strong className="text-[var(--ink)]">M3.2</strong><br />固定策略收口，当前生产决策记录为 {defaultPipelineLabel}，复杂固定候选未晋级。</p>
          <p><strong className="text-[var(--ink)]">M4.1</strong><br />Adaptive 两次复验未证明净收益，生产默认继续使用 fixed，不在此页重新调参。</p>
        </div>
      </section>
    </main>
  );
}
