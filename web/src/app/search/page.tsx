"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { searchLibrary } from "@/lib/api";
import type { SearchResponse, SearchResult } from "@/lib/types";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState("");

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalized = query.trim();
    if (!normalized) {
      setError("请输入要查找的概念、方法或结论。");
      return;
    }
    setIsSearching(true);
    setError("");
    try {
      setResponse(await searchLibrary({ query: normalized }));
    } catch (caught) {
      setResponse(null);
      setError(resolveError(caught, "搜索失败"));
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <main
      id="main-content"
      className="mx-auto flex w-full max-w-7xl flex-col gap-10 px-5 py-10 sm:px-8 sm:py-14"
    >
      <header className="space-y-5 border-b border-[var(--line)] pb-9">
        <p className="editorial-kicker">Search / Page evidence</p>
        <h1 className="page-title">搜索页码证据</h1>
        <p className="page-description">
          结果同时展示论文、章节、页码、原文摘录与各阶段评分。打开证据即可回到
          PDF 对应页核验。
        </p>
      </header>

      <form
        className="grid gap-3 border-b border-[var(--line)] pb-8 sm:grid-cols-[1fr_auto]"
        onSubmit={handleSearch}
      >
        <label className="sr-only" htmlFor="search-query">
          搜索论文
        </label>
        <Input
          id="search-query"
          value={query}
          autoFocus
          placeholder="例如：hybrid retrieval 如何融合 BM25 与向量评分？"
          className="min-h-14 bg-[var(--panel)] text-base"
          onChange={(event) => setQuery(event.target.value)}
        />
        <Button className="min-h-14 px-8" disabled={isSearching} type="submit">
          {isSearching ? "正在检索" : "搜索论文"}
        </Button>
      </form>

      {error ? (
        <p role="alert" className="border-l-4 border-red-600 pl-4 text-sm text-red-800">
          {error}
        </p>
      ) : null}

      {response ? (
        <section aria-live="polite" className="space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-slate-600">
              “{response.query}” 找到 {response.total} 条证据
            </p>
            <p className="font-mono text-xs text-slate-400">
              INDEX {response.index_version}
            </p>
          </div>
          {response.degraded_reason ? (
            <p className="border-l-4 border-amber-600 bg-amber-50 px-4 py-3 text-sm text-amber-950">
              当前使用 BM25 降级检索：{response.degraded_reason}
            </p>
          ) : null}
          {response.results.length ? (
            <ol className="space-y-5">
              {response.results.map((result, index) => (
                <ResultCard
                  key={`${result.passage_id}-${index}`}
                  result={result}
                  rank={index + 1}
                />
              ))}
            </ol>
          ) : (
            <Card className="rounded-none border border-dashed border-[var(--line)] bg-transparent text-center shadow-none">
              <h2 className="font-serif text-2xl">没有匹配的页码证据</h2>
              <p className="mt-3 text-sm text-slate-600">
                尝试更具体的方法名、指标名或论文中的原始术语。
              </p>
            </Card>
          )}
        </section>
      ) : (
        <section className="grid gap-px bg-[var(--line)] md:grid-cols-3">
          {["方法名与模型名", "实验指标与数据集", "原文结论与限制"].map(
            (label, index) => (
              <div key={label} className="bg-[var(--panel)] p-6">
                <p className="font-mono text-xs text-[var(--accent-strong)]">
                  0{index + 1}
                </p>
                <p className="mt-6 font-serif text-xl">{label}</p>
              </div>
            ),
          )}
        </section>
      )}
    </main>
  );
}

function ResultCard({ result, rank }: { result: SearchResult; rank: number }) {
  return (
    <li>
      <article className="grid gap-0 border border-[var(--line)] bg-[var(--panel)] lg:grid-cols-[4rem_1fr_16rem]">
        <div className="border-b border-[var(--line)] p-4 font-mono text-xs text-slate-400 lg:border-r lg:border-b-0">
          {String(rank).padStart(2, "0")}
        </div>
        <div className="min-w-0 p-5 sm:p-7">
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
            <span className="status-pill">
              P. {result.page_start}
              {result.page_end !== result.page_start ? `–${result.page_end}` : ""}
            </span>
            <span>{result.section_title}</span>
            <span>{result.block_type}</span>
          </div>
          <h2 className="mt-4 font-serif text-2xl leading-snug text-slate-950">
            {result.paper_title || "未命名论文"}
          </h2>
          <p className="mt-2 text-sm text-slate-500">
            {result.authors.length ? result.authors.join("、") : "作者未知"}
            {result.year ? ` · ${result.year}` : ""}
          </p>
          <EvidenceQuote text={result.quote_text} />
          <div className="mt-6 flex flex-wrap gap-4">
            <Link
              href={result.paper_url}
              className="inline-flex min-h-10 items-center border-b border-[var(--accent)] text-sm font-semibold text-[var(--accent-strong)]"
            >
              在论文详情中定位
            </Link>
            <a
              href={result.pdf_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex min-h-10 items-center text-sm font-semibold text-slate-600 hover:text-slate-950"
            >
              直接打开 PDF 原页
            </a>
          </div>
        </div>
        <ScorePanel result={result} />
      </article>
    </li>
  );
}

function EvidenceQuote({ text }: { text: string }) {
  const isLong = text.length > 1000;
  return (
    <div className="mt-5">
      <blockquote className="border-l-2 border-[var(--accent)] pl-4 text-sm leading-7 text-slate-700">
        {isLong ? `${text.slice(0, 1000).trimEnd()}…` : text}
      </blockquote>
      {isLong ? (
        <details className="mt-3 text-sm">
          <summary className="min-h-10 cursor-pointer font-semibold text-[var(--accent-strong)]">
            展开完整原文摘录
          </summary>
          <blockquote className="mt-3 border-l-2 border-slate-300 pl-4 leading-7 text-slate-600">
            {text}
          </blockquote>
        </details>
      ) : null}
    </div>
  );
}

function ScorePanel({ result }: { result: SearchResult }) {
  const scoreRows = [
    ["Vector", result.scores.vector],
    ["BM25", result.scores.bm25],
    ["Fusion", result.scores.fusion],
    ["Final", result.scores.final],
  ] as const;
  return (
    <aside className="border-t border-[var(--line)] bg-[var(--panel-muted)] p-5 lg:border-t-0 lg:border-l">
      <p className="editorial-kicker">Score trace</p>
      <dl className="mt-4 divide-y divide-[var(--line)] font-mono text-xs">
        {scoreRows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-3 py-2.5">
            <dt className="text-slate-500">{label}</dt>
            <dd>{formatScore(value)}</dd>
          </div>
        ))}
        <div className="flex justify-between gap-3 py-2.5">
          <dt className="text-slate-500">Rank</dt>
          <dd>#{result.scores.rerank_rank}</dd>
        </div>
      </dl>
      {Object.keys(result.scores.boosts).length ? (
        <div className="mt-4 border-t border-[var(--line)] pt-3">
          <p className="text-xs text-slate-500">Boosts</p>
          {Object.entries(result.scores.boosts).map(([name, value]) => (
            <p
              key={name}
              className="mt-2 flex justify-between gap-3 font-mono text-xs"
            >
              <span>{name}</span>
              <span>{formatScore(value)}</span>
            </p>
          ))}
        </div>
      ) : null}
    </aside>
  );
}

function formatScore(value: number | null) {
  return value === null ? "N/A" : value.toFixed(4);
}

function resolveError(caught: unknown, fallback: string) {
  return caught instanceof Error && caught.message ? caught.message : fallback;
}
