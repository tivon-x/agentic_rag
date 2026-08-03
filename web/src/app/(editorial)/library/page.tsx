"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { FileUpload } from "@/components/FileUpload";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  fetchIndexingJob,
  fetchPapers,
  uploadKnowledgeFiles,
} from "@/lib/api";
import type {
  IndexingJobResponse,
  PaperSummary,
  ParseStatus,
} from "@/lib/types";

const STATUS_LABELS: Record<ParseStatus, string> = {
  queued: "等待解析",
  parsing: "解析中",
  parsed: "解析完成",
  degraded: "降级完成",
  needs_ocr: "需要 OCR",
  failed: "解析失败",
};

const STATUS_FILTERS: Array<{ value: "" | ParseStatus; label: string }> = [
  { value: "", label: "全部状态" },
  { value: "parsed", label: "解析完成" },
  { value: "degraded", label: "降级完成" },
  { value: "needs_ocr", label: "需要 OCR" },
  { value: "failed", label: "解析失败" },
];

export default function LibraryPage() {
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [jobs, setJobs] = useState<IndexingJobResponse[]>([]);
  const [query, setQuery] = useState("");
  const [parseStatus, setParseStatus] = useState<"" | ParseStatus>("");
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadPapers = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await fetchPapers({
        query: query.trim() || undefined,
        parseStatus: parseStatus || undefined,
      });
      setPapers(data.items);
      setError("");
    } catch (caught) {
      setError(resolveError(caught, "论文目录加载失败"));
    } finally {
      setIsLoading(false);
    }
  }, [parseStatus, query]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadPapers();
    }, 180);
    return () => window.clearTimeout(timer);
  }, [loadPapers]);

  useEffect(() => {
    const active = jobs.filter(
      (job) => job.status === "queued" || job.status === "running",
    );
    if (active.length === 0) {
      return;
    }
    const timer = window.setInterval(() => {
      void Promise.all(active.map((job) => fetchIndexingJob(job.id)))
        .then((updated) => {
          setJobs((current) =>
            current.map(
              (job) => updated.find((item) => item.id === job.id) ?? job,
            ),
          );
          if (updated.some((job) => job.status === "completed")) {
            void loadPapers();
          }
        })
        .catch((caught) => setError(resolveError(caught, "任务状态获取失败")));
    }, 1800);
    return () => window.clearInterval(timer);
  }, [jobs, loadPapers]);

  const counts = useMemo(
    () => ({
      total: papers.length,
      ready: papers.filter((paper) => paper.parse_status === "parsed").length,
      attention: papers.filter((paper) =>
        ["degraded", "needs_ocr", "failed"].includes(paper.parse_status),
      ).length,
    }),
    [papers],
  );

  async function handleUpload() {
    if (files.length === 0) {
      setError("请先选择 PDF、Markdown 或文本文件。");
      return;
    }
    setIsUploading(true);
    setError("");
    setNotice("");
    try {
      const created = await uploadKnowledgeFiles({
        files,
        indexMode: "flat",
      });
      const jobIds = Array.from(new Set(created.map((item) => item.job_id)));
      const nextJobs = await Promise.all(jobIds.map(fetchIndexingJob));
      setJobs((current) => [...nextJobs, ...current]);
      setFiles([]);
      setNotice("文件已进入解析队列。解析和索引状态会在这里自动更新。");
      await loadPapers();
    } catch (caught) {
      setError(resolveError(caught, "上传失败"));
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <main
      id="main-content"
      className="mx-auto flex w-full max-w-[90rem] flex-col gap-10 px-5 py-10 sm:px-8 sm:py-14"
    >
      <header className="grid items-end gap-7 border-b border-[var(--ink)] pb-9 lg:grid-cols-[minmax(0,1fr)_auto]">
        <div className="space-y-4">
          <p className="editorial-kicker">Library / Papers</p>
          <h1 className="page-title">论文目录</h1>
          <p className="page-description">
            导入论文后查看解析状态、元数据可信度与章节页码。不同字节的 PDF
            会保留为独立论文，不自动合并修订版。
          </p>
        </div>
        <dl className="grid grid-cols-3 gap-7 border-t border-[var(--line)] pt-4 lg:border-l lg:border-t-0 lg:pl-7 lg:pt-0">
          <Metric label="当前结果" value={counts.total} />
          <Metric label="可检索" value={counts.ready} />
          <Metric label="需处理" value={counts.attention} />
        </dl>
      </header>

      <section className="grid gap-10 lg:grid-cols-[20rem_minmax(0,1fr)]">
        <aside className="space-y-5">
          <FileUpload
            files={files}
            disabled={isUploading}
            actionLabel={isUploading ? "正在提交" : "导入并解析"}
            helperText="PDF 默认使用 PyMuPDF4LLM。失败时尝试 legacy parser，并保留明确原因。"
            onChange={setFiles}
            onSubmit={handleUpload}
          />

          {jobs.length > 0 ? (
            <Card className="space-y-4 rounded-none border-t-2 border-t-[var(--ink-blue)]">
              <h2 className="font-serif text-xl">最近任务</h2>
              {jobs.slice(0, 4).map((job) => (
                <div
                  key={job.id}
                  className="border-t border-[var(--line)] pt-3 text-sm"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-mono text-xs text-[var(--muted-ink)]">
                      {job.id.slice(0, 8)}
                    </span>
                    <JobStatus status={job.status} />
                  </div>
                  {job.error_message ? (
                    <p className="mt-2 text-xs leading-6 text-[var(--signal-red)]">
                      {job.error_message}
                    </p>
                  ) : null}
                </div>
              ))}
            </Card>
          ) : null}
        </aside>

        <div className="min-w-0 space-y-5">
          <div className="grid gap-3 sm:grid-cols-[1fr_12rem_auto]">
            <label className="sr-only" htmlFor="library-query">
              搜索论文目录
            </label>
            <Input
              id="library-query"
              value={query}
              placeholder="按标题、作者或文件名筛选"
              onChange={(event) => setQuery(event.target.value)}
            />
            <label className="sr-only" htmlFor="parse-status">
              解析状态
            </label>
            <select
              id="parse-status"
              value={parseStatus}
              className="min-h-11 border border-[var(--line-strong)] bg-[var(--panel)] px-3 text-sm text-[var(--ink)] transition-[border-color] duration-160 focus:border-[var(--ink-blue)]"
              onChange={(event) =>
                setParseStatus(event.target.value as "" | ParseStatus)
              }
            >
              {STATUS_FILTERS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            <Button variant="secondary" onClick={() => void loadPapers()}>
              刷新
            </Button>
          </div>

          {notice ? (
            <p role="status" className="border-t border-[var(--ink-blue)] pt-3 text-sm text-[var(--ink-blue)]">
              {notice}
            </p>
          ) : null}
          {error ? (
            <p role="alert" className="border-t border-[var(--signal-red)] pt-3 text-sm text-[var(--signal-red)]">
              {error}
            </p>
          ) : null}

          {isLoading ? (
            <p className="border-y border-[var(--line)] py-14 text-center text-sm text-[var(--muted-ink)]">
              正在读取论文目录…
            </p>
          ) : papers.length === 0 ? (
            <Card className="rounded-none border border-dashed border-[var(--line)] bg-transparent text-center shadow-none">
              <h2 className="font-serif text-2xl">目录还是空的</h2>
              <p className="mt-3 text-sm leading-7 text-slate-600">
                从左侧导入一篇 PDF，或清除筛选条件查看全部论文。
              </p>
            </Card>
          ) : (
              <ol className="divide-y divide-[var(--line)] border-y border-[var(--line)]">
              {papers.map((paper, index) => (
                <PaperRow key={paper.id} paper={paper} index={index + 1} />
              ))}
            </ol>
          )}
        </div>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="mt-1 font-serif text-3xl text-slate-950">{value}</dd>
    </div>
  );
}

function PaperRow({ paper, index }: { paper: PaperSummary; index: number }) {
  const confidence = paper.metadata.title?.confidence;
  const reason = paper.fallback_reason || paper.parse_error;
  return (
    <li className="grid gap-4 bg-[var(--panel)] px-4 py-6 sm:grid-cols-[3rem_minmax(0,1fr)_auto] sm:px-5 sm:py-7">
      <span className="font-mono text-xs text-[var(--muted-ink)]">
        {String(index).padStart(2, "0")}
      </span>
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={paper.parse_status} />
          <span className="text-xs text-[var(--muted-ink)]">
            标题置信度 {formatConfidence(confidence)}
          </span>
          <span className="text-xs text-[var(--muted-ink)]">
            来源 {paper.metadata.title?.source ?? "unknown"}
          </span>
        </div>
        <Link
          href={`/papers/${paper.id}`}
          className="mt-3 block font-serif text-xl leading-snug text-[var(--ink)] hover:text-[var(--accent-strong)]"
        >
          {paper.title || paper.file_name}
        </Link>
        <p className="mt-2 text-sm text-[var(--muted-ink)]">
          {formatAuthors(paper.authors)}
          {paper.year ? ` · ${paper.year}` : ""}
          {paper.venue ? ` · ${paper.venue}` : ""}
        </p>
        {reason ? (
          <p className="mt-3 border-t border-[var(--signal-amber)] pt-2 text-xs leading-6 text-[var(--signal-amber)]">
            {reason}
          </p>
        ) : null}
      </div>
      <div className="flex items-center gap-2 sm:flex-col sm:items-end">
        <Link
          href={`/papers/${paper.id}`}
          className="text-link inline-flex"
        >
          查看详情
        </Link>
        <a
          href={`${paper.file_url}#page=1`}
          className="inline-flex min-h-10 items-center text-xs text-[var(--muted-ink)] hover:text-[var(--ink)]"
        >
          打开 PDF
        </a>
      </div>
    </li>
  );
}

function StatusBadge({ status }: { status: ParseStatus }) {
  const warning = ["degraded", "needs_ocr", "failed"].includes(status);
  return (
    <span
      className={
        warning
          ? "inline-flex border border-[var(--signal-amber)] bg-[#fbf4e8] px-2 py-1 text-xs font-semibold text-[var(--signal-amber)]"
          : "status-pill"
      }
    >
      {STATUS_LABELS[status]}
    </span>
  );
}

function JobStatus({ status }: { status: IndexingJobResponse["status"] }) {
  const labels = {
    queued: "排队",
    running: "处理中",
    completed: "完成",
    failed: "失败",
    cancelled: "取消",
  };
  return <span className="text-xs font-semibold">{labels[status]}</span>;
}

function formatConfidence(value: number | undefined) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "未知";
}

function formatAuthors(authors: string[]) {
  return authors.length ? authors.join("、") : "作者未知";
}

function resolveError(caught: unknown, fallback: string) {
  return caught instanceof Error && caught.message ? caught.message : fallback;
}
