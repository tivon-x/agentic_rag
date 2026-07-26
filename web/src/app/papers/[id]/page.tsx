"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { fetchPaper, updatePaperMetadata } from "@/lib/api";
import type { MetadataEvidence, PaperDetail } from "@/lib/types";

type MetadataForm = {
  title: string;
  authors: string;
  year: string;
  venue: string;
  doi: string;
  arxiv_id: string;
};

const EMPTY_FORM: MetadataForm = {
  title: "",
  authors: "",
  year: "",
  venue: "",
  doi: "",
  arxiv_id: "",
};

export default function PaperDetailPage() {
  const params = useParams<{ id: string }>();
  const paperId = params.id;
  const [paper, setPaper] = useState<PaperDetail | null>(null);
  const [form, setForm] = useState<MetadataForm>(EMPTY_FORM);
  const [page, setPage] = useState(1);
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    const rawPage = new URLSearchParams(window.location.search).get("page");
    const requested = Number(rawPage);
    if (Number.isInteger(requested) && requested > 0) {
      setPage(requested);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void fetchPaper(paperId)
      .then((data) => {
        if (!cancelled) {
          setPaper(data);
          setForm(toForm(data));
        }
      })
      .catch((caught) => {
        if (!cancelled) {
          setError(resolveError(caught, "论文详情加载失败"));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [paperId]);

  const currentSection = useMemo(
    () =>
      paper?.sections.find(
        (section) => page >= section.page_start && page <= section.page_end,
      ),
    [page, paper],
  );

  async function handleSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!paper) {
      return;
    }
    setIsSaving(true);
    setError("");
    setNotice("");
    const patch = buildMetadataPatch(form, paper);
    if (Object.keys(patch).length === 0) {
      setIsEditing(false);
      setIsSaving(false);
      setNotice("没有需要保存的修改。");
      return;
    }
    try {
      const updated = await updatePaperMetadata(
        paper.id,
        paper.metadata_version,
        patch,
      );
      setPaper(updated);
      setForm(toForm(updated));
      setIsEditing(false);
      setNotice(
        updated.reindex_job_id
          ? "元数据已保存，检索前缀正在重建。原文摘录不会改变。"
          : "元数据已保存。",
      );
    } catch (caught) {
      setError(resolveError(caught, "元数据保存失败"));
    } finally {
      setIsSaving(false);
    }
  }

  if (error && !paper) {
    return (
      <main
        id="main-content"
        className="mx-auto min-h-[70vh] max-w-3xl px-5 py-20 sm:px-8"
      >
        <p role="alert" className="border-l-4 border-red-600 pl-4 text-red-800">
          {error}
        </p>
        <Link href="/library" className="mt-8 inline-flex border-b border-slate-500">
          返回论文库
        </Link>
      </main>
    );
  }

  if (!paper) {
    return (
      <main
        id="main-content"
        className="mx-auto min-h-[70vh] max-w-7xl px-5 py-20 sm:px-8"
      >
        <p className="text-sm text-slate-500">正在读取论文与页码证据…</p>
      </main>
    );
  }

  const pageCount = Math.max(paper.paper_version?.page_count ?? 1, 1);
  const pdfSrc = `${paper.file_url}#page=${page}&view=FitH`;
  const degradedReason =
    paper.fallback_reason || paper.parse_error || paper.paper_version?.fallback_reason;

  return (
    <main
      id="main-content"
      className="mx-auto flex w-full max-w-[96rem] flex-col gap-8 px-5 py-8 sm:px-8"
    >
      <header className="space-y-5 border-b border-[var(--line)] pb-7">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <Link
            href="/library"
            className="text-sm font-semibold text-[var(--accent-strong)] hover:underline"
          >
            ← 返回论文目录
          </Link>
          <div className="flex items-center gap-3">
            <span className="status-pill">{statusLabel(paper.parse_status)}</span>
            <a
              href={`${paper.file_url}#page=${page}`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex min-h-10 items-center border border-[var(--line)] bg-[var(--panel)] px-4 text-sm font-semibold hover:border-[var(--accent)]"
            >
              在新窗口打开原页
            </a>
          </div>
        </div>
        <div className="grid gap-6 lg:grid-cols-[1fr_auto]">
          <div>
            <p className="editorial-kicker">Paper / {paper.id.slice(0, 10)}</p>
            <h1 className="mt-3 max-w-5xl font-serif text-3xl leading-tight text-slate-950 sm:text-5xl">
              {paper.title || paper.file_name}
            </h1>
            <p className="mt-4 text-sm leading-7 text-slate-600">
              {paper.authors.length ? paper.authors.join("、") : "作者未知"}
              {paper.year ? ` · ${paper.year}` : ""}
              {paper.venue ? ` · ${paper.venue}` : ""}
            </p>
          </div>
          <div className="self-end text-right text-xs leading-6 text-slate-500">
            <p>{paper.file_name}</p>
            <p>{formatBytes(paper.size_bytes)}</p>
          </div>
        </div>
        {degradedReason ? (
          <p className="border-l-4 border-amber-600 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-950">
            解析说明：{degradedReason}
          </p>
        ) : null}
      </header>

      {notice ? (
        <p role="status" className="border-l-4 border-emerald-600 pl-4 text-sm">
          {notice}
        </p>
      ) : null}
      {error ? (
        <p role="alert" className="border-l-4 border-red-600 pl-4 text-sm text-red-800">
          {error}
        </p>
      ) : null}

      <section className="grid min-h-[75vh] gap-6 xl:grid-cols-[18rem_minmax(0,1fr)_23rem]">
        <aside className="max-h-[75vh] overflow-y-auto border border-[var(--line)] bg-[var(--panel)] p-4">
          <div className="flex items-center justify-between gap-3">
            <h2 className="font-serif text-xl">目录</h2>
            <span className="text-xs text-slate-500">
              {paper.sections.length} 节
            </span>
          </div>
          <ol className="mt-4 divide-y divide-[var(--line)]">
            {paper.sections.length ? (
              paper.sections.map((section) => (
                <li key={section.id}>
                  <button
                    type="button"
                    className={`w-full py-3 text-left text-sm leading-6 ${
                      currentSection?.id === section.id
                        ? "font-semibold text-[var(--accent-strong)]"
                        : "text-slate-600 hover:text-slate-950"
                    }`}
                    style={{ paddingLeft: `${Math.max(section.level - 1, 0) * 0.7}rem` }}
                    onClick={() => setPage(section.page_start)}
                  >
                    <span className="block">{section.title}</span>
                    <span className="font-mono text-[0.68rem] text-slate-400">
                      P. {section.page_start}
                      {section.page_end !== section.page_start
                        ? `–${section.page_end}`
                        : ""}
                    </span>
                  </button>
                </li>
              ))
            ) : (
              <li className="py-4 text-sm leading-6 text-slate-500">
                当前解析结果没有可用章节。仍可按页查看原文。
              </li>
            )}
          </ol>
        </aside>

        <div className="flex min-w-0 flex-col border border-[var(--line)] bg-slate-200">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] bg-[var(--panel)] px-4 py-3">
            <div>
              <p className="font-mono text-xs text-slate-500">
                PAGE {page} / {pageCount}
              </p>
              <p className="mt-1 text-sm font-semibold">
                {currentSection?.title ?? "原始 PDF"}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                disabled={page <= 1}
                onClick={() => setPage((value) => Math.max(1, value - 1))}
              >
                上一页
              </Button>
              <label className="flex items-center gap-2 text-sm">
                <span className="sr-only">跳到页码</span>
                <Input
                  type="number"
                  min={1}
                  max={pageCount}
                  value={page}
                  className="w-20"
                  onChange={(event) =>
                    setPage(
                      Math.min(
                        pageCount,
                        Math.max(1, Number(event.target.value) || 1),
                      ),
                    )
                  }
                />
              </label>
              <Button
                variant="secondary"
                disabled={page >= pageCount}
                onClick={() =>
                  setPage((value) => Math.min(pageCount, value + 1))
                }
              >
                下一页
              </Button>
            </div>
          </div>
          <iframe
            key={pdfSrc}
            title={`${paper.title || paper.file_name} 第 ${page} 页`}
            src={pdfSrc}
            className="min-h-[65vh] w-full flex-1"
          />
        </div>

        <aside className="space-y-5">
          <Card className="rounded-none">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="editorial-kicker">Metadata</p>
                <h2 className="mt-2 font-serif text-2xl">书目信息</h2>
              </div>
              {!isEditing ? (
                <Button variant="secondary" onClick={() => setIsEditing(true)}>
                  校正
                </Button>
              ) : null}
            </div>

            {isEditing ? (
              <form className="mt-5 space-y-4" onSubmit={handleSave}>
                <MetadataInput
                  label="标题"
                  value={form.title}
                  onChange={(value) => setForm({ ...form, title: value })}
                />
                <label className="block space-y-2">
                  <span className="text-sm font-semibold">作者，每行一位</span>
                  <textarea
                    value={form.authors}
                    rows={4}
                    className="w-full border border-slate-300 bg-white px-3 py-2 text-sm"
                    onChange={(event) =>
                      setForm({ ...form, authors: event.target.value })
                    }
                  />
                </label>
                <MetadataInput
                  label="年份"
                  value={form.year}
                  type="number"
                  onChange={(value) => setForm({ ...form, year: value })}
                />
                <MetadataInput
                  label="期刊 / 会议"
                  value={form.venue}
                  onChange={(value) => setForm({ ...form, venue: value })}
                />
                <MetadataInput
                  label="DOI"
                  value={form.doi}
                  onChange={(value) => setForm({ ...form, doi: value })}
                />
                <MetadataInput
                  label="arXiv ID"
                  value={form.arxiv_id}
                  onChange={(value) => setForm({ ...form, arxiv_id: value })}
                />
                <div className="flex gap-2">
                  <Button disabled={isSaving} type="submit">
                    {isSaving ? "保存中" : "保存并重建前缀"}
                  </Button>
                  <Button
                    variant="ghost"
                    type="button"
                    onClick={() => {
                      setForm(toForm(paper));
                      setIsEditing(false);
                    }}
                  >
                    取消
                  </Button>
                </div>
              </form>
            ) : (
              <dl className="mt-5 divide-y divide-[var(--line)]">
                <MetadataRow label="标题" evidence={paper.metadata.title} />
                <MetadataRow label="作者" evidence={paper.metadata.authors} />
                <MetadataRow label="年份" evidence={paper.metadata.year} />
                <MetadataRow label="期刊 / 会议" evidence={paper.metadata.venue} />
                <MetadataRow label="DOI" evidence={paper.metadata.doi} />
                <MetadataRow label="arXiv ID" evidence={paper.metadata.arxiv_id} />
              </dl>
            )}
          </Card>

          {paper.paper_version ? (
            <Card className="rounded-none bg-[var(--foreground)] text-white">
              <p className="editorial-kicker !text-emerald-300">Parser record</p>
              <dl className="mt-4 space-y-3 text-xs leading-6 text-slate-300">
                <div>
                  <dt>解析器</dt>
                  <dd className="text-white">
                    {paper.paper_version.parser_name}{" "}
                    {paper.paper_version.parser_version}
                  </dd>
                </div>
                <div>
                  <dt>归一化版本</dt>
                  <dd className="text-white">
                    {paper.paper_version.normalization_version}
                  </dd>
                </div>
                <div>
                  <dt>耗时</dt>
                  <dd className="text-white">
                    {paper.paper_version.duration_ms} ms
                  </dd>
                </div>
              </dl>
            </Card>
          ) : null}
        </aside>
      </section>
    </main>
  );
}

function MetadataInput({
  label,
  value,
  type = "text",
  onChange,
}: {
  label: string;
  value: string;
  type?: "text" | "number";
  onChange: (value: string) => void;
}) {
  return (
    <label className="block space-y-2">
      <span className="text-sm font-semibold">{label}</span>
      <Input
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function MetadataRow({
  label,
  evidence,
}: {
  label: string;
  evidence: MetadataEvidence | undefined;
}) {
  const value = evidence?.value;
  const rendered = Array.isArray(value)
    ? value.join("、") || "未知"
    : value === null || value === undefined || value === ""
      ? "未知"
      : String(value);
  return (
    <div className="py-3">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="mt-1 break-words text-sm leading-6">{rendered}</dd>
      <dd className="mt-1 font-mono text-[0.65rem] text-slate-400">
        {evidence?.source ?? "unknown"} ·{" "}
        {typeof evidence?.confidence === "number"
          ? `${Math.round(evidence.confidence * 100)}%`
          : "N/A"}
      </dd>
    </div>
  );
}

function toForm(paper: PaperDetail): MetadataForm {
  return {
    title: paper.title ?? "",
    authors: paper.authors.join("\n"),
    year: paper.year ? String(paper.year) : "",
    venue: paper.venue ?? "",
    doi: paper.doi ?? "",
    arxiv_id: paper.arxiv_id ?? "",
  };
}

function buildMetadataPatch(
  form: MetadataForm,
  paper: PaperDetail,
): Parameters<typeof updatePaperMetadata>[2] {
  const normalized = {
    title: form.title.trim() || null,
    authors: form.authors
      .split(/\r?\n|;/)
      .map((item) => item.trim())
      .filter(Boolean),
    year: form.year ? Number(form.year) : null,
    venue: form.venue.trim() || null,
    doi: form.doi.trim() || null,
    arxiv_id: form.arxiv_id.trim() || null,
  };
  const current = {
    title: paper.title,
    authors: paper.authors,
    year: paper.year,
    venue: paper.venue,
    doi: paper.doi,
    arxiv_id: paper.arxiv_id,
  };
  return Object.fromEntries(
    Object.entries(normalized).filter(([field, value]) => {
      const previous = current[field as keyof typeof current];
      return JSON.stringify(value) !== JSON.stringify(previous);
    }),
  ) as Parameters<typeof updatePaperMetadata>[2];
}

function statusLabel(status: PaperDetail["parse_status"]) {
  return {
    queued: "等待解析",
    parsing: "解析中",
    parsed: "解析完成",
    degraded: "Legacy 降级",
    needs_ocr: "需要 OCR",
    failed: "解析失败",
  }[status];
}

function formatBytes(value: number) {
  return `${(value / 1024 / 1024).toFixed(2)} MB`;
}

function resolveError(caught: unknown, fallback: string) {
  return caught instanceof Error && caught.message ? caught.message : fallback;
}
