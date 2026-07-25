"use client";

import { useEffect, useState } from "react";

import { FileUpload } from "@/components/FileUpload";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  fetchCorpusProfile,
  fetchIndexingJob,
  saveCorpusProfile,
  uploadKnowledgeFiles,
} from "@/lib/api";
import { text } from "@/lib/i18n";
import type {
  CorpusProfile,
  FileUploadResponse,
  IndexingJobResponse,
} from "@/lib/types";

const EMPTY_PROFILE: CorpusProfile = {
  name: "",
  summary: "",
  coverage: "",
  non_coverage: "",
  usage_notes: "",
  source_examples: [],
  recommended_questions: [],
  forbidden_questions: [],
  domain_keywords: [],
  preferred_answer_style: "",
  primary_entities: [],
};

const LIST_FIELDS = [
  "source_examples",
  "recommended_questions",
  "forbidden_questions",
  "domain_keywords",
  "primary_entities",
] as const;

export default function KnowledgeBasePage() {
  const [profile, setProfile] = useState<CorpusProfile>(EMPTY_PROFILE);
  const [files, setFiles] = useState<File[]>([]);
  const [indexMode, setIndexMode] = useState<"flat" | "hierarchical">("flat");
  const [jobs, setJobs] = useState<IndexingJobResponse[]>([]);
  const [jobNames, setJobNames] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await fetchCorpusProfile();
        if (!cancelled) {
          setProfile(data);
          setError("");
        }
      } catch {
        if (!cancelled) {
          setError(text.kb.loadError);
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const pendingJobs = jobs.filter(
      (job) => job.status === "pending" || job.status === "running",
    );
    if (pendingJobs.length === 0) {
      return;
    }

    const timer = window.setInterval(() => {
      void Promise.all(pendingJobs.map((job) => fetchIndexingJob(job.id)))
        .then((nextJobs) => {
          setJobs((current) => {
            const stable = current.filter(
              (job) => !pendingJobs.some((pending) => pending.id === job.id),
            );
            return [...stable, ...nextJobs].sort((a, b) =>
              b.created_at.localeCompare(a.created_at),
            );
          });
        })
        .catch(() => {
          setError((current) => current || text.kb.pollError);
        });
    }, 2000);

    return () => {
      window.clearInterval(timer);
    };
  }, [jobs]);

  async function handleSaveProfile() {
    setError("");
    setNotice("");
    setIsSaving(true);

    try {
      const saved = await saveCorpusProfile(profile);
      setProfile(saved);
      setNotice(text.kb.profileSaved);
    } catch (caught) {
      setError(resolveKbError(caught, "save"));
    } finally {
      setIsSaving(false);
    }
  }

  async function handleUpload() {
    if (files.length === 0) {
      setError(text.kb.pickFiles);
      return;
    }
    if (files.some((file) => !/\.(pdf|md|txt)$/i.test(file.name))) {
      setError(text.kb.unsupported);
      return;
    }

    setError("");
    setNotice("");
    setIsUploading(true);

    try {
      const createdJobs = await uploadKnowledgeFiles({
        files,
        indexMode,
      });
      setJobNames((current) => ({
        ...current,
        ...Object.fromEntries(
          createdJobs.map((job: FileUploadResponse) => [job.job_id, job.filename]),
        ),
      }));
      const resolvedJobs = await Promise.all(
        createdJobs.map((job: FileUploadResponse) => fetchIndexingJob(job.job_id)),
      );
      setJobs((current) =>
        [...resolvedJobs, ...current].sort((a, b) =>
          b.created_at.localeCompare(a.created_at),
        ),
      );
      setFiles([]);
      setNotice(text.kb.uploadDone);
    } catch (caught) {
      setError(resolveKbError(caught, "upload"));
    } finally {
      setIsUploading(false);
    }
  }

  const runningCount = jobs.filter(
    (job) => job.status === "pending" || job.status === "running",
  ).length;
  const completedCount = jobs.filter((job) => job.status === "completed").length;
  const failedCount = jobs.filter((job) => job.status === "failed").length;

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-6 py-10">
      <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <Card className="space-y-6">
          <div className="space-y-2">
            <p className="status-pill">{text.nav.kb}</p>
            <h1 className="text-3xl font-semibold tracking-tight text-slate-950">
              {text.kb.title}
            </h1>
            <p className="max-w-2xl text-sm leading-7 text-slate-600">
              {text.kb.description}
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Field
              label={text.fields.name}
              value={profile.name}
              onChange={(value) => setProfile({ ...profile, name: value })}
            />
            <Field
              label={text.fields.preferredAnswerStyle}
              value={profile.preferred_answer_style}
              onChange={(value) =>
                setProfile({ ...profile, preferred_answer_style: value })
              }
            />
          </div>

          <TextField
            label={text.fields.summary}
            value={profile.summary}
            onChange={(value) => setProfile({ ...profile, summary: value })}
          />
          <TextField
            label={text.fields.coverage}
            value={profile.coverage}
            onChange={(value) => setProfile({ ...profile, coverage: value })}
          />
          <TextField
            label={text.fields.nonCoverage}
            value={profile.non_coverage}
            onChange={(value) => setProfile({ ...profile, non_coverage: value })}
          />
          <TextField
            label={text.fields.usageNotes}
            value={profile.usage_notes}
            onChange={(value) => setProfile({ ...profile, usage_notes: value })}
          />

          <div className="grid gap-4 md:grid-cols-2">
            {LIST_FIELDS.map((field) => (
              <TextField
                key={field}
                label={fieldLabel(field)}
                value={profile[field].join("\n")}
                onChange={(value) =>
                  setProfile({
                    ...profile,
                    [field]: value
                      .split(/\r?\n|;/)
                      .map((item) => item.trim())
                      .filter(Boolean),
                  })
                }
              />
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button disabled={isSaving} onClick={handleSaveProfile}>
              {isSaving ? text.kb.saving : text.kb.save}
            </Button>
            <label className="text-sm font-medium text-slate-700">
              索引模式
              <select
                className="ml-3 rounded-full border border-slate-300 bg-white px-4 py-2"
                value={indexMode}
                onChange={(event) =>
                  setIndexMode(event.target.value as "flat" | "hierarchical")
                }
              >
                <option value="flat">Flat</option>
                <option value="hierarchical">Hierarchical</option>
              </select>
            </label>
          </div>

          {notice ? <p className="text-sm text-emerald-700">{notice}</p> : null}
          {error ? <p className="text-sm text-rose-700">{error}</p> : null}
        </Card>

        <div className="space-y-6">
          <FileUpload
            files={files}
            disabled={isUploading}
            actionLabel={isUploading ? text.kb.uploading : text.kb.upload}
            helperText={
              isUploading
                ? "文件已上传，正在为每个文件创建后台索引任务。"
                : "建议按同一主题分批上传，便于排查失败任务。"
            }
            onChange={setFiles}
            onSubmit={handleUpload}
          />

          <Card className="space-y-4 bg-slate-950 text-slate-100">
            <div className="space-y-2">
              <p className="text-sm font-semibold tracking-[0.2em] text-emerald-300 uppercase">
                {text.kb.statusTitle}
              </p>
              <p className="text-sm leading-7 text-slate-300">
                {jobs.length === 0
                  ? text.kb.emptyJobs
                  : runningCount > 0
                    ? text.kb.pollingActive
                    : text.kb.pollingIdle}
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <StatusSummaryCard
                label={text.kb.progressSummary.active}
                value={runningCount}
                tone="emerald"
              />
              <StatusSummaryCard
                label={text.kb.progressSummary.completed}
                value={completedCount}
                tone="sky"
              />
              <StatusSummaryCard
                label={text.kb.progressSummary.failed}
                value={failedCount}
                tone="rose"
              />
            </div>

            <div className="space-y-3">
              {jobs.map((job) => (
                <div
                  key={job.id}
                  className="rounded-[24px] border border-white/10 bg-white/8 px-4 py-4"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="space-y-2">
                      <p className="text-sm font-semibold text-white">
                        {jobNames[job.id] ?? "未命名文件任务"}
                      </p>
                      <p className="text-xs text-slate-400">
                        {text.kb.jobId}：{job.id}
                      </p>
                    </div>
                    <StatusBadge status={job.status} />
                  </div>
                  <p className="mt-3 text-sm leading-7 text-slate-300">
                    {text.kb.statusDescriptions[job.status]}
                  </p>
                  <dl className="mt-4 grid gap-2 text-xs text-slate-400 sm:grid-cols-2">
                    <div>
                      <dt>{text.kb.createdAt}</dt>
                      <dd className="mt-1 text-sm text-slate-200">
                        {formatDateTime(job.created_at)}
                      </dd>
                    </div>
                    <div>
                      <dt>{text.kb.updatedAt}</dt>
                      <dd className="mt-1 text-sm text-slate-200">
                        {formatDateTime(job.updated_at)}
                      </dd>
                    </div>
                  </dl>
                  {job.error_message ? (
                    <div className="mt-4 rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3">
                      <p className="text-xs font-semibold tracking-wide text-rose-200">
                        {text.kb.errorDetail}
                      </p>
                      <p className="mt-2 text-sm leading-7 text-rose-100">
                        {job.error_message}
                      </p>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </Card>
        </div>
      </section>
    </main>
  );
}

type FieldProps = {
  label: string;
  value: string;
  onChange: (value: string) => void;
};

function Field({ label, value, onChange }: FieldProps) {
  return (
    <label className="space-y-2">
      <span className="text-sm font-semibold text-slate-800">{label}</span>
      <Input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function TextField({ label, value, onChange }: FieldProps) {
  return (
    <label className="space-y-2">
      <span className="text-sm font-semibold text-slate-800">{label}</span>
      <Textarea value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function fieldLabel(field: (typeof LIST_FIELDS)[number]) {
  switch (field) {
    case "source_examples":
      return text.fields.sourceExamples;
    case "recommended_questions":
      return text.fields.recommendedQuestions;
    case "forbidden_questions":
      return text.fields.forbiddenQuestions;
    case "domain_keywords":
      return text.fields.domainKeywords;
    case "primary_entities":
      return text.fields.primaryEntities;
    default:
      return field;
  }
}

function resolveKbError(caught: unknown, context: "save" | "upload") {
  const fallback =
    context === "save" ? text.kb.saveError : text.kb.uploadError;
  if (!(caught instanceof Error)) {
    return fallback;
  }

  const message = caught.message;
  if (message.includes("Unsupported file type")) {
    return "文件格式不支持，仅支持 PDF、Markdown 和 TXT。";
  }
  if (message.includes("No files uploaded")) {
    return text.kb.pickFiles;
  }
  if (message.includes("Unsupported index mode")) {
    return "索引模式不受支持，请重新选择。";
  }
  if (message.includes("Indexing job not found")) {
    return "索引任务不存在，页面将自动刷新最新状态。";
  }
  if (message.includes("Failed to fetch")) {
    return text.kb.loadError;
  }
  return message || fallback;
}

function formatDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    hour12: false,
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function StatusBadge({ status }: { status: IndexingJobResponse["status"] }) {
  const toneMap = {
    pending: "bg-amber-400/15 text-amber-200 border-amber-300/30",
    running: "bg-emerald-400/15 text-emerald-200 border-emerald-300/30",
    completed: "bg-sky-400/15 text-sky-200 border-sky-300/30",
    failed: "bg-rose-400/15 text-rose-200 border-rose-300/30",
  } as const;

  return (
    <span
      className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold ${toneMap[status]}`}
    >
      {text.kb.status[status]}
    </span>
  );
}

function StatusSummaryCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "emerald" | "sky" | "rose";
}) {
  const toneMap = {
    emerald: "border-emerald-400/30 bg-emerald-400/10 text-emerald-100",
    sky: "border-sky-400/30 bg-sky-400/10 text-sky-100",
    rose: "border-rose-400/30 bg-rose-400/10 text-rose-100",
  } as const;

  return (
    <div className={`rounded-[22px] border px-4 py-4 ${toneMap[tone]}`}>
      <p className="text-xs tracking-wide">{label}</p>
      <p className="mt-2 text-2xl font-semibold">{value}</p>
    </div>
  );
}
