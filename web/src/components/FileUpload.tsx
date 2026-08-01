"use client";

import { useId, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { text } from "@/lib/i18n";
import { cn } from "@/lib/utils";

type Props = {
  files: File[];
  disabled?: boolean;
  actionLabel?: string;
  helperText?: string;
  onChange: (files: File[]) => void;
  onSubmit: () => void;
};

const ACCEPTED_TYPES = ".pdf,.md,.txt";

export function FileUpload({
  files,
  disabled,
  actionLabel,
  helperText,
  onChange,
  onSubmit,
}: Props) {
  const inputId = useId();
  const [isDragging, setIsDragging] = useState(false);

  return (
    <Card className="space-y-4 rounded-none">
      <label
        htmlFor={inputId}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => {
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          onChange(Array.from(event.dataTransfer.files));
        }}
        className={cn(
          "upload-dropzone relative flex min-h-44 cursor-pointer flex-col items-center justify-center border border-dashed px-6 py-8 text-center transition-[border-color,background-color] duration-160",
          isDragging
            ? "border-[var(--ink-blue)] bg-[#eef5fb]"
            : "border-[var(--line-strong)] bg-[var(--panel-muted)]",
        )}
      >
        <input
          id={inputId}
          multiple
          type="file"
          accept={ACCEPTED_TYPES}
          aria-label="选择论文文件"
          className="file-input-visually-hidden"
          onChange={(event) => {
            onChange(Array.from(event.target.files ?? []));
          }}
        />
        <p className="text-base font-semibold text-[var(--ink)]">{text.kb.sections.upload}</p>
        <p className="mt-2 text-sm leading-7 text-[var(--muted-ink)]">
          支持 PDF、Markdown 和 TXT。可以点击选择，也可以直接拖拽到这里。
        </p>
      </label>

      <div className="space-y-2">
        {files.length === 0 ? (
          <p className="text-sm text-[var(--muted-ink)]">还没有选择文件。</p>
        ) : (
          <>
            <p className="text-sm font-semibold text-[var(--ink)]">
              {text.kb.selectedFiles} {files.length} 个
            </p>
            {files.map((file) => (
              <div
                key={`${file.name}-${file.size}`}
                className="border border-[var(--line)] bg-[var(--paper)] px-4 py-3 text-sm text-[var(--ink)]"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="truncate">{file.name}</span>
                  <span className="shrink-0 text-xs text-[var(--muted-ink)]">
                    {formatSize(file.size)}
                  </span>
                </div>
              </div>
            ))}
          </>
        )}
      </div>

      {helperText ? <p className="text-sm text-[var(--muted-ink)]">{helperText}</p> : null}

      <Button disabled={disabled || files.length === 0} onClick={onSubmit}>
        {actionLabel ?? text.kb.upload}
      </Button>
    </Card>
  );
}

function formatSize(size: number) {
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}
