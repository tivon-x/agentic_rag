import type { TextareaHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export function Textarea({
  className,
  ...props
}: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        "min-h-28 w-full border border-[var(--line-strong)] bg-[color:var(--panel-strong)] px-4 py-3 text-sm leading-7 text-[var(--ink)] transition-[border-color,background-color] duration-160 focus:border-[var(--ink-blue)] focus:bg-[var(--paper)]",
        className,
      )}
      {...props}
    />
  );
}
