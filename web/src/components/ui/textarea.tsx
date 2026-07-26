import type { TextareaHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export function Textarea({
  className,
  ...props
}: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        "min-h-28 w-full rounded-md border border-slate-300 bg-[color:var(--panel-strong)] px-4 py-3 text-sm leading-7 text-slate-950 outline-none transition-[border-color,box-shadow] focus:border-emerald-700 focus:ring-2 focus:ring-emerald-700/10",
        className,
      )}
      {...props}
    />
  );
}
