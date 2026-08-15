import type { InputHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export function Input({
  className,
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "w-full border border-[var(--line-strong)] bg-[color:var(--panel-strong)] px-4 py-3 text-base text-[var(--ink)] transition-[border-color,background-color] duration-160 focus:border-[var(--ink-blue)] focus:bg-[var(--paper)] focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-[var(--ink-blue)] sm:text-sm",
        className,
      )}
      {...props}
    />
  );
}
