import type { ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
};

export function Button({
  className,
  variant = "primary",
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex min-h-11 items-center justify-center border px-4 py-2.5 text-sm font-semibold transition-[transform,background-color,color,border-color] duration-160 active:scale-[0.96] motion-reduce:active:scale-100 disabled:cursor-not-allowed disabled:opacity-55 disabled:active:scale-100",
        variant === "primary" &&
          "ink-button-label border-[var(--ink)] bg-[var(--ink)] hover:border-[var(--accent-strong)] hover:bg-[var(--accent-strong)]",
        variant === "secondary" &&
          "border-[var(--line-strong)] bg-[var(--paper)] text-[var(--ink)] hover:border-[var(--ink-blue)] hover:text-[var(--ink-blue)]",
        variant === "ghost" &&
          "border-transparent bg-transparent text-[var(--muted-ink)] hover:border-[var(--line)] hover:text-[var(--ink)]",
        className,
      )}
      {...props}
    />
  );
}
