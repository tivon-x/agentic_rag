import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-lg bg-[color:var(--panel)] p-6 shadow-[0_12px_32px_oklch(0.25_0.02_235_/_0.08)]",
        className,
      )}
      {...props}
    />
  );
}
