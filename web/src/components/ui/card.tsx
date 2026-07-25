import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-[28px] border border-[color:var(--line)] bg-[color:var(--panel)] p-6 shadow-[0_18px_50px_rgba(20,35,45,0.08)] backdrop-blur",
        className,
      )}
      {...props}
    />
  );
}
