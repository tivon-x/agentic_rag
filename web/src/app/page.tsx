import Link from "next/link";

import { Card } from "@/components/ui/card";

const features = [
  {
    number: "01",
    title: "论文目录",
    body: "查看每份 PDF 的解析状态、元数据来源与置信度，需要时直接校正。",
  },
  {
    number: "02",
    title: "页码证据",
    body: "搜索结果保留章节、页码和原文摘录，一步回到 PDF 原页核验。",
  },
  {
    number: "03",
    title: "可恢复索引",
    body: "解析失败或需要 OCR 时保留上一版可用索引，原因对你可见。",
  },
];

export default function Home() {
  return (
    <main
      id="main-content"
      className="mx-auto flex w-full max-w-7xl flex-1 flex-col px-5 py-12 sm:px-8 sm:py-20"
    >
      <section className="grid items-end gap-10 border-b border-[var(--line)] pb-14 lg:grid-cols-[1.35fr_0.65fr]">
        <div className="space-y-6">
          <p className="editorial-kicker">Personal research library</p>
          <h1 className="page-title max-w-4xl">
            每条结论，都能回到论文原页。
          </h1>
          <p className="page-description">
            Paper Index 把散落的 PDF 变成可管理的论文目录。搜索章节与原文证据，
            校正错误元数据，并从结果直接打开对应页码。
          </p>
        </div>
        <div className="flex flex-col gap-3 lg:items-end">
          <Link
            href="/library"
            className="inline-flex min-h-12 items-center justify-center bg-[var(--foreground)] px-6 text-sm font-semibold text-white transition-colors hover:bg-[var(--accent-strong)]"
          >
            打开论文库
          </Link>
          <Link
            href="/search"
            className="inline-flex min-h-12 items-center justify-center border border-[var(--line)] bg-[var(--panel)] px-6 text-sm font-semibold transition-colors hover:border-[var(--accent)] hover:text-[var(--accent-strong)]"
          >
            搜索页码证据
          </Link>
        </div>
      </section>

      <section className="grid gap-px bg-[var(--line)] md:grid-cols-3">
        {features.map((item) => (
          <Card key={item.number} className="rounded-none bg-[var(--panel)] shadow-none">
            <p className="font-mono text-xs text-[var(--accent-strong)]">
              {item.number}
            </p>
            <h2 className="mt-10 font-serif text-2xl text-slate-950">
              {item.title}
            </h2>
            <p className="mt-3 text-sm leading-7 text-slate-600">{item.body}</p>
          </Card>
        ))}
      </section>
    </main>
  );
}
