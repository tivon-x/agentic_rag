import Link from "next/link";

const routes = [
  {
    number: "01",
    label: "论文目录",
    description: "导入文件，查看解析质量、元数据来源和索引状态。",
    href: "/library",
    action: "进入目录",
  },
  {
    number: "02",
    label: "页码搜索",
    description: "按术语、方法或实验指标查找原文证据。",
    href: "/search",
    action: "搜索证据",
  },
  {
    number: "03",
    label: "证据问答",
    description: "向固定检索基线提问，并让每轮回答留下可回看的来源。",
    href: "/chat",
    action: "开始提问",
  },
];

export default function Home() {
  return (
    <main
      id="main-content"
      className="mx-auto flex w-full max-w-[90rem] flex-1 flex-col px-5 py-10 sm:px-8 sm:py-16"
    >
      <section className="grid items-end gap-10 border-b border-[var(--ink)] pb-12 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.65fr)] lg:gap-20">
        <div>
          <p className="editorial-kicker">Paper Index / Research desk</p>
          <h1 className="page-title mt-5 max-w-5xl">
            每条结论，<br className="hidden sm:block" />都能回到论文原页。
          </h1>
          <p className="page-description mt-7">
            这是一个面向个人论文库的固定 RAG 工作台。导入论文，保留章节和页码证据，
            再从搜索或 Chat 回到原文核验，不把模型的判断藏在一个无法追溯的答案里。
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/library" className="ink-button-label inline-flex min-h-12 items-center bg-[var(--ink)] px-6 text-sm font-semibold transition-[background-color,transform] duration-160 hover:bg-[var(--accent-strong)] active:scale-95">
              打开论文库
            </Link>
            <Link href="/chat" className="inline-flex min-h-12 items-center border border-[var(--line-strong)] bg-[var(--paper)] px-6 text-sm font-semibold text-[var(--ink)] transition-[border-color,color,transform] duration-160 hover:border-[var(--ink-blue)] hover:text-[var(--ink-blue)] active:scale-95">
              直接提问
            </Link>
          </div>
        </div>

        <aside className="border-t-2 border-[var(--ink-blue)] pt-4 lg:mb-2">
          <p className="editorial-kicker">Reading contract</p>
          <p className="mt-5 font-serif text-2xl leading-tight">
            证据不是答案末尾的装饰，而是回答可以被复查的入口。
          </p>
          <dl className="mt-8 divide-y divide-[var(--line)] border-y border-[var(--line)]">
            <div className="flex items-baseline justify-between gap-4 py-3">
              <dt className="text-sm text-[var(--muted-ink)]">默认策略</dt>
              <dd className="font-mono text-xs text-[var(--ink-blue)]">v1_flat_rerank</dd>
            </div>
            <div className="flex items-baseline justify-between gap-4 py-3">
              <dt className="text-sm text-[var(--muted-ink)]">证据位置</dt>
              <dd className="font-mono text-xs text-[var(--ink)]">paper / section / page</dd>
            </div>
            <div className="flex items-baseline justify-between gap-4 py-3">
              <dt className="text-sm text-[var(--muted-ink)]">输入格式</dt>
              <dd className="font-mono text-xs text-[var(--ink)]">PDF · MD · TXT</dd>
            </div>
          </dl>
        </aside>
      </section>

      <section className="mt-12" aria-labelledby="workflow-title">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-[var(--line)] pb-4">
          <div>
            <p className="editorial-kicker">The daily loop</p>
            <h2 id="workflow-title" className="mt-2 font-serif text-3xl">
              从文件到原文
            </h2>
          </div>
          <p className="max-w-md text-sm leading-7 text-[var(--muted-ink)]">
            三个入口共享同一套论文、章节、passage 和页码语言，阅读路径不会在页面之间断掉。
          </p>
        </div>
        <ol className="divide-y divide-[var(--line)] border-b border-[var(--line)]">
          {routes.map((route) => (
            <li key={route.number} className="grid gap-5 py-6 sm:grid-cols-[4rem_minmax(12rem,0.45fr)_minmax(0,1fr)_auto] sm:items-center">
              <span className="font-mono text-xs text-[var(--ink-blue)]">{route.number}</span>
              <h3 className="font-serif text-2xl">{route.label}</h3>
              <p className="max-w-xl text-sm leading-7 text-[var(--muted-ink)]">{route.description}</p>
              <Link href={route.href} className="text-link inline-flex w-fit">
                {route.action} <span className="ml-2" aria-hidden="true">↗</span>
              </Link>
            </li>
          ))}
        </ol>
      </section>
    </main>
  );
}
