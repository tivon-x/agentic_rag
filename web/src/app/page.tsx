import Link from "next/link";

import { Card } from "@/components/ui/card";
import { text } from "@/lib/i18n";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-8 px-6 py-10">
      <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <Card className="overflow-hidden border-0 bg-[linear-gradient(135deg,rgba(16,185,129,0.16),rgba(251,191,36,0.10),rgba(255,255,255,0.92))]">
          <div className="space-y-5">
            <p className="inline-flex rounded-full bg-white/80 px-3 py-1 text-sm font-semibold tracking-[0.25em] text-emerald-800 uppercase">
              Agentic RAG
            </p>
            <div className="space-y-3">
              <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-slate-950 sm:text-5xl">
                {text.home.heroTitle}
              </h1>
              <p className="max-w-2xl text-base leading-8 text-slate-700 sm:text-lg">
                {text.home.heroDescription}
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/chat"
                className="inline-flex items-center justify-center rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
              >
                {text.home.chatCta}
              </Link>
              <Link
                href="/kb"
                className="inline-flex items-center justify-center rounded-full border border-slate-300 bg-white/80 px-5 py-3 text-sm font-semibold text-slate-900 transition hover:border-emerald-500 hover:text-emerald-700"
              >
                {text.home.kbCta}
              </Link>
            </div>
          </div>
        </Card>

        <Card className="bg-slate-950 text-slate-50">
          <div className="space-y-4">
            <p className="text-sm font-semibold tracking-[0.25em] text-emerald-300 uppercase">
              {text.home.panelTitle}
            </p>
            <ul className="space-y-3 text-sm leading-7 text-slate-300">
              <li>{text.home.panelItems.streaming}</li>
              <li>{text.home.panelItems.citations}</li>
              <li>{text.home.panelItems.kb}</li>
            </ul>
          </div>
        </Card>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {text.home.featureCards.map((item) => (
          <Card key={item.title} className="bg-white/88">
            <div className="space-y-3">
              <p className="text-sm font-semibold tracking-wide text-emerald-700">
                {item.kicker}
              </p>
              <h2 className="text-xl font-semibold text-slate-950">{item.title}</h2>
              <p className="text-sm leading-7 text-slate-600">{item.body}</p>
            </div>
          </Card>
        ))}
      </section>
    </main>
  );
}
