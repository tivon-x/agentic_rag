import Link from "next/link";

import { evidenceHref, evidencePageLabel, evidenceSectionLabel } from "@/lib/evidence";
import type { ChatEvidence, ChatMessage } from "@/lib/types";

type Props = {
  messages: ChatMessage[];
};

export function EvidenceRail({ messages }: Props) {
  const groups = messages.flatMap((message, index) => {
    if (message.role !== "assistant" || !message.evidence?.length) {
      return [];
    }
    const answerNumber = messages
      .slice(0, index + 1)
      .reduce((count, item) => count + (item.role === "assistant" ? 1 : 0), 0);
    return [{ answerNumber, evidence: message.evidence }];
  });
  const evidenceCount = groups.reduce(
    (count, group) => count + group.evidence.length,
    0,
  );

  return (
    <aside className="evidence-rail" aria-labelledby="evidence-rail-title">
      <div className="flex items-end justify-between gap-4 border-b border-[var(--line)] pb-4">
        <div>
          <p className="editorial-kicker">Evidence rail</p>
          <h2 id="evidence-rail-title" className="mt-2 font-serif text-2xl">
            证据轨
          </h2>
        </div>
        <span className="font-mono text-xs text-[var(--muted-ink)]">
          {evidenceCount} 条
        </span>
      </div>

      {groups.length === 0 ? (
        <div className="evidence-empty mt-5">
          <span className="evidence-marker" aria-hidden="true" />
          <p className="font-serif text-lg">当前会话还没有结构化证据</p>
          <p className="mt-2 text-sm leading-7 text-[var(--muted-ink)]">
            没有证据时不会伪造引用。可以先到搜索页查看论文原文片段。
          </p>
          <Link href="/search" className="text-link mt-4 inline-flex">
            去搜索证据
          </Link>
        </div>
      ) : (
        <div className="mt-5 space-y-8">
          {groups.map((group) => (
            <section key={`answer-${group.answerNumber}`}>
              <div className="flex items-center justify-between gap-3 border-b border-[var(--line)] pb-2">
                <h3 className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--muted-ink)]">
                  回答 {String(group.answerNumber).padStart(2, "0")}
                </h3>
                <span className="font-mono text-xs text-[var(--muted-ink)]">
                  {group.evidence.length} 条证据
                </span>
              </div>
              <ol className="divide-y divide-[var(--line)]">
                {group.evidence.map((evidence, index) => (
                  <EvidenceEntry
                    key={`${group.answerNumber}-${evidence.node_id}`}
                    evidence={evidence}
                    number={index + 1}
                  />
                ))}
              </ol>
            </section>
          ))}
        </div>
      )}
    </aside>
  );
}

export function CitationAccordion(props: Props) {
  return <EvidenceRail {...props} />;
}

function EvidenceEntry({
  evidence,
  number,
}: {
  evidence: ChatEvidence;
  number: number;
}) {
  const href = evidenceHref(evidence);
  const paperLabel = evidence.paper_title || evidence.source;
  const preview =
    evidence.quote.length > 180
      ? `${evidence.quote.slice(0, 180).trimEnd()}…`
      : evidence.quote;

  return (
    <li className="evidence-entry">
      <div className="flex items-start gap-3">
        <span className="evidence-number" aria-hidden="true">
          {String(number).padStart(2, "0")}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
            <p className="font-semibold leading-6 text-[var(--ink)]">{paperLabel}</p>
            <span className="font-mono text-xs text-[var(--ink-blue)]">
              {evidencePageLabel(evidence)}
            </span>
          </div>
          <p className="mt-1 text-xs leading-5 text-[var(--muted-ink)]">
            {evidenceSectionLabel(evidence)}
          </p>
          <p className="mt-3 border-t border-[var(--ink-blue)] pt-3 text-sm leading-7 text-[var(--ink)]">
            “{preview}”
          </p>
          <details className="mt-3 evidence-disclosure">
            <summary>查看完整原文摘录</summary>
            <blockquote className="mt-3 border-t border-[var(--line)] pt-3 text-sm leading-7 text-[var(--muted-ink)]">
              {evidence.quote}
            </blockquote>
          </details>
          {evidence.relevance ? (
            <p className="mt-3 text-xs leading-6 text-[var(--muted-ink)]">
              关联说明：{evidence.relevance}
            </p>
          ) : null}
          {href ? (
            <Link href={href} className="text-link mt-3 inline-flex">
              打开论文原页
            </Link>
          ) : (
            <span className="mt-3 inline-flex text-xs text-[var(--muted-ink)]">
              该来源暂无论文目录链接
            </span>
          )}
        </div>
      </div>
    </li>
  );
}
