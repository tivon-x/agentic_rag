export type AssistantDisplayParts = {
  answer: string;
  reasoningSummary: string;
  confidence: number | null;
  limitations: string;
};

type DisplaySection = "answer" | "reasoningSummary" | "limitations" | "evidence";

const HEADING_PATTERNS: Array<[DisplaySection, RegExp]> = [
  ["reasoningSummary", /^#{1,6}\s*(?:reasoning summary|推理摘要|回答依据)\s*$/iu],
  ["limitations", /^#{1,6}\s*(?:limitations|局限性|限制)\s*$/iu],
  ["evidence", /^#{1,6}\s*(?:evidence|证据|citations|引用)\s*$/iu],
];

export function normalizeInlineMath(content: string): string {
  const normalizeExpression = (expression: string) =>
    expression
      .replace(/\\sqrt\s*\{([^{}]+)\}/gu, "sqrt($1)")
      .replace(/\\(?:text|mathrm|operatorname)\s*\{([^{}]+)\}/gu, "$1")
      .replace(/\{([^{}]+)\}/gu, "$1")
      .replace(/\\([a-zA-Z]+)/gu, "$1")
      .replace(/\s+/gu, " ")
      .trim();

  return content
    .replace(/\$([^$\n]+)\$/gu, (_match, expression: string) =>
      normalizeExpression(expression),
    )
    .replace(/\\sqrt\s*\{([^{}]+)\}/gu, "sqrt($1)")
    .replace(/\\(?:text|mathrm|operatorname)\s*\{([^{}]+)\}/gu, "$1")
    .replace(/\\\(([^\n]+?)\\\)/gu, "$1")
    .replace(/\\\[([\s\S]+?)\\\]/gu, "$1");
}

export function parseAssistantDisplayContent(content: string): AssistantDisplayParts {
  const lines = normalizeInlineMath(content).split(/\r?\n/gu);
  const sections: Record<DisplaySection, string[]> = {
    answer: [],
    reasoningSummary: [],
    limitations: [],
    evidence: [],
  };
  let current: DisplaySection = "answer";
  let confidence: number | null = null;

  for (const line of lines) {
    const heading = HEADING_PATTERNS.find(([, pattern]) => pattern.test(line.trim()));
    if (heading) {
      current = heading[0];
      continue;
    }

    const confidenceMatch = line
      .replace(/\*\*/gu, "")
      .match(
        /^\s*(?:confidence|置信度)\s*[:：]\s*(\d+(?:\.\d+)?)\s*%?\s*$/iu,
      );
    if (confidenceMatch) {
      const parsed = Number(confidenceMatch[1]);
      confidence = Number.isFinite(parsed)
        ? parsed <= 1
          ? Math.round(parsed * 100)
          : Math.round(parsed)
        : null;
      continue;
    }

    sections[current].push(line);
  }

  return {
    answer: cleanSection(sections.answer),
    reasoningSummary: cleanSection(sections.reasoningSummary),
    confidence,
    limitations: cleanSection(sections.limitations),
  };
}

function cleanSection(lines: string[]): string {
  return lines
    .join("\n")
    .replace(/^\s*---\s*$/gmu, "")
    .replace(/\n{3,}/gu, "\n\n")
    .trim();
}
