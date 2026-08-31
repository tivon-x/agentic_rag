import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  normalizeInlineMath,
  parseAssistantDisplayContent,
} from "../src/lib/assistant-display.ts";
import { appendRecoveryHint, toUserError } from "../src/lib/errors.ts";

const math = String.raw`$d_k$ $d_{model}$ $d_k = d_v = d_{model}/h$ $\sqrt{d_k}$`;
assert.equal(
  normalizeInlineMath(math),
  "d_k d_model d_k = d_v = d_model/h sqrt(d_k)",
);

const parsed = parseAssistantDisplayContent(
  "正文回答\n\n## Reasoning summary\n依据检索结果\n\n**Confidence:** 88%\n\n## Limitations\n样本有限\n\n## Evidence\n不应在正文重复",
);
assert.equal(parsed.answer, "正文回答");
assert.equal(parsed.reasoningSummary, "依据检索结果");
assert.equal(parsed.confidence, 88);
assert.equal(parsed.limitations, "样本有限");
assert.equal(
  toUserError(new Error("No index loaded.")),
  "当前还没有可搜索的论文索引，请先前往论文库导入并解析文件。",
);
assert.equal(
  toUserError(new Error("No active index is available.")),
  "当前还没有可搜索的论文索引，请先前往论文库导入并解析文件。",
);
assert.equal(toUserError(new Error("内部解析错误"), "请求失败"), "请求失败。");
assert.equal(toUserError(new Error("provider secret 123"), "请求失败"), "请求失败。");
assert.equal(
  appendRecoveryHint("回答生成失败，回答没有保存。", "请重试。"),
  "回答生成失败，回答没有保存。请重试。",
);

const chatSource = await readFile(
  new URL("../src/components/ChatExperience.tsx", import.meta.url),
  "utf8",
);
const globalStyles = await readFile(
  new URL("../src/app/globals.css", import.meta.url),
  "utf8",
);
const buttonSource = await readFile(
  new URL("../src/components/ui/button.tsx", import.meta.url),
  "utf8",
);
const inputSource = await readFile(
  new URL("../src/components/ui/input.tsx", import.meta.url),
  "utf8",
);
const evaluationSource = await readFile(
  new URL("../src/app/(editorial)/evaluation/page.tsx", import.meta.url),
  "utf8",
);
const evaluationDataSource = await readFile(
  new URL("../src/lib/kite-evaluation.ts", import.meta.url),
  "utf8",
);
const typesSource = await readFile(
  new URL("../src/lib/types.ts", import.meta.url),
  "utf8",
);
const editorialNavSource = await readFile(
  new URL("../src/components/EditorialNav.tsx", import.meta.url),
  "utf8",
);

assert.match(chatSource, /setIsLongWait\(true\)/u);
assert.match(chatSource, /clearTimeout\(timer\)/u);
assert.match(chatSource, /setAttribute\("inert"/u);
assert.match(chatSource, /activeDialog\.focus/u);
assert.match(chatSource, /'summary'/u);
assert.match(chatSource, /isMobile && sessionsOpen \? "chat-session-drawer"/u);
assert.match(chatSource, /isEvidenceModal && evidenceOpen \? "chat-evidence-overlay"/u);
assert.match(globalStyles, /outline: 2px solid var\(--ink-blue\)/u);
assert.match(globalStyles, /font-size: 1rem/u);
assert.match(buttonSource, /active:scale-\[0\.96\]/u);
assert.match(inputSource, /text-base/u);
assert.match(typesSource, /KITE_PIPELINE_KEYS = \["b0", "b1", "b2", "b3"\]/u);
assert.match(typesSource, /pipelines: Record<KitePipelineKey, KitePipelineSummary>/u);
assert.match(typesSource, /pairwise_vs_b1: Record<KitePairwiseKey, KitePairwiseSummary>/u);
assert.match(evaluationDataSource, /formal_run: true/u);
assert.match(evaluationSource, /decision\?\.default_pipeline/u);
assert.match(evaluationSource, /formal_run=false/u);
assert.match(evaluationSource, /未获生产批准/u);
assert.doesNotMatch(evaluationSource, /pipelines\.b1/u);
assert.doesNotMatch(evaluationSource, /key === "b1"/u);
assert.match(editorialNavSource, /usePathname/u);
assert.match(editorialNavSource, /aria-current=\{isCurrentPath\(pathname, href\) \? "page" : undefined\}/u);

console.log("web UI contract checks passed");
