import { Card } from "@/components/ui/card";
import { text } from "@/lib/i18n";

type Props = {
  value: string;
};

export function CitationAccordion({ value }: Props) {
  return (
    <Card className="bg-white/78">
      <details open className="group">
        <summary className="cursor-pointer list-none text-sm font-semibold text-slate-900">
          {text.chat.citations}
        </summary>
        <div className="mt-4 rounded-3xl bg-slate-950 px-5 py-4 text-sm text-slate-100">
          <pre className="prose-block font-sans">{value}</pre>
        </div>
      </details>
    </Card>
  );
}
