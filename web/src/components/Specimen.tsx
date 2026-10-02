import { useState } from "react";
import ModuleTag from "./ModuleTag";
import { SPECIMENS, spread } from "./specimens";
import type { ModelCard } from "../lib/types";
import type { Module } from "../lib/types";

/** Landing example: a real post with the stored outputs of the models. */
const STAGES: [string, Module, string][] = [
  ["Is it a claim at all", "content", "Opinions, jokes and questions are set aside before any verdict, and the interface says so rather than scoring them anyway."],
  ["What the text says", "content", "Attention over the tokens, then each token removed in turn to measure what actually moved the answer, then all five models on the same input."],
  ["How it travelled", "propagation", "The cascade drawn, its shape against the class averages, and the verdict replayed at 20, 40, 60 and 80 per cent of the spread."],
  ["What the text adds", "both", "The content score attached to the root of the cascade, with the verdict shown before and after: the ablation, on your input."],
  ["Who carried it", "propagation", "The influence ranking, and the measurement showing it does not identify who spreads misinformation."],
];

export default function Specimen({ models = [] }: { models?: ModelCard[] }) {
  // random one per page load
  const [pick] = useState(() => Math.floor(Math.random() * SPECIMENS.length));
  const { claim, readings, lesson } = SPECIMENS[pick];
  const gap = spread(readings);
  const loudest = readings.reduce((a, b) => (Math.abs(b.p - 0.5) > Math.abs(a.p - 0.5) ? b : a));

  return (
    <div className="grid gap-x-14 gap-y-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,340px)]">
      <div>
        <p className="eyebrow">Why the reasoning is shown, not just the answer</p>

        <blockquote className="mt-4 border-l-2 border-signal pl-4 text-[18px] leading-snug">
          {claim}
        </blockquote>

        <div className="mt-6 space-y-2">
          {readings.map((r) => (
            <div key={r.model} className="flex items-center gap-3">
              <span className="w-[176px] shrink-0 truncate text-[12.5px]">
                {r.primary && <span className="mr-1 text-signal">▸</span>}
                <span className={r.primary ? "font-semibold" : "text-ink-soft"}>{r.model}</span>
              </span>
              <div className="relative h-[16px] flex-1 bg-sunk">
                <div className="absolute bottom-0 left-1/2 top-0 w-px bg-rule-firm" />
                <div className="absolute inset-y-0"
                     style={{
                       width: `${Math.abs(r.p - 0.5) * 100}%`,
                       left: r.p >= 0.5 ? "50%" : undefined,
                       right: r.p < 0.5 ? "50%" : undefined,
                       background: r.p >= 0.5 ? "var(--color-ink)" : "var(--color-signal)",
                     }} />
              </div>
              <span className="tnum w-[46px] shrink-0 text-right font-mono text-[11.5px] text-ink-faint">
                {r.p.toFixed(3)}
              </span>
            </div>
          ))}
        </div>

        <p className="mt-5 max-w-[66ch] text-[13.5px] leading-relaxed text-ink-soft">
          {lesson}
        </p>
        <p className="mt-3 max-w-[66ch] text-[13px] leading-relaxed text-ink-faint">
          Five models, one sentence, separated on the held-out benchmark by
          three points of macro-F1 and by{" "}
          <span className="tnum font-mono font-semibold text-ink">
            {gap.toFixed(3)}
          </span>{" "}
          here. The loudest of them is{" "}
          {(Math.abs(loudest.p - 0.5) * 200).toFixed(0)}&nbsp;% of the way to an
          answer. That gap is the ceiling content analysis runs into, and the
          reason this system also reads how a claim travelled.
        </p>
      </div>

      <aside className="border-t border-rule-firm pt-5 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
        <p className="eyebrow">What a reading contains</p>
        <ol className="mt-4 space-y-4">
          {STAGES.map(([title, module, body], i) => (
            <li key={title} className="grid grid-cols-[22px_1fr] gap-x-3">
              <span className="tnum pt-[3px] font-mono text-[12px] text-rule-firm">
                {String(i + 1).padStart(2, "0")}
              </span>
              <div>
                <p className="text-[13.5px] font-medium leading-snug">{title}</p>
                <ModuleTag module={module} className="mt-1" />
                <p className="mt-1 text-[12.5px] leading-relaxed text-ink-faint">{body}</p>
              </div>
            </li>
          ))}
        </ol>
        <p className="mt-5 border-t border-rule pt-4 text-[12.5px] leading-relaxed text-ink-faint">
          Stages that do not apply are reported as skipped rather than hidden. A
          text on its own gets the first two. A cascade gets all five.
        </p>
      </aside>
    </div>
  );
}
