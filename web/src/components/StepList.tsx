import { useState } from "react";
import FigureView from "./Figures";
import ModuleTag from "./ModuleTag";
import type { Step } from "../lib/types";

/**
 * The reasoning, as one continuous document.
 *
 * Not cards: a card implies each stage is a separable unit, and these are not.
 * Stage four only means something given stage three. Hairlines and a marginal
 * column carry the separation instead, the way a technical report does.
 *
 * Steps are numbered because this genuinely is a sequence, and skipping one is
 * itself a reported outcome.
 */

const STATUS: Record<Step["status"], { mark: string; tone: string; word: string }> = {
  ok:      { mark: "", tone: "text-ink-faint", word: "" },
  skipped: { mark: "·", tone: "text-ink-faint", word: "not run" },
  warning: { mark: "!", tone: "text-signal", word: "read the caveat" },
};

function Detail({ detail }: { detail: Record<string, any> }) {
  const entries = Object.entries(detail).filter(
    ([, v]) => v !== null && v !== "" && typeof v !== "object",
  );
  if (!entries.length) return null;
  return (
    <dl className="mt-1 flex max-w-[72ch] flex-wrap gap-x-7 gap-y-2">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt className="eyebrow">{key.replace(/_/g, " ")}</dt>
          <dd className={`mt-0.5 text-[13px] ${
            typeof value === "number" ? "tnum font-mono" : "max-w-[46ch] leading-snug text-ink-soft"
          }`}>
            {typeof value === "number" ? Number(value.toFixed(4)) : String(value)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export default function StepList({ steps }: { steps: Step[] }) {
  const [closed, setClosed] = useState<Set<string>>(new Set());
  const toggle = (key: string) =>
    setClosed((c) => {
      const next = new Set(c);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });

  return (
    <ol className="mt-10">
      {steps.map((step, index) => {
        const status = STATUS[step.status];
        const open = !closed.has(step.key);
        return (
          <li key={step.key}
              className="settle border-t border-rule-firm py-6 first:border-t-0 sm:grid sm:grid-cols-[112px_1fr] sm:gap-8"
              style={{ animationDelay: `${index * 45}ms` }}>
            <div className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1 sm:mb-0 sm:block">
              <span className="tnum font-mono text-[26px] font-normal leading-none text-rule-firm">
                {String(index + 1).padStart(2, "0")}
              </span>
              <ModuleTag module={step.module} className="sm:mt-2.5 sm:flex" />
              {status.word && (
                <span className={`block text-[11px] sm:mt-1.5 ${status.tone}`}>
                  {status.mark} {status.word}
                </span>
              )}
            </div>

            <div>
              <button onClick={() => toggle(step.key)} aria-expanded={open}
                      className="group block w-full text-left">
                <h3 className="text-[17px] font-semibold tracking-[-0.011em]">
                  {step.title}
                  <span className="ml-2 text-[13px] font-normal text-ink-faint opacity-0 transition-opacity group-hover:opacity-100">
                    {open ? "collapse" : "expand"}
                  </span>
                </h3>
                <p className="mt-1 max-w-[68ch] text-[14px] leading-relaxed text-ink-soft">
                  {step.summary}
                </p>
              </button>

              {open && (
                <>
                  <Detail detail={step.detail} />
                  {step.note && (
                    <p className="mt-4 max-w-[70ch] border-l-0 bg-sunk px-4 py-3 text-[13px] leading-relaxed">
                      {step.note}
                    </p>
                  )}
                  {/* Two abreast once there is room. Attention beside occlusion
                      is also the comparison the pair exists to invite, so this
                      shortens the page and sharpens the point at once. A graph,
                      a curve, and the model comparison keep the full measure:
                      the last one carries the evidence for every model at once
                      and turns unreadable in half a column. */}
                  <div className="grid gap-x-10 xl:grid-cols-2">
                    {step.figures.map((figure, i) => (
                      <div key={i}
                           className={["graph", "line", "compare"].includes(figure.kind)
                             ? "xl:col-span-2" : "min-w-0"}>
                        <FigureView figure={figure} />
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
