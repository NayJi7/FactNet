import { useEffect, useState } from "react";
import Spreading from "./Spreading";
import type { Step } from "../lib/types";

/**
 * What the interface shows while a reading is computed.
 *
 * The engine reports each stage as it lands, so the ones already finished are
 * shown with their real numbers and the ones still to come are named. The bar
 * stays indeterminate: what is known is which stage is running, not how far
 * through it the engine is, and a bar that implied otherwise would be a lie
 * drawn in pixels.
 *
 * The names here have to match the stages the engine actually appends, in
 * order. They drifted once, listing a figure as though it were a stage and
 * omitting two real ones, which made the counter read "4 of 3".
 */
const TEXT_STAGES = [
  "reading what came in",
  "deciding whether it states a claim",
  "scoring the text, and every other model on it",
];
const CASCADE_STAGES = [
  "reading the shape of the cascade",
  "asking what the text adds to it",
  "ranking who carried it",
];
const VERDICT_STAGE = "settling the verdict";

export default function Pending({ withCascade, done = [], rejoined = false, label,
                                 elapsed = 0 }:
                                { withCascade: boolean; done?: Step[];
                                  rejoined?: boolean; label?: string;
                                  elapsed?: number }) {
  // The reading is older than this page whenever the page was reloaded, so the
  // clock starts where the engine says the work started, not at zero.
  const [seconds, setSeconds] = useState(() => Math.round(elapsed));
  useEffect(() => {
    setSeconds(Math.round(elapsed));
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [elapsed]);

  const stages = [...TEXT_STAGES, ...(withCascade ? CASCADE_STAGES : []), VERDICT_STAGE];
  // a run that stops at the gate emits fewer stages than were named, and one
  // that skips the ablation emits fewer still, so the count follows whichever
  // is larger and never reads "7 of 6"
  const total = Math.max(stages.length, done.length);
  const pending = stages.slice(done.length);

  return (
    <section className="settle" aria-live="polite" aria-busy="true">
      <div className="flex flex-wrap items-center gap-x-9 gap-y-5 border border-rule-firm bg-panel px-7 py-6">
        <Spreading />
        <div className="min-w-[260px] flex-1">
          <p className="eyebrow">{rejoined ? "Reading, already under way" : "Reading"}</p>
          {label && (
            <p className="mt-1.5 max-w-[70ch] text-[13.5px] text-ink-soft">{label}</p>
          )}
          <div className="mt-3 flex flex-wrap items-baseline gap-x-5">
            <span className="tnum font-mono text-[13px] text-ink-faint">
              {done.length} of {total} stages
            </span>
            <span className="tnum font-mono text-[13px] text-ink-faint">
              {seconds}s elapsed
            </span>
          </div>
          <div className="mt-5 h-[6px] max-w-[620px] overflow-hidden bg-sunk">
            <div className="h-full w-1/3 animate-[sweep_1.5s_ease-in-out_infinite] bg-rule-firm" />
          </div>
        </div>
      </div>

      <ol className="mt-8">
        {/* what has already landed, with its real numbers */}
        {done.map((step, index) => (
          <li key={step.key}
              className="settle grid grid-cols-[112px_1fr] gap-8 border-t border-rule py-5 first:border-t-0">
            <span className="tnum font-mono text-[26px] leading-none text-rule-firm">
              {String(index + 1).padStart(2, "0")}
            </span>
            <div>
              <p className="text-[15px] font-medium">{step.title}</p>
              <p className="mt-1 max-w-[68ch] text-[14px] leading-relaxed text-ink-soft">
                {step.summary}
              </p>
            </div>
          </li>
        ))}
        {/* and what is still to come, named but empty */}
        {pending.map((stage, index) => (
          <li key={stage}
              className="grid grid-cols-[112px_1fr] gap-8 border-t border-rule py-5 first:border-t-0">
            <span className="tnum font-mono text-[26px] leading-none text-rule-firm">
              {String(done.length + index + 1).padStart(2, "0")}
            </span>
            <div>
              <p className="text-[15px] text-ink-faint">{stage}</p>
              <div className="mt-2 h-[9px] bg-sunk"
                   style={{ width: `${52 + ((index * 37) % 34)}%` }} />
            </div>
          </li>
        ))}
      </ol>

      <p className="mt-6 max-w-[64ch] text-[12.5px] leading-relaxed text-ink-faint">
        {rejoined
          ? "This reading was already running when the page opened, so it is being "
            + "rejoined rather than started again. Reloading does not cancel it, and "
            + "the stages already computed are shown above."
          : "Usually five to twenty seconds. The first reading of a session is slower: "
            + "the transformers are half a gigabyte each and are loaded from disk on "
            + "demand, then kept in memory."}
      </p>
    </section>
  );
}
