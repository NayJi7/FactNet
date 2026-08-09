import { useEffect, useState } from "react";
import type { Step } from "../lib/types";

/**
 * What the interface shows while a reading is computed.
 *
 * A run takes a few seconds, and the first one of a session takes longer
 * because half a gigabyte of weights is loaded from disk. The wait is spent
 * saying what is coming and how long it usually takes, rather than animating a
 * fake progress bar: the engine returns one answer at the end and has no
 * intermediate progress to report, so claiming otherwise would be a lie drawn
 * in pixels.
 */
const STAGES = [
  "reading what came in",
  "deciding whether it states a claim",
  "scoring the text, and every other model on it",
  "reading the shape of the cascade",
  "replaying the verdict as the spread grew",
  "ranking who carried it",
];

export default function Pending({ withCascade, done = [] }:
                                { withCascade: boolean; done?: Step[] }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const stages = withCascade ? STAGES : STAGES.slice(0, 3);
  const pending = stages.slice(done.length);

  return (
    <section className="settle" aria-live="polite" aria-busy="true">
      <div className="border border-rule-firm bg-panel px-7 py-6">
        <p className="eyebrow">Reading</p>
        <div className="mt-3 flex flex-wrap items-baseline gap-x-5">
          <span className="text-[40px] font-semibold leading-none tracking-[-0.03em] text-rule-firm">
            &mdash;&mdash;
          </span>
          <span className="tnum font-mono text-[13px] text-ink-faint">
            {done.length} of {stages.length} stages
          </span>
          <span className="tnum font-mono text-[13px] text-ink-faint">
            {seconds}s elapsed
          </span>
        </div>
        <div className="mt-6 h-[6px] max-w-[620px] overflow-hidden bg-sunk">
          <div className="h-full w-1/3 animate-[sweep_1.5s_ease-in-out_infinite] bg-rule-firm" />
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
        Usually five to twenty seconds. The first reading of a session is slower:
        the transformers are half a gigabyte each and are loaded from disk on
        demand, then kept in memory.
      </p>
    </section>
  );
}
