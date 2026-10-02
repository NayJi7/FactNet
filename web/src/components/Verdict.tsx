import Breakdown from "./Breakdown";
import type { Trace } from "../lib/types";

/** Final verdict: position on the 0-1 axis, no green/red pill on purpose. */
export default function Verdict({ trace }: { trace: Trace }) {
  const p = trace.verdict;
  const stopped = p === null;
  const percent = stopped ? 50 : p * 100;

  return (
    <section className="settle border border-rule-firm bg-panel px-7 py-6 shadow-[0_1px_0_0_var(--color-rule-firm)]">
      <p className="eyebrow">Reading</p>

      <div className="mt-3 flex flex-wrap items-baseline gap-x-5 gap-y-2">
        <h2 className="text-[52px] font-semibold leading-[0.92] tracking-[-0.038em]">
          {trace.label}
        </h2>
        {!stopped && (
          <span className="tnum font-mono text-[15px] text-ink-soft">
            p(reliable) = {p!.toFixed(3)}
          </span>
        )}
        <span className="text-[13px] text-ink-faint">
          {trace.confidence} confidence
        </span>
      </div>

      {!stopped && (
        <div className="mt-7 max-w-[620px]">
          <div className="relative h-[42px]">
            <div className="absolute inset-y-[17px] left-0 right-0 bg-sunk" />
            <div className="absolute inset-y-[17px] left-0 w-1/2 bg-signal-dim" />
            <div className="absolute bottom-0 left-1/2 top-0 w-px bg-rule-firm" aria-hidden />
            <div className="absolute inset-y-0 w-[3px] bg-ink transition-[left] duration-700"
                 style={{ left: `calc(${percent}% - 1px)`, transitionTimingFunction: "var(--ease-out-quint)" }} />
          </div>
          {/* absolute positioning, justify-between put 0.5 off center */}
          <div className="relative mt-1.5 h-[1.3em] text-[11.5px] text-ink-faint">
            <span className="absolute left-0">misleading</span>
            <span className="tnum absolute left-1/2 -translate-x-1/2 font-mono">0.5</span>
            <span className="absolute right-0">reliable</span>
          </div>
        </div>
      )}

      <Breakdown trace={trace} />

      {trace.warnings.length > 0 && (
        <ul className="mt-6 max-w-[72ch] space-y-2">
          {trace.warnings.map((w, i) => (
            <li key={i} className="flex gap-2.5 text-[13px] leading-relaxed text-ink-soft">
              <span className="mt-[1px] shrink-0 font-mono text-signal">!</span>
              {w}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
