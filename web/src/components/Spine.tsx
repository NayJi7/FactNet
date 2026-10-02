import type { Module } from "../lib/types";

/**
 * Main scores on one axis, sorted by value, next to their baselines
 * (0.920 only means something next to the 0.940 with no edges).
 * Axis starts at the constant-guess floor.
 */
interface Mark {
  label: string;
  value: number;
  module: Module;
  note: string;
  kind: "result" | "baseline" | "floor";
}

const TONE: Record<Module, string> = {
  content: "var(--color-content)",
  propagation: "var(--color-struct)",
  both: "var(--color-ink-faint)",
};

export default function Spine({ marks, unit }: { marks: Mark[]; unit: string }) {
  if (marks.length < 2) return null;
  const floor = 0.3;
  const place = (v: number) => ((v - floor) / (1 - floor)) * 100;

  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 className="text-[17px] font-semibold tracking-[-0.011em]">
          Everything this project measured, on one axis
        </h2>
        <span className="font-mono text-[12px] text-ink-faint">{unit}</span>
      </div>
      <p className="mt-1.5 max-w-[76ch] text-[13px] leading-relaxed text-ink-soft">
        Read down the column, not across it. A score is only worth what it beats,
        and two of the marks here are what the others have to beat.
      </p>

      <ol className="mt-6">
        {marks.map((mark) => {
          const weak = mark.kind !== "result";
          return (
            <li key={mark.label}
                className="grid items-center gap-x-5 border-t border-rule py-2.5
                           sm:grid-cols-[minmax(0,27ch)_1fr_7ch]">
              <div className="min-w-0">
                <p className={`truncate text-[13px] ${weak ? "text-ink-soft" : "font-medium"}`}
                   title={mark.label}>
                  {mark.label}
                </p>
                <p className="truncate text-[11px] text-ink-faint" title={mark.note}>
                  {mark.note}
                </p>
              </div>

              <div className="relative h-[22px]">
                <div className="absolute inset-y-[9px] left-0 right-0 bg-sunk" aria-hidden />
                <div className="absolute inset-y-[9px] left-0 transition-[width] duration-700"
                     style={{
                       width: `${place(mark.value)}%`,
                       background: TONE[mark.module],
                       opacity: weak ? 0.3 : 0.85,
                       transitionTimingFunction: "var(--ease-out-quint)",
                     }} />
                {/* baselines = tick, not bar */}
                {weak && (
                  <div className="absolute inset-y-0 w-px bg-ink-faint"
                       style={{ left: `${place(mark.value)}%` }} aria-hidden />
                )}
              </div>

              <span className={`tnum text-right font-mono text-[13px] ${
                weak ? "text-ink-faint" : ""}`}>
                {mark.value.toFixed(3)}
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-2 border-t border-rule pt-2 text-[11.5px] text-ink-faint">
        The axis runs from {floor.toFixed(1)} to 1. Faint marks are what a result
        has to clear, not results themselves.
      </p>
    </section>
  );
}
