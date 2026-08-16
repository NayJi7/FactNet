const LEVELS = [20, 40, 60, 80, 100];

/**
 * How much of the cascade the system is allowed to see.
 *
 * Offered only where there is a cascade to truncate. The choice applies to the
 * whole reading and not just to the detector: the shape, the influence ranking
 * and the integration all see the same truncated object, because asking what
 * the system would have said early is not a question about one stage.
 *
 * Drawn as one continuous track rather than five separate buttons, because the
 * values are a scale and not a menu: the eye should read left as early and
 * right as complete without being told.
 */
export default function Observed({ value, onChange, disabled }:
                                 { value: number; onChange: (v: number) => void;
                                   disabled?: boolean }) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <span className="eyebrow">Cascade observed</span>
        <span className="text-[11px] text-ink-faint">
          {value === 100 ? "all of it" : "as it stood"}
        </span>
      </div>

      <div className="flex overflow-hidden rounded-[3px] border border-rule"
           role="group" aria-label="Share of the cascade observed">
        {LEVELS.map((level, i) => (
          <button key={level}
                  onClick={() => onChange(level)}
                  disabled={disabled}
                  aria-pressed={value === level}
                  className={`tnum flex-1 py-1.5 font-mono text-[12px] transition-colors
                              disabled:opacity-40 ${i > 0 ? "border-l border-rule" : ""} ${
                    value === level
                      ? "bg-ink font-medium text-paper"
                      : "bg-panel text-ink-faint hover:bg-sunk hover:text-ink-soft"}`}>
            {level}
          </button>
        ))}
      </div>

      <p className="mt-2 max-w-[38ch] text-[12px] leading-relaxed text-ink-faint">
        A detector that needs the whole cascade arrives after the story has
        spread. Cut to a fifth, the reading below is what the system would have
        said while it was still moving.
      </p>
    </div>
  );
}
