import { useState } from "react";

const LEVELS = [20, 40, 60, 80, 100];

/**
 * "% of the cascade observed" picker (20..100), only when there's a cascade.
 * A gauge instead of 5 buttons, it's a proportion. Hover previews the fill.
 */
export default function Observed({ value, onChange, disabled }:
                                 { value: number; onChange: (v: number) => void;
                                   disabled?: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const shown = hover ?? value;

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="eyebrow">Cascade observed</span>
        <span className="tnum font-mono text-[13px] font-medium text-ink">
          {shown}&thinsp;%
        </span>
      </div>

      <div className="flex overflow-hidden rounded-[3px] border border-rule-firm"
           role="radiogroup" aria-label="Share of the cascade observed"
           onMouseLeave={() => setHover(null)}>
        {LEVELS.map((level, i) => (
          <button key={level}
                  onClick={() => onChange(level)}
                  onMouseEnter={() => !disabled && setHover(level)}
                  onFocus={() => !disabled && setHover(level)}
                  onBlur={() => setHover(null)}
                  disabled={disabled}
                  role="radio"
                  aria-checked={value === level}
                  aria-label={`${level} percent`}
                  className={`h-3.5 flex-1 transition-colors duration-150
                              disabled:cursor-not-allowed
                              ${i > 0 ? "border-l border-paper/70" : ""} ${
                    level <= shown
                      ? hover !== null && hover !== value
                        ? "bg-struct/55"
                        : "bg-struct"
                      : "bg-sunk"}`} />
        ))}
      </div>

      <div className="mt-1 flex" aria-hidden="true">
        {LEVELS.map(level => (
          <span key={level}
                onClick={() => !disabled && onChange(level)}
                onMouseEnter={() => !disabled && setHover(level)}
                onMouseLeave={() => setHover(null)}
                className={`tnum flex-1 cursor-pointer text-center font-mono
                            text-[11px] transition-colors ${
                  value === level ? "font-medium text-ink" : "text-ink-faint"}`}>
            {level}&thinsp;%
          </span>
        ))}
      </div>

      <p className="mt-2.5 max-w-[38ch] text-[12px] leading-relaxed text-ink-faint">
        A detector that needs the whole cascade arrives after the story has
        spread. Cut to a fifth, the reading below is what the system would have
        said while it was still moving.
      </p>
    </div>
  );
}
