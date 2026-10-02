import type { Trace } from "../lib/types";

/** Every p(reliable) from the steps on one vertical axis, 0.5 line in the middle. */

interface Reading {
  key: string;
  label: string;
  value: number;
  kind: "content" | "struct" | "final";
  counted: boolean;
}

export function readings(trace: Trace): Reading[] {
  const out: Reading[] = [];
  for (const step of trace.steps) {
    const p = step.detail?.p_reliable;
    if (step.key === "content" && typeof p === "number")
      out.push({ key: "content", label: "text", value: p, kind: "content",
                 counted: step.detail?.counted !== false });
    if (step.key === "structure" && typeof p === "number")
      out.push({ key: "structure", label: "cascade", value: p, kind: "struct",
                 counted: true });
    if (step.key === "integration") {
      const withScore = step.detail?.with_score;
      if (typeof withScore === "number")
        out.push({ key: "integration", label: "combined", value: withScore,
                   kind: "struct", counted: true });
    }
  }
  if (typeof trace.verdict === "number")
    out.push({ key: "verdict", label: "verdict", value: trace.verdict,
               kind: "final", counted: true });
  return out;
}

const STROKE = {
  content: "var(--color-ink-soft)",
  struct: "var(--color-struct)",
  final: "var(--color-ink)",
} as const;

export default function MeasureRail({ trace }: { trace: Trace }) {
  const points = readings(trace);
  if (!points.length) return null;

  // must fit under the input panel on a laptop screen
  const H = 250;
  const y = (v: number) => (1 - v) * H;

  return (
    <div className="mt-8 hidden border-t border-rule-firm pt-5 lg:block">
      <p className="eyebrow mb-3">Readings</p>
      {/* fixed width, otherwise the viewBox scales the text up too */}
      <svg viewBox={`0 0 150 ${H + 26}`} className="w-[196px] overflow-visible"
           role="img" aria-label="Each stage plotted on a shared probability axis">
        {/* misleading half */}
        <rect x="18" y={y(0.5)} width="10" height={H - y(0.5)} fill="var(--color-signal-dim)" />
        <line x1="23" y1="0" x2="23" y2={H} stroke="var(--color-ink-faint)" strokeWidth="1.25" />

        {[1, 0.5, 0].map((v) => (
          <g key={v}>
            <line x1={v === 0.5 ? 14 : 19} y1={y(v)} x2={28} y2={y(v)}
                  stroke={v === 0.5 ? "var(--color-ink-faint)" : "var(--color-rule-firm)"}
                  strokeWidth={v === 0.5 ? 1 : 0.75}
                  strokeDasharray={v === 0.5 ? "2 2" : undefined} />
            <text x="34" y={y(v) + 3} fontSize="9" fill="var(--color-ink-faint)"
                  fontFamily="var(--font-mono)">{v.toFixed(1)}</text>
          </g>
        ))}

        {/* line between steps */}
        {points.length > 1 && (
          <polyline
            points={points.map((p, i) => `${23 + (i + 1) * 0},${y(p.value)}`).join(" ")}
            fill="none" stroke="var(--color-rule-firm)" strokeWidth="0.75" />
        )}

        {/* push labels apart when values are equal (they overlapped) */}
        {(() => {
          const written: number[] = [];
          return points.map((p, i) => {
            let ty = y(p.value);
            while (written.some((used) => Math.abs(used - ty) < 20)) ty += 20;
            written.push(ty);
            return (
              <g key={p.key} className="settle" style={{ animationDelay: `${i * 60}ms` }}>
                <line x1="18" y1={y(p.value)} x2="28" y2={y(p.value)}
                      stroke={STROKE[p.kind]} strokeWidth={p.kind === "final" ? 3 : 2}
                      strokeDasharray={p.counted ? undefined : "2 2"} />
                {/* hollow = not counted */}
                <circle cx="23" cy={y(p.value)} r={p.kind === "final" ? 4 : 3}
                        fill={p.counted ? STROKE[p.kind] : "var(--color-panel)"}
                        stroke={STROKE[p.kind]} strokeWidth={p.counted ? 0 : 1.2} />
                {ty !== y(p.value) && (
                  <line x1="28" y1={y(p.value)} x2="33" y2={ty - 3}
                        stroke="var(--color-rule-firm)" strokeWidth="0.6" />
                )}
                <text x="34" y={ty - 4} fontSize="9.5"
                      fill={p.kind === "final" ? "var(--color-ink)" : "var(--color-ink-soft)"}
                      fontWeight={p.kind === "final" ? 600 : 400}>
                  {p.label}{p.counted ? "" : " ·"}
                </text>
                <text x="34" y={ty + 7} fontSize="10" fontFamily="var(--font-mono)"
                      fill="var(--color-ink-faint)">{p.value.toFixed(3)}</text>
              </g>
            );
          });
        })()}
      </svg>
      <p className="mt-3 max-w-[42ch] text-[11px] leading-snug text-ink-faint">
        One axis, every stage. The washed half is unreliable; a hollow mark is a
        reading the verdict did not use.
      </p>
    </div>
  );
}
