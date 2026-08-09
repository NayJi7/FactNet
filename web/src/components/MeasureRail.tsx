import type { Trace } from "../lib/types";

/**
 * The measurement rail.
 *
 * Every stage that produces a probability plots it against one shared axis, so
 * that disagreement between the content model, the structural model and their
 * integration is a shape rather than a paragraph. The 0.5 boundary is drawn
 * firmly and the unreliable half is washed, because the distance from that line
 * is the only thing the verdict actually reports.
 */

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

  const H = 470;
  const y = (v: number) => (1 - v) * H;

  return (
    <div className="sticky top-8 hidden w-[118px] shrink-0 border-r border-rule pr-4 lg:block">
      <p className="eyebrow mb-3">Readings</p>
      <svg viewBox={`0 0 108 ${H + 26}`} className="w-full overflow-visible"
           role="img" aria-label="Each stage plotted on a shared probability axis">
        {/* the unreliable half, washed rather than outlined */}
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

        {/* the path between stages: the zigzag is the disagreement */}
        {points.length > 1 && (
          <polyline
            points={points.map((p, i) => `${23 + (i + 1) * 0},${y(p.value)}`).join(" ")}
            fill="none" stroke="var(--color-rule-firm)" strokeWidth="0.75" />
        )}

        {points.map((p, i) => (
          <g key={p.key} className="settle" style={{ animationDelay: `${i * 60}ms` }}>
            <line x1="18" y1={y(p.value)} x2="28" y2={y(p.value)}
                  stroke={STROKE[p.kind]} strokeWidth={p.kind === "final" ? 3 : 2}
                  strokeDasharray={p.counted ? undefined : "2 2"} />
            {/* a reading the verdict did not use is drawn hollow: present, not counted */}
            <circle cx="23" cy={y(p.value)} r={p.kind === "final" ? 4 : 3}
                    fill={p.counted ? STROKE[p.kind] : "var(--color-panel)"}
                    stroke={STROKE[p.kind]} strokeWidth={p.counted ? 0 : 1.2} />
            <text x="34" y={y(p.value) - 4} fontSize="9.5"
                  fill={p.kind === "final" ? "var(--color-ink)" : "var(--color-ink-soft)"}
                  fontWeight={p.kind === "final" ? 600 : 400}>
              {p.label}{p.counted ? "" : " ·"}
            </text>
            <text x="34" y={y(p.value) + 7} fontSize="10" fontFamily="var(--font-mono)"
                  fill="var(--color-ink-faint)">{p.value.toFixed(3)}</text>
          </g>
        ))}
      </svg>
      <p className="mt-3 text-[10.5px] leading-snug text-ink-faint">
        One axis, every stage. The washed half is unreliable; a hollow mark is a
        reading the verdict did not use.
      </p>
    </div>
  );
}
