import type { Step, Trace } from "../lib/types";

/**
 * Content vs propagation reading under the verdict, + how much the structural
 * verdict moved when the content score was added (it's a feature, not a
 * weighted average, so there's no weight to show).
 */
interface Mark {
  key: "content" | "propagation" | "both";
  label: string;
  p: number;
  note: string;
  counted: boolean;
}

const COLOUR: Record<Mark["key"], string> = {
  content: "var(--color-content)",
  propagation: "var(--color-struct)",
  both: "var(--color-ink)",
};

function build(steps: Step[]): { marks: Mark[]; moved: number | null; counted: boolean } {
  const at = (key: string) => steps.find((s) => s.key === key);
  const content = at("content");
  const structure = at("structure");
  const integration = at("integration");
  const marks: Mark[] = [];

  if (content && typeof content.detail.p_reliable === "number") {
    marks.push({
      key: "content", label: content.detail.model ?? "Content model",
      p: content.detail.p_reliable,
      note: content.detail.counted === false
        ? "read, and excluded from the verdict"
        : "from the wording alone",
      counted: content.detail.counted !== false,
    });
  }
  if (structure && typeof structure.detail.p_reliable === "number") {
    // structure alone
    const alone = typeof integration?.detail.without_score === "number"
      ? integration.detail.without_score : structure.detail.p_reliable;
    marks.push({
      key: "propagation", label: structure.detail.model ?? "Propagation model",
      p: alone, note: "from the shape of the cascade alone", counted: true,
    });
  }
  if (integration && typeof integration.detail.with_score === "number") {
    marks.push({
      key: "both", label: "The two together", p: integration.detail.with_score,
      note: integration.detail.counted === false
        ? "shown, and excluded from the verdict"
        : "the content score written onto the cascade root",
      counted: integration.detail.counted !== false,
    });
  }
  return {
    marks,
    moved: typeof integration?.detail.delta === "number" ? integration.detail.delta : null,
    counted: integration?.detail.counted !== false,
  };
}

export default function Breakdown({ trace }: { trace: Trace }) {
  const { marks, moved, counted } = build(trace.steps);
  if (marks.length < 2) return null;

  return (
    <div className="mt-8 border-t border-rule pt-5">
      <p className="eyebrow">What each half read</p>

      <div className="mt-3.5 space-y-2.5">
        {marks.map((mark) => (
          <div key={mark.key} className="flex items-center gap-3">
            <span className="w-[180px] shrink-0 truncate text-[12.5px]" title={mark.label}>
              <span className="mr-1.5 inline-block h-2 w-2 align-middle"
                    style={{ background: COLOUR[mark.key] }} aria-hidden />
              <span className={mark.key === "both" ? "font-semibold" : "text-ink-soft"}>
                {mark.label}
              </span>
            </span>
            <div className="relative h-[16px] flex-1 bg-sunk">
              <div className="absolute bottom-0 left-1/2 top-0 w-px bg-rule-firm" aria-hidden />
              <div className="absolute inset-y-0 transition-[width] duration-500"
                   style={{
                     width: `${Math.abs(mark.p - 0.5) * 100}%`,
                     left: mark.p >= 0.5 ? "50%" : undefined,
                     right: mark.p < 0.5 ? "50%" : undefined,
                     background: COLOUR[mark.key],
                     opacity: mark.counted ? 1 : 0.35,
                   }} />
            </div>
            <span className="tnum w-[52px] shrink-0 text-right font-mono text-[11.5px] text-ink-soft">
              {mark.p.toFixed(3)}
            </span>
          </div>
        ))}
      </div>

      {/* same gutters as the rows (180px / 52px). labels absolute, justify-between
          put 0.5 off center */}
      <div className="mt-2 flex items-center gap-3 text-[11.5px] text-ink-faint">
        <span className="w-[180px] shrink-0" aria-hidden />
        <div className="relative h-[1.3em] flex-1">
          <span className="absolute left-0">misleading</span>
          <span className="tnum absolute left-1/2 -translate-x-1/2 font-mono">0.5</span>
          <span className="absolute right-0">reliable</span>
        </div>
        <span className="w-[52px] shrink-0" aria-hidden />
      </div>

      <p className="mt-4 max-w-[80ch] text-[13px] leading-relaxed text-ink-soft">
        {moved === null ? (
          <>
            Only one half could read this input, so the verdict is its reading.
            The other needs what this input does not carry.
          </>
        ) : (
          <>
            Nothing here is a vote or a weighted average. The content score is
            written onto the root of the cascade as one more number per account,
            and the propagation model decides from there. Attaching it moved the
            structural verdict by{" "}
            <span className="tnum font-mono font-semibold text-ink">
              {moved >= 0 ? "+" : ""}{moved.toFixed(3)}
            </span>
            {counted ? " on this cascade." : ", and that reading takes no part in the verdict."}
            {" "}Across a whole benchmark split the same ablation is worth a
            quarter of the remaining error where the cascade carries no text of
            its own, and nothing where it already does.
          </>
        )}
      </p>
    </div>
  );
}
