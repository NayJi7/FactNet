import ModuleTag from "./ModuleTag";
import type { ModelCard } from "../lib/types";

/**
 * The graph detectors and how they differ (features, platform), and why the
 * default is picked automatically. Scores come from /api/models.
 */
const FAMILY: Record<string, { kind: string; what: string }> = {
  "bigcn-upfd-profile": {
    kind: "Bidirectional, account features",
    what: "Reads the cascade twice, along the direction it spread and against "
        + "it. Each account is ten public counters. The detector of record for "
        + "benchmark cascades.",
  },
  "gcn-upfd-profile": {
    kind: "One direction, account features",
    what: "The same input through a plain graph convolution. Kept to show what "
        + "the bidirectional design is worth, which is little: the two sit "
        + "within a seed deviation of each other.",
  },
  "gat-upfd-profile": {
    kind: "Attention over neighbours",
    what: "Weighs each neighbour instead of averaging them. Same input again, "
        + "and the same figure again, which is the point of listing all three.",
  },
  "bigcn-structure": {
    kind: "Shape only",
    what: "Every account feature replaced by a constant, so nothing survives "
        + "but the shape of the cascade. The honest test of the structural "
        + "claim, and the fallback when a cascade arrives without features.",
  },
  "bigcn-collected": {
    kind: "Trained on Bluesky",
    what: "The same architecture trained on the 400 collected cascades. It "
        + "exists because the benchmark detector does not survive the move "
        + "between platforms, and applying it to Bluesky data is the failure "
        + "this project reports.",
  },
};

const ORDER = ["bigcn-upfd-profile", "gcn-upfd-profile", "gat-upfd-profile",
               "bigcn-structure", "bigcn-collected"];

export default function GraphModels({ models }: { models: ModelCard[] }) {
  const listed = ORDER
    .map((key) => models.find((m) => m.key === key))
    .filter((m): m is ModelCard => Boolean(m));
  if (listed.length < 2) return null;

  return (
    <section className="border-t border-rule-firm pt-6">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <p className="eyebrow">The detectors, and what separates them</p>
        <ModuleTag module="propagation" />
      </div>

      <div className="mt-4 space-y-3.5">
        {listed.map((model) => {
          const family = FAMILY[model.key];
          return (
            <div key={model.key}
                 className="grid gap-x-5 gap-y-1 border-t border-rule pt-3 sm:grid-cols-[210px_1fr_84px]">
              <div>
                <p className="text-[13.5px] font-medium leading-snug">
                  {model.primary && <span className="mr-1 text-signal">▸</span>}
                  {model.name}
                </p>
                <p className="text-[11px] text-ink-faint">{family?.kind}</p>
              </div>
              <p className="max-w-[62ch] text-[13px] leading-relaxed text-ink-soft">
                {family?.what}
              </p>
              <p className="tnum font-mono text-[13px] sm:text-right">
                {model.macro_f1?.toFixed(3) ?? "·"}
                <span className="ml-1 text-[10.5px] text-ink-faint">
                  {model.macro_f1 ? "F1" : "not scored"}
                </span>
              </p>
            </div>
          );
        })}
      </div>

      <div className="mt-5 border-t border-rule pt-4">
        <p className="max-w-[86ch] text-[13px] leading-relaxed text-ink-soft">
          <span className="font-medium text-ink">Why the system chooses. </span>
          These figures are not comparable to each other the way the content
          figures are. The first three are measured on the same benchmark split
          and separate by less than a seed deviation. The fourth reads no
          features, so a lower figure is expected and is the finding rather than
          a defect. The fifth is measured on collected Bluesky cascades and on a
          different task, so it cannot be ranked against the others at all. By
          default the system picks by where the cascade came from. Overriding
          that is worth doing once, on a Bluesky cascade with a benchmark
          detector, because watching the transfer fail is more convincing than
          reading that it does.
        </p>
      </div>
    </section>
  );
}
