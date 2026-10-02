import ModuleTag from "./ModuleTag";
import type { ModelCard } from "../lib/types";

/**
 * The 5 content models. None is significantly better than tf-idf, so the
 * default one is a protocol choice. Scores come from /api/models.
 */
const FAMILY: Record<string, { kind: string; what: string }> = {
  tfidf: {
    kind: "Bag of words",
    what: "Counts words and weighs each one. It cannot see order, context or "
        + "negation, and it trains in seconds on a laptop.",
  },
  bert: {
    kind: "Transformer, fine-tuned",
    what: "The original encoder, retrained end to end on the task. It is here "
        + "to show how much the architecture alone is worth.",
  },
  roberta: {
    kind: "Transformer, fine-tuned",
    what: "The same idea with better pre-training. Standard regime: three "
        + "epochs, no warmup, no weight decay.",
  },
  "roberta-opt": {
    kind: "Transformer, tuned harder",
    what: "The same encoder given longer inputs, a warmup and weight decay. It "
        + "wins on the validation split and loses on the test split.",
  },
  "roberta-wd": {
    kind: "Transformer, one extra run",
    what: "A single seed that landed above the rest. Kept visible, and counted "
        + "for nothing, because one seed cannot be told from luck.",
  },
};

const ORDER = ["tfidf", "bert", "roberta", "roberta-opt", "roberta-wd"];

export default function ContentModels({ models }: { models: ModelCard[] }) {
  const listed = ORDER
    .map((key) => models.find((m) => m.key === key))
    .filter((m): m is ModelCard => Boolean(m));
  if (listed.length < 2) return null;

  const reference = listed.find((m) => m.primary);
  const best = listed.reduce((a, b) => ((b.macro_f1 ?? 0) > (a.macro_f1 ?? 0) ? b : a));
  const bag = listed.find((m) => m.key === "tfidf");

  return (
    <section className="border-t border-rule-firm pt-6">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <p className="eyebrow">The five readers, and what separates them</p>
        <ModuleTag module="content" />
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
                {model.macro_f1?.toFixed(3) ?? "-"}
                <span className="ml-1 text-[10.5px] text-ink-faint">F1</span>
              </p>
            </div>
          );
        })}
      </div>

      <div className="mt-5 border-t border-rule pt-4">
        <p className="max-w-[86ch] text-[13px] leading-relaxed text-ink-soft">
          <span className="font-medium text-ink">
            Why {reference?.name ?? "one of them"} is the reference.{" "}
          </span>
          Not because it wins. {bag && best && bag.key !== best.key ? (
            <>
              The bag of words reaches {bag.macro_f1?.toFixed(3)} and the highest
              figure in the column, {best.macro_f1?.toFixed(3)}, belongs to a single
              extra run that the study does not report.{" "}
            </>
          ) : null}
          It is the reference because it is the best of the three transformers run
          under the protocol the study describes, and because a reference has to be
          fixed before the results are read rather than chosen after. The finding is
          that the choice barely matters: no gap between any two of these models
          excludes zero under a paired bootstrap, so the ceiling near 0.63 belongs
          to the task and not to the model you pick.
        </p>
      </div>
    </section>
  );
}
