import ModuleTag from "./ModuleTag";

/**
 * What the two modules are, said once, where everyone lands.
 *
 * Their names are stamped on every stage and every table, and until now
 * nowhere on the site explained what either one reads or what it is worth.
 * A colour that labels something unexplained is decoration.
 *
 * Each panel states three things and no more: what the module reads, what it
 * costs it, and what it cannot do. The last one is the point of the project,
 * so it is not buried at the bottom of a results table.
 */
const PANELS = [
  {
    module: "content" as const,
    reads: "The words of the post itself.",
    how: "A RoBERTa encoder fine-tuned on LIAR, a corpus of short political "
       + "claims, beside a bag of words and three other transformers.",
    limit: "It stops near 0.63 macro-F1 and no transformer here separates from "
         + "the bag of words. The ceiling belongs to the one-sentence format: "
         + "handed a whole news article instead, the same approach reaches 0.805.",
  },
  {
    module: "propagation" as const,
    reads: "The shape of the cascade. Who shared it, from whom, and how far it went.",
    how: "A bidirectional graph convolutional network reading the reshare tree, "
       + "with no access to the wording at all.",
    limit: "It reaches 0.920 on the benchmark, and a model with every edge "
         + "deleted reaches 0.940 on the same input. The structure separates "
         + "from that baseline on one configuration of four.",
  },
];

export default function Modules() {
  return (
    <section className="border-t border-rule-firm pt-6">
      <p className="eyebrow">The two halves of this system</p>
      <p className="mt-2 max-w-[80ch] text-[13.5px] leading-relaxed text-ink-soft">
        A claim can be judged by what it says or by the way it travelled. This
        project builds both, measures each against a baseline that ignores it,
        and reports where each one runs out. Every stage and every table below
        carries the mark of the half it came from.
      </p>

      <div className="mt-6 grid gap-x-10 gap-y-7 lg:grid-cols-2">
        {PANELS.map((panel) => (
          <div key={panel.module} className="border-t-2 pt-3"
               style={{ borderColor: panel.module === "content"
                 ? "var(--color-content)" : "var(--color-struct)" }}>
            <ModuleTag module={panel.module} className="!border-l-0 !pl-0" />
            <p className="mt-2 text-[15px] font-medium leading-snug">{panel.reads}</p>
            <p className="mt-2 max-w-[52ch] text-[13px] leading-relaxed text-ink-soft">
              {panel.how}
            </p>
            <p className="mt-2.5 max-w-[52ch] border-t border-rule pt-2.5 text-[13px] leading-relaxed text-ink-soft">
              <span className="font-medium text-ink">Where it runs out. </span>
              {panel.limit}
            </p>
          </div>
        ))}
      </div>

      <div className="mt-7 border-t border-rule pt-4">
        <ModuleTag module="both" />
        <p className="mt-2 max-w-[80ch] text-[13px] leading-relaxed text-ink-soft">
          The two meet at one point, and it is deliberately narrow. The content
          module's score is written onto the root of the cascade as one more
          number per account, and the propagation model decides from there. It
          is not a vote and not a weighted average: nothing averages the two
          verdicts. Measured as an ablation, that one column recovers a quarter
          of the error left where the cascade carries no text of its own, and
          adds nothing where it already does.
        </p>
      </div>
    </section>
  );
}
