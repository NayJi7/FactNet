import { useEffect, useState } from "react";
import FigureView from "./Figures";
import Pending from "./Pending";
import StepList from "./StepList";
import Verdict from "./Verdict";
import { getSampleDetail, getSamples } from "../lib/api";
import { Account } from "../lib/handle";
import type { Figure, ModelCard, Sample, SampleDetail, Step, Trace } from "../lib/types";

/**
 * What a cascade actually is, before any model is asked about it.
 *
 * The point of this page is that a verdict on an object nobody has looked at
 * teaches nothing. It shows the post that started it, the outlet it links to
 * and how the label was derived from that, how far it travelled, and who
 * carried it. Running a detector is the last thing offered, not the first.
 */

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className="tnum mt-1 font-mono text-[15px]">{value}</dd>
    </div>
  );
}

export default function Cascades({
  models, onAnalyse, busy, trace, arriving, onChange,
}: {
  models: ModelCard[];
  onAnalyse: (id: number, model: string) => void;
  busy: boolean;
  trace: Trace | null;
  arriving: Step[];
  onChange: () => void;
}) {
  const [list, setList] = useState<Sample[]>([]);
  const [chosen, setChosen] = useState(0);
  const [detail, setDetail] = useState<SampleDetail | null>(null);
  const [model, setModel] = useState(models.find((m) => m.primary)?.key ?? "roberta");

  useEffect(() => { getSamples().then((s) => setList(s.samples)).catch(() => {}); }, []);
  useEffect(() => {
    setDetail(null);
    onChange();                       // a reading belongs to one cascade only
    getSampleDetail(chosen).then(setDetail).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chosen]);

  // the caption is the engine's, not a second copy written here: two wordings
  // of the same figure drift apart the moment one of them is edited
  const graphFigure: Figure | null = detail && {
    kind: "graph", title: "Every account, and who they took it from",
    data: detail.graph,
    caption: detail.graph_caption ?? "",
  };

  return (
    <div className="lg:grid lg:grid-cols-[290px_1fr] lg:gap-12">
      <nav className="lg:sticky lg:top-8 lg:self-start">
        <p className="eyebrow mb-3">Four collected cascades</p>
        <ul className="space-y-px">
          {list.map((s) => (
            <li key={s.id}>
              <button onClick={() => setChosen(s.id)}
                      aria-current={chosen === s.id}
                      className={`w-full border-l-2 py-2.5 pl-3 pr-2 text-left transition-colors ${
                        chosen === s.id
                          ? "border-ink bg-panel"
                          : "border-transparent hover:bg-panel/60"
                      }`}>
                <span className="flex items-baseline justify-between gap-2">
                  <span className="font-mono text-[12.5px]">{s.source_domain}</span>
                  <span className="tnum font-mono text-[11px] text-ink-faint">
                    {s.accounts}
                  </span>
                </span>
                <span className="mt-0.5 block text-[11.5px] leading-snug text-ink-faint">
                  {s.label === 0 ? "linked to a low-rated outlet" : "linked to a vetted outlet"}
                </span>
              </button>
            </li>
          ))}
        </ul>
        <p className="mt-4 max-w-[34ch] text-[12px] leading-relaxed text-ink-faint">
          Kept beside the code, so a demonstration never depends on the network.
          One of them the detector gets wrong, on purpose.
        </p>
      </nav>

      {detail && (
        <article className="mt-10 min-w-0 lg:mt-0">
          <header className="border-b border-rule-firm pb-6">
            <p className="eyebrow">The post that started it</p>
            <p className="mt-3 max-w-[68ch] text-[19px] leading-snug">{detail.text}</p>
            <p className="mt-3 flex flex-wrap items-baseline gap-x-4 text-[12.5px] text-ink-faint">
              <Account handle={detail.source_handle} />
              <span>{new Date(detail.created_at).toLocaleDateString("en-GB",
                { day: "numeric", month: "long", year: "numeric" })}</span>
              {detail.url && (
                <a href={detail.url} target="_blank" rel="noreferrer"
                   className="underline underline-offset-2 hover:text-ink">
                  open on Bluesky
                </a>
              )}
            </p>
          </header>

          <section className="border-b border-rule-firm py-5">
            <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
              <div className="max-w-[46ch]">
                <p className="eyebrow">Ask a model</p>
                <p className="mt-1.5 text-[13px] leading-relaxed text-ink-soft">
                  A cascade is read differently from a bare sentence: text and shape
                  are scored apart, then combined, then replayed as the spread grew.
                </p>
              </div>
              <div className="flex flex-wrap items-end gap-3">
                <label className="min-w-52">
                  <span className="eyebrow mb-1.5 block">Content model</span>
                  <select value={model} onChange={(e) => setModel(e.target.value)}
                          className="w-full border border-rule bg-panel p-2 text-[13px]">
                    {models.map((m) => (
                      <option key={m.key} value={m.key} disabled={!m.available}>
                        {m.name}{m.macro_f1 ? ` — macro-F1 ${m.macro_f1}` : ""}
                      </option>
                    ))}
                  </select>
                </label>
                <button onClick={() => onAnalyse(detail.id, model)} disabled={busy}
                        className="bg-ink px-6 py-2.5 text-[13.5px] font-medium text-paper transition-opacity hover:opacity-90 disabled:opacity-40">
                  {busy ? "Reading" : "Read this cascade"}
                </button>
              </div>
            </div>
          </section>

          {busy && (
            <section className="border-b border-rule-firm py-7">
              <Pending withCascade done={arriving} />
            </section>
          )}

          {!busy && trace && (
            <section className="border-b border-rule-firm py-7">
              <Verdict trace={trace} />
              <StepList steps={trace.steps} />
            </section>
          )}

          <section className="border-b border-rule py-6">
            <p className="eyebrow mb-3">Where the label comes from</p>
            <p className="max-w-[70ch] text-[14px] leading-relaxed">
              The post links to{" "}
              <span className="font-mono font-semibold">{detail.source_domain}</span>, rated{" "}
              <span className={detail.source_label === "misleading" ? "text-signal" : ""}>
                {detail.source_label === "misleading"
                  ? "low for factual reporting"
                  : "high for factual reporting"}
              </span>
              . The label describes that outlet, not this post: an account
              debunking an unreliable article is scored like one relaying it, and
              that residual noise is a known property of labelling by source.
            </p>
          </section>

          <section className="border-b border-rule py-6">
            <p className="eyebrow mb-4">How far it travelled</p>
            <dl className="flex flex-wrap gap-x-12 gap-y-4">
              <Field label="accounts" value={detail.shape.accounts} />
              <Field label="direct shares" value={detail.shape.direct} />
              <Field label="depth" value={detail.shape.depth} />
              <Field label="widest level" value={detail.shape.breadth} />
              <Field label="reposts" value={detail.reposts} />
              <Field label="replies" value={detail.replies} />
              <Field label="likes" value={detail.likes} />
            </dl>
            {graphFigure && <FigureView figure={graphFigure} />}
          </section>

          <section className="border-b border-rule py-6">
            <p className="eyebrow mb-1">Who carried it</p>
            <p className="mb-3 text-[12.5px] text-ink-faint">
              The first {detail.people_shown} of {detail.shape.accounts}, in the order
              the cascade reached them.
            </p>
            <div className="max-h-[300px] overflow-y-auto">
              <table className="w-full text-[12.5px]">
                <thead className="sticky top-0 bg-paper">
                  <tr className="border-b border-rule-firm text-left">
                    {["account", "role", "hops", "followers", "posts"].map((c) => (
                      <th key={c} className="eyebrow pb-1.5 pr-5 font-medium">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {detail.people.map((p, i) => (
                    <tr key={i} className="border-b border-rule last:border-0">
                      <td className="py-1.5 pr-5"><Account handle={p.handle} /></td>
                      <td className="py-1.5 pr-5 text-ink-faint">{p.kind}</td>
                      <td className="tnum py-1.5 pr-5 font-mono">{p.hops ?? "—"}</td>
                      <td className="tnum py-1.5 pr-5 font-mono">
                        {p.followers.toLocaleString("en")}
                      </td>
                      <td className="tnum py-1.5 pr-5 font-mono">
                        {p.posts.toLocaleString("en")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

        </article>
      )}
    </div>
  );
}
