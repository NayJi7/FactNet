import { useEffect, useState } from "react";
import FigureView from "./Figures";
import Observed from "./Observed";
import Pending from "./Pending";
import StepList from "./StepList";
import Verdict from "./Verdict";
import { getSampleDetail, getSamples } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import Failed from "./Failed";
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

const BLUESKY = "#0085FF";

/** The platform's mark, drawn rather than fetched: no request, and it inherits
 *  the colour of the text it sits in. */
function Butterfly({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size * (501 / 568)} viewBox="0 0 568 501"
         fill="currentColor" aria-hidden className="shrink-0">
      <path d="M123.121 33.664C188.241 82.553 258.281 181.68 284 234.873c25.719-53.192
               95.759-152.32 160.879-201.209C491.866-1.611 568-28.906 568 57.947c0
               17.346-9.945 145.713-15.778 166.555-20.275 72.453-94.155 90.933-159.875
               79.748C507.222 323.8 536.444 388.56 473.333 453.32c-119.86 122.992-172.272-30.859-185.702-70.281-2.462-7.227-3.614-10.608-3.631-7.733-.017-2.875-1.169.506-3.631
               7.733-13.43 39.422-65.842 193.273-185.702 70.281-63.111-64.76-33.89-129.52
               80.986-149.071-65.72 11.185-139.6-7.295-159.875-79.748C9.945 203.66 0
               75.293 0 57.947 0-28.906 76.135-1.611 123.121 33.664Z" />
    </svg>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className="tnum mt-1 font-mono text-[15px]">{value}</dd>
    </div>
  );
}

export default function Cascades({
  models, graphModels, onAnalyse, busy, trace, arriving, onChange, rejoined, jobLabel, elapsed,
}: {
  models: ModelCard[];
  graphModels: ModelCard[];
  onAnalyse: (id: number, model: string, graphModel?: string,
              observed?: number) => void;
  busy: boolean;
  trace: Trace | null;
  arriving: Step[];
  onChange: () => void;
  rejoined?: boolean;
  jobLabel?: string;
  elapsed?: number;
}) {
  const [chosen, setChosen] = useState(0);
  const [model, setModel] = useState(models.find((m) => m.primary)?.key ?? "roberta");
  const [graphModel, setGraphModel] = useState("");
  const [observed, setObserved] = useState(100);

  const samples = useAsync(() => getSamples().then((s) => s.samples), []);
  const list: Sample[] = samples.data ?? [];

  const loaded = useAsync<SampleDetail>(() => getSampleDetail(chosen), [chosen]);
  const detail = loaded.data;

  // a reading belongs to one cascade only
  useEffect(() => { onChange(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [chosen]);

  // the caption is the engine's, not a second copy written here: two wordings
  // of the same figure drift apart the moment one of them is edited
  const graphFigure: Figure | null = detail && {
    kind: "graph", title: "Every account, and who they took it from",
    data: detail.graph,
    caption: detail.graph_caption ?? "",
  };

  return (
    <div className="lg:grid lg:grid-cols-[290px_1fr] lg:gap-12">
      {/* top-28 clears the sticky header, which is 7rem tall. At top-8 the first
          entry of the list slid under it as soon as the page scrolled. The same
          offset is used by every other sticky column in the interface. */}
      <nav className="lg:sticky lg:top-28 lg:self-start">
        <p className="eyebrow mb-3">Pick one</p>
        <ul className="space-y-px">
          {list.map((s) => (
            <li key={s.id}>
              {/* Changing cascade mid-reading would swap the object under the
                  running job and land its verdict on the wrong page, so the
                  list is held until the reading finishes. */}
              <button onClick={() => setChosen(s.id)}
                      aria-current={chosen === s.id}
                      disabled={busy && chosen !== s.id}
                      title={busy && chosen !== s.id
                        ? "A reading is under way on another cascade." : undefined}
                      className={`w-full border-l-2 py-2.5 pl-3 pr-2 text-left transition-colors ${
                        chosen === s.id
                          ? "border-ink bg-panel"
                          : busy
                            ? "border-transparent opacity-40"
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
        {samples.error && !list.length && (
          <button onClick={samples.reload}
                  className="mt-3 w-full border border-rule px-3 py-2 text-left text-[12px]
                             leading-relaxed text-ink-soft transition-colors hover:bg-panel">
            The list did not load. Tap to try again.
          </button>
        )}

        {/* The controls sit with the list rather than beside the post, so the
            column that chooses what to read is also the column that decides how,
            which is the arrangement the post reader already uses. */}
        {detail && (
          <div className="mt-6 border-t border-rule pt-4">
            <p className="eyebrow">Ask a model</p>
            <p className="mt-1.5 max-w-[34ch] text-[12.5px] leading-relaxed text-ink-soft">
              A cascade is read differently from a bare sentence: text and shape
              are scored apart, then combined, then replayed as the spread grew.
            </p>

            <label className="mt-3.5 block">
              <span className="eyebrow mb-1.5 block">Content model</span>
              <select value={model} onChange={(e) => setModel(e.target.value)}
                      className="w-full border border-rule bg-panel p-2 text-[13px]">
                {models.map((m) => (
                  <option key={m.key} value={m.key} disabled={!m.available}>
                    {m.name}{m.macro_f1 ? ` \u2014 macro-F1 ${m.macro_f1}` : ""}
                  </option>
                ))}
              </select>
            </label>

            <div className="mt-4">
              <Observed value={observed} onChange={setObserved} disabled={busy} />
            </div>

            <button onClick={() =>
                      onAnalyse(detail.id, model, graphModel || undefined, observed)}
                    disabled={busy}
                    className="mt-3 w-full bg-ink px-6 py-2.5 text-[13.5px] font-medium
                               text-paper transition-opacity hover:opacity-90 disabled:opacity-40">
              {busy ? "Reading" : "Read this cascade"}
            </button>

            {/* The detector choice belongs here more than anywhere else: the
                demonstration it exists for is forcing the benchmark model onto a
                Bluesky cascade, and this is the tab that has cascades. */}
            <details className="mt-4 border-t border-rule pt-3">
              <summary className="cursor-pointer select-none text-[13px] font-medium text-ink">
                Choose the propagation detector yourself
              </summary>
              <div className="mt-3 space-y-2">
                <select value={graphModel} onChange={(e) => setGraphModel(e.target.value)}
                        aria-label="Propagation detector"
                        className="w-full border border-rule bg-panel p-2 text-[13px]">
                  <option value="">Let the system choose (recommended)</option>
                  {graphModels
                    .filter((m) => m.available && m.key !== "bigcn-upfd-profile-score")
                    .map((m) => (
                      <option key={m.key} value={m.key}>
                        {m.name}{m.macro_f1 ? ` \u2014 macro-F1 ${m.macro_f1}` : ""}
                      </option>
                    ))}
                </select>
                <p className="max-w-[34ch] text-[12.5px] leading-relaxed text-ink-faint">
                  These are collected Bluesky cascades. Forcing a benchmark
                  detector onto one is worth doing once: it is the cross-platform
                  failure the article reports, and it is more convincing watched
                  than read.
                </p>
              </div>
            </details>
          </div>
        )}
      </nav>

      {loaded.error && !detail && (
        <div className="mt-10 min-w-0 lg:mt-0">
          <Failed error={loaded.error} onRetry={loaded.reload} what="this cascade" />
        </div>
      )}

      {detail && (
        <article className="mt-10 min-w-0 lg:mt-0">
          <header className="border-b border-rule-firm pb-6">
            <p className="eyebrow">The post that started it</p>
            <p className="mt-3 max-w-[68ch] text-[19px] leading-snug">{detail.text}</p>
            <p className="mt-3 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[12.5px] text-ink-faint">
              {detail.source_name && (
                <span className="font-medium text-ink">{detail.source_name}</span>
              )}
              <Account handle={detail.source_handle} />
              <span>{new Date(detail.created_at).toLocaleDateString("en-GB",
                { day: "numeric", month: "long", year: "numeric" })}</span>
            </p>

            {/* What the post did on the platform, beside what the collector
                reached. The two differ by a lot and the gap is the point: a
                cascade here is a sample of the diffusion, never all of it. */}
            <div className="mt-4 flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
              <dl className="flex flex-wrap gap-x-8 gap-y-3">
                {([["likes", detail.likes], ["replies", detail.replies],
                   ["reposts", detail.reposts],
                   ["accounts collected", detail.shape.accounts]] as [string, number][])
                  .map(([label, value]) => (
                    <div key={label}>
                      <dt className="eyebrow">{label}</dt>
                      <dd className="tnum mt-1 font-mono text-[17px] leading-none">
                        {value.toLocaleString("en")}
                      </dd>
                    </div>
                  ))}
              </dl>
              {detail.url && (
                <a href={detail.url} target="_blank" rel="noreferrer"
                   /* the platform's own blue, so the button reads as a way out
                      to Bluesky rather than as one more control of this page */
                   style={{ backgroundColor: BLUESKY }}
                   className="inline-flex items-center gap-2 px-4 py-2 text-[13px]
                              font-medium text-white transition-opacity hover:opacity-85">
                  <Butterfly />
                  Open on Bluesky
                  <span aria-hidden className="text-[14px] leading-none opacity-70">&#8599;</span>
                </a>
              )}
            </div>
          </header>

          {busy && (
            <section className="border-b border-rule-firm py-7">
              <Pending withCascade done={arriving} rejoined={rejoined} label={jobLabel}
                       elapsed={elapsed} />
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
