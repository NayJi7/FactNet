import { useEffect, useState } from "react";
import Failed from "./Failed";
import { useAsync } from "../lib/useAsync";

export default function DataView() {
  const [here, setHere] = useState("benchmarks");
  const loaded = useAsync<any>(() => fetch("/api/data").then((r) => {
    if (!r.ok) throw new Error(`the engine answered ${r.status}`);
    return r.json();
  }), []);
  const data = loaded.data;
  // the index follows the reader rather than the reader hunting the index
  useEffect(() => {
    if (!data) return;
    const spotter = new IntersectionObserver(
      (entries) => {
        const first = entries.filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (first) setHere(first.target.id.replace(/^d-/, ""));
      },
      { rootMargin: "-14% 0px -70% 0px" },
    );
    document.querySelectorAll('[id^="d-"]').forEach((n) => spotter.observe(n));
    return () => spotter.disconnect();
  }, [data]);

  if (loaded.error && !data) {
    return <Failed error={loaded.error} onRetry={loaded.reload} what="the data" />;
  }
  if (!data) return null;
  const c = data.collected;

  const SECTIONS: [string, string][] = [
    ["benchmarks", "Public benchmarks"],
    ["collected", "What we collected ourselves"],
    ["features", "How each account is described"],
    ["labels", "How the labels were obtained"],
  ];

  return (
    <div className="xl:grid xl:grid-cols-[172px_1fr] xl:gap-x-12">
      <nav className="hidden xl:block">
        <div className="sticky top-28 pb-4">
          <p className="eyebrow mb-1.5">On this page</p>
          <ul>
            {SECTIONS.map(([id, label]) => (
              <li key={id}>
                <a href={`#d-${id}`}
                   className={`block border-l py-1 pl-2.5 text-[12px] leading-snug transition-colors ${
                     here === id
                       ? "border-ink font-medium text-ink"
                       : "border-rule text-ink-faint hover:text-ink-soft"}`}>
                  {label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </nav>

    <div className="min-w-0 space-y-8">
      <section id="d-benchmarks" className="scroll-mt-24 border-t border-rule-firm pt-6">
        <h3 className="mb-3 text-[15px] font-semibold text-ink">
          Public benchmarks
        </h3>
        <div className="space-y-4">
          {data.benchmarks.map((b: any) => (
            <div key={b.name}
                 className="grid gap-x-6 gap-y-1 border-t border-rule pt-3 sm:grid-cols-[210px_1fr]">
              <div>
                <p className="font-semibold leading-snug text-ink">{b.name}</p>
                <p className="font-mono tnum text-[12px] text-ink-soft">{b.size}</p>
                <p className="text-[11px] text-ink-faint">{b.role}</p>
              </div>
              <p className="max-w-[70ch] text-[13px] leading-relaxed text-ink-soft">
                <span className="text-ink-faint">Labels: </span>{b.labels}. {b.note}
              </p>
            </div>
          ))}
        </div>
      </section>

      {c?.available && (
        <>
          <section id="d-collected" className="scroll-mt-24 border-t border-rule-firm pt-6">
            <h3 className="text-[15px] font-semibold text-ink">
              What we collected ourselves
            </h3>
            <p className="mt-2 max-w-[74ch] text-[13px] leading-relaxed text-ink-soft">
              {c.cascades} cascades over {c.accounts.toLocaleString()} accounts,
              linked to {c.domains} outlets whose factual record is already rated.
              Balanced by cascade count, and not remotely balanced by size.
            </p>

            {/* The asymmetry is the finding of this sample, so it is drawn. Two
                classes on one axis of accounts per cascade, which is the axis
                the size confound is measured on. */}
            <div className="mt-5 space-y-3">
              {c.classes.map((row: any) => {
                const widest = Math.max(...c.classes.map((x: any) => x.mean_accounts));
                return (
                  <div key={row.name} className="grid items-center gap-x-5 sm:grid-cols-[110px_1fr_74px]">
                    <span className="text-[13px]">{row.name}</span>
                    <div className="relative h-[20px] bg-sunk">
                      <div className="absolute inset-y-0 left-0 transition-[width] duration-700"
                           style={{ width: `${(row.mean_accounts / widest) * 100}%`,
                                    background: row.name === "reliable"
                                      ? "var(--color-struct)" : "var(--color-signal)",
                                    transitionTimingFunction: "var(--ease-out-quint)" }} />
                    </div>
                    <span className="tnum text-right font-mono text-[13px]">
                      {row.mean_accounts}
                    </span>
                  </div>
                );
              })}
              <p className="flex flex-wrap justify-between gap-x-6 border-t border-rule pt-2 text-[11.5px] text-ink-faint">
                <span>mean accounts per cascade</span>
                <span>
                  medians: {c.classes.map((x: any) => `${x.name} ${x.median_accounts}`).join(", ")}
                </span>
              </p>
            </div>

            <p className="mt-4 max-w-[74ch] text-[13px] leading-relaxed text-ink-soft">
              A credible outlet here carries a cascade six times the size of an
              unreliable one, and the medians are further apart still. That
              asymmetry is a finding rather than a defect, and it runs opposite
              to GossipCop. It is also the reason a one-parameter rule on size
              beats the detector until the sizes are matched away.
            </p>
          </section>

          <section id="d-features" className="scroll-mt-24 border-t border-rule-firm pt-6">
            <h3 className="mb-1 text-[15px] font-semibold text-ink">
              How each account is described
            </h3>
            <p className="mb-3 text-[12px] text-ink-soft">
              Share of accounts carrying a value in each slot.
            </p>
            <div className="space-y-1.5">
              {c.feature_coverage.map((f: any) => (
                <div key={f.feature} className="flex items-center gap-3">
                  <span className="w-40 shrink-0 text-[13px]">{f.feature}</span>
                  <div className="h-4 flex-1 overflow-hidden bg-sunk">
                    <div
                      className={`h-full ${f.share > 0 ? "bg-struct" : ""}`}
                      style={{ width: `${f.share * 100}%` }}
                    />
                  </div>
                  <span className="w-14 shrink-0 text-right font-mono tnum text-[12px] text-ink-soft">
                    {(f.share * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
            <p className="mt-3 max-w-[74ch] text-[13px] leading-relaxed text-ink-soft">
              Three slots stay empty because Bluesky exposes no verification flag,
              no geolocation and no list membership. The layout keeps them so a
              model trained on the Twitter benchmark reads the same ten columns.
            </p>
          </section>

          <section id="d-labels" className="scroll-mt-24 border-t border-rule-firm pt-6">
            <h3 className="mb-3 text-[15px] font-semibold text-ink">
              How the labels were obtained
            </h3>
            <ol className="space-y-2.5">
              {data.protocol.map((rule: string, i: number) => (
                <li key={i} className="flex gap-3 text-[13px] leading-relaxed text-ink-soft">
                  <span className="font-mono tnum text-ink-faint">{i + 1}</span>
                  <span>{rule}</span>
                </li>
              ))}
            </ol>
            <div className="mt-4 flex flex-wrap gap-1.5 border-t border-rule pt-3">
              {c.top_domains.map(([domain, count]: [string, number]) => (
                <span key={domain}
                      className="rounded-full bg-sunk px-2.5 py-1 font-mono tnum text-[12px] text-ink">
                  {domain} <span className="text-ink-faint">{count}</span>
                </span>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
    </div>
  );
}
