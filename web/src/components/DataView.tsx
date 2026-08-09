import { useEffect, useState } from "react";

export default function DataView() {
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    fetch("/api/data").then((r) => r.json()).then(setData).catch(() => {});
  }, []);
  if (!data) return null;
  const c = data.collected;

  return (
    <div className="space-y-8">
      <section className="border-t border-rule-firm pt-6">
        <h3 className="mb-3 text-[15px] font-semibold text-ink">
          Public benchmarks
        </h3>
        <div className="space-y-3">
          {data.benchmarks.map((b: any) => (
            <div key={b.name} className="border-l-2 border-struct pl-3">
              <div className="flex flex-wrap items-baseline gap-x-3">
                <span className="font-semibold text-ink">{b.name}</span>
                <span className="font-mono tnum text-[12px] text-ink-soft">{b.size}</span>
                <span className="text-[12px] text-ink-faint">{b.role}</span>
              </div>
              <p className="mt-0.5 text-[13px] leading-relaxed text-ink-soft">
                Labels: {b.labels}. {b.note}
              </p>
            </div>
          ))}
        </div>
      </section>

      {c?.available && (
        <>
          <section className="border-t border-rule-firm pt-6">
            <h3 className="text-[15px] font-semibold text-ink">
              What we collected ourselves
            </h3>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              {[
                ["cascades", c.cascades],
                ["accounts", c.accounts.toLocaleString()],
                ["rated outlets", c.domains],
              ].map(([label, value]) => (
                <div key={label as string} className="bg-sunk p-3">
                  <div className="font-mono text-[22px] font-medium tnum text-ink">{value}</div>
                  <div className="text-[12px] text-ink-soft">{label}</div>
                </div>
              ))}
            </div>

            <table className="mt-4 w-full text-[13px]">
              <thead>
                <tr className="border-b border-rule text-left text-ink-soft">
                  <th className="py-1.5 font-medium">Class</th>
                  <th className="py-1.5 font-medium">Cascades</th>
                  <th className="py-1.5 font-medium">Mean accounts</th>
                  <th className="py-1.5 font-medium">Median</th>
                </tr>
              </thead>
              <tbody>
                {c.classes.map((row: any) => (
                  <tr key={row.name} className="border-b border-rule last:border-0">
                    <td className="py-1.5">{row.name}</td>
                    <td className="py-1.5 font-mono tnum">{row.cascades}</td>
                    <td className="py-1.5 font-mono tnum">{row.mean_accounts}</td>
                    <td className="py-1.5 font-mono tnum">{row.median_accounts}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-[13px] leading-relaxed text-ink-soft">
              The classes are balanced by cascade count and wildly unbalanced by
              size. That asymmetry is a finding, not a defect: on this platform the
              credible outlets carry the far larger cascades, the reverse of what
              GossipCop shows.
            </p>
          </section>

          <section className="border-t border-rule-firm pt-6">
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
            <p className="mt-3 text-[13px] leading-relaxed text-ink-soft">
              Three slots stay empty because Bluesky exposes no verification flag,
              no geolocation and no list membership. The layout keeps them so a
              model trained on the Twitter benchmark reads the same ten columns.
            </p>
          </section>

          <section className="border-t border-rule-firm pt-6">
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
  );
}
