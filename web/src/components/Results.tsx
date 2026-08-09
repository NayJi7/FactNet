import { useEffect, useState } from "react";

interface Table {
  key: string;
  title: string;
  unit: string;
  columns: string[];
  rows: (string | number | null)[][];
  best: string | null;
  reading: string;
}
interface Headline { value: string; label: string; note: string }

export default function Results() {
  const [data, setData] = useState<{ headlines: Headline[]; tables: Table[] } | null>(null);

  useEffect(() => {
    fetch("/api/results").then((r) => r.json()).then(setData).catch(() => {});
  }, []);
  if (!data) return null;

  return (
    <div className="space-y-8">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {data.headlines.map((h) => (
          <div key={h.label} className="border-t border-rule-firm pt-4">
            <div className="font-mono text-[26px] font-medium tnum text-ink">{h.value}</div>
            <div className="mt-1 text-[13px] font-medium text-ink">{h.label}</div>
            <div className="mt-1 text-[12px] leading-snug text-ink-soft">{h.note}</div>
          </div>
        ))}
      </div>

      {data.tables.map((table) => {
        const bestColumn = table.best ? table.columns.indexOf(table.best) : -1;
        return (
          <section key={table.key} className="border-t border-rule-firm pt-6">
            <h3 className="text-[15px] font-semibold text-ink">{table.title}</h3>
            <p className="mb-3 text-[12px] text-ink-faint">{table.unit}</p>
            <div className="overflow-x-auto">
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="border-b border-rule text-left text-ink-soft">
                    {table.columns.map((c, i) => (
                      <th key={c}
                          className={`py-2 pr-5 font-medium ${i === bestColumn ? "text-signal" : ""}`}>
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {table.rows.map((row, r) => (
                    <tr key={r} className="border-b border-rule last:border-0">
                      {row.map((cell, c) => (
                        <td key={c}
                            className={`py-2 pr-5 ${c >= 1 ? "font-mono tnum" : ""} ${
                              c === bestColumn ? "font-semibold text-signal" : ""
                            }`}>
                          {cell === null ? "—" : cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 border-t border-rule pt-3 text-[13px] leading-relaxed text-ink-soft">
              {table.reading}
            </p>
          </section>
        );
      })}
    </div>
  );
}
