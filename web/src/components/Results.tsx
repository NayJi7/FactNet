import { useEffect, useState } from "react";
import Spine from "./Spine";
import { MODULE_NAMES } from "./ModuleTag";
import Failed from "./Failed";
import { useAsync } from "../lib/useAsync";
import type { Module } from "../lib/types";

/**
 * Results tab. Summary axis first, then the tables grouped by module, each cell
 * with a small bar, and a side index that follows the scroll (it was 14 tables
 * in a row before, unreadable).
 */
interface Table {
  key: string;
  short?: string;
  module: Module;
  title: string;
  unit: string;
  columns: string[];
  rows: (string | number | null)[][];
  best: string | null;
  reading: string;
}

const ORDER: Module[] = ["propagation", "content", "both"];

/** first number in a cell, if any */
function magnitude(cell: unknown): number | null {
  if (typeof cell === "number") return cell;
  if (typeof cell !== "string") return null;
  const found = /^[+-]?\d*\.?\d+/.exec(cell.trim());
  return found ? Number(found[0]) : null;
}

/** bar scale per column: macro-F1 on a fixed 0-1 axis, the rest relative to the column */
function scales(table: Table) {
  const shared = /macro-f1/i.test(table.unit);
  return table.columns.map((_, column) => {
    const values = table.rows
      .map((row) => magnitude(row[column]))
      .filter((v): v is number => v !== null);
    if (values.length < 2 || column === 0) return null;
    if (shared && values.every((v) => v >= 0 && v <= 1)) return { lo: 0, hi: 1 };
    const lo = Math.min(...values), hi = Math.max(...values);
    return hi > lo ? { lo, hi } : null;
  });
}

function TableView({ table }: { table: Table }) {
  const bestColumn = table.best ? table.columns.indexOf(table.best) : -1;
  const columnScale = scales(table);

  return (
    <section id={`t-${table.key}`} className="scroll-mt-6">
      <h3 className="text-[15px] font-semibold text-ink">{table.title}</h3>
      <p className="mb-3 text-[12px] text-ink-faint">{table.unit}</p>

      <div className="overflow-x-auto">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-rule text-left text-ink-soft">
              {table.columns.map((c, i) => (
                <th key={c} className={`whitespace-nowrap py-2 pr-5 font-medium ${
                  i === bestColumn ? "text-signal" : ""}`}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, r) => (
              <tr key={r} className="border-b border-rule last:border-0">
                {row.map((cell, c) => {
                  const scale = columnScale[c];
                  const value = scale ? magnitude(cell) : null;
                  const numeric = typeof cell === "number"
                    || /^[+-]?\d*\.?\d+|^\[/.test(String(cell));
                  return (
                    <td key={c} className="relative whitespace-nowrap py-2 pr-5">
                      {/* thin line, not a filled block (differences are small, don't exaggerate) */}
                      {value !== null && scale && (
                        <span aria-hidden
                              className="absolute bottom-[5px] left-0 h-[2px] bg-rule-firm"
                              style={{ width: `${Math.max(0, Math.min(1,
                                (value - scale.lo) / (scale.hi - scale.lo))) * 82}%` }} />
                      )}
                      <span className={`relative ${numeric ? "font-mono tnum" : ""} ${
                        c === bestColumn ? "font-semibold text-signal" : ""}`}>
                        {cell === null || cell === "" ? "·" : cell}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-3 max-w-[74ch] border-t border-rule pt-3 text-[13px] leading-relaxed text-ink-soft">
        {table.reading}
      </p>
    </section>
  );
}

export default function Results() {
  const [here, setHere] = useState("");
  const loaded = useAsync<{ spine: { marks: any[]; unit: string }; tables: Table[] }>(
    () => fetch("/api/results").then((r) => {
      if (!r.ok) throw new Error(`the engine answered ${r.status}`);
      return r.json();
    }), []);
  const data = loaded.data;

  // highlight the current section in the index
  useEffect(() => {
    if (!data) return;
    const spotter = new IntersectionObserver(
      (entries) => {
        const first = entries.filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (first) setHere(first.target.id.replace(/^t-/, ""));
      },
      { rootMargin: "-8% 0px -70% 0px" },
    );
    data.tables.forEach((t) => {
      const node = document.getElementById(`t-${t.key}`);
      if (node) spotter.observe(node);
    });
    return () => spotter.disconnect();
  }, [data]);

  if (loaded.error && !data) {
    return <Failed error={loaded.error} onRetry={loaded.reload} what="the results" />;
  }
  if (!data) return null;

  return (
    <div className="xl:grid xl:grid-cols-[172px_1fr] xl:gap-x-12">
      <nav className="hidden xl:block">
        <div className="sticky top-28 pb-4">
          <p className="eyebrow mb-1.5">On this page</p>
          {ORDER.map((module) => {
            const tables = data.tables.filter((t) => t.module === module);
            if (!tables.length) return null;
            return (
              <div key={module} className="mb-5">
                <p className={`mb-1 pl-2.5 text-[11px] font-medium uppercase tracking-[0.09em] ${
                  module === "content" ? "text-content"
                  : module === "propagation" ? "text-struct" : "text-ink-faint"}`}>
                  {MODULE_NAMES[module]}
                </p>
                <ul>
                  {tables.map((t) => (
                    <li key={t.key}>
                      <a href={`#t-${t.key}`}
                         className={`block border-l py-1 pl-2.5 text-[12px] leading-snug
                                     transition-colors ${
                           here === t.key
                             ? "border-ink font-medium text-ink"
                             : "border-rule text-ink-faint hover:text-ink-soft"}`}>
                        {t.short ?? t.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
          <p className="mt-4 border-t border-rule pt-3 text-[11.5px] leading-relaxed text-ink-faint">
            Every table is built from the file its experiment wrote, so a rerun
            changes this page and nothing has to be copied across.
          </p>
        </div>
      </nav>

      <div className="min-w-0 space-y-12">
        <Spine marks={data.spine?.marks ?? []} unit={data.spine?.unit ?? "macro-F1"} />

        {ORDER.map((module) => {
          const tables = data.tables.filter((t) => t.module === module);
          if (!tables.length) return null;
          return (
            <div key={module} className="space-y-9">
              <h2 className={`border-t-2 pt-3 text-[13px] font-semibold uppercase tracking-[0.11em] ${
                module === "content" ? "border-content text-content"
                : module === "propagation" ? "border-struct text-struct"
                : "border-rule-firm text-ink-faint"}`}>
                {MODULE_NAMES[module]}
              </h2>
              {tables.map((t) => <TableView key={t.key} table={t} />)}
            </div>
          );
        })}
      </div>
    </div>
  );
}
