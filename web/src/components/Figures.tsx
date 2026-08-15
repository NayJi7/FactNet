import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import ForceGraph2D from "react-force-graph-2d";
import { Account, personGlyph, shortHandle } from "../lib/handle";
import type { Figure } from "../lib/types";

const INK = "#2b3550";
const INK_SOFT = "#5d6480";
const FAINT = "#8e93a8";
const SIGNAL = "#d08a2c";
const STRUCT = "#3f7f97";
const RULE = "#dde3e2";

const axis = { fontSize: 10.5, fill: FAINT, fontFamily: "var(--font-mono)" };
const tip = {
  contentStyle: {
    fontSize: 11.5, borderRadius: 3, border: `1px solid ${RULE}`,
    boxShadow: "none", fontFamily: "var(--font-mono)", padding: "6px 8px",
  },
};

/** Weighted tokens, set inline. A sentence is read as a sentence. */
function Tokens({ data }: { data: Figure["data"] }) {
  const tokens: [string, number][] = data.tokens ?? [];
  const signed = Boolean(data.signed);
  return (
    // no ch cap: set two abreast, the column is already the measure
    <p className="font-mono text-[13px] leading-[2.1]">
      {tokens.map(([token, weight], i) => {
        const m = Math.min(1, Math.abs(weight));
        const colour = signed ? (weight >= 0 ? STRUCT : SIGNAL) : INK_SOFT;
        return (
          <span key={i} title={`${signed ? "effect" : "attention"} ${weight.toFixed(3)}`}
                className="px-[3px] py-[1px]"
                style={{
                  boxShadow: m > 0.06 ? `inset 0 -${Math.round(m * 12 + 2)}px 0 0 ${colour}${
                    Math.round(m * 90 + 25).toString(16).padStart(2, "0")}` : undefined,
                }}>
            {token.trim() === "" ? "·" : token}
          </span>
        );
      })}
    </p>
  );
}

function Bars({ data }: { data: Figure["data"] }) {
  const keys: string[] = data.keys ?? ["this"];
  const palette: Record<string, string> = {
    this: INK, misleading: SIGNAL, reliable: STRUCT,
  };
  return (
    <ResponsiveContainer width="100%" height={Math.max(150, (data.rows?.length ?? 3) * 30)}>
      <BarChart data={data.rows} layout="vertical" margin={{ left: 0, right: 30, top: 4 }}>
        <CartesianGrid horizontal={false} stroke={RULE} />
        <XAxis type="number" tick={axis} axisLine={{ stroke: RULE }} tickLine={false} />
        <YAxis dataKey="metric" type="category" width={122} tick={{ fontSize: 12, fill: INK }}
               axisLine={false} tickLine={false} />
        <Tooltip {...tip} cursor={{ fill: "rgba(0,0,0,0.02)" }} />
        {keys.length > 1 && <Legend wrapperStyle={{ fontSize: 11.5, paddingTop: 6 }} />}
        {keys.map((k) => (
          <Bar key={k} dataKey={k} fill={palette[k] ?? FAINT} radius={[0, 1, 1, 0]} barSize={keys.length > 1 ? 7 : 11}>
            {keys.length === 1 && data.rows.map((r: any, i: number) => (
              <Cell key={i} fill={r[k] < 0 ? SIGNAL : INK} />
            ))}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function Series({ data }: { data: Figure["data"] }) {
  const points = data.series?.[0]?.points ?? [];
  return (
    <ResponsiveContainer width="100%" height={190}>
      <LineChart data={points} margin={{ left: -14, right: 14, top: 10, bottom: 2 }}>
        <CartesianGrid stroke={RULE} vertical={false} />
        <XAxis dataKey={data.x} unit="%" tick={axis} axisLine={{ stroke: RULE }} tickLine={false} />
        <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tick={axis} axisLine={false} tickLine={false} />
        <Tooltip {...tip} formatter={(v: number) => v.toFixed(3)} />
        <Line type="monotone" dataKey={data.y} stroke={INK} strokeWidth={1.75}
              dot={{ r: 3, fill: INK, strokeWidth: 0 }}
              activeDot={{ r: 4.5 }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** A small control, in the same language as the tabs: text, underlined when on. */
function Control({ on, onClick, children, title }: {
  on?: boolean; onClick: () => void; children: React.ReactNode; title?: string;
}) {
  return (
    <button onClick={onClick} title={title} aria-pressed={on}
            className={`border-b-2 px-1.5 pb-1 text-[12.5px] transition-colors ${
              on ? "border-ink font-medium text-ink"
                 : "border-transparent text-ink-faint hover:text-ink-soft"}`}>
      {children}
    </button>
  );
}

function Cascade({ data }: { data: Figure["data"] }) {
  const box = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const [width, setWidth] = useState(0);
  const HEIGHT = 520;

  const allLevels: { depth: number; accounts: number }[] = data.levels ?? [];
  const maxHop = allLevels.length ? allLevels[allLevels.length - 1].depth : 0;

  const [limit, setLimit] = useState(maxHop);
  const [radial, setRadial] = useState(true);
  const [playing, setPlaying] = useState(false);

  useEffect(() => { setLimit(maxHop); setPlaying(false); }, [maxHop]);

  useEffect(() => {
    if (!box.current) return;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(box.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!playing) return;
    if (limit >= maxHop) { setPlaying(false); return; }
    const timer = setTimeout(() => setLimit((n) => n + 1), 900);
    return () => clearTimeout(timer);
  }, [playing, limit, maxHop]);

  const graph = useMemo(() => {
    const all: any[] = data.nodes ?? [];
    let radius = 0;
    const links: any[] = (data.links ?? []).filter(
      (l: any) => l.source !== l.target,
    );
    const nodes = all
      .filter((n) => n.depth >= 0 && n.depth <= limit)
      .map((n) => ({ ...n }));

    const kept = new Set(nodes.map((n) => n.id));
    const edges = links
      .filter((l) => kept.has(l.source) && kept.has(l.target))
      .map((l) => ({ ...l }));

    if (radial) {
      // The rings are placed here rather than left to the layout engine: these
      // graphs contain cycles and self-replies, so the library's radial mode
      // gives up on them silently. Pinning also removes the shake, since
      // nothing is simulated when the filter changes.
      const byDepth = new Map<number, any[]>();
      for (const n of nodes) {
        if (!byDepth.has(n.depth)) byDepth.set(n.depth, []);
        byDepth.get(n.depth)!.push(n);
      }
      const parentOf = new Map<number, number>();
      const at = new Map<number, any>(nodes.map((n) => [n.id, n]));
      for (const l of edges) {
        const from = at.get(l.source);
        const to = at.get(l.target);
        if (from && to && from.depth < to.depth && !parentOf.has(to.id))
          parentOf.set(to.id, from.id);
      }
      const angle = new Map<number, number>();
      const step = Math.max(58, Math.min(112, width / 8));
      radius = Math.max(...[...byDepth.keys()]) * step;
      for (const depth of [...byDepth.keys()].sort((a, b) => a - b)) {
        const level = byDepth.get(depth)!;
        level.sort((a, b) => (angle.get(parentOf.get(a.id) ?? -1) ?? 0)
                           - (angle.get(parentOf.get(b.id) ?? -1) ?? 0));
        level.forEach((n, i) => {
          const theta = depth === 0 ? 0 : (2 * Math.PI * i) / level.length;
          angle.set(n.id, theta);
          n.fx = Math.cos(theta) * depth * step;
          n.fy = Math.sin(theta) * depth * step;
        });
      }
    }
    return { nodes, links: edges, radius };
  }, [data, limit, radial, width]);

  // the handful of accounts worth printing a name for: the source, and the
  // largest audiences on screen. Everything else stays a dot, or the picture
  // becomes a wall of text.
  // Nothing is written on the picture. A ring holds hundreds of nodes a few
  // pixels apart, so any label lands on its neighbour; the handle is on hover
  // and the largest audiences are listed under the figure, where they read.
  const biggest = useMemo(
    () => [...graph.nodes]
            .filter((n: any) => !n.root && n.followers > 0)
            .sort((a: any, b: any) => b.followers - a.followers)
            .slice(0, 3),
    [graph],
  );

  // counts of what is actually on screen, not of the whole cascade
  const shownLevels = useMemo(() => {
    const counts = new Map<number, number>();
    for (const n of graph.nodes) counts.set(n.depth, (counts.get(n.depth) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => a[0] - b[0]);
  }, [graph]);
  const mix = useMemo(() => ({
    reposts: graph.links.filter((l: any) => l.kind !== "reply").length,
    replies: graph.links.filter((l: any) => l.kind === "reply").length,
  }), [graph]);

  // With pinned rings the extent is known exactly, so the zoom is computed
  // rather than requested: zoomToFit runs before the coordinates are applied
  // and, finding every node at the origin, magnifies the picture to nothing.
  const refit = () => {
    const g = graphRef.current;
    if (!g) return;
    if (radial) {
      const span = graph.radius * 2 + 90;
      g.centerAt(0, 0, 0);
      g.zoom(span > 0 ? Math.min(1.6, Math.min(width, HEIGHT) / span) : 1, 0);
    } else {
      g.zoomToFit(400, 34);
    }
  };
  useEffect(() => {
    if (!width) return;
    const timer = setTimeout(refit, 60);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, limit, radial]);

  const shade = (d: number) =>
    d <= 0 ? SIGNAL : d === 1 ? STRUCT : d === 2 ? "#79a6b6" : "#b3c1c6";

  return (
    <div>
      <div className="mb-2.5 flex flex-wrap items-end justify-between gap-x-6 gap-y-2 border-b border-rule pb-1.5">
        <div className="flex flex-wrap items-end gap-x-1 gap-y-1">
          <span className="eyebrow mr-2 pb-1">show</span>
          {allLevels.map(({ depth }) => (
            <Control key={depth} on={limit === depth}
                     onClick={() => { setPlaying(false); setLimit(depth); }}
                     title={depth === 0 ? "the source alone"
                                        : `up to ${depth} hop${depth > 1 ? "s" : ""} away`}>
              {depth === 0 ? "source" : `+${depth}`}
            </Control>
          ))}
        </div>
        <div className="flex flex-wrap items-end gap-x-1 gap-y-1">
          <Control on={radial} onClick={() => setRadial((v) => !v)}
                   title="rings by hop, or a free layout">
            rings
          </Control>
          <Control on={playing} onClick={() => { setLimit(0); setPlaying(true); }}
                   title="replay the spread one hop at a time">
            replay
          </Control>
          <Control onClick={refit}>fit</Control>
        </div>
      </div>

      {/* The layout is a disc, so a wide short box wastes its corners whatever
          the zoom. Narrowing the canvas and standing the legend beside it uses
          the row instead, and buys the disc more height at the same time. */}
      <div className="xl:grid xl:grid-cols-[minmax(0,1fr)_248px] xl:gap-7">
      <div ref={box} className="overflow-hidden border border-rule bg-panel"
           style={{ height: HEIGHT }}>
        {width > 0 && (
          <ForceGraph2D
            width={width}
            height={HEIGHT}
            graphData={graph}
            backgroundColor="rgba(0,0,0,0)"
            nodeRelSize={3}
            nodeVal={(n: any) => 1 + Math.sqrt(n.reposted_by ?? 0) * 2.2}
            nodeColor={(n: any) => shade(n.depth)}
            nodeLabel={(n: any) =>
              `${personGlyph(11)} <b>${shortHandle(n.label)}</b> · ` +
              `${n.depth === 0 ? "the source" : `${n.depth} hop${n.depth > 1 ? "s" : ""} away`}` +
              (n.reposted_by ? ` · reposted by ${n.reposted_by}` : "") +
              (n.replied_to_by ? ` · replied to by ${n.replied_to_by}` : "")}
            linkColor={(l: any) => (l.kind === "reply" ? "#e6ebea" : "#9fb2b0")}
            linkWidth={(l: any) => (l.kind === "reply" ? 0.5 : 1.2)}
            d3VelocityDecay={0.3}
            cooldownTicks={radial ? 0 : 110}
            warmupTicks={radial ? 0 : 20}
            enableNodeDrag={false}
            ref={graphRef}
            onEngineStop={refit}
          />
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-start justify-between gap-x-8 gap-y-3
                      xl:mt-0 xl:flex-col xl:flex-nowrap xl:justify-start xl:gap-y-6">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 xl:flex-col xl:items-start xl:gap-y-2">
          {shownLevels.map(([depth, accounts]) => (
            <span key={depth} className="flex items-center gap-1.5 text-[12px] xl:w-full">
              <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ background: shade(depth) }} aria-hidden />
              <span className="text-ink-soft">
                {depth === 0 ? "the source" : `${depth} hop${depth > 1 ? "s" : ""}`}
              </span>
              <span className="tnum font-mono text-ink-faint xl:ml-auto">{accounts}</span>
            </span>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-[12px] text-ink-faint
                        xl:flex-col xl:items-start xl:gap-y-2">
          {biggest.length > 0 && (
            <span className="flex flex-wrap items-center gap-x-3 xl:flex-col xl:items-start xl:gap-y-1">
              <span>largest audiences here:</span>
              {biggest.map((n: any) => (
                <span key={n.id} className="whitespace-nowrap">
                  <Account handle={n.label} className="text-ink-soft" />{" "}
                  <span className="tnum font-mono">
                    {n.followers.toLocaleString("en")}
                  </span>
                </span>
              ))}
            </span>
          )}
          <span>
            <span className="tnum font-mono">{mix.reposts}</span> reposts,{" "}
            <span className="tnum font-mono">{mix.replies}</span> replies shown
          </span>
        </div>
      </div>
      </div>
    </div>
  );
}

/** Models on one axis. The spread is the finding, so the axis is shared. */
function Compare({ data }: { data: Figure["data"] }) {
  const rows = [...(data.rows ?? [])].sort((a, b) => b.p_reliable - a.p_reliable);
  return (
    <div>
      <div className="relative">
        <div className="absolute bottom-0 left-[168px] right-[64px] top-0 border-l border-dashed border-rule-firm"
             style={{ left: "calc(168px + (100% - 232px) * 0.5)" }} aria-hidden />
        <div className="space-y-2">
          {rows.map((row: any) => (
            <div key={row.key} className="flex items-center gap-3">
              <span className="w-[160px] shrink-0 truncate text-[12.5px]"
                    title={row.model}>
                {row.primary && <span className="mr-1 text-signal">▸</span>}
                <span className={row.primary ? "font-semibold" : "text-ink-soft"}>{row.model}</span>
              </span>
              <div className="relative h-[18px] flex-1 bg-sunk">
                <div className="absolute inset-y-0 transition-[width] duration-500"
                     style={{
                       width: `${Math.abs(row.p_reliable - 0.5) * 100}%`,
                       left: row.p_reliable >= 0.5 ? "50%" : undefined,
                       right: row.p_reliable < 0.5 ? "50%" : undefined,
                       background: row.p_reliable >= 0.5 ? INK : SIGNAL,
                     }} />
              </div>
              <span className="tnum w-[56px] shrink-0 text-right font-mono text-[11.5px] text-ink-soft">
                {row.p_reliable.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-2.5 flex items-baseline justify-between border-t border-rule pt-2">
        <span className="text-[11.5px] text-ink-faint">
          bars run outward from the 0.5 boundary
        </span>
        {typeof data.spread === "number" && (
          <span className="text-[12px]">
            spread <span className="tnum font-mono font-semibold">{data.spread.toFixed(3)}</span>
          </span>
        )}
      </div>
      <Disagreement rows={rows} why={data.why ?? []} />
    </div>
  );
}

/**
 * Why the models differ on this post, which is the question a viewer asks the
 * moment they see five answers and one input.
 *
 * The answer is never that one model is better: on the held-out benchmark none
 * of these separates from any other. It is that they weigh different objects,
 * so what each one actually weighed is shown, measured on this input rather
 * than described in general. A token in ink pushed that model towards
 * reliable, one in amber pushed it the other way.
 */
function Disagreement({ rows, why }: { rows: any[]; why: string[] }) {
  const withTokens = rows.filter((r) => r.tokens?.length);
  if (!withTokens.length) return null;

  return (
    <div className="mt-6 border-t border-rule-firm pt-4">
      <p className="eyebrow">Why they disagree</p>

      <div className="mt-3 space-y-3">
        {withTokens.map((row: any) => (
          <div key={row.key} className="grid gap-x-4 gap-y-1 sm:grid-cols-[160px_1fr]">
            <div className="min-w-0">
              <p className="truncate text-[12.5px]" title={row.model}>
                {row.primary && <span className="mr-1 text-signal">▸</span>}
                <span className={row.primary ? "font-semibold" : "text-ink-soft"}>
                  {row.model}
                </span>
              </p>
              <p className={`text-[11px] ${row.fragmented ? "text-signal" : "text-ink-faint"}`}>
                reads {row.basis}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5">
              {row.tokens.map(([token, value]: [string, number], i: number) => (
                <span key={`${token}-${i}`}
                      className="inline-flex items-baseline gap-1 font-mono text-[11.5px]">
                  <span className="border-b-2 px-0.5"
                        style={{
                          borderColor: value >= 0 ? INK : SIGNAL,
                          // the weight of the mark is the size of the effect, so
                          // a glance ranks them without reading a single number
                          opacity: 0.45 + 0.55 * Math.min(1, Math.abs(value)),
                        }}>
                    {token}
                  </span>
                  <span className="tnum text-[10.5px] text-ink-faint">
                    {value >= 0 ? "+" : ""}{value.toFixed(2)}
                  </span>
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {why.length > 0 && (
        <div className="mt-4 space-y-2 border-t border-rule pt-3">
          {why.map((line, i) => (
            <p key={i} className="max-w-[78ch] text-[12.5px] leading-relaxed text-ink-soft">
              {line}
            </p>
          ))}
        </div>
      )}
      <p className="mt-2 text-[11px] text-ink-faint">
        A mark underlined in ink pushed that model towards reliable, one in amber
        towards misleading. Darker means it mattered more.
      </p>
    </div>
  );
}

function Table({ data }: { data: Figure["data"] }) {
  const columns: string[] = data.columns ?? [];
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12.5px]">
        <thead>
          <tr className="border-b border-rule-firm text-left">
            {columns.map((c) => (
              <th key={c} className="eyebrow pb-1.5 pr-5 font-medium">{c.replace(/_/g, " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(data.rows ?? []).map((row: any, i: number) => (
            <tr key={i} className="border-b border-rule last:border-0">
              {columns.map((c, j) => (
                <td key={c} className={`py-1.5 pr-5 ${j > 0 ? "tnum font-mono text-ink-soft" : ""}`}>
                  {typeof row[c] === "number" ? row[c].toFixed(3)
                    : c === "account" ? <Account handle={String(row[c])} />
                    : row[c]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function FigureView({ figure }: { figure: Figure }) {
  const body = {
    tokens: <Tokens data={figure.data} />,
    bars: <Bars data={figure.data} />,
    line: <Series data={figure.data} />,
    graph: <Cascade data={figure.data} />,
    compare: <Compare data={figure.data} />,
    table: <Table data={figure.data} />,
  }[figure.kind];

  return (
    <figure className="mt-5">
      <figcaption className="eyebrow mb-2.5">{figure.title}</figcaption>
      {body}
      {figure.caption && (
        <p className="mt-2.5 max-w-[72ch] text-[12.5px] leading-relaxed text-ink-faint">
          {figure.caption}
        </p>
      )}
    </figure>
  );
}
