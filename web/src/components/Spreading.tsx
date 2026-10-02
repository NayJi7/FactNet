/**
 * Loading animation: a small cascade spreading from the source, then clearing.
 * (replaced the three dots)
 *
 * - the source stays, everything grows from it each cycle
 * - a node never shows up before its parent
 * - branches start one after another with different speeds
 * - one shared cycle so it all clears at once (per-node timing just flickered)
 *
 * Pure CSS animation, no JS per frame.
 */
const RINGS = [
  { count: 1, radius: 0, dot: 5.5 },
  { count: 5, radius: 30, dot: 3.6 },
  { count: 11, radius: 58, dot: 2.6 },
  { count: 15, radius: 84, dot: 1.9 },
];
const CYCLE = 2.4;                // s
const BRANCH_GAP = [0.10, 0.24];
const HOP = [0.07, 0.19];         // delay per hop

/** seeded noise so the layout is the same on every render */
function noise(seed: number): number {
  let t = (seed * 1013 + 0x6d2b79f5) >>> 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

interface Dot {
  x: number; y: number; r: number;
  at: number; ring: number; parent: Dot | null;
  branch: number;
}

function layout(): { dots: Dot[]; edges: [Dot, Dot][] } {
  const dots: Dot[] = [];
  const edges: [Dot, Dot][] = [];
  const previous: Dot[][] = [];
  let seed = 0;

  // place the nodes
  RINGS.forEach((ring, depth) => {
    const here: Dot[] = [];
    for (let i = 0; i < ring.count; i++) {
      const a = noise(seed++), b = noise(seed++), c = noise(seed++);
      const angle = ((i + 0.5) / ring.count) * Math.PI * 2
                  + depth * 0.7 + (a - 0.5) * (1.7 / ring.count) * Math.PI;
      const radius = ring.radius * (0.84 + b * 0.32);
      const dot: Dot = {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        r: ring.dot * (depth === 0 ? 1 : 0.75 + c * 0.5),
        at: 0, ring: depth, parent: null, branch: depth === 1 ? i : -1,
      };
      here.push(dot);
      dots.push(dot);
      if (depth > 0) {
        // parent = closest node in the previous ring
        const parents = previous[depth - 1];
        dot.parent = parents.reduce((best, p) =>
          (p.x - dot.x) ** 2 + (p.y - dot.y) ** 2 <
          (best.x - dot.x) ** 2 + (best.y - dot.y) ** 2 ? p : best, parents[0]);
        edges.push([dot.parent, dot]);
      }
    }
    previous.push(here);
  });

  // branch start times + speeds
  const limbs = RINGS[1].count;
  const starts: number[] = [];
  const tempo: number[] = [];
  let clock = 0;
  for (let b = 0; b < limbs; b++) {
    starts.push(clock);
    clock += BRANCH_GAP[0] + noise(500 + b) * (BRANCH_GAP[1] - BRANCH_GAP[0]);
    // fast / slow / fast...
    tempo.push((b % 2 ? 1.35 : 0.75) * (0.85 + noise(700 + b) * 0.3));
  }

  // arrival times, parents first (depth order is enough)
  dots.forEach((dot, i) => {
    if (!dot.parent) return;
    if (dot.branch < 0) dot.branch = dot.parent.branch;
    const own = tempo[dot.branch] ?? 1;
    dot.at = dot.ring === 1
      ? starts[dot.branch]
      : dot.parent.at + (HOP[0] + noise(1000 + i) * (HOP[1] - HOP[0])) * own;
  });

  return { dots, edges };
}

const { dots, edges } = layout();
const SOURCE = dots[0];

// one keyframe per element instead of animation-delay, otherwise the fade out
// gets delayed too and it clears in a wave
const HOLD = 78, GONE = 88;      // % of the cycle

function sheet(): string {
  const at = (t: number) => Math.min(HOLD - 6, (t / CYCLE) * 100);
  const rules: string[] = [];
  dots.slice(1).forEach((dot, i) => {
    const a = at(dot.at);
    rules.push(`@keyframes sp-d${i}{`
      + `0%,${a}%{transform:scale(0);opacity:0}`
      + `${a + 4}%{transform:scale(1.3);opacity:1}`
      + `${a + 7}%{transform:scale(1)}`
      + `${HOLD}%{transform:scale(1);opacity:1}`
      + `${GONE}%,100%{transform:scale(1);opacity:0}}`);
  });
  edges.forEach(([from, to], i) => {
    const a = at(from.at), b = at(to.at);
    rules.push(`@keyframes sp-e${i}{`
      + `0%,${a}%{stroke-dashoffset:var(--len);opacity:0}`
      + `${a + 1}%{opacity:1}`
      + `${Math.max(a + 2, b)}%{stroke-dashoffset:0;opacity:1}`
      + `${HOLD}%{stroke-dashoffset:0;opacity:1}`
      + `${GONE}%,100%{stroke-dashoffset:0;opacity:0}}`);
  });
  return rules.join("");
}

const SHEET = sheet();

export default function Spreading({ size = 168 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="-100 -100 200 200" aria-hidden
         className="spreading shrink-0 overflow-visible">
      <style>{SHEET}</style>

      {RINGS.slice(1).map((ring) => (
        <circle key={ring.radius} cx="0" cy="0" r={ring.radius}
                fill="none" stroke="var(--color-rule)" strokeWidth="0.5"
                strokeDasharray="2 4" opacity="0.55" />
      ))}

      {edges.map(([from, to], i) => {
        const length = Math.hypot(to.x - from.x, to.y - from.y);
        return (
          <line key={i} x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                stroke="var(--color-struct)" strokeWidth="0.7" strokeLinecap="round"
                strokeDasharray={length} strokeDashoffset={length}
                style={{
                  animation: `sp-e${i} ${CYCLE}s linear infinite both`,
                  opacity: 0.5 + 0.4 * (1 - to.ring / RINGS.length),
                  // @ts-expect-error a CSS custom property in a style object
                  "--len": length,
                }} />
        );
      })}

      {dots.slice(1).map((dot, i) => (
        <circle key={i} cx={dot.x} cy={dot.y} r={dot.r}
                fill="var(--color-struct)"
                opacity={0.92 - dot.ring * 0.13}
                style={{ transformOrigin: `${dot.x}px ${dot.y}px`,
                         animation: `sp-d${i} ${CYCLE}s cubic-bezier(0.22,1,0.36,1) infinite both` }} />
      ))}

      {/* source, always visible */}
      <circle cx={SOURCE.x} cy={SOURCE.y} r={SOURCE.r} fill="var(--color-signal)" />
      <circle cx={SOURCE.x} cy={SOURCE.y} r={SOURCE.r} fill="none"
              stroke="var(--color-signal)" strokeWidth="1"
              style={{ animation: `spread-source ${CYCLE}s ease-out infinite` }} />
    </svg>
  );
}
