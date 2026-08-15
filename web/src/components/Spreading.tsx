/**
 * A cascade drawing itself, as the wait.
 *
 * The old placeholder was three dots, which say "something is happening" and
 * nothing else. This says what is happening: a post leaves one account and is
 * carried outward, hop by hop, which is the object the propagation model reads
 * and the reason this project exists.
 *
 * Three rules keep it a propagation rather than a shimmer of dots.
 *
 * The source never leaves. It is where the post came from, so it is drawn once
 * and stays for good, and every cycle grows out of it again.
 *
 * Nobody arrives before whoever passed it to them. Arrival times are walked
 * down the tree from the root, so a child always lands after its parent and
 * after the edge between them finishes drawing.
 *
 * The limbs leave one after another, each at its own rate, alternating fast and
 * slow. A post does not reach five accounts in the same instant, and two
 * branches of one cascade rarely travel at the same speed, so a picture where
 * every limb fills together is describing something that does not happen.
 *
 * Everything that is not the source shares one cycle, so the whole cascade
 * clears at the same moment and starts over together. Giving each node its own
 * tempo, which an earlier version did, dissolves the picture into nodes
 * flickering on and off with no reading at all.
 *
 * The motion is pure CSS, so nothing is driven from JavaScript and a busy page
 * cannot make it stutter.
 */
const RINGS = [
  { count: 1, radius: 0, dot: 5.5 },
  { count: 5, radius: 30, dot: 3.6 },
  { count: 11, radius: 58, dot: 2.6 },
  { count: 15, radius: 84, dot: 1.9 },
];
const CYCLE = 2.4;            // one spread and one clearing, shared by all
const BRANCH_GAP = [0.10, 0.24];  // between one branch leaving and the next
const HOP = [0.07, 0.19];         // how long a post sits before being passed on

/** Deterministic noise from an integer, so the figure never re-rolls itself. */
function noise(seed: number): number {
  let t = (seed * 1013 + 0x6d2b79f5) >>> 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

interface Dot {
  x: number; y: number; r: number;
  at: number; ring: number; parent: Dot | null;
  branch: number;                 // which limb off the source it hangs from
}

function layout(): { dots: Dot[]; edges: [Dot, Dot][] } {
  const dots: Dot[] = [];
  const edges: [Dot, Dot][] = [];
  const previous: Dot[][] = [];
  let seed = 0;

  // 1. place every account, irregularly but not randomly
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
        // whoever it was nearest to in the ring before is who passed it on
        const parents = previous[depth - 1];
        dot.parent = parents.reduce((best, p) =>
          (p.x - dot.x) ** 2 + (p.y - dot.y) ** 2 <
          (best.x - dot.x) ** 2 + (best.y - dot.y) ** 2 ? p : best, parents[0]);
        edges.push([dot.parent, dot]);
      }
    }
    previous.push(here);
  });

  // 2. the limbs leave one after another rather than all at once, and each
  //    carries its own tempo: a post does not reach five people in the same
  //    instant, and two branches of one cascade rarely move at the same rate.
  const limbs = RINGS[1].count;
  const starts: number[] = [];
  const tempo: number[] = [];
  let clock = 0;
  for (let b = 0; b < limbs; b++) {
    starts.push(clock);
    clock += BRANCH_GAP[0] + noise(500 + b) * (BRANCH_GAP[1] - BRANCH_GAP[0]);
    // alternating, so consecutive limbs never move at the same rate
    tempo.push((b % 2 ? 1.35 : 0.75) * (0.85 + noise(700 + b) * 0.3));
  }

  // 3. then walk the arrivals down each limb, so nobody is reached before the
  //    account that reached them. Depth order is enough: a parent is always in
  //    the ring above, and its own time is already settled when we get here.
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

/**
 * One keyframe per element.
 *
 * A shared keyframe plus a per-element delay shifts the whole timeline, the
 * clearing included, so the cascade dissolves in a rolling wave instead of at
 * once. Writing the arrival into each element's own keyframe leaves the fade
 * window identical for all of them, which is what makes the picture clear
 * together and start over as one spread.
 */
const HOLD = 78, GONE = 88;      // per cent of the cycle: fade window, shared

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

      {/* the source, which never leaves: the post came from somewhere, and that
          somewhere does not stop existing between two spreads */}
      <circle cx={SOURCE.x} cy={SOURCE.y} r={SOURCE.r} fill="var(--color-signal)" />
      <circle cx={SOURCE.x} cy={SOURCE.y} r={SOURCE.r} fill="none"
              stroke="var(--color-signal)" strokeWidth="1"
              style={{ animation: `spread-source ${CYCLE}s ease-out infinite` }} />
    </svg>
  );
}
