import { useEffect, useRef, useState } from "react";

/**
 * A figure that arrives rather than appears.
 *
 * The header states what the instrument is equipped with, and those four
 * numbers are the first thing anyone sees. Counting them up reads as the page
 * taking stock of itself, which is what it is in fact doing: none of them is
 * written down, they are all answers from the engine.
 *
 * The easing is fast at the start and slow at the end, so the eye catches the
 * order of magnitude immediately and the exact value settles a moment later.
 * A viewer who asked for less motion gets the number and no animation.
 */
const DURATION = 900;

export default function CountUp({ value, className = "" }:
                                { value: number; className?: string }) {
  // Starts at nothing, not at the answer. The caller only mounts this once the
  // engine has replied, so seeding the state with the value it was given left
  // the first render already holding the final figure and nothing to count.
  const [shown, setShown] = useState(0);
  const from = useRef(0);

  useEffect(() => {
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const start = from.current;
    if (reduced || start === value) { setShown(value); from.current = value; return; }

    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / DURATION);
      // quintic ease-out, the same curve the rest of the interface moves on
      const eased = 1 - Math.pow(1 - t, 5);
      setShown(Math.round(start + (value - start) * eased));
      if (t < 1) raf = requestAnimationFrame(tick);
      else from.current = value;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  return <span className={className}>{shown.toLocaleString("en")}</span>;
}
