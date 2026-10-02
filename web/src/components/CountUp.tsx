import { useEffect, useRef, useState } from "react";

/** Count-up animation for the header numbers. Off with prefers-reduced-motion. */
const DURATION = 900;

export default function CountUp({ value, className = "" }:
                                { value: number; className?: string }) {
  // start at 0, not at value (otherwise there's nothing to animate)
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
      // ease-out quint
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
