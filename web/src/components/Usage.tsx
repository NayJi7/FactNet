import { useEffect, useState } from "react";
import { getUsage, type Usage as Left } from "../lib/api";

export default function Usage() {
  const [left, setLeft] = useState<Left | null>(null);

  useEffect(() => {
    const load = () => getUsage().then(setLeft).catch(() => setLeft(null));
    load();
    window.addEventListener("factnet:usage", load);
    return () => window.removeEventListener("factnet:usage", load);
  }, []);

  if (!left) return null;
  const { reading, fetch } = left;

  return (
    <p className="mt-3 max-w-[54ch] text-[12px] leading-relaxed text-ink-faint">
      Hosted on a personal server. Everything already on the site, the collected
      cascades and the example posts, is free to explore. What you bring yourself
      is capped at {reading.limit} readings and {fetch.limit} Bluesky links a day:{" "}
      <span className="tnum font-mono text-ink-soft">{reading.left}</span> and{" "}
      <span className="tnum font-mono text-ink-soft">{fetch.left}</span> left today.
    </p>
  );
}
