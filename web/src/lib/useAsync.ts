import { useEffect, useState } from "react";

/**
 * One way of loading data, with the two things every panel here was missing.
 *
 * Each view used to fetch in its own effect and swallow the rejection, which
 * had a failure mode nobody would diagnose from the screen: reloading while a
 * request was in flight aborts it, the panel renders nothing at all, and
 * clicking the same item again re-runs no effect because the value it is keyed
 * on has not changed. The reading looked broken while the engine was fine, and
 * the only way out was another reload.
 *
 * So two guarantees. A failure becomes a value the caller has to render rather
 * than a silence, and `reload` re-runs the load even when nothing else changed,
 * which is what lets a person recover without touching the address bar.
 *
 * A response that arrives after the inputs moved on is dropped: the panel
 * belongs to the request the viewer is waiting for, not to the one they left.
 */
export interface Async<T> {
  data: T | null;
  error: string;
  loading: boolean;
  reload: () => void;
}

export function useAsync<T>(load: () => Promise<T>, deps: unknown[] = []): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError("");
    load().then(
      (value) => { if (live) { setData(value); setLoading(false); } },
      (cause) => {
        if (!live) return;
        setData(null);
        setLoading(false);
        // an aborted request is what a reload looks like from here, and calling
        // that a failure would be a lie: it simply never finished
        const message = cause instanceof Error ? cause.message : String(cause);
        setError(/abort|fetch|network|load failed/i.test(message)
          ? "That request did not finish, most likely because the page reloaded "
            + "while it was in flight."
          : message || "The engine is not answering.");
      },
    );
    return () => { live = false; };
    // load is rebuilt on every render, so the caller declares what it depends on
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt]);

  return { data, error, loading, reload: () => setAttempt((a) => a + 1) };
}
