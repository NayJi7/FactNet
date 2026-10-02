import { useEffect, useState } from "react";

/**
 * Small fetch hook. Errors are returned (not swallowed) and reload() refetches
 * even if the deps didn't change. Before this, a reload mid-request left an
 * empty panel and clicking again did nothing. Stale responses are ignored.
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
        // aborted = page reload, not an error
        const message = cause instanceof Error ? cause.message : String(cause);
        setError(/abort|fetch|network|load failed/i.test(message)
          ? "That request did not finish, most likely because the page reloaded "
            + "while it was in flight."
          : message || "The engine is not answering.");
      },
    );
    return () => { live = false; };
    // caller passes the deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, attempt]);

  return { data, error, loading, reload: () => setAttempt((a) => a + 1) };
}
