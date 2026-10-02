/** Error state for a panel: the message + a retry button (no auto retry). */
export default function Failed({ error, onRetry, what = "this" }:
                               { error: string; onRetry: () => void; what?: string }) {
  return (
    <div className="border border-rule-firm bg-panel px-6 py-5">
      <p className="eyebrow">Nothing came back</p>
      <p className="mt-2 max-w-[62ch] text-[13.5px] leading-relaxed text-ink-soft">
        {error}
      </p>
      <button onClick={onRetry}
              className="mt-4 border border-ink bg-ink px-4 py-2 text-[13px] font-medium
                         text-paper transition-opacity hover:opacity-85">
        Load {what} again
      </button>
    </div>
  );
}
