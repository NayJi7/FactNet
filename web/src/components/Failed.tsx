/**
 * What a panel shows when its data did not arrive.
 *
 * The alternative, which this replaces, was nothing at all: a blank column and
 * a person clicking the same thing repeatedly with no way to know whether the
 * engine had died, the request had been cut, or they had misread the interface.
 *
 * It states what happened and offers the one action that helps. It does not
 * apologise, and it does not retry on its own: a silent retry loop against an
 * engine that is genuinely down is worse than a button.
 */
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
