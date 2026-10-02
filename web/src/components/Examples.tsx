/**
 * Example posts, one per path through the pipeline (claim, opinion -> gate,
 * non-english warning...). Labelled by what they are, not by the verdict.
 * Keep in sync with EXAMPLE_POSTS in serve/samples.py (there's a test).
 */
const EXAMPLES: [string, string][] = [
  ["a checkable false claim",
   "Doctors confirm the new vaccine has caused thousands of deaths that health "
   + "agencies refuse to report."],
  ["an attributed study",
   "Researchers at the university published a study showing the drug reduced "
   + "symptoms in 2019 trials."],
  ["an opinion, not a claim",
   "Honestly I think this is the funniest thing I have seen all week, my whole "
   + "family loved it."],
  ["a post in another language",
   "En 2024, le ministère a confirmé que le taux de chômage avait baissé selon "
   + "trois sources officielles."],
];

export default function Examples({ onPick, disabled }:
                                 { onPick: (text: string) => void; disabled?: boolean }) {
  return (
    <div>
      <p className="eyebrow mb-1.5">Or try one</p>
      <div className="flex flex-wrap gap-1.5">
        {EXAMPLES.map(([label, text]) => (
          <button key={label}
                  onClick={() => onPick(text)}
                  disabled={disabled}
                  title={text}
                  className="border border-rule px-2 py-1 text-[11.5px] text-ink-soft
                             transition-colors hover:border-rule-firm hover:bg-sunk
                             disabled:opacity-40">
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
