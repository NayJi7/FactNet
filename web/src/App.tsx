import { useEffect, useState } from "react";
import Cascades from "./components/Cascades";
import DataView from "./components/DataView";
import Specimen from "./components/Specimen";
import Institutions from "./components/Institutions";
import InputPanel from "./components/InputPanel";
import Pending from "./components/Pending";
import MeasureRail from "./components/MeasureRail";
import Results from "./components/Results";
import StepList from "./components/StepList";
import Verdict from "./components/Verdict";
import { fetchCascade, getModels, streamVerdict } from "./lib/api";
import type { ModelCard, Step, Trace } from "./lib/types";

type Tab = "verdict" | "cascades" | "data" | "results";
const TABS: Record<Tab, string> = {
  verdict: "Read a post",
  cascades: "Cascades",
  data: "The data",
  results: "What we measured",
};

// One measure for the whole page. It gains enough on a large display to stand
// two figures abreast, and no more: past that the paragraphs, which cap
// themselves in ch, sit in a widening pool of white and the page reads emptier
// than the narrow version it replaced.
const SHELL = "mx-auto w-full max-w-[1180px] xl:max-w-[1340px]";

export default function App() {
  const [models, setModels] = useState<ModelCard[]>([]);
  const [graphModels, setGraphModels] = useState<ModelCard[]>([]);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [busy, setBusy] = useState(false);
  const [arriving, setArriving] = useState<Step[]>([]);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<Tab>("verdict");
  const [corpus, setCorpus] = useState<{ cascades: number; accounts: number } | null>(null);

  useEffect(() => {
    getModels()
      .then((m) => { setModels(m.content); setGraphModels(m.graph); })
      .catch(() => setError(
        "The engine is not answering.",
      ));
    fetch("/api/data").then((r) => r.json())
      .then((d) => d.collected?.available && setCorpus(d.collected))
      .catch(() => {});
  }, []);

  const guard = async (work: () => Promise<Trace>) => {
    setBusy(true);
    setError("");
    setArriving([]);
    try { setTrace(await work()); }
    catch (e) { setError((e as Error).message); setTrace(null); }
    finally { setBusy(false); setArriving([]); }
  };

  const run = (payload: any) =>
    guard(() => streamVerdict(payload, (s) => setArriving((all) => [...all, s])));
  const fromUrl = (url: string) =>
    guard(async () => {
      const { cascade } = await fetchCascade(url);
      return streamVerdict({ cascade, origin: "bluesky" },
                           (s) => setArriving((all) => [...all, s]));
    });

  return (
    // a column so the footer sits at the bottom of the window when the page is
    // shorter than it, instead of floating halfway up with dead space beneath
    <div className="relative flex min-h-screen flex-col overflow-x-hidden">
      {/* the mark, ambient rather than applied: fixed, barely there, and set a
          little off square so it reads as a watermark and not as a stamp */}
      <div aria-hidden
           className="pointer-events-none fixed bottom-[7vh] right-[3vw] -z-10 hidden select-none lg:block">
        <img src="/logos/logo-txt.png" alt=""
             className="w-[min(420px,32vw)] max-w-none opacity-[0.085]"
             style={{ transform: "rotate(-6.5deg)" }} />
      </div>
      <header className="border-b border-rule-firm bg-panel">
        <div className={`${SHELL} flex flex-wrap items-end justify-between gap-x-10 gap-y-5 px-8 pb-6 pt-7`}>
          <div className="flex items-center gap-4">
            <img src="/logos/logo.png" alt="" aria-hidden
                 className="h-[52px] w-[52px] shrink-0 object-contain" />
            <div>
              <h1 className="text-[30px] font-semibold leading-none tracking-[-0.035em]">
                FactNet
              </h1>
              <p className="mt-1.5 text-[13.5px] text-ink-soft">
                Reading a claim, and the way it travelled
              </p>
            </div>
          </div>
          {/* what is loaded, stated rather than implied: an instrument says what
              it is equipped with before it says what it found */}
          <dl className="flex flex-wrap gap-x-9 gap-y-3">
            {[
              ["content models", models.length || "-"],
              ["propagation models", graphModels.length || "-"],
              ["collected cascades", corpus?.cascades ?? "-"],
              ["accounts", corpus ? corpus.accounts.toLocaleString("en") : "-"],
            ].map(([label, value]) => (
              <div key={String(label)}>
                <dt className="eyebrow">{label}</dt>
                <dd className="tnum mt-1 font-mono text-[17px] leading-none">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div className={`${SHELL} px-8`}>
          <nav className="flex gap-7 border-t border-rule pt-3">
            {(Object.keys(TABS) as Tab[]).map((name) => (
              <button key={name}
                      onClick={() => { setTab(name); setTrace(null); setError(""); }}
                      className={`-mb-px border-b-2 pb-2.5 text-[13.5px] transition-colors ${
                        tab === name
                          ? "border-ink font-medium"
                          : "border-transparent text-ink-faint hover:text-ink-soft"
                      }`}>
                {TABS[name]}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className={`${SHELL} flex-1 px-8 py-10`}>
        {tab === "cascades" && (
        <Cascades models={models} busy={busy} trace={trace} arriving={arriving}
                  onChange={() => { setTrace(null); setError(""); }}
                  onAnalyse={(id, model) => run({ sample_id: id, model })} />
      )}
      {tab === "data" && <DataView />}
        {tab === "results" && <Results />}

        {tab === "verdict" && (
          <div className="lg:grid lg:grid-cols-[336px_1fr] lg:gap-12">
            {/* input and readings share one sticky column. The rail used to sit
                in a third column of its own, which cost width on the right and
                left a tall empty strip on the left once the panel was scrolled
                past. Stacked, the column stays occupied and the reading gets
                everything else. */}
            <div className="lg:sticky lg:top-8 lg:self-start">
              {models.length > 0 && (
                <InputPanel models={models} graphModels={graphModels}
                            busy={busy} onRun={run} onFetch={fromUrl} />
              )}
              {trace && <MeasureRail trace={trace} />}
            </div>

            <div className="mt-10 min-w-0 lg:mt-0">
              {error && (
                <p className="border border-signal/40 bg-signal-dim/40 px-4 py-3 text-[13px]">
                  {error}
                </p>
              )}
              {busy && <Pending withCascade={false} done={arriving} />}
              {!busy && !trace && !error && <Specimen />}
              {!busy && trace && (
                <>
                  <Verdict trace={trace} />
                  <StepList steps={trace.steps} />
                </>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="mt-8 border-t border-rule-firm bg-panel">
        <div className={`${SHELL} flex flex-wrap items-center justify-between gap-x-12 gap-y-7 px-8 py-8`}>
          <div className="flex flex-wrap items-center gap-x-10 gap-y-6">
            {/* the lockup carries its own wordmark, so no label accompanies it */}
            <img src="/logos/logo-txt.png" alt="FactNet"
                 className="h-[86px] w-auto object-contain" />
            <span className="hidden h-12 w-px bg-rule sm:block" aria-hidden />
            <Institutions />
          </div>
          <p className="max-w-[54ch] text-[12px] leading-relaxed text-ink-faint">
            A research prototype. The propagation model is trained on 400 cascades
            collected from Bluesky and two public benchmarks. It is not a deployable
            classifier, and a reading here is an argument, not a ruling.
          </p>
        </div>
      </footer>
    </div>
  );
}
