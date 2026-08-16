import { useEffect, useState } from "react";
import Cascades from "./components/Cascades";
import CountUp from "./components/CountUp";
import DataView from "./components/DataView";
import Specimen from "./components/Specimen";
import Models from "./components/Models";
import Institutions from "./components/Institutions";
import InputPanel from "./components/InputPanel";
import Pending from "./components/Pending";
import MeasureRail from "./components/MeasureRail";
import Results from "./components/Results";
import StepList from "./components/StepList";
import Verdict from "./components/Verdict";
import { fetchCascade, followJob, getCurrentJob, getModels, streamVerdict } from "./lib/api";
import type { JobInfo } from "./lib/api";
import type { ModelCard, Step, Trace } from "./lib/types";

type Tab = "verdict" | "cascades" | "models" | "data" | "results";
const TABS: Record<Tab, string> = {
  verdict: "Read a post",
  cascades: "Cascades",
  models: "Our models",
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
  // whether the run under way will produce structural stages, so the skeleton
  // names the six that are coming rather than the three a bare text would give
  const [expectCascade, setExpectCascade] = useState(false);
  // the reading the server is doing, when there is one. Non-null means this
  // page is following work it may not have started itself.
  const [job, setJob] = useState<JobInfo | null>(null);
  // true when this page joined a reading it did not start, which is what a
  // reload during a run now produces
  const [rejoined, setRejoined] = useState(false);
  // The header follows the page down rather than leaving it. Past a threshold
  // it drops what only matters on arrival, the counters, and keeps what is
  // needed all the way down: where you are and how to get elsewhere. Hysteresis
  // on the two thresholds, so a page resting near the boundary cannot flicker.
  const [compact, setCompact] = useState(false);

  useEffect(() => {
    getModels()
      .then((m) => { setModels(m.content); setGraphModels(m.graph); })
      .catch(() => setError(
        "The engine is not answering.",
      ));
    // the counters are cosmetic, so a failure here is retried quietly rather
    // than shown: what it must not do is sit at "-" for the rest of the session
    const counters = (tries = 2): void => {
      fetch("/api/data").then((r) => r.json())
        .then((d) => d.collected?.available && setCorpus(d.collected))
        .catch(() => { if (tries > 0) setTimeout(() => counters(tries - 1), 1200); });
    };
    counters();

    // A reading belongs to the server, not to this tab. If one is under way,
    // this page rejoins it instead of showing an idle screen: reloading during
    // a run used to lose the result and leave a page that looked broken.
    getCurrentJob()
      .then(({ job }) => {
        if (!job) return;
        setTab(job.tab as Tab);
        setExpectCascade(job.tab === "cascades" || job.stages > 4);
        setRejoined(true);
        guard(() => followJob(job.id, onStep, setJob), undefined, job);
      })
      .catch(() => {});
    // guard and onStep are stable for the life of the page
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const watch = () => setCompact((was) => (was ? window.scrollY > 60 : window.scrollY > 140));
    watch();
    window.addEventListener("scroll", watch, { passive: true });
    return () => window.removeEventListener("scroll", watch);
  }, []);

  const onStep = (s: Step) => setArriving((all) => [...all, s]);

  /** Back to the start, which is the reading tab with nothing in it. */
  const home = () => { setTab("verdict"); setTrace(null); setError(""); };

  const guard = async (work: () => Promise<Trace>, cascade?: boolean,
                       joined?: JobInfo) => {
    setBusy(true);
    setError("");
    setArriving([]);
    if (cascade !== undefined) { setExpectCascade(cascade); setRejoined(false); }
    if (joined) setJob(joined);
    try { setTrace(await work()); }
    catch (e) { setError((e as Error).message); setTrace(null); }
    finally { setBusy(false); setArriving([]); setJob(null); setRejoined(false); }
  };

  const run = (payload: any) =>
    guard(() => streamVerdict(payload, onStep, setJob),
          // sample_id can be 0, so its presence is what counts, not its truth
          Boolean(payload.cascade) || payload.sample_id !== undefined);
  const fromUrl = (url: string) =>
    guard(async () => {
      const { cascade } = await fetchCascade(url);
      return streamVerdict({ cascade, origin: "bluesky" }, onStep, setJob);
    }, true);

  return (
    // a column so the footer sits at the bottom of the window when the page is
    // shorter than it, instead of floating halfway up with dead space beneath
    <div className="relative flex min-h-screen flex-col overflow-x-clip">
      {/* the mark, ambient rather than applied: fixed, barely there, and set a
          little off square so it reads as a watermark and not as a stamp */}
      {/* decoration belongs to the empty screen only. At 8 % it is a texture
          behind prose and a distraction behind a column of figures, and every
          view here fills with figures the moment it has something to show. */}
      <div aria-hidden
           className={`pointer-events-none fixed bottom-[7vh] right-[3vw] -z-10 select-none ${
             tab === "verdict" && !trace && !busy ? "hidden lg:block" : "hidden"}`}>
        <img src="/logos/logo-txt.png" alt=""
             className="w-[min(420px,32vw)] max-w-none opacity-[0.085]"
             style={{ transform: "rotate(-6.5deg)" }} />
      </div>
      <header className="sticky top-0 z-20 border-b border-rule-firm bg-panel">
        <div className={`${SHELL} flex flex-wrap items-center justify-between gap-x-10 px-8 transition-[padding] duration-300 ${
          compact ? "gap-y-2 pb-2 pt-2.5" : "items-end gap-y-5 pb-6 pt-7"}`}
             style={{ transitionTimingFunction: "var(--ease-out-quint)" }}>
          <button onClick={home} disabled={busy}
                  title={busy ? "A reading is under way" : "Back to the start"}
                  className="flex items-center gap-4 text-left transition-opacity hover:opacity-80 disabled:opacity-100">
            <img src="/logos/logo.png" alt="" aria-hidden
                 className={`shrink-0 object-contain transition-all duration-300 ${
                   compact ? "h-[30px] w-[30px]" : "h-[52px] w-[52px]"}`}
                 style={{ transitionTimingFunction: "var(--ease-out-quint)" }} />
            <div>
              <h1 className={`font-semibold leading-none tracking-[-0.035em] transition-all duration-300 ${
                compact ? "text-[19px]" : "text-[30px]"}`}
                  style={{ transitionTimingFunction: "var(--ease-out-quint)" }}>
                FactNet
              </h1>
              {!compact && (
                <p className="mt-1.5 text-[13.5px] text-ink-soft">
                  Reading a claim <span className="text-content">by its text</span> and{" "}
                  <span className="text-struct">by the way it travelled</span>
                </p>
              )}
            </div>
          </button>

          {/* compressed, the strapline moves to the right where the counters were */}
          {compact && (
            <p className="hidden text-[12px] text-ink-soft sm:block">
              Reading a claim <span className="text-content">by its text</span> and{" "}
              <span className="text-struct">by the way it travelled</span>
            </p>
          )}
          {/* what is loaded, stated rather than implied: an instrument says what
              it is equipped with before it says what it found */}
          {!compact && (
          <dl className="flex flex-wrap gap-x-9 gap-y-3">
            {([
              ["content models", models.length],
              ["propagation models", graphModels.length],
              ["collected cascades", corpus?.cascades],
              ["accounts", corpus?.accounts],
            ] as [string, number | undefined][]).map(([label, value]) => (
              <div key={label}>
                <dt className="eyebrow">{label}</dt>
                <dd className="tnum mt-1 font-mono text-[17px] leading-none">
                  {value ? <CountUp value={value} /> : "-"}
                </dd>
              </div>
            ))}
          </dl>
          )}
        </div>
        <div className={`${SHELL} px-8`}>
          {/* While a reading runs the views are locked to the one it belongs
              to. Leaving would not stop the work, but it would hide the stages
              as they land and leave the viewer wondering where the answer went. */}
          <nav className={`flex flex-wrap items-center gap-x-7 gap-y-2 border-t border-rule transition-[padding] duration-300 ${
            compact ? "pt-1.5" : "pt-3"}`}
               style={{ transitionTimingFunction: "var(--ease-out-quint)" }}>
            {(Object.keys(TABS) as Tab[]).map((name) => (
              <button key={name}
                      disabled={busy && name !== tab}
                      title={busy && name !== tab
                        ? "A reading is under way. This opens again when it lands."
                        : undefined}
                      onClick={() => { setTab(name); setTrace(null); setError(""); }}
                      className={`-mb-px border-b-2 transition-all duration-300 ${
                        compact ? "pb-1.5 text-[12.5px]" : "pb-2.5 text-[13.5px]"} ${
                        tab === name
                          ? "border-ink font-medium"
                          : busy
                            ? "cursor-not-allowed border-transparent text-rule-firm"
                            : "border-transparent text-ink-faint hover:text-ink-soft"
                      }`}>
                {TABS[name]}
              </button>
            ))}
            {busy && (
              <span className="-mb-px ml-auto flex items-center gap-2 pb-2.5 text-[12px] text-ink-faint">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-signal" />
                reading {job?.label ? <span className="text-ink-soft">{job.label}</span> : null}
              </span>
            )}
          </nav>
        </div>
      </header>

      <main className={`${SHELL} flex-1 px-8 py-10`}>
        {tab === "cascades" && (
        <Cascades models={models} graphModels={graphModels} busy={busy} trace={trace} arriving={arriving}
                  rejoined={rejoined} jobLabel={job?.label} elapsed={job?.elapsed}
                  onChange={() => { setTrace(null); setError(""); }}
                  onAnalyse={(id, model, graph_model) =>
                    run({ sample_id: id, model, graph_model })} />
      )}
      {tab === "models" && <Models models={models} graphModels={graphModels} />}
      {tab === "data" && <DataView />}
        {tab === "results" && <Results />}

        {tab === "verdict" && (
          <div className="lg:grid lg:grid-cols-[336px_1fr] lg:gap-12">
            {/* input and readings share one sticky column. The rail used to sit
                in a third column of its own, which cost width on the right and
                left a tall empty strip on the left once the panel was scrolled
                past. Stacked, the column stays occupied and the reading gets
                everything else. */}
            <div className="lg:sticky lg:top-28 lg:self-start">
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
              {busy && <Pending withCascade={expectCascade} done={arriving}
                                rejoined={rejoined} label={job?.label}
                                elapsed={job?.elapsed} />}
              {!busy && !trace && !error && <Specimen models={models} />}
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
            <button onClick={home} disabled={busy} title="Back to the start"
                    className="transition-opacity hover:opacity-80 disabled:opacity-100">
              <img src="/logos/logo-txt.png" alt="FactNet"
                   className="h-[86px] w-auto object-contain" />
            </button>
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
