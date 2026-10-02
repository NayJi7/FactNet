import { useEffect, useState } from "react";
import ContentModels from "./ContentModels";
import GraphModels from "./GraphModels";
import Modules from "./Modules";
import { ModuleLegend } from "./ModuleTag";
import type { ModelCard } from "../lib/types";

/** Models tab: the two modules, then content models, then graph models.
 *  (used to be under the example on the first page, nobody scrolled there) */
const SECTIONS: [string, string][] = [
  ["halves", "The two halves"],
  ["content", "Content readers"],
  ["graph", "Propagation detectors"],
];

export default function Models({ models, graphModels }:
                               { models: ModelCard[]; graphModels: ModelCard[] }) {
  const [here, setHere] = useState("halves");

  useEffect(() => {
    const spotter = new IntersectionObserver(
      (entries) => {
        const first = entries.filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (first) setHere(first.target.id.replace(/^m-/, ""));
      },
      { rootMargin: "-14% 0px -70% 0px" },
    );
    document.querySelectorAll('[id^="m-"]').forEach((n) => spotter.observe(n));
    return () => spotter.disconnect();
  }, [models, graphModels]);

  return (
    <div className="xl:grid xl:grid-cols-[172px_1fr] xl:gap-x-12">
      <nav className="hidden xl:block">
        <div className="sticky top-28 pb-4">
          <p className="eyebrow mb-1.5">On this page</p>
          <ul>
            {SECTIONS.map(([id, label]) => (
              <li key={id}>
                <a href={`#m-${id}`}
                   className={`block border-l py-1 pl-2.5 text-[12px] leading-snug transition-colors ${
                     here === id
                       ? "border-ink font-medium text-ink"
                       : "border-rule text-ink-faint hover:text-ink-soft"}`}>
                  {label}
                </a>
              </li>
            ))}
          </ul>
          <p className="mt-5 border-t border-rule pt-3 text-[11.5px] leading-relaxed text-ink-faint">
            Every figure on this page is the one the model earned on a held-out
            split, served by the engine rather than written into the page.
          </p>
        </div>
      </nav>

      <div className="min-w-0 space-y-10">
        <div>
          <p className="max-w-[80ch] text-[14px] leading-relaxed text-ink-soft">
            Eleven models are loaded here, and a verdict from any of them looks
            the same on screen. This page is what separates them: which half of
            the system each one belongs to, what it actually reads, and what it
            scored on a held-out benchmark.
          </p>
          <ModuleLegend className="mt-6 border-t border-rule pt-5" />
        </div>

        <div id="m-halves" className="scroll-mt-32"><Modules /></div>
        <div id="m-content" className="scroll-mt-32"><ContentModels models={models} /></div>
        <div id="m-graph" className="scroll-mt-32"><GraphModels models={graphModels} /></div>
      </div>
    </div>
  );
}
