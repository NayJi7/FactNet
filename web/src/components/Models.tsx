import ContentModels from "./ContentModels";
import GraphModels from "./GraphModels";
import Modules from "./Modules";
import { ModuleLegend } from "./ModuleTag";
import type { ModelCard } from "../lib/types";

/**
 * Everything the reader might want to know about what is doing the reading.
 *
 * These three blocks used to sit stacked under the opening example, where the
 * page had already said its piece and nobody scrolled that far. A viewer who
 * wonders which of eleven names to trust has a question, and a question
 * deserves a place to go rather than a paragraph they have to find.
 *
 * The order answers the question in the order it is asked: which half is this,
 * then who reads the text, then who reads the cascade.
 */
export default function Models({ models, graphModels }:
                               { models: ModelCard[]; graphModels: ModelCard[] }) {
  return (
    <div className="space-y-10">
      <div>
        <p className="max-w-[80ch] text-[14px] leading-relaxed text-ink-soft">
          Eleven models are loaded here, and a verdict from any of them looks the
          same on screen. This page is what separates them: which half of the
          system each one belongs to, what it actually reads, and what it scored
          on a held-out benchmark. Every figure below is the one the model earned,
          served by the engine rather than written into this page.
        </p>
        <ModuleLegend className="mt-6 border-t border-rule pt-5" />
      </div>

      <Modules />
      <ContentModels models={models} />
      <GraphModels models={graphModels} />
    </div>
  );
}
