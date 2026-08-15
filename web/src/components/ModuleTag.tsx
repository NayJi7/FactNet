import type { Module } from "../lib/types";

/**
 * Which half of the project a stage or a table comes from.
 *
 * The work is split into two modules written up as two papers, and a viewer
 * looking at a number here has no way to tell which paper to check it against
 * unless the interface says so. The mark is a coloured rule and a word, not a
 * filled badge: it has to be findable when scanning down the page and quiet
 * enough to ignore while reading across it.
 */
export const MODULE_NAMES: Record<Module, string> = {
  content: "Content",
  propagation: "Propagation",
  both: "Both modules",
};

export const MODULE_BLURBS: Record<Module, string> = {
  content: "Reads the text of the post itself. Ayman Ouguerd and Louaye Saghir.",
  propagation:
    "Reads the shape of the cascade and who carried it. Adam Terrak and Abdelah El Harsal.",
  both: "Where a content score is handed to the propagation model.",
};

export default function ModuleTag({ module, className = "" }:
                                  { module: Module | ""; className?: string }) {
  if (!module) return null;
  return (
    <span className={`module-tag module-${module} ${className}`}
          title={MODULE_BLURBS[module]}>
      {MODULE_NAMES[module]}
    </span>
  );
}

/** The key, shown once per view so the colours mean something on first sight. */
export function ModuleLegend({ className = "" }: { className?: string }) {
  return (
    <div className={`flex flex-wrap items-start gap-x-8 gap-y-3 ${className}`}>
      {(["propagation", "content", "both"] as Module[]).map((m) => (
        <div key={m} className="max-w-[30ch]">
          <ModuleTag module={m} />
          <p className="mt-1 pl-[calc(0.5em+2px)] text-[11.5px] leading-snug text-ink-faint">
            {MODULE_BLURBS[m]}
          </p>
        </div>
      ))}
    </div>
  );
}
