import type { Module } from "../lib/types";

/** Content / Propagation / Both tag (left border + word, not a badge). */
export const MODULE_NAMES: Record<Module, string> = {
  content: "Content",
  propagation: "Propagation",
  both: "Both modules",
};

export const MODULE_BLURBS: Record<Module, string> = {
  content: "Reads the text of the post itself.",
  propagation:
    "Reads the shape of the cascade and who carried it.",
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

/** legend */
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
