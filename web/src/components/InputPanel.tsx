import { useState } from "react";
import type { ModelCard } from "../lib/types";

type Mode = "text" | "url" | "cascade";

const SCHEMA_EXAMPLE = `{
  "text": "the claim the post makes",
  "source_handle": "someone.bsky.social",
  "nodes": [
    {"did": "a", "handle": "source", "kind": "source",
     "profile": [0,0, 7679, 22, 0, 14729, 629, 20, 24, 118]},
    {"did": "b", "handle": "resharer", "kind": "repost"}
  ],
  "edges": [{"source": "a", "target": "b", "kind": "repost"}]
}`;

export default function InputPanel({
  models, graphModels, busy, onRun, onFetch,
}: {
  models: ModelCard[];
  graphModels: ModelCard[];
  busy: boolean;
  onRun: (payload: any) => void;
  onFetch: (url: string) => void;
}) {
  const [mode, setMode] = useState<Mode>("text");
  const [text, setText] = useState(
    "The unemployment rate has doubled since last year, according to sources online.",
  );
  const [url, setUrl] = useState("");
  const [cascade, setCascade] = useState("");
  const [model, setModel] = useState(models.find((m) => m.primary)?.key ?? "roberta");
  const [error, setError] = useState("");
  const [graphModel, setGraphModel] = useState("");

  const submit = () => {
    setError("");
    const graph_model = graphModel || undefined;
    if (mode === "text") return onRun({ text, model, origin: "text" });
    if (mode === "url") return onFetch(url);
    try {
      onRun({ cascade: JSON.parse(cascade), model, graph_model, origin: "cascade" });
    } catch {
      setError("That is not valid JSON. Check for a trailing comma or a missing brace.");
    }
  };

  const chosen = models.find((m) => m.key === model);

  return (
    <div className="space-y-4">
      <div className="flex border-b border-rule-firm">
        {(["text", "url", "cascade"] as Mode[]).map((option) => (
          <button
            key={option}
            onClick={() => setMode(option)}
            className={`-mb-px border-b-2 px-3 py-2 text-[13px] transition-colors ${
              mode === option
                ? "border-ink font-medium text-ink"
                : "border-transparent text-ink-faint hover:text-ink-soft"
            }`}
          >
            {{ text: "Post", url: "Bluesky link", cascade: "Cascade" }[option]}
          </button>
        ))}
      </div>

      {mode === "text" && (
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          className="w-full border border-rule bg-panel p-3 outline-none transition-colors focus:border-struct resize-y text-[14px] leading-relaxed"
          placeholder="Paste the text of a post"
        />
      )}

      {mode === "url" && (
        <div className="space-y-2">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="w-full border border-rule bg-panel p-3 outline-none transition-colors focus:border-struct font-mono text-[12.5px]"
            placeholder="https://bsky.app/profile/handle/post/xxxxx"
          />
          <p className="text-[12.5px] leading-relaxed text-ink-faint">
            The cascade is rebuilt live: who reshared, who replied, and each
            account's public counters. Only Bluesky is supported, because it is the
            one major platform whose reshare graph is open. X charges for it and
            Threads exposes no equivalent endpoint.
          </p>
        </div>
      )}

      {mode === "cascade" && (
        <div className="space-y-2">
          <textarea
            value={cascade}
            onChange={(e) => setCascade(e.target.value)}
            rows={8}
            spellCheck={false}
            className="w-full border border-rule bg-panel p-3 outline-none transition-colors focus:border-struct resize-y font-mono text-[11.5px] leading-relaxed"
            placeholder={SCHEMA_EXAMPLE}
          />
          <details className="text-[12.5px] text-ink-faint">
            <summary className="cursor-pointer select-none font-medium text-ink">
              What the format is
            </summary>
            <div className="mt-2 space-y-2 leading-relaxed">
              <p>
                The same records the collector writes, so a cascade from any
                platform can be read here if it is expressed this way. One node must
                carry <code className="font-mono">"kind": "source"</code>; edges point
                from the account shared to the account that shared it.
              </p>
              <p>
                <code className="font-mono">profile</code> is optional and holds ten
                numbers: verified, geo, followers, follows, listed, posts, account age
                in days, then the lengths of the handle, display name and description.
                Leave it out and the cascade is read on its shape alone, which the
                interface will tell you.
              </p>
              <button
                onClick={() => setCascade(SCHEMA_EXAMPLE)}
                className="border border-rule px-2 py-1 font-medium transition-colors hover:bg-sunk"
              >
                Fill in the example
              </button>
            </div>
          </details>
        </div>
      )}

      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-52 flex-1">
          <span className="eyebrow mb-1.5 block">
            Content model
          </span>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full border border-rule bg-panel p-2 text-[13px]"
          >
            {models.map((m) => (
              <option key={m.key} value={m.key} disabled={!m.available}>
                {m.name}
                {m.macro_f1 ? ` — macro-F1 ${m.macro_f1}` : ""}
                {m.primary ? " (of record)" : ""}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={submit}
          disabled={busy}
          className="bg-ink px-6 py-2.5 text-[13.5px] font-medium text-paper transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {busy ? "Reading" : "Read it"}
        </button>
      </div>

      {chosen?.note && (
        <p className="text-[12.5px] leading-relaxed text-ink-faint">{chosen.note}</p>
      )}

      <details className="border-t border-rule pt-3">
        <summary className="cursor-pointer select-none text-[13px] font-medium text-ink">
          Choose the propagation detector yourself
        </summary>
        <div className="mt-3 space-y-2">
          <select
            value={graphModel}
            onChange={(e) => setGraphModel(e.target.value)}
            className="w-full border border-rule bg-panel p-2 text-[13px]"
          >
            <option value="">
              Let the system choose, by platform (recommended)
            </option>
            {graphModels
              .filter((m) => m.available && m.key !== "bigcn-upfd-profile-score")
              .map((m) => (
                <option key={m.key} value={m.key}>
                  {m.name}
                  {m.macro_f1 ? ` — macro-F1 ${m.macro_f1}` : ""}
                </option>
              ))}
          </select>
          <p className="text-[12.5px] leading-relaxed text-ink-faint">
            Forcing the benchmark detector onto a Bluesky cascade is worth doing
            once: it is the cross-platform failure the article reports, and it is
            more convincing watched than read.
          </p>
        </div>
      </details>
      {error && <p className="text-[13px] text-signal">{error}</p>}

    </div>
  );
}
