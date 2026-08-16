"""A small curated set of cascades that always works, network or not.

A demonstration that depends on a live API is a demonstration that can fail in
front of an audience, so a handful of real collected cascades are kept beside
the code and offered as ready-made examples. They are chosen for contrast
rather than for flattery: one that the detector reads correctly, one it gets
wrong, one large and one small, so that the interface cannot be mistaken for a
showcase of successes.

    uv run python -m factnet.serve.samples      # rebuild the selection
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky"
COLLECTED = DATA / "cascades-by-source.jsonl"
CURATED = Path(__file__).resolve().parents[3] / "data" / "samples.jsonl"


def load() -> list[dict]:
    if not CURATED.exists():
        return []
    with CURATED.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summaries() -> list[dict]:
    """What the interface lists in its example picker."""
    out = []
    for index, cascade in enumerate(load()):
        out.append({"id": index,
                    "handle": cascade.get("source_handle", ""),
                    "text": (cascade.get("text", "") or "")[:180],
                    "accounts": len(cascade.get("nodes", [])),
                    "source_domain": cascade.get("source_domain", ""),
                    "label": cascade.get("label"),
                    "why": cascade.get("_why", "")})
    return out


def detail(index: int) -> dict | None:
    """Everything about one cascade, so a reader can see what a cascade is.

    The dashboard shows this before any model runs: the post that started it,
    what it linked to, how far it travelled and who carried it. A verdict on an
    object nobody has looked at teaches nothing.
    """
    loaded = load()
    if not 0 <= index < len(loaded):
        return None
    cascade = loaded[index]

    from factnet.ingestion.to_graph import cascade_to_pyg
    from factnet.serve.structure import depths, edge_kinds, graph_figure, node_followers, shape

    data = cascade_to_pyg(cascade)
    nodes = cascade["nodes"]
    order = sorted(range(len(nodes)), key=lambda i: nodes[i].get("kind") != "source")
    handles = [nodes[i].get("handle", "") for i in order]
    found = depths(data)

    # a readable sample of the participants rather than all of them: the point
    # is to show what kind of account is in here, not to list forty thousand
    people = [{"handle": handles[i] or f"A{i}",
               "kind": nodes[order[i]].get("kind", ""),
               "hops": found.get(i, None),
               "followers": int(nodes[order[i]].get("profile", [0] * 10)[2]),
               "posts": int(nodes[order[i]].get("profile", [0] * 10)[5])}
              for i in range(min(len(order), 40))]

    figure = graph_figure(data, handles, edge_kinds(cascade),
                          node_followers(cascade, order))
    return {
        "id": index,
        "uri": cascade.get("uri", ""),
        "url": _web_url(cascade.get("uri", ""), cascade.get("source_handle", "")),
        "source_handle": cascade.get("source_handle", ""),
        "text": cascade.get("text", ""),
        "created_at": cascade.get("created_at", ""),
        "reposts": cascade.get("repost_count", 0),
        "replies": cascade.get("reply_count", 0),
        "likes": cascade.get("like_count", 0),
        "source_domain": cascade.get("source_domain", ""),
        "source_label": cascade.get("source_label", ""),
        "label": cascade.get("label"),
        "why": cascade.get("_why", ""),
        "shape": shape(data),
        "graph": figure.data,
        "graph_caption": figure.caption,
        "people": people,
        "people_shown": len(people),
        "source_name": next((n.get("display_name", "") for n in nodes
                             if n.get("kind") == "source"), ""),
    }




def record(index: int) -> dict | None:
    """One cascade as a reader would paste it, and nothing more.

    The stored record carries the label, the annotator's verdict and several
    bookkeeping keys. None of them is an input: the pipeline never reads a
    label, and showing one in an example would suggest the system is handed the
    answer. What is left is exactly the four fields a cascade needs.
    """
    loaded = load()
    if not 0 <= index < len(loaded):
        return None
    cascade = loaded[index]
    return {
        "text": cascade.get("text", ""),
        "source_handle": cascade.get("source_handle", ""),
        "nodes": [{k: v for k, v in node.items() if k in
                   ("did", "handle", "kind", "profile")}
                  for node in cascade.get("nodes", [])],
        "edges": [{k: v for k, v in edge.items() if k in
                   ("source", "target", "kind")}
                  for edge in cascade.get("edges", [])],
    }


def smallest() -> int:
    """The index of the shortest cascade, which is the one to offer as an example."""
    loaded = load()
    if not loaded:
        return 0
    return min(range(len(loaded)), key=lambda i: len(loaded[i].get("nodes", [])))


def _web_url(uri: str, handle: str) -> str:
    """The post as a person would open it, rebuilt from the AT record key."""
    if not uri or "/" not in uri:
        return ""
    return f"https://bsky.app/profile/{handle}/post/{uri.rsplit('/', 1)[-1]}"


def build(count_per_bucket: int = 1) -> list[dict]:
    """Pick contrasting cascades and record why each was kept."""
    from factnet.ingestion.to_graph import cascade_to_pyg
    from factnet.serve.structure import pick_checkpoint, verdict

    cascades = [json.loads(line) for line in COLLECTED.open(encoding="utf-8")
                if line.strip()]
    cascades = [c for c in cascades if c.get("label") is not None
                and len(c.get("nodes", [])) > 5]

    checkpoint, _, _ = pick_checkpoint("bluesky", has_features=True)
    scored = []
    for cascade in cascades:
        probability = verdict(cascade_to_pyg(cascade), checkpoint)
        predicted = 1 if probability >= 0.5 else 0
        scored.append((cascade, probability, predicted == cascade["label"]))

    buckets: dict[str, list] = {
        "correct and large": [s for s in scored if s[2] and len(s[0]["nodes"]) > 100],
        "correct and small": [s for s in scored if s[2] and len(s[0]["nodes"]) <= 40],
        "wrong": [s for s in scored if not s[2]],
        "misleading source": [s for s in scored if s[0]["label"] == 0],
    }
    reasons = {
        "correct and large": "A large cascade the detector reads correctly.",
        "correct and small": "A small cascade, where the structural signal is thin.",
        "wrong": "The detector is wrong on this one. Kept deliberately: a model at "
                 "0.768 macro-F1 is wrong about one cascade in four.",
        "misleading source": "Linked to an outlet rated low for factual reporting.",
    }

    chosen, seen = [], set()
    for name, entries in buckets.items():
        for cascade, probability, _ in sorted(entries, key=lambda s: -len(s[0]["nodes"])):
            if cascade["uri"] in seen:
                continue
            seen.add(cascade["uri"])
            chosen.append(dict(cascade, _why=reasons[name],
                               _p_reliable=round(probability, 4)))
            if len([c for c in chosen if c["_why"] == reasons[name]]) >= count_per_bucket:
                break
    return chosen


def main() -> None:
    chosen = build()
    CURATED.parent.mkdir(parents=True, exist_ok=True)
    with CURATED.open("w", encoding="utf-8") as handle:
        for cascade in chosen:
            handle.write(json.dumps(cascade, ensure_ascii=False) + "\n")
    print(f"{len(chosen)} cascades written to {CURATED}")
    for entry in summaries():
        print(f"  {entry['accounts']:>4} accounts  "
              f"{'misleading' if entry['label'] == 0 else 'reliable':<10s} "
              f"{entry['source_domain']:<22s} {entry['why']}")


if __name__ == "__main__":
    main()
