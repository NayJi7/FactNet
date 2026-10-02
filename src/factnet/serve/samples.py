"""A few collected cascades shipped with the app, so the demo works offline.

Picked to show both sides: big/small, one the model gets right, one it gets wrong.

    uv run python -m factnet.serve.samples      # rebuild the selection
"""

from __future__ import annotations

import json
from pathlib import Path

# must match the "Or try one" buttons exactly (web/src/components/Examples.tsx)
EXAMPLE_POSTS = (
    "Doctors confirm the new vaccine has caused thousands of deaths that health "
    "agencies refuse to report.",
    "Researchers at the university published a study showing the drug reduced "
    "symptoms in 2019 trials.",
    "Honestly I think this is the funniest thing I have seen all week, my whole "
    "family loved it.",
    "En 2024, le ministère a confirmé que le taux de chômage avait baissé selon "
    "trois sources officielles.",
)

DATA = Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky"
COLLECTED = DATA / "cascades-by-source.jsonl"
CURATED = Path(__file__).resolve().parents[3] / "data" / "samples.jsonl"


def load() -> list[dict]:
    if not CURATED.exists():
        return []
    with CURATED.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summaries() -> list[dict]:
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
    """Full info on one sample (post, link, shape, accounts) for the cascades tab."""
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

    # first 40 is enough to get an idea
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
    """Sample stripped down to the input fields (no label, it isn't an input)."""
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
    """Index of the smallest cascade."""
    loaded = load()
    if not loaded:
        return 0
    return min(range(len(loaded)), key=lambda i: len(loaded[i].get("nodes", [])))


def _web_url(uri: str, handle: str) -> str:
    """at:// uri -> bsky.app link"""
    if not uri or "/" not in uri:
        return ""
    return f"https://bsky.app/profile/{handle}/post/{uri.rsplit('/', 1)[-1]}"


def build(count_per_bucket: int = 1) -> list[dict]:
    """Pick one cascade per bucket, with the reason in _why."""
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
