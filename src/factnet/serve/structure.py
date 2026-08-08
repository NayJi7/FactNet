"""Read a cascade: its verdict, its shape, and how early that verdict appears.

Three things are asked of a propagation graph here, and each one corresponds to
a result the article reports rather than to a display idea.

The verdict itself comes from a detector chosen for the platform the cascade
comes from, because a model trained on the benchmark does not survive the move
to another platform: applying the wrong one would show a confident number
computed on nothing.

The shape is set against the class averages measured on the benchmark, which is
what makes a single cascade legible: a reader cannot tell whether ninety
accounts is many without knowing what many looks like.

The early-detection curve re-runs the verdict on truncated versions of this
cascade, so the claim that the signal is present from a fifth of the diffusion
is demonstrated on the object in front of the viewer instead of quoted.
"""

from __future__ import annotations

from collections import deque

import torch
from torch_geometric.data import Data

from factnet.serve.build import load_graph_model
from factnet.serve.trace import Figure

# Average cascade shape per class, as the article measures it. The two corpora
# disagree in direction, and that is not a detail to smooth over: on GossipCop
# false stories reach twice the accounts of true ones, while on PolitiFact the
# true ones are the larger. A single pair of reference values would therefore be
# a fiction, so the comparison is always drawn against the corpus the detector
# was trained on.
CLASS_SHAPES = {
    "politifact": {
        "misleading": {"accounts": 114.9, "depth": 3.62, "breadth": 71.8, "direct": 62.7},
        "reliable":   {"accounts": 146.6, "depth": 3.89, "breadth": 87.7, "direct": 79.4},
    },
    "gossipcop": {
        "misleading": {"accounts": 76.2, "depth": 2.48, "breadth": 62.2, "direct": 60.6},
        "reliable":   {"accounts": 38.9, "depth": 2.54, "breadth": 27.9, "direct": 26.1},
    },
    # measured on our own 400 labelled cascades, and the direction is reversed
    # again: on Bluesky the credible outlets carry the far larger cascades
    "bluesky": {
        "misleading": {"accounts": 43.9, "depth": 1.4, "breadth": 40.3, "direct": 40.3},
        "reliable":   {"accounts": 259.0, "depth": 3.0, "breadth": 224.9, "direct": 224.9},
    },
}
TRUNCATIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def _adjacency(data: Data) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(data.num_nodes)]
    for source, target in data.edge_index.t().tolist():
        adjacency[source].append(target)
        adjacency[target].append(source)
    return adjacency


def depths(data: Data) -> dict[int, int]:
    """Hops from the source, breadth first, the root being node zero."""
    adjacency = _adjacency(data)
    seen = {0: 0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen[neighbour] = seen[node] + 1
                queue.append(neighbour)
    return seen


def shape(data: Data) -> dict[str, float]:
    found = depths(data)
    levels: dict[int, int] = {}
    for depth in found.values():
        levels[depth] = levels.get(depth, 0) + 1
    return {"accounts": data.num_nodes,
            "depth": max(found.values(), default=0),
            "breadth": max(levels.values(), default=0),
            "direct": levels.get(1, 0)}


def truncate(data: Data, fraction: float) -> Data:
    """The cascade as it stood after a fraction of it had spread.

    Arrival times are not recorded, so breadth-first order stands in for them,
    which is the same proxy the early-detection study uses.
    """
    keep_count = max(1, int(round(fraction * data.num_nodes)))
    order = sorted(depths(data).items(), key=lambda kv: (kv[1], kv[0]))
    keep = {node for node, _ in order[:keep_count]}
    remap = {node: rank for rank, node in enumerate(sorted(keep))}

    pairs = [(remap[s], remap[t]) for s, t in data.edge_index.t().tolist()
             if s in keep and t in keep]
    edge_index = (torch.tensor(pairs, dtype=torch.long).t().contiguous()
                  if pairs else torch.empty((2, 0), dtype=torch.long))
    return Data(x=data.x[sorted(keep)], edge_index=edge_index)


def as_model_expects(data: Data, model: torch.nn.Module) -> Data:
    """Present the cascade in the width the chosen detector was trained on.

    The structure-only detector reads a single constant column, so a cascade
    carrying the ten account features has to be stripped before it reaches it.
    Passing the wrong width used to fail inside the first linear layer, well
    away from the decision that caused it.
    """
    expected = int(getattr(model, "in_dim", data.x.size(1)))
    if data.x.size(1) == expected:
        return data
    if expected == 1:
        return Data(x=torch.ones(data.num_nodes, 1), edge_index=data.edge_index)
    raise ValueError(f"the detector reads {expected} features, the cascade carries "
                     f"{data.x.size(1)}")


@torch.no_grad()
def verdict(data: Data, checkpoint: str) -> float:
    """Probability the cascade carries reliable content."""
    model = load_graph_model(checkpoint)
    shaped = as_model_expects(data, model)
    batch = torch.zeros(shaped.num_nodes, dtype=torch.long)
    logits = model(shaped.x, shaped.edge_index, batch)
    return float(torch.softmax(logits, dim=1)[0, 1])


def early_curve(data: Data, checkpoint: str) -> Figure:
    """The verdict this model would have given at each stage of the diffusion."""
    points = []
    for fraction in TRUNCATIONS:
        partial = truncate(data, fraction)
        points.append({"observed": round(100 * fraction),
                       "accounts": partial.num_nodes,
                       "p_reliable": round(verdict(partial, checkpoint), 4)})
    return Figure(
        kind="line", title="What the verdict was, as the cascade grew",
        data={"series": [{"name": "p(reliable)", "points": points}],
              "x": "observed", "y": "p_reliable"},
        caption="Breadth-first order stands in for arrival times, which the "
                "collector does not record. The article reports the verdict "
                "holding from a fifth of the cascade observed.")


def shape_figure(measured: dict[str, float], corpus: str = "politifact") -> Figure:
    """This cascade against what each class looks like on a given corpus."""
    reference = CLASS_SHAPES[corpus]
    rows = [{"metric": metric, "this": measured[metric],
             "misleading": reference["misleading"][metric],
             "reliable": reference["reliable"][metric]}
            for metric in ("accounts", "depth", "breadth", "direct")]
    return Figure(
        kind="bars", title=f"Shape against the {corpus} class averages",
        data={"rows": rows, "keys": ["this", "misleading", "reliable"], "corpus": corpus},
        caption="The two corpora disagree in direction: on GossipCop false stories "
                "reach twice the accounts of true ones, on PolitiFact the true ones "
                "are larger. The comparison is drawn against the corpus this "
                "detector was trained on.")


def graph_figure(data: Data, handles: list[str] | None = None,
                 kinds: list[str] | None = None,
                 followers: list[int] | None = None) -> Figure:
    """Nodes and links for the layout, carrying what makes the picture readable.

    Depth alone is not enough to tell the story: a cascade in which one account
    passes the post to two hundred others looks the same as one in which two
    hundred accounts each pass it to one, unless the number of people who took
    it from each account is drawn too.
    """
    found = depths(data)
    # Reposts and replies are not the same act and must not be counted as one:
    # a repost pushes the post to an account's own followers, a reply does not
    # travel at all, and on a doubtful claim a reply is very often a correction.
    # Two thirds of the edges collected here are replies.
    kinds = list(kinds or [])
    onward: dict[int, int] = {}
    answered: dict[int, int] = {}
    for position, (source, target) in enumerate(data.edge_index.t().tolist()):
        if found.get(source, 0) >= found.get(target, 0):
            continue
        kind = kinds[position] if position < len(kinds) else "repost"
        bucket = answered if kind == "reply" else onward
        bucket[source] = bucket.get(source, 0) + 1

    audience = list(followers or [])
    nodes = [{"id": index,
              "label": (handles[index] if handles and index < len(handles) else f"A{index}"),
              "depth": found.get(index, -1),
              "reposted_by": onward.get(index, 0),
              "replied_to_by": answered.get(index, 0),
              # who an account could reach, which is what makes a name worth
              # printing on the picture rather than hidden behind a hover
              "followers": int(audience[index]) if index < len(audience) else 0,
              "root": index == 0}
             for index in range(data.num_nodes)]
    links = [{"source": s, "target": t,
              "kind": kinds[i] if i < len(kinds) else "repost"}
             for i, (s, t) in enumerate(data.edge_index.t().tolist())]
    levels: dict[int, int] = {}
    for depth in found.values():
        levels[depth] = levels.get(depth, 0) + 1
    return Figure(kind="graph", title="The cascade",
                  data={"nodes": nodes, "links": links,
                        "levels": [{"depth": d, "accounts": n}
                                   for d, n in sorted(levels.items())]},
                  caption="Every dot is an account. A solid line is a repost, "
                          "which carries the post to that account's own followers; "
                          "a faint line is a reply, which travels nowhere and is "
                          "often a correction. Rings are hops from the source, and "
                          "a dot is as large as the number of accounts that "
                          "reposted it from there.")


CHECKPOINT_FOR = {
    "bigcn-upfd-profile": "bigcn-upfd-profile.pt",
    "bigcn-collected": "bigcn-collected.pt",
    "bigcn-structure": "bigcn-structure.pt",
    "gcn-upfd-profile": "gcn-upfd-profile.pt",
    "gat-upfd-profile": "gat-upfd-profile.pt",
}


def pick_checkpoint(origin: str, has_features: bool,
                    forced: str | None = None) -> tuple[str, str, str]:
    """Which detector applies, and why.

    Returns the checkpoint file, the registry key, and the reason to display.
    A forced choice is honoured, because being able to apply the benchmark
    detector to a Bluesky cascade and watch it collapse is the point of the
    transfer experiment; the reason string then says plainly that the model is
    being used outside the platform it was trained on.
    """
    if forced and forced in CHECKPOINT_FOR:
        mismatched = origin in ("bluesky", "url", "cascade") and "upfd" in forced
        return (CHECKPOINT_FOR[forced], forced,
                "Chosen by hand. This detector was trained on the benchmark and is "
                "being applied to another platform, which is exactly the transfer "
                "the article shows collapsing: read the verdict as a demonstration "
                "of that failure, not as an assessment of this cascade."
                if mismatched else "Chosen by hand.")
    if not has_features:
        return ("bigcn-structure.pt", "bigcn-structure",
                "Account features are missing or constant on this input, so only "
                "the shape of the cascade is read.")
    if origin == "bluesky":
        return ("bigcn-collected.pt", "bigcn-collected",
                "Trained on collected Bluesky cascades. The benchmark detector is "
                "not used here because it does not transfer between platforms.")
    return ("bigcn-upfd-profile.pt", "bigcn-upfd-profile",
            "The benchmark detector, applied to a benchmark cascade.")
