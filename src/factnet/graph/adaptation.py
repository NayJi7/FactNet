"""Can the transfer failure be repaired, and what does it teach?

Applying a detector trained on Twitter cascades to Bluesky ones makes it
collapse to a single class. This module asks what follows from that, in three
steps whose order matters.

The first is a diagnostic, and it comes first because the other two depend on
its answer: trained and tested on the collected cascades alone, does cascade
shape carry any signal about the credibility of the linked source? If it does
not, the failure is not one of transfer at all but of the construct, and no
amount of adaptation would help.

The second asks how much labelled data from the new platform is needed to
recover, by fine-tuning the benchmark-trained model on a growing fraction of
the collected set.

The third avoids new labels entirely: absolute sizes differ by an order of
magnitude between the platforms, so the model is trained on ratios instead,
the share of accounts one hop from the source and the depth relative to the
cascade's own size, which are dimensionless and therefore portable.

    uv run python -m factnet.graph.adaptation
"""

from __future__ import annotations

import statistics as st
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from factnet.graph.models import BiGCN
from factnet.ingestion.to_graph import cascade_to_pyg, read_cascades

DATA = Path(__file__).resolve().parents[3] / "data" / "raw"
COLLECTED = DATA / "bluesky" / "cascades-by-source.jsonl"
SEEDS = (0, 1, 2)


def shape_features(data: Data) -> torch.Tensor:
    """Per-node features that do not depend on the platform's scale.

    Absolute counts do not survive the move: a Bluesky cascade holds three
    times the accounts of a GossipCop one. Ratios do, so every node carries its
    depth relative to the cascade's own depth, its share of the cascade's
    nodes, and whether it sits one hop from the source, which is the quantity
    that separated the classes on the benchmark.
    """
    n = data.num_nodes
    adj: list[list[int]] = [[] for _ in range(n)]
    for s, t in data.edge_index.t().tolist():
        adj[s].append(t)
        adj[t].append(s)

    depth = {0: 0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in depth:
                depth[nxt] = depth[node] + 1
                queue.append(nxt)
    max_depth = max(depth.values()) or 1
    direct = sum(1 for d in depth.values() if d == 1)

    rows = []
    for node in range(n):
        d = depth.get(node, max_depth)
        rows.append([d / max_depth,                       # relative position
                     1.0 if d == 1 else 0.0,              # one hop from the source
                     len(adj[node]) / n,                  # share of the cascade touched
                     direct / n])                         # fan-out of this cascade
    return torch.tensor(rows, dtype=torch.float)


def as_shape(dataset) -> list[Data]:
    return [Data(x=shape_features(d), edge_index=d.edge_index, y=d.y) for d in dataset]


def collected() -> list[Data]:
    return [cascade_to_pyg(c) for c in read_cascades(COLLECTED) if c.get("label") is not None]


def split(graphs: list[Data], fraction: float, seed: int) -> tuple[list[Data], list[Data]]:
    """Stratified split, so both parts keep the class balance."""
    generator = torch.Generator().manual_seed(seed)
    train, test = [], []
    for label in (0, 1):
        subset = [g for g in graphs if int(g.y) == label]
        order = torch.randperm(len(subset), generator=generator).tolist()
        cut = int(fraction * len(subset))
        train += [subset[i] for i in order[:cut]]
        test += [subset[i] for i in order[cut:]]
    return train, test


def fit(train_list: list[Data], in_dim: int, epochs: int = 60, seed: int = 0,
        start_from: BiGCN | None = None, lr: float = 0.01) -> BiGCN:
    torch.manual_seed(seed)
    model = start_from if start_from is not None else BiGCN(in_dim, 64, 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loader = DataLoader(train_list, batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            F.cross_entropy(model(batch.x, batch.edge_index, batch.batch), batch.y).backward()
            opt.step()
    return model


@torch.no_grad()
def score(model: BiGCN, graphs: list[Data]) -> float:
    model.eval()
    ys, ps = [], []
    for batch in DataLoader(graphs, batch_size=128):
        ps += model(batch.x, batch.edge_index, batch.batch).argmax(dim=1).tolist()
        ys += batch.y.tolist()
    return f1_score(ys, ps, average="macro")


def diagnostic(graphs: list[Data]) -> None:
    """Does cascade shape predict source credibility within Bluesky at all?"""
    scores = []
    for seed in SEEDS:
        train, test = split(graphs, 0.6, seed)
        scores.append(score(fit(train, graphs[0].x.size(1), seed=seed), test))
    print(f"  trained and tested on collected data only : macro-F1 "
          f"{st.fmean(scores):.3f} (+/- {st.pstdev(scores):.3f})")


def how_much_data(graphs: list[Data], upfd) -> None:
    """How many collected cascades does recovery need, and does the benchmark help?

    Each budget is spent twice, once fine-tuning the benchmark-trained model and
    once training from scratch on the same cascades, so that the two columns
    answer a question the first alone cannot: whether starting from the
    benchmark is worth anything at all.
    """
    print(f"  {'cascades':>8s} {'fine-tuned':>11s} {'from scratch':>13s}")
    for fraction in (0.0, 0.1, 0.25, 0.5):
        tuned, fresh = [], []
        for seed in SEEDS:
            train, test = split(graphs, 0.5, seed)          # test half is fixed
            base = fit(list(upfd), upfd.num_features, seed=seed)
            if fraction == 0:
                tuned.append(score(base, test))
                fresh.append(float("nan"))
                continue
            subset, _ = split(train, fraction / 0.5, seed)
            tuned.append(score(fit(subset, upfd.num_features, epochs=30, seed=seed,
                                   start_from=base, lr=0.002), test))
            fresh.append(score(fit(subset, upfd.num_features, seed=seed), test))
        n = int(fraction * len(graphs))
        scratch = "-" if fraction == 0 else f"{st.fmean(fresh):.3f}"
        print(f"  {n:>8d} {st.fmean(tuned):>11.3f} {scratch:>13s}")


def portable_features(graphs: list[Data], upfd_train, upfd_test) -> None:
    """Train on the benchmark with dimensionless features, test on Bluesky."""
    tr, te, col = as_shape(upfd_train), as_shape(upfd_test), as_shape(graphs)
    in_dom, ood = [], []
    for seed in SEEDS:
        model = fit(tr, 4, seed=seed)
        in_dom.append(score(model, te))
        ood.append(score(model, col))
    print(f"  in-domain {st.fmean(in_dom):.3f} | transferred to Bluesky "
          f"{st.fmean(ood):.3f} (+/- {st.pstdev(ood):.3f})")


def main():
    graphs = collected()
    upfd_train = UPFD(str(DATA / "upfd"), "gossipcop", "profile", split="train")
    upfd_test = UPFD(str(DATA / "upfd"), "gossipcop", "profile", split="test")
    print(f"collected cascades: {len(graphs)}  |  UPFD train: {len(upfd_train)}\n")

    print("1. Diagnostic: is there any signal to learn on this platform?")
    diagnostic(graphs)
    print("\n2. Adaptation: how much labelled data does recovery need?")
    how_much_data(graphs, upfd_train)
    print("\n3. Portable features: ratios instead of absolute sizes")
    portable_features(graphs, upfd_train, upfd_test)


if __name__ == "__main__":
    main()
