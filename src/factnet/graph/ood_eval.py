"""Cross-platform test: a detector trained on Twitter, evaluated on Bluesky.

This is the experiment the collected sample was gathered for. A model is trained
on the benchmark cascades and then applied, unchanged, to cascades collected
from a different platform whose labels come from the credibility of the source
rather than from the benchmark's fact-checkers.

Two variants are run, and the distinction matters for what may be concluded.
The first keeps the account features, which are computed differently on each
platform because the two expose different counters: a drop there cannot be
attributed to structure alone. The second replaces every node feature by a
constant, so the model can only read the shape of the cascade; that variant is
the honest test of the project's structural claim, since nothing but the
topology survives the move between platforms.

    uv run python -m factnet.graph.ood_eval
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from factnet.graph.models import BiGCN
from factnet.ingestion.to_graph import cascade_to_pyg, read_cascades

ROOT = str(Path(__file__).resolve().parents[3] / "data" / "raw" / "upfd")
DATA = Path(__file__).resolve().parents[3] / "data" / "raw"
COLLECTED = DATA / "bluesky" / "cascades-by-source.jsonl"
IFFY = DATA / "iffy-index.csv"
SEEDS = (0, 1, 2)


def strip_features(dataset) -> list[Data]:
    """Replace node features by a constant, leaving only the graph shape."""
    return [Data(x=torch.ones(d.num_nodes, 1), edge_index=d.edge_index, y=d.y)
            for d in dataset]


def collected_graphs() -> tuple[list[Data], list[str]]:
    """The collected cascades as graphs, with the source domain of each."""
    graphs, domains = [], []
    for cascade in read_cascades(COLLECTED):
        if cascade.get("label") is None:
            continue
        graphs.append(cascade_to_pyg(cascade))
        domains.append(cascade.get("source_domain", ""))
    return graphs, domains


def very_low_domains() -> set[str]:
    with IFFY.open(encoding="utf-8") as handle:
        return {r["Domain"].strip().lower() for r in csv.DictReader(handle)
                if (r.get("MBFC Fact") or "").strip() == "VL"}


def train(train_list: list[Data], in_dim: int, epochs: int = 60, seed: int = 0) -> BiGCN:
    torch.manual_seed(seed)
    model = BiGCN(in_dim, 64, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loader = DataLoader(train_list, batch_size=128, shuffle=True)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            F.cross_entropy(model(batch.x, batch.edge_index, batch.batch), batch.y).backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model: BiGCN, graphs: list[Data]) -> tuple[float, float]:
    model.eval()
    ys, ps = [], []
    for batch in DataLoader(graphs, batch_size=128):
        ps += model(batch.x, batch.edge_index, batch.batch).argmax(dim=1).tolist()
        ys += batch.y.tolist()
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def run(dataset: str = "gossipcop") -> None:
    train_ds = UPFD(ROOT, dataset, "profile", split="train")
    test_ds = UPFD(ROOT, dataset, "profile", split="test")
    collected, domains = collected_graphs()
    strict = very_low_domains()
    strict_idx = [i for i, d in enumerate(domains) if d in strict
                  or int(collected[i].y) == 1]  # the reliable side is unchanged
    print(f"trained on UPFD {dataset}: {len(train_ds)} cascades")
    print(f"collected: {len(collected)} cascades, "
          f"of which {len(strict_idx)} in the strict subset\n")

    for variant in ("account features", "structure only"):
        rows = []
        for seed in SEEDS:
            if variant == "account features":
                model = train(list(train_ds), train_ds.num_features, seed=seed)
                in_domain, ood = list(test_ds), collected
            else:
                model = train(strip_features(train_ds), 1, seed=seed)
                in_domain, ood = strip_features(test_ds), strip_features(collected)
            _, f_in = evaluate(model, in_domain)
            _, f_ood = evaluate(model, ood)
            _, f_strict = evaluate(model, [ood[i] for i in strict_idx])
            rows.append((f_in, f_ood, f_strict))
        mean = [sum(c) / len(c) for c in zip(*rows, strict=True)]
        print(f"{variant:18s} in-domain {mean[0]:.3f} | Bluesky {mean[1]:.3f} "
              f"| strict subset {mean[2]:.3f} | gap {mean[0] - mean[1]:+.3f}")


def main():
    run()


if __name__ == "__main__":
    main()
