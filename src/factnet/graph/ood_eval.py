"""Train on UPFD (twitter), test as-is on our Bluesky cascades.

Two runs: with the profile features (not comparable across platforms) and with
constant features, i.e. structure only, which is the fair one.

    uv run python -m factnet.graph.ood_eval
"""

from __future__ import annotations

import csv
import json
import statistics as st
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
RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"


def strip_features(dataset) -> list[Data]:
    """x = 1"""
    return [Data(x=torch.ones(d.num_nodes, 1), edge_index=d.edge_index, y=d.y)
            for d in dataset]


def collected_graphs() -> tuple[list[Data], list[str]]:
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
                  or int(collected[i].y) == 1]
    print(f"trained on UPFD {dataset}: {len(train_ds)} cascades")
    print(f"collected: {len(collected)} cascades, "
          f"of which {len(strict_idx)} in the strict subset\n")

    table = {}
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
        cols = list(zip(*rows, strict=True))
        mean = [st.fmean(c) for c in cols]
        sd = [st.pstdev(c) for c in cols]
        table[variant] = {k: {"mean": round(m, 4), "std": round(s, 4)}
                          for k, m, s in zip(("in_domain", "bluesky", "strict"),
                                             mean, sd, strict=True)}
        print(f"{variant:18s} in-domain {mean[0]:.3f} ±{sd[0]:.3f} "
              f"| Bluesky {mean[1]:.3f} ±{sd[1]:.3f} "
              f"| strict subset {mean[2]:.3f} ±{sd[2]:.3f} "
              f"| gap {mean[0] - mean[1]:+.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "ood_transfer.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"\nwritten to {path}")


def main():
    run()


if __name__ == "__main__":
    main()
