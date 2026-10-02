"""Is the 0.773 on Bluesky really about cascade shape? Two suspects:

- size: reliable outlets are big, misleading ones are tiny, so maybe the number
  of accounts is enough -> one-threshold rule as a baseline
- memorisation: same accounts in train and test -> split by domain instead
4 conditions: (cascade or domain split) x (all sizes or size matched).

    uv run python -u -m factnet.graph.confound
"""

from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

from factnet.graph.models import BiGCN
from factnet.ingestion.to_graph import cascade_to_pyg

COLLECTED = (Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky"
             / "cascades-by-source.jsonl")
SEEDS = (0, 1, 2)
RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
DONE = "CONFOUND-COMPLETE"


def load() -> list[dict]:
    with COLLECTED.open(encoding="utf-8") as handle:
        return [c for c in (json.loads(line) for line in handle if line.strip())
                if c.get("label") is not None]


def size_matched(rows: list[dict], tolerance: float = 0.25) -> list[dict]:
    """Match each misleading cascade with a reliable one of similar size, so size
    can't predict the label anymore."""
    smaller = sorted((len(r["nodes"]), i) for i, r in enumerate(rows) if r["label"] == 0)
    larger = sorted((len(r["nodes"]), i) for i, r in enumerate(rows) if r["label"] == 1)
    taken: set[int] = set()
    keep: list[dict] = []
    for size, index in smaller:
        allowance = max(5.0, tolerance * size)
        best = None
        for other_size, other in larger:
            if other in taken:
                continue
            distance = abs(size - other_size)
            if best is None or distance < best[0]:
                best = (distance, other)
            if other_size > size + allowance:
                break
        if best and best[0] <= allowance:
            taken.add(best[1])
            keep += [rows[index], rows[best[1]]]
    return keep


def split_by_cascade(rows: list[dict], seed: int, fraction: float = 0.6):
    generator = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []
    for label in (0, 1):
        subset = [r for r in rows if r["label"] == label]
        generator.shuffle(subset)
        cut = int(fraction * len(subset))
        train += subset[:cut]
        test += subset[cut:]
    return train, test


def split_by_domain(rows: list[dict], seed: int, fraction: float = 0.6):
    """no domain in both train and test"""
    generator = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []
    for label in (0, 1):
        domains = sorted({r.get("source_domain", "") for r in rows if r["label"] == label})
        generator.shuffle(domains)
        cut = max(1, int(fraction * len(domains)))
        held = set(domains[:cut])
        for row in rows:
            if row["label"] != label:
                continue
            (train if row.get("source_domain", "") in held else test).append(row)
    return train, test


def size_rule(train: list[dict], test: list[dict]) -> float:
    """one threshold on n_accounts"""
    truth = [r["label"] for r in train]
    sizes = [len(r["nodes"]) for r in train]
    best = (-1.0, 1)
    for threshold in range(1, max(sizes, default=2) + 1):
        score = f1_score(truth, [int(s >= threshold) for s in sizes], average="macro")
        if score > best[0]:
            best = (score, threshold)
    return f1_score([r["label"] for r in test],
                    [int(len(r["nodes"]) >= best[1]) for r in test], average="macro")


def detector(train: list[dict], test: list[dict], seed: int) -> float:
    torch.manual_seed(seed)
    graphs = [cascade_to_pyg(r) for r in train]
    model = BiGCN(graphs[0].x.size(1), 64, 2)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(60):
        model.train()
        for batch in DataLoader(graphs, batch_size=64, shuffle=True):
            optimiser.zero_grad()
            F.cross_entropy(model(batch.x, batch.edge_index, batch.batch),
                            batch.y).backward()
            optimiser.step()

    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for batch in DataLoader([cascade_to_pyg(r) for r in test], batch_size=128):
            predicted += model(batch.x, batch.edge_index, batch.batch).argmax(dim=1).tolist()
            truth += batch.y.tolist()
    return f1_score(truth, predicted, average="macro")


def main() -> None:
    rows = load()
    matched = size_matched(rows)
    print(f"{len(rows)} labelled cascades, {len(matched)} after size matching "
          f"({len(matched) // 2} pairs)\n", flush=True)

    conditions = [
        ("cascade split, all sizes", rows, split_by_cascade),
        ("cascade split, size matched", matched, split_by_cascade),
        ("domain split, all sizes", rows, split_by_domain),
        ("domain split, size matched", matched, split_by_domain),
    ]

    table = {}
    print(f"  {'condition':<30s} {'size rule':>14s} {'Bi-GCN':>15s}", flush=True)
    print("  " + "-" * 53, flush=True)
    for name, data, splitter in conditions:
        rule_scores, model_scores = [], []
        for seed in SEEDS:
            train, test = splitter(data, seed)
            if not train or not test or len({r["label"] for r in train}) < 2:
                continue
            rule_scores.append(size_rule(train, test))
            model_scores.append(detector(train, test, seed))
        if not rule_scores:
            print(f"  {name:<30s} {'n/a':>11s} {'n/a':>10s}", flush=True)
            continue
        table[name] = {
            "size_rule": {"mean": round(st.fmean(rule_scores), 4),
                          "std": round(st.pstdev(rule_scores), 4)},
            "detector": {"mean": round(st.fmean(model_scores), 4),
                         "std": round(st.pstdev(model_scores), 4)},
            "test_cascades": len(test),
        }
        print(f"  {name:<30s} {st.fmean(rule_scores):>7.3f} "
              f"+/-{st.pstdev(rule_scores):<6.3f} "
              f"{st.fmean(model_scores):>7.3f} +/-{st.pstdev(model_scores):<6.3f}",
              flush=True)

    print("\n  chance on a balanced binary task is 0.500", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "confound.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"  written to {path}", flush=True)
    print(DONE, flush=True)


if __name__ == "__main__":
    main()
