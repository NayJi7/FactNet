"""With `profile`, what are the 10 numbers on the root (it's a news item, not an account)?

Turns out almost every root vector is identical to some account's vector, i.e.
it's the publisher's account. Not leakage, but the model can learn which
publishers are on which side (same outlet confound as on our Bluesky data).

    uv run python -m factnet.graph.root_identity
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch_geometric.datasets import UPFD

from factnet.graph.train_upfd import ROOT

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
SAMPLE = 300


def inspect(name: str) -> dict:
    graphs = [g for split in ("train", "val", "test")
              for g in UPFD(ROOT, name, "profile", split=split)]
    roots = np.array([g.x[0].numpy() for g in graphs])
    labels = np.array([int(g.y) for g in graphs])
    others = np.concatenate([g.x[1:].numpy() for g in graphs if g.num_nodes > 1])

    distinct = len({tuple(np.round(r, 8)) for r in roots})
    pool = {tuple(np.round(u, 8)) for u in others[:60_000]}
    checked = min(SAMPLE, len(roots))
    matched = sum(1 for r in roots[:checked] if tuple(np.round(r, 8)) in pool)

    return {
        "cascades": len(graphs),
        "distinct_root_vectors": distinct,
        "root_matches_an_account_vector": {"matched": matched, "checked": checked},
        "class_means": {str(c): [round(float(v), 4) for v in roots[labels == c].mean(0)]
                        for c in (0, 1)},
    }


def main() -> None:
    table = {}
    for name in ("politifact", "gossipcop"):
        res = inspect(name)
        table[name] = res
        m = res["root_matches_an_account_vector"]
        print(f"=== {name}: {res['cascades']} cascades")
        print(f"    distinct root vectors            : "
              f"{res['distinct_root_vectors']} of {res['cascades']}")
        print(f"    root equals an account elsewhere : "
              f"{m['matched']} of {m['checked']} sampled")
        for c, label in ((0, "misleading"), (1, "reliable")):
            row = res["class_means"][str(c)]
            print(f"    class mean, {label:10s}         : "
                  + " ".join(f"{v:.4f}" for v in row))

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "root_identity.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
