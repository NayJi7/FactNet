"""Control for adaptation.py: fine-tune vs scratch with the same training budget.

In adaptation.py fine-tuning got 30 epochs at lr 0.002 and scratch 60 at 0.01,
so fine-tuning was just under-trained maybe. Here same epochs and seeds, and
fine-tuning gets a small lr grid on top.

    uv run python -m factnet.graph.adaptation_control
"""

from __future__ import annotations

import copy
import json
import statistics as st
from pathlib import Path

from torch_geometric.datasets import UPFD

from factnet.graph.adaptation import DATA, SEEDS, collected, fit, score, split

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
BUDGETS = (0.1, 0.25, 0.5)
GRID = (0.01, 0.005, 0.002)
EPOCHS = 60


def main() -> None:
    graphs = collected()
    upfd = UPFD(str(DATA / "upfd"), "gossipcop", "profile", split="train")
    in_dim = upfd.num_features
    print(f"collected {len(graphs)} cascades | UPFD train {len(upfd)} "
          f"| {EPOCHS} epochs for every arm\n")

    base = {seed: fit(list(upfd), in_dim, epochs=EPOCHS, seed=seed) for seed in SEEDS}

    rows = []
    header = (f"{'cascades':>9s} {'scratch':>16s} {'published FT':>16s} "
              f"{'matched FT':>16s} {'best-lr FT':>16s}")
    print(header)
    print("-" * len(header))

    for fraction in BUDGETS:
        arms: dict[str, list[float]] = {k: [] for k in
                                        ("scratch", "published", "matched", "best")}
        for seed in SEEDS:
            train, test = split(graphs, 0.5, seed)
            subset, _ = split(train, fraction / 0.5, seed)

            arms["scratch"].append(
                score(fit(subset, in_dim, epochs=EPOCHS, seed=seed), test))
            arms["published"].append(
                score(fit(subset, in_dim, epochs=30, seed=seed,
                          start_from=copy.deepcopy(base[seed]), lr=0.002), test))
            arms["matched"].append(
                score(fit(subset, in_dim, epochs=EPOCHS, seed=seed,
                          start_from=copy.deepcopy(base[seed]), lr=0.01), test))
            arms["best"].append(max(
                score(fit(subset, in_dim, epochs=EPOCHS, seed=seed,
                          start_from=copy.deepcopy(base[seed]), lr=lr), test)
                for lr in GRID))

        n = int(fraction * len(graphs))
        cells = {k: (st.fmean(v), st.pstdev(v)) for k, v in arms.items()}
        rows.append({"cascades": n,
                     **{k: {"mean": round(m, 4), "std": round(s, 4)}
                        for k, (m, s) in cells.items()}})
        print(f"{n:>9d} " + " ".join(
            f"{cells[k][0]:.3f} ±{cells[k][1]:.3f}".rjust(16)
            for k in ("scratch", "published", "matched", "best")))

    print("\nIf 'best-lr FT' reaches 'scratch', the published gap was the protocol.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "adaptation_control.json"
    path.write_text(json.dumps({"epochs": EPOCHS, "grid": list(GRID),
                                "rows": rows}, indent=2) + "\n")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
