"""Was the benchmark initialisation obstructive, or just under-trained?

The adaptation result says that starting from the benchmark-trained detector is
worse than starting from nothing, at every budget of target labels, and that
the gap never closes. Read literally it is a strong claim: what the model
learned on Twitter cascades actively prevents it learning Bluesky ones.

The two arms were not given the same chance. Fine-tuning ran for 30 epochs at a
learning rate of 0.002 and training from scratch for 60 at 0.01, so the model
carrying the benchmark's weights also got half the steps at a fifth the step
size. A handicap that size produces an under-trained model whether or not the
initialisation is harmful, and the published table cannot tell the two apart.

Here every arm gets the same epochs on the same cascades with the same seeds,
and the fine-tuned arm is additionally given the best of a small learning-rate
grid, which if anything favours the claim being tested. If the gap survives
that, it is a property of the initialisation. If it closes, it was the
protocol.

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
