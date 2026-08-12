"""Were the two platforms ever describing accounts in the same units?

The transfer experiment trains on UPFD's ten account counters and tests on the
same ten counters recomputed from Bluesky. That is only a domain-gap
measurement if the two sets of ten numbers mean the same thing. They do not.

Three of the ten counters have no equivalent on the target platform and are
written as zero by the collector. Of the seven that remain, several sit two to
four orders of magnitude away from their benchmark counterparts, because the
collector's ``log1p(x)/15`` compression was written to mirror the benchmark's
normalisation without the benchmark's normalisation ever having been checked.

A model handed inputs that far outside its training marginals will collapse to
one class whatever the domain gap is, so the collapse on its own establishes
nothing. This module separates the two causes. It reports which dimensions are
structurally unavailable, how far the survivors are from the benchmark, and
what the transfer scores once the survivors are quantile-aligned so that only
the genuine difference between the platforms is left to measure.

    uv run python -m factnet.graph.feature_alignment
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD

from factnet.graph.adaptation import COLLECTED, DATA, SEEDS
from factnet.graph.ood_eval import evaluate as ood_evaluate
from factnet.graph.ood_eval import train as ood_train
from factnet.ingestion.to_graph import cascade_to_pyg, read_cascades

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
QUANTILES = np.linspace(0, 1, 257)


def stack(graphs) -> np.ndarray:
    return torch.cat([g.x for g in graphs]).numpy()


def dead_dimensions(matrix: np.ndarray) -> list[int]:
    """Counters the collector never populates, so they carry nothing."""
    return [i for i in range(matrix.shape[1]) if float(np.ptp(matrix[:, i])) == 0.0]


def marginal_gap(source: np.ndarray, target: np.ndarray) -> list[dict]:
    """Per-counter distance between the two platforms, on the median."""
    rows = []
    for i in range(source.shape[1]):
        s, t = float(np.median(source[:, i])), float(np.median(target[:, i]))
        ratio = (max(s, t) / min(s, t)) if min(s, t) > 0 else float("inf")
        rows.append({"dim": i, "upfd": round(s, 8), "collected": round(t, 8),
                     "ratio": None if ratio == float("inf") else round(ratio, 1)})
    return rows


def quantile_align(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Map each target counter onto the benchmark's marginal for that counter.

    Rank-preserving by construction, so the ordering of accounts within a
    counter is untouched and only the scale is moved. Dimensions that are
    constant on either side are left alone, there being no ranks to map.
    """
    out = target.copy()
    for i in range(target.shape[1]):
        if float(np.ptp(target[:, i])) == 0.0 or float(np.ptp(source[:, i])) == 0.0:
            continue
        ranks = target[:, i].argsort().argsort() / max(1, len(target) - 1)
        out[:, i] = np.quantile(source[:, i], QUANTILES)[
            np.clip((ranks * (len(QUANTILES) - 1)).astype(int), 0, len(QUANTILES) - 1)]
    return out


def realign(graphs: list[Data], source: np.ndarray) -> list[Data]:
    """Re-emit the collected cascades with counters on the benchmark's scale."""
    flat = quantile_align(stack(graphs), source)
    out, cursor = [], 0
    for g in graphs:
        n = g.num_nodes
        out.append(Data(x=torch.tensor(flat[cursor:cursor + n], dtype=torch.float),
                        edge_index=g.edge_index, y=g.y))
        cursor += n
    return out


def main() -> None:
    upfd = UPFD(str(DATA / "upfd"), "gossipcop", "profile", split="train")
    collected = [cascade_to_pyg(c) for c in read_cascades(COLLECTED)
                 if c.get("label") is not None]
    U, C = stack(upfd), stack(collected)

    dead = dead_dimensions(C)
    print(f"collected cascades {len(collected)}  |  UPFD train {len(upfd)}\n")
    print(f"1. Counters the collector never populates: {dead} "
          f"({len(dead)} of {C.shape[1]})")
    print("   These have no Bluesky equivalent, so the model receives constants.\n")

    print("2. Marginal distance on the counters that do carry values")
    print(f"   {'dim':>4s} {'UPFD median':>14s} {'collected':>12s} {'ratio':>10s}")
    gaps = marginal_gap(U, C)
    for row in gaps:
        tag = "  (dead)" if row["dim"] in dead else ""
        ratio = "-" if row["ratio"] is None else f"{row['ratio']:.0f}x"
        print(f"   {row['dim']:>4d} {row['upfd']:>14.8f} {row['collected']:>12.6f} "
              f"{ratio:>10s}{tag}")

    print("\n3. Transfer, before and after putting the counters on one scale")
    print("   Trained exactly as the published transfer result, so the first row")
    print("   reproduces it and the second is comparable to it.\n")
    aligned = realign(collected, U)
    raw_scores, fixed_scores = [], []
    for seed in SEEDS:
        model = ood_train(list(upfd), upfd.num_features, seed=seed)
        raw_scores.append(ood_evaluate(model, collected)[1])
        fixed_scores.append(ood_evaluate(model, aligned)[1])
    print(f"   as collected      macro-F1 {st.fmean(raw_scores):.3f} "
          f"(+/- {st.pstdev(raw_scores):.3f})")
    print(f"   quantile-aligned  macro-F1 {st.fmean(fixed_scores):.3f} "
          f"(+/- {st.pstdev(fixed_scores):.3f})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "feature_alignment.json"
    path.write_text(json.dumps({
        "dead_dimensions": dead,
        "marginals": gaps,
        "transfer_raw": {"mean": round(st.fmean(raw_scores), 4),
                         "std": round(st.pstdev(raw_scores), 4),
                         "seeds": [round(s, 4) for s in raw_scores]},
        "transfer_aligned": {"mean": round(st.fmean(fixed_scores), 4),
                             "std": round(st.pstdev(fixed_scores), 4),
                             "seeds": [round(s, 4) for s in fixed_scores]},
    }, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
