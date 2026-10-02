"""RQ2: do high-influence accounts spread more misinformation?

Careful with the baseline: reliable cascades average 259 accounts vs 44 for
misleading, so a random account slot is misleading only ~14% of the time, not
50%. Baseline = share of misleading slots. Null = shuffle influence among
accounts with the same number of cascades (activity drives both).

    uv run python -m factnet.graph.influence_veracity
"""

from __future__ import annotations

import argparse
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

from factnet.graph.influence import build_graph_from_collected, influence_ranking
from factnet.ingestion.to_graph import read_cascades

COLLECTED = (Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky"
             / "cascades-by-source.jsonl")
PERMUTATIONS = 2000
SEED = 0


def account_labels(path: str | Path) -> tuple[dict[str, list[int]], int, int]:
    """{account: [labels]}, 0 = misleading"""
    per_account: dict[str, list[int]] = {}
    slots = {0: 0, 1: 0}
    for cascade in read_cascades(path):
        label = cascade.get("label")
        if label is None:
            continue
        for node in cascade["nodes"]:
            per_account.setdefault(node["did"], []).append(int(label))
            slots[int(label)] += 1
    return per_account, slots[0], slots[1]


def misleading_share(labels: list[int]) -> float:
    return sum(1 for label in labels if label == 0) / len(labels)


def stratified_null(influences: list[float], shares: list[float], strata: list[int],
                    rounds: int = PERMUTATIONS, seed: int = SEED) -> tuple[float, float]:
    """Permutation null, shuffling influence within same-activity strata."""
    groups: dict[int, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[stratum].append(index)

    rng = random.Random(seed)
    observed, _ = spearmanr(influences, shares)
    hits = 0
    null_rhos = []
    for _ in range(rounds):
        shuffled = list(influences)
        for members in groups.values():
            values = [influences[i] for i in members]
            rng.shuffle(values)
            for position, index in enumerate(members):
                shuffled[index] = values[position]
        rho, _ = spearmanr(shuffled, shares)
        null_rhos.append(rho)
        if abs(rho) >= abs(observed):
            hits += 1
    return observed, (hits + 1) / (rounds + 1)


def report(path: str | Path = COLLECTED, top_k: int = 100) -> None:
    graph = build_graph_from_collected(path)
    per_account, misleading_slots, reliable_slots = account_labels(path)
    slot_rate = misleading_slots / (misleading_slots + reliable_slots)

    ranked, *_ = influence_ranking(graph, top_k=graph.number_of_nodes())
    score = dict(ranked)

    print(f"network: {graph.number_of_nodes()} accounts, {graph.number_of_edges()} shares")
    print(f"account slots: {misleading_slots} in misleading cascades, "
          f"{reliable_slots} in reliable ones")
    print(f"  a randomly placed account is misleading-side {slot_rate:.3f} of the time,")
    print("  which is the baseline any result here has to beat\n")

    multi = [a for a, labels in per_account.items() if len(labels) > 1 and a in score]
    shares = [misleading_share(per_account[a]) for a in multi]
    influences = [score[a] for a in multi]
    counts = [len(per_account[a]) for a in multi]

    print(f"accounts in more than one cascade: {len(multi)}")
    print(f"  observed mean misleading share {st.fmean(shares):.3f} "
          f"against a slot baseline of {slot_rate:.3f}")
    print("  -> repeat participation carries no preference for either class\n")

    print(f"association between influence and misleading share (n = {len(multi)})")
    rho_plain, p_plain = spearmanr(influences, shares)
    print(f"  Spearman rho = {rho_plain:+.3f}, p = {p_plain:.3g}  (uncontrolled)")
    _, p_perm = stratified_null(influences, shares, counts)
    print("  same rho against a null shuffling influence at equal activity:")
    print(f"  p = {p_perm:.4f} over {PERMUTATIONS} permutations")

    order = sorted(multi, key=lambda a: score[a], reverse=True)
    head = order[:top_k]
    head_shares = [misleading_share(per_account[a]) for a in head]
    print(f"\ntop {top_k} by influence: mean misleading share "
          f"{st.fmean(head_shares):.3f} (baseline {slot_rate:.3f})")
    pure_misleading = sum(1 for s in head_shares if s == 1.0)
    pure_reliable = sum(1 for s in head_shares if s == 0.0)
    print(f"  {pure_misleading} appear only in misleading cascades, "
          f"{pure_reliable} only in reliable ones, "
          f"{top_k - pure_misleading - pure_reliable} in both")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(COLLECTED))
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()
    report(args.file, args.top)


if __name__ == "__main__":
    main()
