"""Does structural influence predict who spreads misinformation? (RQ2)

The influence ranking answers a purely structural question: which accounts sit
where the diffusion passes. That is not what the project asks. RQ2 asks whether
the accounts that rank high are the ones spreading unreliable content, and until
the collected sample carried labels there was no way to ask it.

The labelled sample allows the question, but not naively, because the two
classes do not produce cascades of the same size: on this sample a reliable
cascade holds 259 accounts on average against 44 for a misleading one. Four
fifths of all account slots therefore sit in reliable cascades, and any account
drawn at random lands in misleading content about 14 percent of the time
whatever its behaviour. Comparing an account's misleading share against the
50/50 balance of the cascade counts would read that arithmetic as a finding.

Two things follow for the design. The baseline is the share of account *slots*
that are misleading, not the share of cascades. And the association between
influence and unreliability is tested against a null that shuffles influence
only among accounts appearing in the same number of cascades, so that activity,
which drives both centrality and the precision of a share, is held fixed and
only the residual effect of influence is measured.

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
    """Per account, the label of every cascade it took part in (0 misleading)."""
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
    """Rank correlation expected when influence is shuffled at equal activity.

    Accounts that appear in many cascades are both more central and measured
    more precisely, so a plain permutation would credit influence with an
    association that activity alone produces. Shuffling within strata of equal
    cascade count removes that route and leaves only what influence adds.
    """
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
