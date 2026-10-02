"""influence_veracity, but also controlling for cascade size.

Reliable cascades are big, so being in one gives you more edges (higher reach /
pagerank) and a lower misleading share, with no behaviour involved. Stratifying
on activity alone doesn't fix that, so here the null shuffles within
(activity, cascade size) strata.

    uv run python -m factnet.graph.influence_size_control
"""

from __future__ import annotations

import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

from factnet.graph.influence import build_graph_from_collected, influence_ranking
from factnet.graph.influence_veracity import COLLECTED, PERMUTATIONS, SEED, misleading_share
from factnet.ingestion.to_graph import read_cascades

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
SIZE_BUCKETS = 4


def account_context(path) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """{account: labels}, {account: cascade sizes}"""
    labels: dict[str, list[int]] = defaultdict(list)
    sizes: dict[str, list[int]] = defaultdict(list)
    for cascade in read_cascades(path):
        label = cascade.get("label")
        if label is None:
            continue
        size = len(cascade["nodes"])
        for node in cascade["nodes"]:
            labels[node["did"]].append(int(label))
            sizes[node["did"]].append(size)
    return dict(labels), dict(sizes)


def _buckets(values: list[float], count: int) -> list[int]:
    """quantile buckets"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0] * len(values)
    for position, index in enumerate(order):
        out[index] = min(count - 1, position * count // len(values))
    return out


def double_stratified_null(influences, shares, strata, rounds=PERMUTATIONS,
                           seed=SEED) -> tuple[float, float, float]:
    """-> (rho, p, mean null rho)"""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[stratum].append(index)

    rng = random.Random(seed)
    observed, _ = spearmanr(influences, shares)
    hits, null_rhos = 0, []
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
    return observed, (hits + 1) / (rounds + 1), st.fmean(null_rhos)


def within_strata_rho(influences, shares, strata) -> float:
    """Spearman after centring each stratum on its mean rank.

    Needed because lots of strata have a single account, and the permutation null
    can't shuffle those, so it isn't centred on 0 anymore.
    """
    from scipy.stats import rankdata

    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[stratum].append(index)

    ri, rs = rankdata(influences), rankdata(shares)
    ci, cs = [], []
    for members in groups.values():
        if len(members) < 2:
            continue
        mi = st.fmean(ri[m] for m in members)
        ms = st.fmean(rs[m] for m in members)
        ci += [ri[m] - mi for m in members]
        cs += [rs[m] - ms for m in members]
    return float(spearmanr(ci, cs).statistic)


def main() -> None:
    graph = build_graph_from_collected(COLLECTED)
    labels, sizes = account_context(COLLECTED)
    ranked, *_ = influence_ranking(graph, top_k=graph.number_of_nodes())
    score = dict(ranked)

    multi = [a for a in labels if len(labels[a]) > 1 and a in score]
    influences = [score[a] for a in multi]
    shares = [misleading_share(labels[a]) for a in multi]
    activity = [len(labels[a]) for a in multi]
    mean_size = [st.fmean(sizes[a]) for a in multi]

    print(f"accounts in more than one cascade: {len(multi)}\n")

    print("1. Does the route exist?")
    rho_is, p_is = spearmanr(influences, mean_size)
    rho_sm, p_sm = spearmanr(mean_size, shares)
    print(f"   influence  vs mean cascade size : rho {rho_is:+.3f} (p {p_is:.3g})")
    print(f"   mean size  vs misleading share  : rho {rho_sm:+.3f} (p {p_sm:.3g})")
    print("   If both are strong, size alone can manufacture the association.\n")

    print("2. The association, under three nulls")
    rho_plain, p_plain = spearmanr(influences, shares)
    print(f"   uncontrolled                    : rho {rho_plain:+.3f} (p {p_plain:.3g})")

    activity_only = [(a,) for a in activity]
    _, p_act, null_act = double_stratified_null(influences, shares, activity_only)
    print(f"   shuffled at equal activity      : p {p_act:.4f} "
          f"(null mean rho {null_act:+.3f})")

    size_bucket = _buckets(mean_size, SIZE_BUCKETS)
    both = list(zip(activity, size_bucket, strict=True))
    _, p_both, null_both = double_stratified_null(influences, shares, both)
    print(f"   equal activity and cascade size : p {p_both:.4f} "
          f"(null mean rho {null_both:+.3f})")

    print("\n3. Why the null is not centred on zero")
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(both):
        groups[s].append(i)
    singleton = sum(len(m) for m in groups.values() if len(m) == 1)
    print(f"   strata: {len(groups)}, of which {sum(1 for m in groups.values() if len(m) == 1)}"
          f" hold one account")
    print(f"   accounts a shuffle cannot move: {singleton} of {len(multi)} "
          f"({100 * singleton / len(multi):.1f}\\%)")
    print("   A shuffle inside a stratum of one leaves that account where it was,")
    print("   so those accounts keep their contribution to rho under the null.")

    within = within_strata_rho(influences, shares, both)
    print("\n4. The effect measured inside the strata directly")
    print(f"   rho on influence and share, both centred within their stratum: "
          f"{within:+.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "influence_size_control.json"
    path.write_text(json.dumps({
        "accounts": len(multi),
        "influence_vs_size": {"rho": round(rho_is, 4), "p": float(f"{p_is:.4g}")},
        "size_vs_share": {"rho": round(rho_sm, 4), "p": float(f"{p_sm:.4g}")},
        "uncontrolled": {"rho": round(rho_plain, 4), "p": float(f"{p_plain:.4g}")},
        "activity_null": {"p": round(p_act, 4)},
        "activity_and_size_null": {"p": round(p_both, 4),
                                   "null_mean_rho": round(null_both, 4)},
        "within_strata_rho": round(within, 4),
        "accounts_in_singleton_strata": singleton,
    }, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
