"""How much does the influence ranking owe to the weights we chose?

The score combines reach, PageRank and k-core position under weights that are a
stated design choice and are not fitted. An unfitted weight is a free parameter,
and a reader is entitled to ask whether the ranking would survive a different
one. The article claims it does. This measures the claim rather than asserting
it, by rebuilding the score on the collected network under seven alternative
weightings and comparing each ranking against the reference.

Two statistics, because they fail differently. Spearman correlation says whether
the whole ordering moves. The overlap of the top hundred says whether the part
anyone would act on moves, which a global correlation can hide.

    uv run python -m factnet.graph.influence_weights
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from scipy.stats import spearmanr

from factnet.graph.influence import _normalise, build_graph_from_collected
from factnet.graph.influence_veracity import COLLECTED

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
REFERENCE = (0.5, 0.3, 0.2)
ALTERNATIVES = [
    (0.7, 0.2, 0.1), (0.6, 0.2, 0.2), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3),
    (0.34, 0.33, 0.33), (0.3, 0.5, 0.2), (0.2, 0.3, 0.5),
]
TOP_K = 100


def components(graph):
    reach = _normalise({n: len(nx.descendants(graph, n)) for n in graph})
    pagerank = _normalise(nx.pagerank(graph))
    core = _normalise(nx.core_number(nx.Graph(graph)))
    return reach, pagerank, core


def score(parts, weights) -> dict:
    reach, pagerank, core = parts
    a, b, c = weights
    return {n: a * reach[n] + b * pagerank[n] + c * core[n] for n in reach}


def main() -> None:
    graph = build_graph_from_collected(COLLECTED)
    parts = components(graph)
    nodes = list(parts[0])
    reference = score(parts, REFERENCE)
    top = set(sorted(reference, key=reference.get, reverse=True)[:TOP_K])

    print(f"network: {graph.number_of_nodes()} accounts, "
          f"reference weights {REFERENCE}\n")
    print(f"{'weights':>22s} {'Spearman':>10s} {'top-100 shared':>16s}")
    rows = []
    for weights in ALTERNATIVES:
        other = score(parts, weights)
        rho = float(spearmanr([reference[n] for n in nodes],
                              [other[n] for n in nodes]).statistic)
        shared = len(top & set(sorted(other, key=other.get, reverse=True)[:TOP_K]))
        rows.append({"weights": list(weights), "spearman": round(rho, 5),
                     "top_k_shared": shared})
        print(f"{str(weights):>22s} {rho:>10.5f} {shared:>16d}")

    worst_rho = min(r["spearman"] for r in rows)
    least_shared = min(r["top_k_shared"] for r in rows)
    most_shared = max(r["top_k_shared"] for r in rows)
    print(f"\nworst correlation {worst_rho:.5f}, "
          f"top-100 overlap between {least_shared} and {most_shared}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "influence_weights.json"
    path.write_text(json.dumps({"reference": list(REFERENCE), "top_k": TOP_K,
                                "rows": rows, "worst_spearman": worst_rho,
                                "top_k_shared_range": [least_shared, most_shared]},
                               indent=2) + "\n")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
