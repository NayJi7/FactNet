"""Dashboard: browse propagation cascades, their spreaders and their verdicts.

Reads either a benchmark dataset (UPFD) or a file of cascades collected from
Bluesky, draws the selected cascade, ranks the accounts that carry it, and
shows the content credibility score alongside the structural verdict.

    uv run streamlit run src/factnet/viz/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from factnet.ingestion.to_graph import cascade_to_networkx, read_cascades
from factnet.viz.cascade_view import draw, from_pyg, spreader_table

DATA = Path(__file__).resolve().parents[3] / "data" / "raw"
LABELS = {0: "fake", 1: "real"}


@st.cache_resource(show_spinner=False)
def load_benchmark(name: str, split: str):
    from torch_geometric.datasets import UPFD
    return UPFD(str(DATA / "upfd"), name, "profile", split=split)


@st.cache_data(show_spinner=False)
def load_collected(path: str):
    return read_cascades(path)


def cascade_metrics(graph) -> dict[str, int]:
    depths = {}
    stack = [(next(iter(graph.nodes)), 0)]
    seen = set()
    undirected = graph.to_undirected(as_view=True)
    while stack:
        node, depth = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        depths[node] = depth
        stack.extend((n, depth + 1) for n in undirected.neighbors(node) if n not in seen)
    counts: dict[int, int] = {}
    for depth in depths.values():
        counts[depth] = counts.get(depth, 0) + 1
    return {"accounts": graph.number_of_nodes(), "shares": graph.number_of_edges(),
            "depth": max(depths.values(), default=0), "breadth": max(counts.values(), default=0)}


def main() -> None:
    st.set_page_config(page_title="FactNet dashboard", layout="wide")
    st.title("Misinformation propagation")

    source = st.sidebar.radio("Source", ["Benchmark (UPFD)", "Collected (Bluesky)"])

    if source.startswith("Benchmark"):
        name = st.sidebar.selectbox("Dataset", ["politifact", "gossipcop"])
        split = st.sidebar.selectbox("Split", ["test", "train", "val"])
        dataset = load_benchmark(name, split)
        index = st.sidebar.number_input("Cascade", 0, len(dataset) - 1, 0)
        data = dataset[int(index)]
        graph = from_pyg(data)
        verdict = LABELS.get(int(data.y), "unknown")
        headline, credibility = f"{name} / {split} / cascade {int(index)}", None
    else:
        default = str(DATA / "bluesky" / "cascades.jsonl")
        path = st.sidebar.text_input("Collected file", default)
        if not Path(path).exists():
            st.info("No collected file yet. Run the Bluesky collector first.")
            return
        cascades = load_collected(path)
        labels = [f"{i}: @{c['source_handle']} ({len(c['nodes'])} accounts)"
                  for i, c in enumerate(cascades)]
        choice = st.sidebar.selectbox("Cascade", range(len(cascades)),
                                      format_func=lambda i: labels[i])
        cascade = cascades[choice]
        graph = cascade_to_networkx(cascade)
        verdict = LABELS.get(cascade.get("label"), "unlabelled")
        credibility = cascade.get("credibility")
        headline = f"@{cascade['source_handle']}"
        st.caption(cascade.get("text", ""))

    metrics = cascade_metrics(graph)
    columns = st.columns(5)
    columns[0].metric("Accounts", metrics["accounts"])
    columns[1].metric("Shares", metrics["shares"])
    columns[2].metric("Depth", metrics["depth"])
    columns[3].metric("Breadth", metrics["breadth"])
    columns[4].metric("Label", verdict)
    if credibility is not None:
        st.progress(float(credibility), text=f"content credibility {credibility:.2f}")

    left, right = st.columns([3, 2])
    with left:
        st.pyplot(draw(graph, title=headline), use_container_width=True)
    with right:
        st.subheader("Top spreaders")
        st.dataframe(spreader_table(graph), hide_index=True, use_container_width=True)
        st.caption("influence = 0.5 reach + 0.3 PageRank + 0.2 k-core")


if __name__ == "__main__":
    main()
