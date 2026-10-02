"""Train + save the CPU models the dashboard uses (same configs as the reports).

Transformers come from the Kaggle GPU runs, not from here.

    uv run python -m factnet.serve.build
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from factnet.graph.models import GAT, GCN, BiGCN
from factnet.serve.registry import MODELS

DATA = Path(__file__).resolve().parents[3] / "data" / "raw"
COLLECTED = DATA / "bluesky" / "cascades-by-source.jsonl"
GRAPHS = MODELS / "graph"
SEED = 0


def build_tfidf() -> None:
    """TF-IDF + logreg, same as the baseline in the report."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from factnet.nlp.data import load_liar

    train = load_liar("train")
    # named steps, content.py reads the vectoriser and coefs by name
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(train["text"], train["y"])
    joblib.dump(pipeline, MODELS / "tfidf.joblib")
    print(f"  tfidf.joblib  ({len(train)} training statements)")


def _fit(model, graphs: list[Data], epochs: int = 60, lr: float = 0.01) -> torch.nn.Module:
    torch.manual_seed(SEED)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loader = DataLoader(graphs, batch_size=128, shuffle=True)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimiser.zero_grad()
            F.cross_entropy(model(batch.x, batch.edge_index, batch.batch),
                            batch.y).backward()
            optimiser.step()
    return model


def _save(model: torch.nn.Module, name: str, in_dim: int, architecture: str) -> None:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim,
                "architecture": architecture}, GRAPHS / name)
    print(f"  graph/{name}")


def constant_features(dataset) -> list[Data]:
    """x = ones, structure only."""
    return [Data(x=torch.ones(d.num_nodes, 1), edge_index=d.edge_index, y=d.y)
            for d in dataset]


def build_graph_models() -> None:
    train = UPFD(str(DATA / "upfd"), "politifact", "profile", split="train")
    dim = train.num_features
    graphs = list(train)

    _save(_fit(BiGCN(dim, 64, 2), graphs), "bigcn-upfd-profile.pt", dim, "BiGCN")
    _save(_fit(GCN(dim, 64, 2), graphs), "gcn-upfd-profile.pt", dim, "GCN")
    _save(_fit(GAT(dim, 64, 2), graphs), "gat-upfd-profile.pt", dim, "GAT")

    stripped = constant_features(train)
    _save(_fit(BiGCN(1, 64, 2), stripped), "bigcn-structure.pt", 1, "BiGCN")

    # +1 column for the content score on the root
    from factnet.graph.integration import attach_scores
    scored = attach_scores(train, [0.0] * len(train))
    _save(_fit(BiGCN(dim + 1, 64, 2), scored), "bigcn-upfd-profile-score.pt",
          dim + 1, "BiGCN")

    if COLLECTED.exists():
        from factnet.ingestion.to_graph import cascade_to_pyg, read_cascades
        collected = [cascade_to_pyg(c) for c in read_cascades(COLLECTED)
                     if c.get("label") is not None]
        if collected:
            width = collected[0].x.size(1)
            _save(_fit(BiGCN(width, 64, 2), collected),
                  "bigcn-collected.pt", width, "BiGCN")
            # bluesky version of the scored model
            scored_collected = attach_scores(collected, [0.0] * len(collected))
            _save(_fit(BiGCN(width + 1, 64, 2), scored_collected),
                  "bigcn-collected-score.pt", width + 1, "BiGCN")


@lru_cache(maxsize=8)
def load_graph_model(name: str) -> torch.nn.Module:
    blob = torch.load(GRAPHS / name, weights_only=False)
    cls = {"BiGCN": BiGCN, "GCN": GCN, "GAT": GAT}[blob["architecture"]]
    model = cls(blob["in_dim"], 64, 2)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    # in_dim is saved in the checkpoint (first param shape gives hidden size, not input)
    model.in_dim = blob["in_dim"]
    return model


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    print("content models")
    build_tfidf()
    print("graph models")
    build_graph_models()
    manifest = sorted(p.name for p in GRAPHS.glob("*.pt"))
    (MODELS / "manifest.json").write_text(
        json.dumps({"graph": manifest, "tfidf": "tfidf.joblib"}, indent=2),
        encoding="utf-8")
    print(f"\n{len(manifest)} graph models and the bag-of-words model written to {MODELS}")


if __name__ == "__main__":
    main()
