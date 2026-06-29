"""LIAR dataset loading (content-based verdict).

Downloads the LIAR dataset on first use if it is missing, then loads the TSVs and
maps the 6-way veracity labels to a binary reliable / misleading target, aligned
with the project framing.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

LIAR_COLS = [
    "id", "label", "statement", "subject", "speaker", "job", "state", "party",
    "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context",
]
# misleading (0) vs reliable (1)
MISLEADING = {"pants-fire", "false", "barely-true"}
ROOT = "data/raw/liar"
LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"


def _ensure_liar(root: str = ROOT) -> None:
    """Download and extract the LIAR dataset if it is not already present."""
    path = Path(root)
    if (path / "train.tsv").exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LIAR_URL) as response:  # noqa: S310 (trusted URL)
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(path)


def load_liar(split: str = "train", binary: bool = True) -> pd.DataFrame:
    _ensure_liar()
    df = pd.read_csv(f"{ROOT}/{split}.tsv", sep="\t", header=None, names=LIAR_COLS)
    df = df[["statement", "subject", "label"]].dropna(subset=["statement"]).copy()
    df = df.rename(columns={"statement": "text"})
    if binary:
        df["y"] = (~df["label"].isin(MISLEADING)).astype(int)  # 1 = reliable
    else:
        df["y"] = df["label"]
    return df
