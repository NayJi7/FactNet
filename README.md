# UM-FactNet

Social media misinformation detection. The system reads social media posts and flags
misleading content, models how misinformation spreads across a user network, and ranks
accounts by how much they drive that spread.

Internship project at Universiti Malaya (Faculty of Computer Science and Information
Technology), summer 2026. Built by a team of four: Adam Terrak, Abdelah El Harsal,
Louaye Saghir, Ayman Ouguerd. Supervisors: Dr. Suraya Hamid and Dr. Norjihan Abdul Ghani.

The full proposal is in `docs/proposal/`.

## What it does

The project has two layers:

- A reading layer (NLP): a text classifier that decides whether a post is reliable or
  misleading, based on a fine-tuned transformer.
- A network layer (graph): a model of how a post propagates through the network, plus a
  score that ranks accounts by their influence on that propagation.

Scope details (target platform, language, type of misinformation, dataset) are being
settled with the supervisors.

## Layout

```
src/factnet/
  ingestion/   data loading and collection (public datasets + real scraping)
  nlp/         text classification model
  graph/       propagation graph model
  scoring/     account influence scoring
  viz/         results visualisation
data/          raw / interim / processed (contents are not versioned)
models/        trained model artifacts (not versioned)
notebooks/     exploration and experiments
tests/
docs/          proposal, literature review
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/). Install it, then from the repo root:

```
uv sync
```

This creates a virtual environment and installs the dev tools. Run things inside it with
`uv run`:

```
uv run pytest
uv run ruff check
```

Heavy ML libraries (PyTorch, Transformers, PyTorch Geometric) are added when we reach the
phase that needs them. On a machine without a GPU, training runs on Google Colab or Kaggle.

## Working together

- `main` is the integration branch; keep it working.
- One branch per task, open a pull request, get a quick review before merging.
- Clear notebook outputs before committing.
- Datasets and model weights stay out of git; share them another way.
