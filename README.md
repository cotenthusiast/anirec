> **Note:** Susume is under active development. Models beyond the popularity baseline are not yet implemented.

# anirec / Susume

> A modular anime recommendation engine. Built for scale.

Susume is a modular recommendation system for anime, built on DuckDB for out-of-core processing of large-scale datasets (50M+ ratings). Designed for clean extensibility: swap in any model by subclassing a single interface.

## Project Structure

```
anirec/
├── configs/
│   └── default.yaml              # paths, thresholds, hyperparameters
├── src/
│   └── anirec/
│       ├── config.py              # YAML config loader
│       ├── data/
│       │   ├── prepare.py         # raw CSV → cleaned Parquet
│       │   ├── split.py           # train / val / test splitting
│       │   └── loader.py          # shared Parquet loading helpers
│       ├── models/
│       │   ├── base.py            # abstract Recommender interface
│       │   └── popularity.py      # popularity baselines
│       └── eval/
│           └── metrics.py         # recall@k, ndcg@k
├── scripts/
│   ├── prepare.py                 # CLI: raw data → parquet
│   ├── split.py                   # CLI: parquet → train/val/test
│   ├── evaluate.py                # CLI: run & score a model
│   └── inspect_data.py            # CLI: quick EDA
├── data/                          # gitignored, bring your own
│   ├── raw/                       # drop CSVs here
│   └── processed/                 # parquet outputs land here
├── tests/
├── notebooks/
├── reports/
├── pyproject.toml
└── .gitignore
```

## Data

Susume is built around the [andrewgatchalian/myanimelist-user-ratings](https://www.kaggle.com/datasets/andrewgatchalian/myanimelist-user-ratings) dataset (50M+ ratings). You need to download it yourself before running anything.

```bash
pip install kaggle
kaggle datasets download andrewgatchalian/myanimelist-user-ratings -p data/raw --unzip
```

You will need a Kaggle account and API token. Place your `kaggle.json` credentials file at `~/.kaggle/kaggle.json` before running.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### 1. Prepare data

Drop your raw CSVs into `data/raw/`, then:

```bash
python scripts/prepare.py
```

Column names are auto-detected across common anime dataset formats (MAL, Kaggle, etc.).

### 2. Split into train / val / test

```bash
python scripts/split.py
```

Uses deterministic leave-one-out: one held-out item per user for test, one for val, rest for training.

### 3. Evaluate a model

```bash
python scripts/evaluate.py --model popularity_filtered --k 10
```

### 4. Quick data inspection

```bash
python scripts/inspect_data.py
```

## Adding a New Model

1. Create `src/anirec/models/your_model.py`
2. Subclass `anirec.models.base.Recommender`
3. Implement `fit(train_path, **kwargs)` and `recommend(user_ids, k)`
4. Register it in `scripts/evaluate.py`

## Configuration

All paths, thresholds, and hyperparameters live in `configs/default.yaml`. Scripts read from it by default, or pass `--config path/to/custom.yaml`.

## Future Directions

- **Jellyfin plugin:** a native C# plugin that surfaces Susume recommendations directly inside Jellyfin, with one-click requesting via Jellyseerr
- **Browser extension:** if the Jellyfin integration gets traction, a lightweight extension for surfacing recommendations on MAL, AniList, and similar sites
- **MAL API scraper:** scrape recent anime and user ratings via the MAL API to supplement the dataset and keep the model current