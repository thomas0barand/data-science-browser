# Data Science Browser - Scientific Literature Search

Project for building a semantic search engine for scientific publications.

## Project Structure

```
.
├── configs/              # Configuration files
│   └── config.py
├── data/                 # Data files
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── valid.tsv
├── notebooks/            # Jupyter notebooks for exploration
├── outputs/              # Generated outputs
│   ├── models/          # Saved models
│   ├── figures/         # Plots and visualizations
│   └── results/         # Results files
├── scripts/              # Standalone scripts
│   └── run_pipeline.py
├── src/                  # Source code
│   ├── __init__.py
│   ├── data_loader.py   # Data loading
│   ├── eda.py           # Exploratory data analysis
│   ├── preprocessing.py # Data preprocessing
│   ├── training.py      # Training pipeline
│   ├── evaluation.py    # Evaluation metrics
│   ├── utils.py         # Utility functions
│   └── models/          # Model implementations
│       ├── __init__.py
│       ├── base_model.py
│       └── embedding_model.py
├── tests/               # Unit tests
├── consigne.ipynb       # Project instructions
├── pyproject.toml       # Poetry dependencies
└── README.md
```

## Installation

```bash
poetry install
```

## Usage

### Run the full pipeline

```bash
poetry run python scripts/run_pipeline.py
```

### Or use individual modules

```python
from src.data_loader import DataLoader
from src.models.embedding_model import EmbeddingModel
from src.evaluation import Evaluator

# Load data
loader = DataLoader("data")
loader.load_all()

# Train model
model = EmbeddingModel()
model.fit(loader.corpus, loader.queries)

# Evaluate
evaluator = Evaluator(model)
results = evaluator.evaluate(loader.validation)
```

## Project Goal

Build a search engine that finds semantically similar scientific articles. Given a query article, the system ranks candidate articles to retrieve the most relevant citations.