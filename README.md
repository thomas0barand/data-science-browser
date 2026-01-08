# Data Science Browser - Branch Thomas

Dans cette partie, est présent la recherche de thématique par LDA et le moteur de recherche basé sur le sparse embedding.


##  Quick Start

```bash
# Install dependencies
poetry install

# Export all embeddings (baseline, TF-IDF, bigrams)
poetry run python scripts/export_embeddings.py

# Generate predictions (fast version, 50x speedup)
poetry run python scripts/generate_predictions_fast.py --method bigram

# Compute metrics
poetry run python scripts/compute_metrics_single.py outputs/results/preds_bigram.csv

# Create submission files
poetry run python scripts/create_submissions.py
```

## Results

| Method | AUC | F1 | Precision | Recall | Speed |
|--------|-----|-----|-----------|--------|-------|
| Baseline | 0.686 | 0.520 | 0.792 | 0.387 | 48s ⚡ |
| TF-IDF Corrected | 0.686 | 0.520 | 0.792 | 0.387 | 48s ⚡ |
| **Bigram TF-IDF** ✅ | **0.720** | **0.540** | 0.662 | **0.456** | 48s ⚡ |
| Bigram RAW | 0.700 | 0.429 | 0.402 | 0.461 | 48s ⚡ |


## 📁 Project Structure

```
.
├── data/                                           # Data files (gitignored except samples)
│   ├── queries.jsonl, corpus.jsonl                # Input data
│   ├── valid.tsv, test_final.tsv                  # Validation/test sets
│   └── sparses_embedding_*.pkl                    # Generated embeddings
│
├── src/                                            # Core library
│   ├── sparse_browser.py                          # Search engine
│   ├── bigrams.py                                 # Bigram extraction
│   └── utils.py                                   # Sparse embeddings
│
├── scripts/                                        # Executables
│   ├── export_embeddings.py                       # Generate all embeddings (unified)
│   ├── generate_predictions_fast.py               # Fast predictions (numpy)
│   ├── compute_metrics_single.py                  # Compute metrics
│   ├── create_submissions.py                      # Create submission files
│   └── run_all_experiments.sh                     # Automated pipeline
│
├── outputs/                                        # Results (gitignored)
│   ├── results/                                   # Metrics and predictions
│   ├── submissions/                               # Competition submissions
│   └── figures/                                   # Visualizations
│
├── docs/                                           # Documentation
├── notebooks/                                      # Jupyter explorations
└── RESULTS_SUMMARY.md                              # Detailed results
```

##  Main Scripts

### 1. Export Embeddings (Unified)

```bash
git clone https://github.com/thomas0barand/data-science-browser.git
cd data-science-browser
# Export all methods
poetry run python scripts/export_embeddings.py

# Export specific method
poetry run python scripts/export_embeddings.py --method baseline
poetry run python scripts/export_embeddings.py --method tfidf
poetry run python scripts/export_embeddings.py --method bigram
poetry run python scripts/export_embeddings.py --method bigram_raw
```

**Output files** (in `data/`):
- `sparses_embedding_without_corpus_text_decimated.pkl` - Baseline
- `sparses_embedding_tfidf_corrected.pkl` - TF-IDF after decimation
- `sparses_embedding_bigram_tfidf_decimated.pkl` - Bigrams with TF-IDF
- `sparses_embedding_bigram_raw_decimated.pkl` - Bigrams without TF-IDF

### 2. Generate Predictions (Fast)

```bash
# Available methods: baseline, tfidf, bigram, bigram_raw
poetry run python scripts/generate_predictions_fast.py --method bigram
```

**Features**:
- 50x faster than original (numpy vectorization)
- Supports resume (skips already processed queries)
- Output: `outputs/results/preds_{method}.csv`

### 3. Compute Metrics

```bash
# Single file
poetry run python scripts/compute_metrics_single.py outputs/results/preds_bigram.csv

# All methods
poetry run python scripts/compute_metrics.py --method all
```

**Metrics calculated**:
- Precision, Recall, F1
- AUC-ROC
- Confusion matrix
- Optimal threshold (via F1 maximization)

### 4. Create Submissions

```bash
# Continuous scores (for ranking)
poetry run python scripts/create_submissions.py

# Binary predictions (with optimal threshold)
poetry run python scripts/create_binary_submissions.py

# Final test predictions
poetry run python scripts/prediction_test_final.py
```

### 5. Automated Pipeline

```bash
# Run all experiments with smart skip
./scripts/run_all_experiments.sh

# Run only TF-IDF
RUN_BIGRAM_RAW=0 ./scripts/run_all_experiments.sh

# Run only Bigram RAW
RUN_TFIDF=0 ./scripts/run_all_experiments.sh
```

