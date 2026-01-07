# Scripts Guide

## Prediction Generation

The main script for generating predictions is now unified: `generate_predictions.py`

### Usage

#### Generate predictions

```bash
# Baseline method (raw frequencies, decimated)
poetry run python scripts/generate_predictions.py --method baseline

# TF-IDF method
poetry run python scripts/generate_predictions.py --method tfidf

# Bigram TF-IDF method
poetry run python scripts/generate_predictions.py --method bigram
```

#### Evaluate predictions

```bash
# Evaluate with default threshold (0.1)
poetry run python scripts/generate_predictions.py --mode evaluate --method tfidf

# Evaluate with custom threshold
poetry run python scripts/generate_predictions.py --mode evaluate --method baseline --threshold 0.15

# Custom output path
poetry run python scripts/generate_predictions.py --mode evaluate --method bigram --output my_results.csv
```

### Test multiple thresholds

```bash
# Test 100 thresholds (0.00 to 0.99) for baseline
poetry run python scripts/test_thresholds.py --method baseline

# Test thresholds for TF-IDF
poetry run python scripts/test_thresholds.py --method tfidf

# Test thresholds for bigrams
poetry run python scripts/test_thresholds.py --method bigram
```

## Embeddings Export

### Sparse embeddings (baseline and TF-IDF)

```bash
poetry run python scripts/export_sparse_embeddings.py
```

This script generates both baseline and TF-IDF embeddings with decimation options.

### Bigram embeddings

```bash
poetry run python scripts/export_bigram_embeddings.py
```

## Submissions

### Continuous scores (for ranking)

```bash
poetry run python scripts/create_submissions.py
```

Generates submission files with continuous similarity scores for all methods.

### Binary predictions (threshold-based)

```bash
poetry run python scripts/create_binary_submissions.py
```

Generates binary predictions using optimal ROC thresholds for all methods.

### Final test predictions

```bash
poetry run python scripts/prediction_test_final.py
```

## Visualizations

```bash
poetry run python scripts/visualizations.py
```

Generates:
- Vocabulary distribution before/after decimation
- Query similarity example

## Output Structure

```
outputs/
├── results/
│   ├── preds_baseline.csv       # Baseline predictions
│   ├── preds_tfidf.csv          # TF-IDF predictions
│   ├── preds_bigram.csv         # Bigram predictions
│   ├── evaluation_*.csv         # Evaluation results
│   └── results_summary_*.json   # Threshold test results
├── submissions/
│   ├── submission_*.csv         # Competition submissions
│   └── test_final_predictions.csv
└── figures/
    ├── vocab_distribution_*.png
    └── query_similarity_example.png
```

## Methods Configuration

The methods are configured in `generate_predictions.py`:

| Method   | Input File | Output File | Description |
|----------|-----------|-------------|-------------|
| baseline | `sparses_embedding_without_corpus_text_decimated.pkl` | `preds_baseline.csv` | Raw frequencies with decimation |
| tfidf    | `sparses_embedding_tfidf.pkl` | `preds_tfidf.csv` | TF-IDF weighted embeddings |
| bigram   | `sparses_embedding_bigram_tfidf.pkl` | `preds_bigram.csv` | Bigram with TF-IDF |

## Workflow

1. **Export embeddings** (if not already done):
   ```bash
   poetry run python scripts/export_sparse_embeddings.py
   poetry run python scripts/export_bigram_embeddings.py
   ```

2. **Generate predictions**:
   ```bash
   poetry run python scripts/generate_predictions.py --method tfidf
   ```

3. **Evaluate and find optimal threshold** (optional):
   ```bash
   poetry run python scripts/test_thresholds.py --method tfidf
   ```

4. **Create submissions**:
   ```bash
   poetry run python scripts/create_submissions.py
   poetry run python scripts/create_binary_submissions.py
   ```

5. **Generate visualizations** (optional):
   ```bash
   poetry run python scripts/visualizations.py
   ```

