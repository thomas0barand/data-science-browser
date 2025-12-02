# Evaluation Guide

## Overview

This guide explains how to evaluate sparse embedding predictions against the validation set (`valid.tsv`).

## Quick Start

### 1. Generate Predictions

First, generate predictions for all queries:

```bash
poetry run python3 scripts/generate_predictions_sparse.py --mode generate
```

This creates: `outputs/results/preds_sparse_embedding_without_corpus_text_decimated.csv`

### 2. Evaluate Predictions

Evaluate against `valid.tsv` with a threshold:

```bash
poetry run python3 scripts/generate_predictions_sparse.py --mode evaluate
```

Default threshold is 0.1 (outputs 1 if cosine similarity > 0.1, else 0).

## Command-Line Options

### Generate Mode

```bash
poetry run python3 scripts/generate_predictions_sparse.py --mode generate
```

Generates predictions for all query-corpus pairs.

### Evaluate Mode

```bash
poetry run python3 scripts/generate_predictions_sparse.py \
    --mode evaluate \
    --predictions-csv outputs/results/preds_sparse_embedding_without_corpus_text_decimated.csv \
    --output outputs/results/evaluation_results.csv \
    --threshold 0.1
```

**Options**:
- `--predictions-csv`: Path to predictions CSV file (default: uses latest generated file)
- `--output`: Output path for evaluation results (default: `outputs/results/evaluation_results.csv`)
- `--threshold`: Threshold for binary classification (default: 0.1)

## Output Format

The evaluation creates a CSV with:

| Column | Description |
|--------|-------------|
| `query_id` | Query document ID |
| `corpus_id` | Corpus document ID |
| `ground_truth` | True label from valid.tsv (0 or 1) |
| `similarity` | Cosine similarity from predictions |
| `prediction` | Binary prediction (1 if similarity > threshold, else 0) |

**Example**:
```csv
query_id,corpus_id,ground_truth,similarity,prediction
query123,doc456,1,0.234567,1
query123,doc789,0,0.045678,0
```

## Evaluation Metrics

The script automatically calculates:

### Confusion Matrix
- **True Positives (TP)**: Correctly predicted relevant documents
- **False Positives (FP)**: Incorrectly predicted as relevant
- **True Negatives (TN)**: Correctly predicted as not relevant
- **False Negatives (FN)**: Incorrectly predicted as not relevant

### Metrics
- **Accuracy**: `(TP + TN) / Total`
- **Precision**: `TP / (TP + FP)` - How many predicted positives are correct
- **Recall**: `TP / (TP + FN)` - How many actual positives were found
- **F1 Score**: `2 * (Precision * Recall) / (Precision + Recall)` - Harmonic mean

## Example Output

```
======================================================================
EVALUATION METRICS
======================================================================

Threshold: 0.1
Total pairs: 20950

Confusion Matrix:
  True Positives:    1327
  False Positives:    347
  True Negatives:   17153
  False Negatives:   2123

Metrics:
  Accuracy:  0.8821
  Precision: 0.7927
  Recall:    0.3846
  F1 Score:  0.5180
======================================================================
```

## Choosing a Threshold

The threshold determines when to classify a query-corpus pair as relevant:
- **Lower threshold** (e.g., 0.05): More predictions → Higher recall, lower precision
- **Higher threshold** (e.g., 0.3): Fewer predictions → Lower recall, higher precision

### Testing Multiple Thresholds

```bash
# Test threshold 0.05
poetry run python3 scripts/generate_predictions_sparse.py \
    --mode evaluate --threshold 0.05 --output outputs/results/eval_0.05.csv

# Test threshold 0.15
poetry run python3 scripts/generate_predictions_sparse.py \
    --mode evaluate --threshold 0.15 --output outputs/results/eval_0.15.csv

# Test threshold 0.2
poetry run python3 scripts/generate_predictions_sparse.py \
    --mode evaluate --threshold 0.2 --output outputs/results/eval_0.2.csv
```

Or use the helper script:

```bash
poetry run python3 scripts/test_thresholds.py
```

## Understanding Results

### High Accuracy, Low Recall
- Model is conservative
- Misses many relevant documents (high FN)
- Predictions are reliable when made (high precision)

### High Recall, Low Precision
- Model is aggressive
- Finds most relevant documents
- Many false positives

### Balanced F1 Score
- Good compromise between precision and recall
- Optimal threshold depends on use case

## Troubleshooting

### Missing Query-Corpus Pairs

If you see:
```
⚠ Missing X query-corpus pairs in predictions
```

This means some pairs in `valid.tsv` don't exist in your predictions CSV. These are treated as prediction=0.

**Possible causes**:
- Incomplete prediction generation (was interrupted)
- Different embeddings used for prediction vs validation
- Queries or corpus docs not in the embeddings

### Low Performance

If metrics are poor:
1. **Check vocabulary size**: Decimation may have removed important terms
2. **Adjust threshold**: Try multiple values
3. **Inspect false negatives**: Look at missed relevant documents
4. **Check embeddings**: Ensure corpus includes both title and text

## Next Steps

1. **Analyze errors**: Look at false positives and false negatives
2. **Tune threshold**: Find optimal value for your use case
3. **Compare methods**: Test with different embedding strategies
4. **Feature engineering**: Try different preprocessing (stemming, stop words, etc.)

## Complete Workflow

```bash
# 1. Generate embeddings (if not done)
poetry run python3 scripts/export_sparse_embeddings.py

# 2. Generate predictions
poetry run python3 scripts/generate_predictions_sparse.py --mode generate

# 3. Evaluate with default threshold
poetry run python3 scripts/generate_predictions_sparse.py --mode evaluate

# 4. Test other thresholds
poetry run python3 scripts/test_thresholds.py

# 5. Analyze results
# Load evaluation_*.csv files for detailed analysis
```

