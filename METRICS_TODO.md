# TODO: Generate Metrics for All Methods

## Current Status

✅ **Baseline (decimated)** - DONE
- Predictions: `preds_sparse_embedding_without_corpus_text_decimated.csv`
- Metrics: Available in `metrics_preds_sparse_embedding_without_corpus_text_decimated.json`

✅ **TF-IDF** - Predictions generated
- Predictions: `preds_tfidf.csv` ✓
- Embeddings: `sparses_embedding_tfidf.pkl` ✓
- Metrics: ❌ NOT YET COMPUTED

⚠️ **Bigram** - Need predictions
- Embeddings: `sparses_embedding_bigram_tfidf.pkl` ✓
- Predictions: ❌ NOT YET GENERATED
- Metrics: ❌ NOT YET COMPUTED

---

## What You Need to Do

### Step 1: Generate Bigram Predictions

The bigram embeddings exist but predictions haven't been generated yet.

```bash
cd /Users/thomas/study/data-science-browser
poetry run python scripts/generate_predictions.py --method bigram
```

**Time estimate**: ~5-15 minutes depending on your data size

This will create: `outputs/results/preds_bigram.csv`

### Step 2: Compute Metrics for All Methods

Once bigram predictions are generated, compute metrics for all three methods:

```bash
# Compute metrics for all methods at once
poetry run python scripts/compute_metrics.py --method all

# Or compute individually:
poetry run python scripts/compute_metrics.py --method baseline
poetry run python scripts/compute_metrics.py --method tfidf
poetry run python scripts/compute_metrics.py --method bigram
```

**Time estimate**: ~2-5 minutes per method

This will:
- Find optimal threshold for each method (maximizing F1-score)
- Calculate: Precision, Recall, F1-Score, AUC
- Save summary to: `outputs/results/metrics_summary.json`
- Display comparison table

### Step 3: Review Results

After running, you'll see output like:

```
======================================================================
COMPARISON TABLE
======================================================================
Method               Precision    Recall       F1           AUC         
----------------------------------------------------------------------
Baseline             0.8305       0.3606       0.5023       0.6855      
TF-IDF               0.8421       0.4123       0.5512       0.7234      
Bigram TF-IDF        0.8567       0.4456       0.5823       0.7456      
======================================================================
```

Results saved to: `outputs/results/metrics_summary.json`

---

## Required Metrics (from consigne.ipynb)

The script will compute all required metrics:

1. ✅ **Precision** (précision)
2. ✅ **Recall** (rappel)  
3. ✅ **F1-Score** (f-mesure)
4. ✅ **AUC** (Area Under ROC Curve)

Plus additional metrics:
- Accuracy
- Confusion matrix (TP, FP, TN, FN)
- Optimal threshold

---

## Quick Commands Summary

```bash
# 1. Generate bigram predictions
poetry run python scripts/generate_predictions.py --method bigram

# 2. Compute all metrics
poetry run python scripts/compute_metrics.py --method all

# 3. View results
cat outputs/results/metrics_summary.json
```

---

## Troubleshooting

### If embeddings are missing:

```bash
# Generate sparse embeddings (baseline + tfidf)
poetry run python scripts/export_sparse_embeddings.py

# Generate bigram embeddings
poetry run python scripts/export_bigram_embeddings.py
```

### If you want to use a specific threshold:

```bash
# Use threshold 0.15 for all methods
poetry run python scripts/compute_metrics.py --method all --threshold 0.15
```

### Check what files exist:

```bash
# Check predictions
ls -lh outputs/results/preds_*.csv

# Check embeddings
ls -lh data/sparses_embedding*.pkl

# Check metrics
ls -lh outputs/results/metrics*.json
```

---

## Next Steps After Metrics

Once you have metrics for all three methods:

1. Compare the methods and choose the best one
2. Generate final predictions for test set:
   ```bash
   poetry run python scripts/create_submissions.py
   ```
3. Create visualizations:
   ```bash
   poetry run python scripts/visualizations.py
   ```

