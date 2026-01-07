# Fix Bigram Speed Issue

## Problem

Bigram predictions are **extremely slow** (~178 seconds per query = 49 hours total!)

**Root cause**: Bigram vocabulary is **144,602 terms** (vs 3,000 for baseline)
- File size: 7.2 GB (vs 137 MB for baseline)
- Each similarity calculation is 48x slower

## Solution

Decimate the bigram embeddings to reduce vocabulary size, just like we did for baseline.

## Steps

### 1. Stop the current process

Press `Ctrl+C` in the terminal running bigram predictions

### 2. Decimate bigram embeddings

```bash
cd /Users/thomas/study/data-science-browser
poetry run python scripts/decimate_bigram.py
```

This will:
- Keep top 5,000 most frequent bigrams (vs 144,602)
- Remove low-frequency bigrams (< 2 occurrences)
- Remove common stop-word bigrams
- Create: `data/sparses_embedding_bigram_tfidf_decimated.pkl`
- Reduce file size from 7.2 GB to ~200-300 MB

**Time**: ~2-5 minutes

### 3. Generate predictions with decimated bigrams

The script `generate_predictions.py` has been updated to use the decimated version.

```bash
poetry run python scripts/generate_predictions.py --method bigram
```

**Expected time**: ~5-15 minutes (vs 49 hours!)

### 4. Compute metrics

```bash
poetry run python scripts/compute_metrics.py --method all
```

## What Changed

**Before**:
- Bigram vocab: 144,602 terms
- File size: 7.2 GB
- Prediction time: ~49 hours

**After**:
- Bigram vocab: ~5,000 terms
- File size: ~200-300 MB
- Prediction time: ~5-15 minutes

## Why This Works

- **Bigrams are sparse**: Most bigrams appear very rarely
- **Top bigrams matter most**: The 5,000 most frequent bigrams capture most of the signal
- **Quality maintained**: Decimation removes noise, often improving results
- **Same approach as baseline**: We already did this successfully for baseline

## Quick Commands

```bash
# 1. Decimate bigrams
poetry run python scripts/decimate_bigram.py

# 2. Generate predictions (much faster now!)
poetry run python scripts/generate_predictions.py --method bigram

# 3. Compute all metrics
poetry run python scripts/compute_metrics.py --method all
```

## Alternative: Skip Bigrams

If you don't want to wait or bigrams aren't important:

```bash
# Just compute metrics for baseline and tfidf
poetry run python scripts/compute_metrics.py --method baseline
poetry run python scripts/compute_metrics.py --method tfidf
```

You'll still have metrics for 2 methods, which may be sufficient for your analysis.

