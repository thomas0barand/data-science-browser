# Results Summary

## AUC-ROC Scores on Validation Set

| Method | AUC-ROC | Optimal Threshold | TPR (Recall) | FPR | Predicted 1 (%) |
|--------|---------|-------------------|--------------|-----|-----------------|
| Raw Frequencies | **0.6858** | 0.0716 | 0.3878 | 0.0204 | 8.5% |
| TF-IDF | **0.7139** | 0.0534 | 0.4588 | 0.0446 | 11.3% |
| **Bigrams + TF-IDF** | **0.7260** ✅ | 0.0081 | 0.4991 | 0.0677 | 14.0% |

**Best Method**: Bigrams + TF-IDF (AUC = 0.7260, +5.9% vs baseline)

### ROC Curve Visualization

![ROC Curve](outputs/figures/roc_curve_bigram.png)

The optimal threshold (red dot) is determined using **Youden's J statistic** (TPR - FPR), maximizing the trade-off between true positive rate and false positive rate.

## Submission Files

### Continuous Scores (for ranking evaluation)
- `submission_raw.csv` - Raw frequency embeddings
- `submission_tfidf.csv` - TF-IDF weighted embeddings  
- `submission_bigram.csv` - Bigrams with TF-IDF

### Binary Predictions (0 or 1, using optimal ROC threshold)
- `submission_raw_binary.csv` - Threshold: 0.0716
- `submission_tfidf_binary.csv` - Threshold: 0.0534
- `submission_bigram_binary.csv` - Threshold: 0.0081 ✅

All files contain 8,978 predictions (300 queries × ~30 candidates each).

## Why Bigrams + TF-IDF Works Best

1. **Captures multi-word terms**: "neural network", "machine learning", "deep learning"
2. **Better semantic matching**: Preserves phrase structure lost in unigrams
3. **More discriminative**: Vocabulary combines unigrams + bigrams (144K terms)
4. **TF-IDF weighting**: Reduces impact of common terms, emphasizes rare discriminative features

## Implementation Files

### Core Code
- `src/bigrams.py` - Bigram extraction and TF-IDF computation
- `scripts/export_bigram_embeddings.py` - Generate bigram embeddings
- `scripts/create_submissions.py` - Create continuous score submissions
- `scripts/create_binary_submissions.py` - Create binary predictions with ROC threshold

### Data
- `data/sparses_embedding_bigram_tfidf.pkl` - Bigram embeddings (7.2 GB)
- `outputs/figures/roc_curve_bigram.png` - ROC curve visualization

### Documentation
- `docs/bigrams_implementation.md` - Implementation guide

## Usage

```bash
# 1. Generate bigram embeddings
poetry run python scripts/export_bigram_embeddings.py

# 2. Create continuous score submissions (for AUC-ROC evaluation)
poetry run python scripts/create_submissions.py

# 3. Create binary submissions (0/1 predictions with optimal ROC threshold)
poetry run python scripts/create_binary_submissions.py
```

