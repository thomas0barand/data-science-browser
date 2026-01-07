# Bigrams Implementation

## Principle

**Bigrams** capture sequences of two consecutive words, preserving multi-word expressions that unigrams lose.

### Example
```
Text: "neural network architecture"

Unigrams: ["neural", "network", "architecture"]
Bigrams:  ["neural_network", "network_architecture"]
Combined: ["neural", "network", "architecture", "neural_network", "network_architecture"]
```

### Advantage
- Captures multi-word terms: "neural network", "machine learning", "deep learning"
- Better semantic discrimination than unigrams alone

### Trade-off
- Vocabulary size increases significantly (×3-5)
- More sparse vectors

## Implementation

### Files Created
- `src/bigrams.py`: Bigram extraction and TF-IDF functions
- `scripts/export_bigram_embeddings.py`: Generate bigram embeddings
- `scripts/create_submissions.py`: Create CSV submissions with AUC-ROC evaluation

## Usage

### 1. Generate bigram embeddings
```bash
poetry run python scripts/export_bigram_embeddings.py
```
Output: `data/sparses_embedding_bigram_tfidf.pkl`

### 2. Create submissions with AUC-ROC
```bash
poetry run python scripts/create_submissions.py
```
Outputs:
- `outputs/submissions/submission_raw.csv` (raw frequencies)
- `outputs/submissions/submission_tfidf.csv` (TF-IDF)
- `outputs/submissions/submission_bigram.csv` (bigrams + TF-IDF)

Each method is evaluated with **AUC-ROC** on `valid.tsv`.

## AUC-ROC Metric

**AUC-ROC** (Area Under ROC Curve) measures ranking quality:
- **1.0**: Perfect ranking
- **0.5**: Random ranking
- Higher is better

Perfect for this task: rank relevant documents (label=1) above non-relevant (label=0).

