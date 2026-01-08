# Data Science Browser - Branch Thomas

Semantic search engine for scientific publications using sparse embeddings and TF-IDF.

## 🎯 Best Results

**Bigrams + TF-IDF**: AUC = 0.720, F1 = 0.540, Recall = 0.456

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for detailed comparison.

## 🚀 Quick Start

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

## 📊 Methods Comparison

| Method | AUC | F1 | Precision | Recall | Speed |
|--------|-----|-----|-----------|--------|-------|
| Baseline | 0.686 | 0.520 | 0.792 | 0.387 | 48s ⚡ |
| TF-IDF Corrected | 0.686 | 0.520 | 0.792 | 0.387 | 48s ⚡ |
| **Bigram TF-IDF** ✅ | **0.720** | **0.540** | 0.662 | **0.456** | 48s ⚡ |
| Bigram RAW | 0.700 | 0.429 | 0.402 | 0.461 | 48s ⚡ |

### Key Findings

1. **Bigrams > Unigrams**: +5% AUC, +18% recall by capturing semantic context
2. **TF-IDF = Raw** on decimated vocabulary: filtering already does the discriminative work  
3. **TF-IDF crucial for bigrams**: prevents common phrase noise
4. **Numpy vectorization**: 50x speedup (40min → 48sec)

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

## 🔧 Main Scripts

### 1. Export Embeddings (Unified)

```bash
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

## 🎓 Implementation Details

### Sparse Embeddings

Simple bag-of-words with term frequencies:

```python
from src.utils import sparse_embedding

text = "machine learning neural networks"
embedding = sparse_embedding(text)  # {'machine': 1, 'learning': 1, 'neural': 1, 'networks': 1}
```

### Bigrams

Extract word pairs to capture context:

```python
from src.bigrams import bigram_embedding

text = "machine learning"
embedding = bigram_embedding(text)  
# {'machine': 1, 'learning': 1, 'machine learning': 1}
```

### TF-IDF Weighting

Down-weight common terms, up-weight rare discriminative terms:

```python
from src.utils import tfidf_weights

embeddings = [sparse_embedding(text) for text in texts]
tfidf_embeddings, vocab = tfidf_weights(embeddings)
```

### Cosine Similarity

Efficient sparse dot product:

```python
from src.sparse_browser import cosine_similarity_sparse

similarity = cosine_similarity_sparse(query_vec, doc_vec, query_norm, doc_norm)
```

### Vocabulary Decimation

Reduce vocabulary size intelligently:
- Remove stop words
- Min/max frequency filtering  
- Keep top-K most frequent
- Stemming (optional)

**Parameters** (baseline/TF-IDF):
- `max_vocab_size=3000`
- `min_freq=3`, `max_freq=350`
- `use_stemming=True`

**Parameters** (bigrams):
- `max_vocab=5000`
- `min_freq=2`
- Remove common bigrams

## 📊 Performance Optimization

### Numpy Vectorization

**Before** (40 minutes):
```python
for corpus_id in corpus_ids:
    similarity = cosine_similarity_sparse(query, corpus[corpus_id], ...)
    results.append(similarity)
```

**After** (48 seconds):
```python
# Vectorized batch computation
similarities = corpus_matrix @ query_vector.T / norms  # All at once!
```

**Key optimizations**:
1. Convert to numpy arrays (`float32`)
2. Pre-compute norms once
3. Batch matrix multiplication
4. Eliminate Python loops

## 🧪 Experimental Results

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for:
- Detailed metrics comparison
- Confusion matrices
- ROC curves
- Why bigrams win
- Why TF-IDF corrected equals baseline

## 📚 Documentation

- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) - Complete results analysis
- [docs/bigrams_implementation.md](docs/bigrams_implementation.md) - Bigram details
- [docs/tfidf_implementation.md](docs/tfidf_implementation.md) - TF-IDF details
- [docs/sparse_browser_guide.md](docs/sparse_browser_guide.md) - Search engine guide

## 🔬 Key Insights

1. **Context matters**: Bigrams capture "machine learning" as a single concept, not just "machine" + "learning"

2. **Vocabulary filtering > TF-IDF**: On a well-filtered vocabulary (stop words removed, frequency filtered, stemmed), TF-IDF provides no additional benefit over raw frequencies

3. **TF-IDF essential for bigrams**: Without it, common phrases dominate and precision drops to 40%

4. **Precision vs Recall trade-off**: 
   - Baseline: High precision (79%), low recall (39%) → good for filtering
   - Bigrams: Medium precision (66%), higher recall (46%) → better for search

5. **Speed matters**: Simple numpy optimization → 50x speedup with identical results

## 🏆 Best Practices Learned

- Always test TF-IDF after decimation, not before
- Bigrams need more aggressive filtering than unigrams
- Vectorization > everything for performance
- Measure both AUC and F1 (different use cases)
- Resume capability crucial for long experiments

## 📝 Installation

```bash
# Clone repo
git clone <repo-url>
cd data-science-browser

# Install dependencies
poetry install

# Verify installation
poetry run python -c "from src import sparse_browser; print('OK')"
```

## 🎯 Competition Workflow

```bash
# 1. Export best embeddings (bigram TF-IDF)
poetry run python scripts/export_embeddings.py --method bigram

# 2. Generate predictions
poetry run python scripts/generate_predictions_fast.py --method bigram

# 3. Create submissions
poetry run python scripts/create_submissions.py
poetry run python scripts/create_binary_submissions.py

# 4. Submit files from outputs/submissions/
ls -lh outputs/submissions/
```

## 📄 License

Academic project - École Centrale de Lyon

---

**Author**: Thomas (Branch thomas)  
**Best Model**: Bigrams + TF-IDF (AUC = 0.720)  
**Last Updated**: 2026-01-08
