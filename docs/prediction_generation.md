# Prediction Generation for Sparse Embeddings

## Overview

This document explains how to generate predictions (similarity scores) for all query-corpus pairs using sparse embeddings.

## Output Format

The script generates a CSV file with the following structure:

- **Rows**: One per query (query_id in first column)
- **Columns**: One per corpus document (corpus_id as header)
- **Values**: Cosine similarity scores (0.000000 to 1.000000)

Example:
```csv
query_id,corpus_doc1,corpus_doc2,corpus_doc3,...
query1,0.123456,0.000000,0.456789,...
query2,0.234567,0.345678,0.000000,...
```

## Scripts

### 1. Generate Predictions (Main Script)

**File**: `scripts/generate_predictions.py`

Generates predictions for all queries against all corpus documents.

```bash
poetry run python3 scripts/generate_predictions.py
```

**Features**:
- ✓ Incremental saving (writes each query result immediately)
- ✓ Resume capability (skips already processed queries)
- ✓ Progress bar with ETA
- ✓ Memory efficient (doesn't load all results in memory)
- ✓ Optimized sparse cosine similarity

**Output**: `outputs/results/preds_sparse_embedding_without_corpus_text.csv`

### 2. Check Progress

**File**: `scripts/check_prediction_progress.py`

Check how many queries have been processed.

```bash
poetry run python3 scripts/check_prediction_progress.py
```

Shows:
- Number of queries processed
- File size
- First and last query IDs
- Total similarities computed

### 3. Test on Small Dataset

**File**: `scripts/generate_predictions_small.py`

Test the pipeline on the small dataset (10 queries, 10 corpus docs).

```bash
poetry run python3 scripts/generate_predictions_small.py
```

**Output**: `outputs/results/preds_sparse_embedding_small.csv`

## Expected Performance

### Dataset Size
- **Queries**: ~1,000
- **Corpus**: ~25,657 documents
- **Total comparisons**: ~25,657,000

### Processing Time
With optimizations (sparse cosine + pre-computed norms):
- ~1,500-3,000 queries/second on modern hardware
- **Estimated total time**: 5-15 minutes

### File Size
- **Small dataset** (10×10): < 1 KB
- **Full dataset** (1000×25657): ~200-300 MB

## How Resume Works

The script checks the existing CSV file and:
1. Reads all processed query IDs from the first column
2. Compares against the full list of queries
3. Only processes queries not yet in the CSV
4. Appends new results to the existing file

**Example**:
```bash
# First run - processes 100 queries, then interrupted
poetry run python3 scripts/generate_predictions.py
# Progress: 100/1000 queries

# Second run - automatically resumes from query 101
poetry run python3 scripts/generate_predictions.py
# Progress: 900/1000 queries (skipped 100)
```

## Optimizations Applied

### 1. Sparse Cosine Similarity
Only computes dot product for non-zero elements (skips ~95% of multiplications).

### 2. Pre-computed Norms
All document norms are calculated once at startup and reused.

### 3. Incremental Saving
Each query result is written immediately to disk (no memory buildup).

### 4. File Flush
`f.flush()` after each query ensures data is saved even if interrupted.

## Storage Management

### Incremental Saving Strategy
```python
with csv_file.open('a', newline='') as f:
    writer = csv.writer(f)
    for query in queries:
        row = compute_similarities(query)
        writer.writerow(row)
        f.flush()  # Force write to disk
```

### Benefits
- ✓ No memory overflow (only one query in memory at a time)
- ✓ Safe interruption (progress saved continuously)
- ✓ Fast resume (skips completed queries)

## Troubleshooting

### Script Interrupted
Just run it again - it will automatically resume:
```bash
poetry run python3 scripts/generate_predictions.py
```

### Check Progress Manually
```bash
# Count processed queries (subtract 1 for header)
wc -l outputs/results/preds_sparse_embedding_without_corpus_text.csv

# View first few lines
head -5 outputs/results/preds_sparse_embedding_without_corpus_text.csv

# View last processed query
tail -1 outputs/results/preds_sparse_embedding_without_corpus_text.csv
```

### Clear and Restart
```bash
# Delete output file to start fresh
rm outputs/results/preds_sparse_embedding_without_corpus_text.csv

# Run script
poetry run python3 scripts/generate_predictions.py
```

### Memory Issues
The script should not have memory issues as it processes one query at a time. If you encounter problems:
- Check available disk space (need ~300 MB for output)
- Verify embeddings load successfully
- Monitor with: `poetry run python3 scripts/check_prediction_progress.py`

## Next Steps

After generating predictions, you can:
1. Load the CSV for evaluation
2. Calculate metrics (e.g., precision@k, recall@k, MAP)
3. Compare with other embedding methods
4. Analyze which queries perform well/poorly

## Example: Loading Results

```python
import pandas as pd

# Load predictions
df = pd.read_csv('outputs/results/preds_sparse_embedding_without_corpus_text.csv', 
                 index_col='query_id')

# Get top 10 documents for a query
query_id = 'some_query_id'
top_10 = df.loc[query_id].nlargest(10)

print(f"Top 10 results for {query_id}:")
for corpus_id, score in top_10.items():
    print(f"  {corpus_id}: {score:.4f}")
```

## Performance Monitoring

During execution, you'll see:
```
Processing 1000 queries...
Each query compares against 25657 corpus documents
Total comparisons: 25,657,000

Queries:  45%|████▌     | 450/1000 [02:15<02:45,  3.32query/s]
```

This shows:
- Current progress (450/1000)
- Time elapsed (2:15)
- Estimated time remaining (2:45)
- Processing speed (3.32 queries/second)

