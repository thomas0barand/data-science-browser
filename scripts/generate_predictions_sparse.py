#!/usr/bin/env python3
"""
Generate predictions for all queries and save to CSV incrementally.
Supports resuming from previous runs.

CSV Format:
- Rows: queries (query_id in first column)
- Columns: corpus documents (corpus_id as header)
- Values: cosine similarity scores
"""

import sys
import csv
import json
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sparse_browser import load_embeddings, precompute_norms, cosine_similarity_sparse  # pyright: ignore[reportMissingImports]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"
OUTPUT_FILE = OUTPUT_DIR / "preds_sparse_embedding_without_corpus_text_decimated.csv"

def load_processed_queries(csv_path: Path) -> set:
    """Load already processed query IDs from existing CSV."""
    if not csv_path.exists():
        return set()
    
    processed = set()
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)  # Skip header
            for row in reader:
                if row:  # Skip empty rows
                    processed.add(row[0])  # First column is query_id
        except StopIteration:
            pass  # Empty file
    
    return processed

def get_query_ids(ids: list[str]) -> list[str]:
    """Extract query IDs from embeddings that exist in queries.jsonl."""
    query_ids_set = set()
    queries_path = DATA_DIR / "queries.jsonl"
    
    print(f"  Reading {queries_path.name}...")
    with queries_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            query = json.loads(line)
            query_ids_set.add(query['_id'])
    
    # Return only IDs that are in both the embeddings and queries
    query_ids = [qid for qid in ids if qid in query_ids_set]
    return sorted(query_ids)  # Sort for consistent ordering

def get_corpus_ids(ids: list[str]) -> list[str]:
    """Extract corpus document IDs from embeddings that exist in corpus.jsonl."""
    corpus_ids_set = set()
    corpus_path = DATA_DIR / "corpus.jsonl"
    
    print(f"  Reading {corpus_path.name}...")
    with corpus_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            corpus_ids_set.add(doc['_id'])
    
    # Return only IDs that are in both the embeddings and corpus
    corpus_ids = [cid for cid in ids if cid in corpus_ids_set]
    return sorted(corpus_ids)  # Sort for consistent ordering

def calculate_auc(y_true: list, y_scores: list) -> float:
    """Calculate ROC AUC score using Wilcoxon-Mann-Whitney statistic."""
    if len(set(y_true)) < 2:
        return 0.0
    
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    # Count how many times a positive score is greater than a negative score
    pairs = list(zip(y_scores, y_true))
    
    auc_sum = 0.0
    for i, (score_i, label_i) in enumerate(pairs):
        if label_i == 1:  # Positive example
            for score_j, label_j in pairs:
                if label_j == 0:  # Negative example
                    if score_i > score_j:
                        auc_sum += 1.0
                    elif score_i == score_j:
                        auc_sum += 0.5
    
    return auc_sum / (n_pos * n_neg)


def load_predictions(predictions_csv: Path, valid_tsv: Path) -> tuple[dict[str, dict[str, float]], float]:
    """
    Load predictions CSV into memory and calculate AUC.
    
    Args:
        predictions_csv: Path to predictions CSV file
        valid_tsv: Path to validation TSV file for AUC calculation
        
    Returns:
        Tuple of (predictions dict, auc_score)
        - predictions: Dictionary mapping query_id -> {corpus_id: similarity}
        - auc_score: Pre-calculated AUC score
    """
    print("="*70)
    print("LOADING PREDICTIONS")
    print("="*70)
    
    print(f"\nLoading predictions from {predictions_csv.name}...")
    file_size_mb = predictions_csv.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Count total rows for progress bar
    print("  Counting rows...")
    with predictions_csv.open('r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1
    print(f"  Total queries: {total_rows}")
    
    predictions = {}
    
    with predictions_csv.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        corpus_ids = header[1:]
        
        print(f"  Corpus columns: {len(corpus_ids)}")
        print("  Loading predictions...")
        
        for row in tqdm(reader, total=total_rows, desc="  Processing queries", unit="query"):
            if not row:
                continue
            query_id = row[0]
            similarities = {cid: float(val) for cid, val in zip(corpus_ids, row[1:])}
            predictions[query_id] = similarities
    
    print(f"  ✓ Loaded predictions for {len(predictions)} queries")
    
    # Calculate AUC from validation set
    print(f"\nCalculating AUC from {valid_tsv.name}...")
    y_true = []
    y_scores = []
    
    with valid_tsv.open('r', encoding='utf-8') as f:
        next(f)  # Skip header
        
        for line in tqdm(f, desc="  Processing validation pairs", unit="pair"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            query_id, corpus_id, ground_truth = parts[0], parts[1], parts[2]
            
            if query_id in predictions and corpus_id in predictions[query_id]:
                similarity = predictions[query_id][corpus_id]
            else:
                similarity = 0.0
            
            y_true.append(int(ground_truth))
            y_scores.append(similarity)
    
    auc = calculate_auc(y_true, y_scores)
    print(f"  ✓ AUC Score: {auc:.4f}\n")
    
    return predictions, auc


def evaluate_predictions(
    predictions: dict[str, dict[str, float]],
    valid_tsv: Path,
    output_csv: Path,
    threshold: float = 0.1,
    auc: float = 0.0
):
    """
    Evaluate predictions based on valid.tsv.
    
    For each (query_id, corpus_id) pair in valid.tsv:
    - Looks up the cosine similarity from predictions
    - Outputs 1 if similarity > threshold, else 0
    
    Args:
        predictions: Dictionary mapping query_id -> {corpus_id: similarity}
        valid_tsv: Path to valid.tsv file
        output_csv: Path to output CSV file
        threshold: Threshold for binary classification (default: 0.1)
        auc: Pre-calculated AUC score (default: 0.0)
    """
    print("="*70)
    print(f"EVALUATING PREDICTIONS (threshold={threshold})")
    print("="*70)
    
    # Process valid.tsv
    print(f"\nProcessing {valid_tsv.name}...")
    results = []
    missing_pairs = 0
    
    with valid_tsv.open('r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in tqdm(f, desc="Evaluating", unit="pair"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            query_id, corpus_id, ground_truth = parts[0], parts[1], parts[2]
            
            # Look up prediction
            if query_id in predictions and corpus_id in predictions[query_id]:
                similarity = predictions[query_id][corpus_id]
                prediction = 1 if similarity > threshold else 0
            else:
                # Missing pair - default to 0
                similarity = 0.0
                prediction = 0
                missing_pairs += 1
            
            results.append({
                'query_id': query_id,
                'corpus_id': corpus_id,
                'ground_truth': ground_truth,
                'similarity': similarity,
                'prediction': prediction
            })
    
    print(f"  ✓ Processed {len(results)} pairs")
    if missing_pairs > 0:
        print(f"  ⚠ Missing {missing_pairs} query-corpus pairs in predictions")
    
    # Save results
    print(f"\nSaving results to {output_csv}...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'corpus_id', 'ground_truth', 'similarity', 'prediction'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"  ✓ Saved {len(results)} evaluations")
    
    # Calculate metrics and concatenate into a dict
    correct = sum(1 for r in results if int(r['ground_truth']) == r['prediction'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    true_positives = sum(1 for r in results if r['prediction'] == 1 and int(r['ground_truth']) == 1)
    false_positives = sum(1 for r in results if r['prediction'] == 1 and int(r['ground_truth']) == 0)
    false_negatives = sum(1 for r in results if r['prediction'] == 0 and int(r['ground_truth']) == 1)
    true_negatives = sum(1 for r in results if r['prediction'] == 0 and int(r['ground_truth']) == 0)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "threshold": threshold,
        "total_pairs": total,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    print(f"\nThreshold: {metrics['threshold']}")
    print(f"Total pairs: {metrics['total_pairs']}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']:6d}")
    print(f"  False Positives: {metrics['false_positives']:6d}")
    print(f"  True Negatives:  {metrics['true_negatives']:6d}")
    print(f"  False Negatives: {metrics['false_negatives']:6d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")
    print("="*70)
    return metrics

def main():
    print("="*70)
    print("GENERATING PREDICTIONS FOR ALL QUERIES")
    print("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    print("\nLoading embeddings...")
    pkl_path = DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl"
    ids, vocab, matrix = load_embeddings(pkl_path)
    print(f"✓ Loaded {len(ids)} embeddings with vocab size {len(vocab)}")
    
    # Pre-compute norms
    print("\nPre-computing norms...")
    precomputed_norms = precompute_norms(matrix)
    print(f"✓ Pre-computed {len(precomputed_norms)} norms")
    
    # Separate query and corpus IDs
    print("\nIdentifying queries and corpus documents...")
    query_ids = get_query_ids(ids)
    corpus_ids = get_corpus_ids(ids)
    print(f"✓ Found {len(query_ids)} queries")
    print(f"✓ Found {len(corpus_ids)} corpus documents")
    
    # Create ID to index mapping for fast lookup
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    
    # Check for existing progress
    processed_queries = load_processed_queries(OUTPUT_FILE)
    remaining_queries = [qid for qid in query_ids if qid not in processed_queries]
    
    print(f"\nProgress check:")
    if processed_queries:
        print(f"  ✓ Already processed: {len(processed_queries)} queries")
        print(f"  ✓ Remaining: {len(remaining_queries)} queries")
        print(f"  ✓ Will resume from: {remaining_queries[0][:40]}..." if remaining_queries else "  ✓ All done!")
    else:
        print(f"  ✓ Starting fresh: {len(query_ids)} queries to process")
    
    if not remaining_queries:
        print(f"\n✓ All queries already processed!")
        print(f"✓ Output file: {OUTPUT_FILE}")
        return
    
    # Create or append to CSV
    file_exists = OUTPUT_FILE.exists() and len(processed_queries) > 0
    mode = 'a' if file_exists else 'w'
    
    print(f"\n{'='*70}")
    print(f"PROCESSING QUERIES")
    print(f"{'='*70}")
    
    with OUTPUT_FILE.open(mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            header = ['query_id'] + corpus_ids
            writer.writerow(header)
            f.flush()
            print(f"✓ Created CSV header: 1 query_id + {len(corpus_ids)} corpus columns")
        else:
            print(f"✓ Appending to existing CSV")
        
        # Process remaining queries with progress bar
        print(f"\nProcessing {len(remaining_queries)} queries...")
        print(f"Each query compares against {len(corpus_ids)} corpus documents")
        print(f"Total comparisons: {len(remaining_queries) * len(corpus_ids):,}\n")
        
        for query_id in tqdm(remaining_queries, desc="Queries", unit="query"):
            # Get query vector
            query_idx = id_to_idx[query_id]
            query_vec = matrix[query_idx]
            query_norm = precomputed_norms[query_idx]
            
            # Calculate similarities with all corpus documents
            row = [query_id]
            for corpus_id in corpus_ids:
                corpus_idx = id_to_idx[corpus_id]
                corpus_vec = matrix[corpus_idx]
                corpus_norm = precomputed_norms[corpus_idx]
                
                similarity = cosine_similarity_sparse(
                    query_vec, corpus_vec, 
                    query_norm, corpus_norm
                )
                row.append(f"{similarity:.6f}")
            
            # Write row immediately (incremental save)
            writer.writerow(row)
            f.flush()  # Force write to disk after each query
    
    print(f"\n{'='*70}")
    print("COMPLETED!")
    print(f"{'='*70}")
    print(f"✓ Total queries processed: {len(query_ids)}")
    print(f"✓ Corpus documents: {len(corpus_ids)}")
    print(f"✓ Total similarities computed: {len(query_ids) * len(corpus_ids):,}")
    print(f"✓ Results saved to: {OUTPUT_FILE}")
    print(f"✓ CSV dimensions: {len(query_ids)} rows x {len(corpus_ids) + 1} columns (including query_id)")
    
    # Show file size
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"✓ File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate or evaluate sparse embedding predictions")
    parser.add_argument(
        '--mode',
        choices=['generate', 'evaluate'],
        default='generate',
        help='Mode: generate predictions or evaluate against valid.tsv'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Threshold for binary classification (default: 0.1)'
    )
    parser.add_argument(
        '--predictions-csv',
        type=str,
        help='Path to predictions CSV file (for evaluation mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for evaluation results'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        main()
    elif args.mode == 'evaluate':
        predictions_csv = Path(args.predictions_csv) if args.predictions_csv else OUTPUT_FILE
        output_csv = Path(args.output) if args.output else OUTPUT_DIR / "evaluation_results.csv"
        valid_tsv = DATA_DIR / "valid.tsv"
        
        predictions, auc = load_predictions(predictions_csv, valid_tsv)
        evaluate_predictions(
            predictions=predictions,
            valid_tsv=valid_tsv,
            output_csv=output_csv,
            threshold=args.threshold,
            auc=auc
        )

