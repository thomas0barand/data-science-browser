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

def evaluate_predictions(
    predictions_csv: Path,
    valid_tsv: Path,
    output_csv: Path,
    threshold: float = 0.1
):
    """
    Evaluate predictions based on valid.tsv.
    
    For each (query_id, corpus_id) pair in valid.tsv:
    - Looks up the cosine similarity from predictions_csv
    - Outputs 1 if similarity > threshold, else 0
    
    Args:
        predictions_csv: Path to predictions CSV file
        valid_tsv: Path to valid.tsv file
        output_csv: Path to output CSV file
        threshold: Threshold for binary classification (default: 0.1)
    """
    print("="*70)
    print("EVALUATING PREDICTIONS")
    print("="*70)
    
    # Load predictions CSV into memory
    print(f"\nLoading predictions from {predictions_csv.name}...")
    predictions = {}
    
    with predictions_csv.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        corpus_ids = header[1:]  # Skip 'query_id' column
        
        print(f"  Corpus columns: {len(corpus_ids)}")
        
        # Create mapping: {query_id: {corpus_id: similarity}}
        for row in reader:
            if not row:
                continue
            query_id = row[0]
            similarities = {corpus_ids[i]: float(row[i+1]) for i in range(len(corpus_ids))}
            predictions[query_id] = similarities
    
    print(f"  ✓ Loaded predictions for {len(predictions)} queries")
    
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
                prediction = 0
                missing_pairs += 1
            
            results.append({
                'query_id': query_id,
                'corpus_id': corpus_id,
                'ground_truth': ground_truth,
                'similarity': similarity if query_id in predictions and corpus_id in predictions[query_id] else 0.0,
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
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
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
    
    print(f"\nThreshold: {threshold}")
    print(f"Total pairs: {total}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {true_positives:6d}")
    print(f"  False Positives: {false_positives:6d}")
    print(f"  True Negatives:  {true_negatives:6d}")
    print(f"  False Negatives: {false_negatives:6d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("="*70)

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
        
        evaluate_predictions(
            predictions_csv=predictions_csv,
            valid_tsv=valid_tsv,
            output_csv=output_csv,
            threshold=args.threshold
        )

