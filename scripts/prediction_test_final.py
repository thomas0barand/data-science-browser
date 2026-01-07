#!/usr/bin/env python3
"""Generate predictions for test_final.tsv with cosine similarity scores."""

import sys
import csv
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"

def load_predictions_simple(predictions_csv: Path) -> dict[str, dict[str, float]]:
    """Load predictions CSV into memory."""
    print("="*70)
    print("LOADING PREDICTIONS")
    print("="*70)
    
    print(f"\nLoading predictions from {predictions_csv.name}...")
    file_size_mb = predictions_csv.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
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
    
    print(f"  ✓ Loaded predictions for {len(predictions)} queries\n")
    return predictions

def generate_test_predictions(
    predictions: dict[str, dict[str, float]],
    test_tsv: Path,
    output_csv: Path
):
    """
    Generate predictions for test_final.tsv.
    
    For each (query_id, corpus_id) pair in test_final.tsv:
    - Looks up the cosine similarity from predictions
    - Outputs the similarity score in the last column as CSV
    
    Args:
        predictions: Dictionary mapping query_id -> {corpus_id: similarity}
        test_tsv: Path to test_final.tsv file
        output_csv: Path to output CSV file
    """
    print("="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)
    
    print(f"\nProcessing {test_tsv.name}...")
    results = []
    missing_pairs = 0
    
    with test_tsv.open('r', encoding='utf-8') as f:
        header_line = next(f)
        
        for line in tqdm(f, desc="Processing test pairs", unit="pair"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            query_id = parts[0]
            corpus_id = parts[1]
            
            if query_id in predictions and corpus_id in predictions[query_id]:
                similarity = predictions[query_id][corpus_id]
            else:
                similarity = 0.0
                missing_pairs += 1
            
            results.append({
                'query-id': query_id,
                'corpus-id': corpus_id,
                'score': similarity
            })
    
    print(f"  ✓ Processed {len(results)} pairs")
    if missing_pairs > 0:
        print(f"  ⚠ Missing {missing_pairs} query-corpus pairs in predictions")
    
    print(f"\nSaving results to {output_csv}...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query-id', 'corpus-id', 'score'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"  ✓ Saved {len(results)} predictions to CSV format")
    
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total pairs: {len(results)}")
    print(f"Missing pairs: {missing_pairs}")
    print(f"Coverage: {(len(results) - missing_pairs) / len(results) * 100:.2f}%")
    
    # Score distribution
    scores = [r['score'] for r in results if r['score'] > 0]
    if scores:
        print(f"\nScore statistics (non-zero):")
        print(f"  Count: {len(scores)}")
        print(f"  Min: {min(scores):.6f}")
        print(f"  Max: {max(scores):.6f}")
        print(f"  Mean: {sum(scores) / len(scores):.6f}")
    
    print("="*70)
    
    return len(results)

def main():
    predictions_csv = OUTPUT_DIR / "preds_sparse_embedding_without_corpus_text_decimated.csv"
    test_tsv = DATA_DIR / "test_final.tsv"
    submissions_dir = PROJECT_ROOT / "outputs" / "submissions"
    output_csv = submissions_dir / "test_final_predictions.csv"
    
    print("\n" + "="*70)
    print("TEST FINAL PREDICTIONS GENERATION")
    print("="*70)
    print(f"\nPredictions: {predictions_csv.name}")
    print(f"Test file: {test_tsv.name}")
    print(f"Output: {output_csv.name}\n")
    
    predictions = load_predictions_simple(predictions_csv)
    
    num_rows = generate_test_predictions(
        predictions=predictions,
        test_tsv=test_tsv,
        output_csv=output_csv
    )
    
    print(f"\n✅ Done! Output saved to: {output_csv}")
    print(f"   Full path: {output_csv.absolute()}")
    
    expected_rows = 8978
    if num_rows == expected_rows:
        print(f"\n✓ Validation passed: {num_rows} rows (matches expected {expected_rows})")
    else:
        print(f"\n⚠ Warning: {num_rows} rows (expected {expected_rows})")
    
    print(f"\n📦 Ready for submission! The file has the correct CSV format.")
    print(f"   You can zip it if needed:")
    print(f"   cd {submissions_dir} && zip submission.zip {output_csv.name}\n")

if __name__ == "__main__":
    main()
