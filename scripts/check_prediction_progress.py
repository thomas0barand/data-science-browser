#!/usr/bin/env python3
"""Check progress of prediction generation."""

from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "results" / "preds_sparse_embedding_without_corpus_text.csv"

def main():
    if not OUTPUT_FILE.exists():
        print(f"❌ File not found: {OUTPUT_FILE}")
        print("Run generate_predictions.py to start processing.")
        return
    
    # Count lines
    with OUTPUT_FILE.open('r') as f:
        lines = sum(1 for _ in f)
    
    # Get file size
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    
    # Read first and last query IDs
    with OUTPUT_FILE.open('r') as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader, None)
        
        # Get last query
        last_query = None
        for row in reader:
            last_query = row[0] if row else last_query
    
    num_corpus = len(header) - 1  # Exclude query_id column
    num_queries_processed = lines - 1  # Exclude header
    
    print("="*70)
    print("PREDICTION GENERATION PROGRESS")
    print("="*70)
    print(f"\n📄 File: {OUTPUT_FILE.name}")
    print(f"📊 File size: {size_mb:.2f} MB")
    print(f"\n✓ Queries processed: {num_queries_processed}")
    print(f"✓ Corpus documents: {num_corpus}")
    print(f"✓ Similarities computed: {num_queries_processed * num_corpus:,}")
    
    if first_row:
        print(f"\n🔹 First query: {first_row[0][:50]}...")
    if last_query:
        print(f"🔹 Last query:  {last_query[:50]}...")
    
    print(f"\n💡 To resume: poetry run python3 scripts/generate_predictions.py")
    print("="*70)

if __name__ == "__main__":
    main()

