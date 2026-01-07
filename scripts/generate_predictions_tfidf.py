#!/usr/bin/env python3
"""
Generate predictions using TF-IDF embeddings.
Identical to generate_predictions_sparse.py but uses TF-IDF embeddings.
"""

import sys
import csv
import json
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sparse_browser import load_embeddings, precompute_norms, cosine_similarity_sparse

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"
OUTPUT_FILE = OUTPUT_DIR / "preds_tfidf.csv"

def load_processed_queries(csv_path: Path) -> set:
    """Load already processed query IDs from existing CSV."""
    if not csv_path.exists():
        return set()
    
    processed = set()
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            for row in reader:
                if row:
                    processed.add(row[0])
        except StopIteration:
            pass
    
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
    
    query_ids = [qid for qid in ids if qid in query_ids_set]
    return sorted(query_ids)

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
    
    corpus_ids = [cid for cid in ids if cid in corpus_ids_set]
    return sorted(corpus_ids)

def main():
    print("="*70)
    print("GENERATING TF-IDF PREDICTIONS FOR ALL QUERIES")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load TF-IDF embeddings
    print("\nLoading TF-IDF embeddings...")
    pkl_path = DATA_DIR / "sparses_embedding_tfidf.pkl"
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
    
    # Create ID to index mapping
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
    print(f"PROCESSING QUERIES WITH TF-IDF")
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
        
        # Process remaining queries
        print(f"\nProcessing {len(remaining_queries)} queries...")
        print(f"Each query compares against {len(corpus_ids)} corpus documents")
        print(f"Total comparisons: {len(remaining_queries) * len(corpus_ids):,}\n")
        
        for query_id in tqdm(remaining_queries, desc="Queries", unit="query"):
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
            
            writer.writerow(row)
            f.flush()
    
    print(f"\n{'='*70}")
    print("COMPLETED!")
    print(f"{'='*70}")
    print(f"✓ Total queries processed: {len(query_ids)}")
    print(f"✓ Corpus documents: {len(corpus_ids)}")
    print(f"✓ Total similarities computed: {len(query_ids) * len(corpus_ids):,}")
    print(f"✓ Results saved to: {OUTPUT_FILE}")
    print(f"✓ CSV dimensions: {len(query_ids)} rows x {len(corpus_ids) + 1} columns")
    
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"✓ File size: {file_size_mb:.2f} MB")
    print(f"\n💡 Next step: Evaluate with valid.tsv using:")
    print(f"   poetry run python scripts/generate_predictions_sparse.py --mode evaluate --predictions-csv {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

