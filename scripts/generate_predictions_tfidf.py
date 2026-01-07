#!/usr/bin/env python3
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

def load_processed_queries(csv_path: Path):
    """Load processed queries from a CSV file."""
    if not csv_path.exists():
        return set()
    processed = set()
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader)
            for row in reader:
                if row:
                    processed.add(row[0])
        except StopIteration:
            pass
    return processed

def get_query_ids(ids):
    """Get query IDs from a list of IDs."""
    query_ids_set = set()
    with (DATA_DIR / "queries.jsonl").open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                query_ids_set.add(json.loads(line)['_id'])
    return sorted([qid for qid in ids if qid in query_ids_set])

def get_corpus_ids(ids):
    """Get corpus IDs from a list of IDs."""
    corpus_ids_set = set()
    with (DATA_DIR / "corpus.jsonl").open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                corpus_ids_set.add(json.loads(line)['_id'])
    return sorted([cid for cid in ids if cid in corpus_ids_set])

def main():
    print("Generating TF-IDF predictions")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading embeddings...")
    ids, vocab, matrix = load_embeddings(DATA_DIR / "sparses_embedding_tfidf.pkl")
    print(f"✓ {len(ids)} docs, {len(vocab)} terms")
    
    print("Pre-computing norms...")
    precomputed_norms = precompute_norms(matrix)
    
    print("Identifying queries and corpus...")
    query_ids = get_query_ids(ids)
    corpus_ids = get_corpus_ids(ids)
    print(f"✓ {len(query_ids)} queries, {len(corpus_ids)} corpus")
    
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    
    processed = load_processed_queries(OUTPUT_FILE)
    remaining = [qid for qid in query_ids if qid not in processed]
    
    if processed:
        print(f"✓ Already: {len(processed)}, Remaining: {len(remaining)}")
    else:
        print(f"✓ Starting fresh: {len(query_ids)} queries")
    
    if not remaining:
        print("✓ All queries processed!")
        return
    
    mode = 'a' if OUTPUT_FILE.exists() and processed else 'w'
    
    with OUTPUT_FILE.open(mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if mode == 'w':
            writer.writerow(['query_id'] + corpus_ids)
        
        print(f"Processing {len(remaining)} queries...")
        for qid in tqdm(remaining, desc="Queries"):
            q_idx = id_to_idx[qid]
            q_vec, q_norm = matrix[q_idx], precomputed_norms[q_idx]
            
            row = [qid]
            for cid in corpus_ids:
                c_idx = id_to_idx[cid]
                sim = cosine_similarity_sparse(q_vec, matrix[c_idx], q_norm, precomputed_norms[c_idx])
                row.append(f"{sim:.6f}")
            
            writer.writerow(row)
            f.flush()
    
    print(f"\n✓ Done! Saved to {OUTPUT_FILE}")
    print(f"  {len(query_ids)} queries x {len(corpus_ids)} corpus")

if __name__ == "__main__":
    main()
