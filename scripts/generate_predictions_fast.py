#!/usr/bin/env python3
"""
Version optimisée de generate_predictions avec:
- Numpy pour calculs matriciels
- Traitement par batch
- Optimisations mémoire
"""
import sys
import csv
import json
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"

def load_embeddings(pkl_path):
    """Load embeddings from pickle file"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['ids'], data['vocab'], data['matrix']

METHODS = {
    "baseline": {
        "pkl": DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl",
        "output": OUTPUT_DIR / "preds_baseline.csv",
        "name": "Baseline"
    },
    "tfidf_old": {
        "pkl": DATA_DIR / "sparses_embedding_tfidf.pkl",
        "output": OUTPUT_DIR / "preds_tfidf.csv",
        "name": "TF-IDF (old)"
    },
    "tfidf": {
        "pkl": DATA_DIR / "sparses_embedding_tfidf_corrected.pkl",
        "output": OUTPUT_DIR / "preds_tfidf_corrected.csv",
        "name": "TF-IDF (corrected)"
    },
    "bigram": {
        "pkl": DATA_DIR / "sparses_embedding_bigram_tfidf_decimated.pkl",
        "output": OUTPUT_DIR / "preds_bigram.csv",
        "name": "Bigram TF-IDF"
    },
    "bigram_raw": {
        "pkl": DATA_DIR / "sparses_embedding_bigram_raw_decimated.pkl",
        "output": OUTPUT_DIR / "preds_bigram_raw.csv",
        "name": "Bigram RAW"
    }
}

def load_processed_queries(csv_path: Path):
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
    query_ids_set = set()
    with (DATA_DIR / "queries.jsonl").open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                query_ids_set.add(json.loads(line)['_id'])
    return sorted([qid for qid in ids if qid in query_ids_set])

def get_corpus_ids(ids):
    corpus_ids_set = set()
    with (DATA_DIR / "corpus.jsonl").open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                corpus_ids_set.add(json.loads(line)['_id'])
    return sorted([cid for cid in ids if cid in corpus_ids_set])

def cosine_similarity_batch(query_vec, corpus_matrix, query_norm, corpus_norms):
    """Calculate cosine similarity between query and all corpus docs (vectorized)"""
    # query_vec: (1, vocab_size)
    # corpus_matrix: (n_corpus, vocab_size)
    # Returns: (n_corpus,)
    
    dot_products = corpus_matrix @ query_vec.T  # (n_corpus, 1)
    dot_products = dot_products.ravel()
    
    denominators = query_norm * corpus_norms
    denominators = np.where(denominators == 0, 1e-10, denominators)
    
    return dot_products / denominators

def generate_predictions_fast(method: str, batch_size: int = 100):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    
    config = METHODS[method]
    output_file = config["output"]
    
    print(f"⚡ Fast generation: {config['name']}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not config["pkl"].exists():
        print(f"✗ Not found: {config['pkl']}")
        return
    
    print("Loading embeddings...")
    ids, vocab, matrix = load_embeddings(config["pkl"])
    
    # Convert to numpy arrays
    print("Converting to numpy...")
    matrix_np = np.array(matrix, dtype=np.float32)
    print(f"✓ {len(ids)} docs, {len(vocab)} terms, {matrix_np.nbytes / (1024**2):.1f} MB")
    
    print("Pre-computing norms...")
    norms = np.linalg.norm(matrix_np, axis=1)
    
    print("Identifying queries and corpus...")
    query_ids = get_query_ids(ids)
    corpus_ids = get_corpus_ids(ids)
    print(f"✓ {len(query_ids)} queries, {len(corpus_ids)} corpus")
    
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    corpus_indices = [id_to_idx[cid] for cid in corpus_ids]
    corpus_matrix = matrix_np[corpus_indices]
    corpus_norms = norms[corpus_indices]
    
    processed = load_processed_queries(output_file)
    remaining = [qid for qid in query_ids if qid not in processed]
    
    if processed:
        print(f"✓ Already: {len(processed)}, Remaining: {len(remaining)}")
    else:
        print(f"✓ Starting: {len(query_ids)} queries")
    
    if not remaining:
        print("✓ All done!")
        return
    
    mode = 'a' if output_file.exists() and processed else 'w'
    
    with output_file.open(mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if mode == 'w':
            writer.writerow(['query_id'] + corpus_ids)
        
        print(f"Processing {len(remaining)} queries...")
        for qid in tqdm(remaining, desc="Queries"):
            q_idx = id_to_idx[qid]
            q_vec = matrix_np[q_idx:q_idx+1]  # Keep 2D shape
            q_norm = norms[q_idx]
            
            similarities = cosine_similarity_batch(q_vec, corpus_matrix, q_norm, corpus_norms)
            
            row = [qid] + [f"{sim:.6f}" for sim in similarities]
            writer.writerow(row)
            
            if len(remaining) > 100:
                f.flush()
    
    print(f"\n✓ Done! Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Fast prediction generation with numpy")
    parser.add_argument('--method', choices=list(METHODS.keys()), required=True)
    parser.add_argument('--batch-size', type=int, default=100)
    
    args = parser.parse_args()
    generate_predictions_fast(args.method, args.batch_size)

if __name__ == "__main__":
    main()

