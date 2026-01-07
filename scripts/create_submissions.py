#!/usr/bin/env python3
"""Create submission CSV files for test_final.tsv using different methods."""

import sys
import csv
import json
import pickle
import math
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sparse_browser import precompute_norms, cosine_similarity_sparse
from sklearn.metrics import roc_auc_score

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "submissions"

def load_embeddings(pkl_path):
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    return data["ids"], data["vocab"], data["matrix"]

def load_test_final(path):
    """Load test_final.tsv and return dict[query_id][corpus_id] structure."""
    test_data = {}
    with path.open("r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            query_id, corpus_id = parts[0], parts[1]
            if query_id not in test_data:
                test_data[query_id] = {}
            test_data[query_id][corpus_id] = 0  # Placeholder
    return test_data

def load_valid_tsv(path):
    """Load valid.tsv for AUC-ROC calculation."""
    valid_data = {}
    with path.open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            query_id, corpus_id, label = parts[0], parts[1], int(parts[2])
            if query_id not in valid_data:
                valid_data[query_id] = {}
            valid_data[query_id][corpus_id] = label
    return valid_data

def calculate_predictions(ids, matrix, precomputed_norms, test_data):
    """Calculate cosine similarity predictions."""
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    predictions = {}
    
    for query_id in tqdm(test_data.keys(), desc="Processing queries"):
        if query_id not in id_to_idx:
            predictions[query_id] = {cid: 0.0 for cid in test_data[query_id]}
            continue
        
        query_idx = id_to_idx[query_id]
        query_vec = matrix[query_idx]
        query_norm = precomputed_norms[query_idx]
        
        predictions[query_id] = {}
        for corpus_id in test_data[query_id]:
            if corpus_id not in id_to_idx:
                predictions[query_id][corpus_id] = 0.0
                continue
            
            corpus_idx = id_to_idx[corpus_id]
            corpus_vec = matrix[corpus_idx]
            corpus_norm = precomputed_norms[corpus_idx]
            
            similarity = cosine_similarity_sparse(
                query_vec, corpus_vec, query_norm, corpus_norm
            )
            predictions[query_id][corpus_id] = similarity
    
    return predictions

def save_submission(predictions, test_data, file_path):
    """Save predictions in submission format."""
    with file_path.open('w', encoding='utf-8') as f:
        f.write("RowId,query-id,corpus-id,score\n")
        row_id = 1
        for query_id in test_data.keys():
            for corpus_id in test_data[query_id].keys():
                score = predictions[query_id].get(corpus_id, 0.0)
                f.write(f"{row_id},{query_id},{corpus_id},{score}\n")
                row_id += 1

def evaluate_on_valid(ids, matrix, precomputed_norms, valid_data):
    """Calculate AUC-ROC on validation set."""
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    
    y_true = []
    y_scores = []
    
    for query_id in tqdm(valid_data.keys(), desc="Evaluating on valid"):
        if query_id not in id_to_idx:
            continue
        
        query_idx = id_to_idx[query_id]
        query_vec = matrix[query_idx]
        query_norm = precomputed_norms[query_idx]
        
        for corpus_id, label in valid_data[query_id].items():
            if corpus_id not in id_to_idx:
                continue
            
            corpus_idx = id_to_idx[corpus_id]
            corpus_vec = matrix[corpus_idx]
            corpus_norm = precomputed_norms[corpus_idx]
            
            similarity = cosine_similarity_sparse(
                query_vec, corpus_vec, query_norm, corpus_norm
            )
            
            y_true.append(label)
            y_scores.append(similarity)
    
    if len(y_true) > 0 and len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_scores)
        return auc
    return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    test_data = load_test_final(DATA_DIR / "test_final.tsv")
    valid_data = load_valid_tsv(DATA_DIR / "valid.tsv")
    
    print(f"Loaded {len(test_data)} test queries")
    print(f"Loaded {len(valid_data)} validation queries")
    
    methods = [
        ("Raw Frequencies", DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl", "submission_raw.csv"),
        ("TF-IDF", DATA_DIR / "sparses_embedding_tfidf.pkl", "submission_tfidf.csv"),
        ("Bigrams TF-IDF", DATA_DIR / "sparses_embedding_bigram_tfidf.pkl", "submission_bigram.csv"),
    ]
    
    for method_name, pkl_path, output_name in methods:
        print(f"\n{'='*70}")
        print(f"Method: {method_name}")
        print(f"{'='*70}")
        
        if not pkl_path.exists():
            print(f"⚠️  Embeddings not found: {pkl_path}")
            continue
        
        # Load embeddings
        print("Loading embeddings...")
        ids, vocab, matrix = load_embeddings(pkl_path)
        print(f"✓ {len(ids)} docs, {len(vocab)} terms")
        
        # Pre-compute norms
        print("Pre-computing norms...")
        precomputed_norms = precompute_norms(matrix)
        
        # Evaluate on validation set
        print("Calculating AUC-ROC on validation set...")
        auc = evaluate_on_valid(ids, matrix, precomputed_norms, valid_data)
        if auc is not None:
            print(f"✓ AUC-ROC: {auc:.4f}")
        else:
            print("⚠️  Could not calculate AUC-ROC")
        
        # Generate predictions
        print("Generating predictions for test set...")
        predictions = calculate_predictions(ids, matrix, precomputed_norms, test_data)
        
        # Save submission
        output_path = OUTPUT_DIR / output_name
        save_submission(predictions, test_data, output_path)
        print(f"✓ Saved: {output_path}")
        
        # Stats
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()

