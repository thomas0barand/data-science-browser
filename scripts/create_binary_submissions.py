#!/usr/bin/env python3
"""Create binary submission CSV files using optimal ROC threshold."""

import sys
import pickle
import math
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sparse_browser import precompute_norms, cosine_similarity_sparse
from sklearn.metrics import roc_curve, roc_auc_score

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "submissions"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

def load_embeddings(pkl_path):
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    return data["ids"], data["vocab"], data["matrix"]

def load_test_final(path):
    test_data = {}
    with path.open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            query_id, corpus_id = parts[0], parts[1]
            if query_id not in test_data:
                test_data[query_id] = {}
            test_data[query_id][corpus_id] = 0
    return test_data

def load_valid_tsv(path):
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

def get_scores_on_valid(ids, matrix, precomputed_norms, valid_data):
    """Get continuous scores and labels on validation set."""
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    
    y_true = []
    y_scores = []
    
    for query_id in tqdm(valid_data.keys(), desc="Computing scores on valid"):
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
    
    return np.array(y_true), np.array(y_scores)

def find_optimal_threshold(y_true, y_scores):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Youden's J = TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    return optimal_threshold, fpr, tpr, thresholds, optimal_idx

def plot_roc_curve(fpr, tpr, auc_score, optimal_idx, method_name, save_path):
    """Plot ROC curve with optimal threshold marked."""
    plt.figure(figsize=(10, 8))
    
    # ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    
    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    # Optimal point
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
             label=f'Optimal threshold\n(FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f})')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {method_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {save_path}")
    plt.close()

def calculate_predictions_binary(ids, matrix, precomputed_norms, test_data, threshold):
    """Calculate binary predictions using threshold."""
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    predictions = {}
    
    for query_id in tqdm(test_data.keys(), desc="Generating binary predictions"):
        if query_id not in id_to_idx:
            predictions[query_id] = {cid: 0 for cid in test_data[query_id]}
            continue
        
        query_idx = id_to_idx[query_id]
        query_vec = matrix[query_idx]
        query_norm = precomputed_norms[query_idx]
        
        predictions[query_id] = {}
        for corpus_id in test_data[query_id]:
            if corpus_id not in id_to_idx:
                predictions[query_id][corpus_id] = 0
                continue
            
            corpus_idx = id_to_idx[corpus_id]
            corpus_vec = matrix[corpus_idx]
            corpus_norm = precomputed_norms[corpus_idx]
            
            similarity = cosine_similarity_sparse(
                query_vec, corpus_vec, query_norm, corpus_norm
            )
            
            # Binary prediction based on threshold
            predictions[query_id][corpus_id] = 1 if similarity >= threshold else 0
    
    return predictions

def save_submission(predictions, test_data, file_path):
    """Save binary predictions in submission format."""
    with file_path.open('w', encoding='utf-8') as f:
        f.write("RowId,query-id,corpus-id,score\n")
        row_id = 1
        for query_id in test_data.keys():
            for corpus_id in test_data[query_id].keys():
                score = predictions[query_id].get(corpus_id, 0)
                f.write(f"{row_id},{query_id},{corpus_id},{score}\n")
                row_id += 1

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test and validation data
    test_data = load_test_final(DATA_DIR / "test_final.tsv")
    valid_data = load_valid_tsv(DATA_DIR / "valid.tsv")
    
    print(f"Loaded {len(test_data)} test queries")
    print(f"Loaded {len(valid_data)} validation queries")
    
    methods = [
        ("Raw Frequencies", DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl", 
         "submission_raw_binary.csv", "roc_curve_raw.png"),
        ("TF-IDF", DATA_DIR / "sparses_embedding_tfidf.pkl", 
         "submission_tfidf_binary.csv", "roc_curve_tfidf.png"),
        ("Bigrams TF-IDF", DATA_DIR / "sparses_embedding_bigram_tfidf.pkl", 
         "submission_bigram_binary.csv", "roc_curve_bigram.png"),
    ]
    
    best_auc = 0
    best_method_info = None
    
    for method_name, pkl_path, output_name, roc_filename in methods:
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
        
        # Get scores on validation set
        print("Computing scores on validation set...")
        y_true, y_scores = get_scores_on_valid(ids, matrix, precomputed_norms, valid_data)
        
        # Calculate AUC-ROC
        auc = roc_auc_score(y_true, y_scores)
        print(f"✓ AUC-ROC: {auc:.4f}")
        
        # Track best method
        if auc > best_auc:
            best_auc = auc
            best_method_info = (method_name, y_true, y_scores, auc, roc_filename)
        
        # Find optimal threshold
        print("Finding optimal threshold...")
        optimal_threshold, fpr, tpr, thresholds, optimal_idx = find_optimal_threshold(y_true, y_scores)
        print(f"✓ Optimal threshold: {optimal_threshold:.4f}")
        print(f"  - TPR (Recall): {tpr[optimal_idx]:.4f}")
        print(f"  - FPR: {fpr[optimal_idx]:.4f}")
        
        # Generate binary predictions for test set
        print("Generating binary predictions for test set...")
        predictions = calculate_predictions_binary(ids, matrix, precomputed_norms, test_data, optimal_threshold)
        
        # Save submission
        output_path = OUTPUT_DIR / output_name
        save_submission(predictions, test_data, output_path)
        print(f"✓ Saved: {output_path}")
        
        # Count predictions
        total_preds = sum(len(preds) for preds in predictions.values())
        ones_count = sum(sum(1 for v in preds.values() if v == 1) for preds in predictions.values())
        print(f"  - Total predictions: {total_preds}")
        print(f"  - Predicted 1: {ones_count} ({100*ones_count/total_preds:.1f}%)")
        print(f"  - Predicted 0: {total_preds - ones_count} ({100*(1-ones_count/total_preds):.1f}%)")
    
    # Plot ROC curve for best method
    if best_method_info:
        print(f"\n{'='*70}")
        print(f"BEST METHOD: {best_method_info[0]} (AUC = {best_method_info[3]:.4f})")
        print(f"{'='*70}")
        
        method_name, y_true, y_scores, auc, roc_filename = best_method_info
        _, fpr, tpr, _, optimal_idx = find_optimal_threshold(y_true, y_scores)
        
        roc_path = FIGURES_DIR / roc_filename
        plot_roc_curve(fpr, tpr, auc, optimal_idx, method_name, roc_path)

if __name__ == "__main__":
    main()

