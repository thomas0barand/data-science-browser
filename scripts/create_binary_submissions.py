#!/usr/bin/env python3
import sys
import pickle
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
            if len(parts) >= 2:
                qid, cid = parts[0], parts[1]
                if qid not in test_data:
                    test_data[qid] = {}
                test_data[qid][cid] = 0
    return test_data

def load_valid_tsv(path):
    valid_data = {}
    with path.open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, cid, label = parts[0], parts[1], int(parts[2])
                if qid not in valid_data:
                    valid_data[qid] = {}
                valid_data[qid][cid] = label
    return valid_data

def get_scores_on_valid(ids, matrix, precomputed_norms, valid_data):
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    y_true, y_scores = [], []
    
    for qid in tqdm(valid_data.keys(), desc="Computing scores"):
        if qid not in id_to_idx:
            continue
        q_idx = id_to_idx[qid]
        q_vec, q_norm = matrix[q_idx], precomputed_norms[q_idx]
        
        for cid, label in valid_data[qid].items():
            if cid in id_to_idx:
                c_idx = id_to_idx[cid]
                sim = cosine_similarity_sparse(q_vec, matrix[c_idx], q_norm, precomputed_norms[c_idx])
                y_true.append(label)
                y_scores.append(sim)
    
    return np.array(y_true), np.array(y_scores)

def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx], fpr, tpr, thresholds, optimal_idx

def plot_roc_curve(fpr, tpr, auc_score, optimal_idx, method_name, save_path):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
             label=f'Optimal (FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {method_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()

def calculate_predictions_binary(ids, matrix, precomputed_norms, test_data, threshold):
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    predictions = {}
    
    for qid in tqdm(test_data.keys(), desc="Generating predictions"):
        if qid not in id_to_idx:
            predictions[qid] = {cid: 0 for cid in test_data[qid]}
            continue
        
        q_idx = id_to_idx[qid]
        q_vec, q_norm = matrix[q_idx], precomputed_norms[q_idx]
        predictions[qid] = {}
        
        for cid in test_data[qid]:
            if cid in id_to_idx:
                c_idx = id_to_idx[cid]
                sim = cosine_similarity_sparse(q_vec, matrix[c_idx], q_norm, precomputed_norms[c_idx])
                predictions[qid][cid] = 1 if sim >= threshold else 0
            else:
                predictions[qid][cid] = 0
    
    return predictions

def save_submission(predictions, test_data, file_path):
    with file_path.open('w', encoding='utf-8') as f:
        f.write("RowId,query-id,corpus-id,score\n")
        row_id = 1
        for qid in test_data.keys():
            for cid in test_data[qid].keys():
                score = predictions[qid].get(cid, 0)
                f.write(f"{row_id},{qid},{cid},{score}\n")
                row_id += 1

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    test_data = load_test_final(DATA_DIR / "test_final.tsv")
    valid_data = load_valid_tsv(DATA_DIR / "valid.tsv")
    
    print(f"{len(test_data)} test queries, {len(valid_data)} validation queries")
    
    methods = [
        ("Raw", DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl", 
         "submission_raw_binary.csv", "roc_curve_raw.png"),
        ("TF-IDF", DATA_DIR / "sparses_embedding_tfidf.pkl", 
         "submission_tfidf_binary.csv", "roc_curve_tfidf.png"),
        ("Bigram", DATA_DIR / "sparses_embedding_bigram_tfidf.pkl", 
         "submission_bigram_binary.csv", "roc_curve_bigram.png"),
    ]
    
    best_auc, best_info = 0, None
    
    for name, pkl_path, output_name, roc_filename in methods:
        print(f"\n{name}:")
        if not pkl_path.exists():
            print(f"  ⚠️ Not found")
            continue
        
        ids, vocab, matrix = load_embeddings(pkl_path)
        print(f"  {len(ids)} docs, {len(vocab)} terms")
        precomputed_norms = precompute_norms(matrix)
        
        y_true, y_scores = get_scores_on_valid(ids, matrix, precomputed_norms, valid_data)
        auc = roc_auc_score(y_true, y_scores)
        print(f"  AUC-ROC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc, best_info = auc, (name, y_true, y_scores, auc, roc_filename)
        
        threshold, fpr, tpr, _, opt_idx = find_optimal_threshold(y_true, y_scores)
        print(f"  Threshold: {threshold:.4f} (TPR={tpr[opt_idx]:.3f}, FPR={fpr[opt_idx]:.3f})")
        
        predictions = calculate_predictions_binary(ids, matrix, precomputed_norms, test_data, threshold)
        save_submission(predictions, test_data, OUTPUT_DIR / output_name)
        
        ones = sum(sum(1 for v in p.values() if v == 1) for p in predictions.values())
        total = sum(len(p) for p in predictions.values())
        print(f"  Predicted 1: {ones}/{total} ({100*ones/total:.1f}%)")
    
    if best_info:
        print(f"\nBest: {best_info[0]} (AUC={best_info[3]:.4f})")
        _, fpr, tpr, _, opt_idx = find_optimal_threshold(best_info[1], best_info[2])
        plot_roc_curve(fpr, tpr, best_info[3], opt_idx, best_info[0], FIGURES_DIR / best_info[4])

if __name__ == "__main__":
    main()

