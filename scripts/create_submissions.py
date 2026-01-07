#!/usr/bin/env python3
import sys
import pickle
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

def calculate_predictions(ids, matrix, precomputed_norms, test_data):
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    predictions = {}
    
    for qid in tqdm(test_data.keys(), desc="Processing"):
        if qid not in id_to_idx:
            predictions[qid] = {cid: 0.0 for cid in test_data[qid]}
            continue
        
        q_idx = id_to_idx[qid]
        q_vec, q_norm = matrix[q_idx], precomputed_norms[q_idx]
        
        predictions[qid] = {}
        for cid in test_data[qid]:
            if cid in id_to_idx:
                c_idx = id_to_idx[cid]
                sim = cosine_similarity_sparse(q_vec, matrix[c_idx], q_norm, precomputed_norms[c_idx])
                predictions[qid][cid] = sim
            else:
                predictions[qid][cid] = 0.0
    
    return predictions

def save_submission(predictions, test_data, file_path):
    with file_path.open('w', encoding='utf-8') as f:
        f.write("RowId,query-id,corpus-id,score\n")
        row_id = 1
        for qid in test_data.keys():
            for cid in test_data[qid].keys():
                score = predictions[qid].get(cid, 0.0)
                f.write(f"{row_id},{qid},{cid},{score}\n")
                row_id += 1

def evaluate_on_valid(ids, matrix, precomputed_norms, valid_data):
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
    y_true, y_scores = [], []
    
    for qid in tqdm(valid_data.keys(), desc="Evaluating"):
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
    
    if len(y_true) > 0 and len(set(y_true)) > 1:
        return roc_auc_score(y_true, y_scores)
    return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    test_data = load_test_final(DATA_DIR / "test_final.tsv")
    valid_data = load_valid_tsv(DATA_DIR / "valid.tsv")
    
    print(f"{len(test_data)} test queries, {len(valid_data)} validation queries")
    
    methods = [
        ("Raw", DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl", "submission_raw.csv"),
        ("TF-IDF", DATA_DIR / "sparses_embedding_tfidf.pkl", "submission_tfidf.csv"),
        ("Bigram", DATA_DIR / "sparses_embedding_bigram_tfidf.pkl", "submission_bigram.csv"),
    ]
    
    for name, pkl_path, output_name in methods:
        print(f"\n{name}:")
        if not pkl_path.exists():
            print(f"  ⚠️ Not found: {pkl_path}")
            continue
        
        ids, vocab, matrix = load_embeddings(pkl_path)
        print(f"  {len(ids)} docs, {len(vocab)} terms")
        
        precomputed_norms = precompute_norms(matrix)
        
        auc = evaluate_on_valid(ids, matrix, precomputed_norms, valid_data)
        if auc:
            print(f"  AUC-ROC: {auc:.4f}")
        
        predictions = calculate_predictions(ids, matrix, precomputed_norms, test_data)
        output_path = OUTPUT_DIR / output_name
        save_submission(predictions, test_data, output_path)
        print(f"  ✓ Saved: {output_path.name}")

if __name__ == "__main__":
    main()

