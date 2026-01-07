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
OUTPUT_FILE = OUTPUT_DIR / "preds_sparse_embedding_without_corpus_text_decimated.csv"

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

def calculate_auc(y_true, y_scores):
    if len(set(y_true)) < 2:
        return 0.0
    
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    pairs = list(zip(y_scores, y_true))
    auc_sum = 0.0
    for score_i, label_i in pairs:
        if label_i == 1:
            for score_j, label_j in pairs:
                if label_j == 0:
                    if score_i > score_j:
                        auc_sum += 1.0
                    elif score_i == score_j:
                        auc_sum += 0.5
    
    return auc_sum / (n_pos * n_neg)

def load_predictions(predictions_csv: Path, valid_tsv: Path):
    print("Loading predictions...")
    
    with predictions_csv.open('r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1
    
    predictions = {}
    with predictions_csv.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        corpus_ids = header[1:]
        
        for row in tqdm(reader, total=total_rows, desc="Loading"):
            if row:
                predictions[row[0]] = {cid: float(val) for cid, val in zip(corpus_ids, row[1:])}
    
    print(f"✓ Loaded {len(predictions)} queries")
    
    print("Calculating AUC...")
    y_true, y_scores = [], []
    with valid_tsv.open('r', encoding='utf-8') as f:
        next(f)
        for line in tqdm(f, desc="AUC"):
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 3:
                    qid, cid, label = parts[0], parts[1], parts[2]
                    score = predictions.get(qid, {}).get(cid, 0.0)
                    y_true.append(int(label))
                    y_scores.append(score)
    
    auc = calculate_auc(y_true, y_scores)
    print(f"✓ AUC: {auc:.4f}\n")
    return predictions, auc

def evaluate_predictions(predictions, valid_tsv: Path, output_csv: Path, threshold=0.1, auc=0.0):
    print(f"Evaluating with threshold={threshold}")
    
    results = []
    with valid_tsv.open('r', encoding='utf-8') as f:
        next(f)
        for line in tqdm(f, desc="Evaluating"):
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 3:
                    qid, cid, label = parts[0], parts[1], parts[2]
                    score = predictions.get(qid, {}).get(cid, 0.0)
                    pred = 1 if score > threshold else 0
                    results.append({
                        'query_id': qid, 'corpus_id': cid, 
                        'ground_truth': label, 'similarity': score, 'prediction': pred
                    })
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'corpus_id', 'ground_truth', 'similarity', 'prediction'])
        writer.writeheader()
        writer.writerows(results)
    
    correct = sum(1 for r in results if int(r['ground_truth']) == r['prediction'])
    tp = sum(1 for r in results if r['prediction'] == 1 and int(r['ground_truth']) == 1)
    fp = sum(1 for r in results if r['prediction'] == 1 and int(r['ground_truth']) == 0)
    tn = sum(1 for r in results if r['prediction'] == 0 and int(r['ground_truth']) == 0)
    fn = sum(1 for r in results if r['prediction'] == 0 and int(r['ground_truth']) == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "threshold": threshold,
        "total_pairs": len(results),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": correct / len(results),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    
    return metrics

def generate_predictions():
    print("Generating predictions")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading embeddings...")
    ids, vocab, matrix = load_embeddings(DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl")
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
        print(f"✓ Starting: {len(query_ids)} queries")
    
    if not remaining:
        print("✓ All done!")
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['generate', 'evaluate'], default='generate')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--predictions-csv', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    if args.mode == 'generate':
        generate_predictions()
    else:
        predictions_csv = Path(args.predictions_csv) if args.predictions_csv else OUTPUT_FILE
        output_csv = Path(args.output) if args.output else OUTPUT_DIR / "evaluation_results.csv"
        valid_tsv = DATA_DIR / "valid.tsv"
        
        predictions, auc = load_predictions(predictions_csv, valid_tsv)
        evaluate_predictions(predictions, valid_tsv, output_csv, args.threshold, auc)
