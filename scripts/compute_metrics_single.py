#!/usr/bin/env python3
"""Compute metrics for a single predictions file"""
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"

def calculate_auc(y_true, y_scores):
    """Calculate AUC using Wilcoxon-Mann-Whitney statistic"""
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

def load_predictions(predictions_csv):
    """Load predictions from CSV"""
    import csv
    predictions = {}
    with predictions_csv.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        corpus_ids = header[1:]
        
        for row in tqdm(reader, desc="Loading predictions"):
            if row:
                predictions[row[0]] = {cid: float(val) for cid, val in zip(corpus_ids, row[1:])}
    
    return predictions, corpus_ids

def calculate_metrics_at_threshold(predictions, valid_tsv, threshold, auc):
    """Calculate metrics at a specific threshold"""
    y_true, y_pred = [], []
    
    with valid_tsv.open('r', encoding='utf-8') as f:
        next(f)
        for line in f:
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 3:
                    qid, cid, label = parts[0], parts[1], int(parts[2])
                    score = predictions.get(qid, {}).get(cid, 0.0)
                    pred = 1 if score > threshold else 0
                    y_true.append(label)
                    y_pred.append(pred)
    
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def main():
    parser = argparse.ArgumentParser(description="Compute metrics for a predictions file")
    parser.add_argument('predictions_file', type=str, help='Path to predictions CSV file')
    parser.add_argument('--threshold', type=float, help='Fixed threshold (default: find optimal)')
    
    args = parser.parse_args()
    
    predictions_csv = Path(args.predictions_file)
    valid_tsv = DATA_DIR / "valid.tsv"
    
    if not predictions_csv.exists():
        print(f"✗ File not found: {predictions_csv}")
        sys.exit(1)
    
    print(f"Loading predictions from {predictions_csv.name}...")
    predictions, corpus_ids = load_predictions(predictions_csv)
    
    print("Calculating AUC...")
    y_true, y_scores = [], []
    with valid_tsv.open('r', encoding='utf-8') as f:
        next(f)
        for line in tqdm(f, desc="AUC"):
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 3:
                    qid, cid, label = parts[0], parts[1], int(parts[2])
                    score = predictions.get(qid, {}).get(cid, 0.0)
                    y_true.append(label)
                    y_scores.append(score)
    
    auc = calculate_auc(y_true, y_scores)
    print(f"✓ AUC: {auc:.4f}")
    
    # Find optimal threshold if not provided
    if args.threshold is None:
        print("\nFinding optimal threshold...")
        best_f1 = 0
        best_threshold = 0.1
        best_metrics = None
        
        for thresh in tqdm([0.01 * i for i in range(100)], desc="Testing"):
            metrics = calculate_metrics_at_threshold(predictions, valid_tsv, thresh, auc)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = thresh
                best_metrics = metrics
        
        print(f"✓ Optimal threshold: {best_threshold:.3f}")
    else:
        best_threshold = args.threshold
        best_metrics = calculate_metrics_at_threshold(predictions, valid_tsv, best_threshold, auc)
    
    # Display results
    print(f"\n{'='*50}")
    print(f"METRICS")
    print(f"{'='*50}")
    print(f"Threshold:  {best_metrics['threshold']:.3f}")
    print(f"Precision:  {best_metrics['precision']:.4f}")
    print(f"Recall:     {best_metrics['recall']:.4f}")
    print(f"F1-Score:   {best_metrics['f1']:.4f}")
    print(f"AUC:        {best_metrics['auc']:.4f}")
    print(f"Accuracy:   {best_metrics['accuracy']:.4f}")
    print(f"{'='*50}")
    print(f"TP: {best_metrics['true_positives']}, FP: {best_metrics['false_positives']}")
    print(f"TN: {best_metrics['true_negatives']}, FN: {best_metrics['false_negatives']}")
    print(f"{'='*50}")
    
    # Save results
    output_file = OUTPUT_DIR / f"metrics_{predictions_csv.stem}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(best_metrics, f, indent=4)
    
    print(f"\n✓ Saved to: {output_file}")

if __name__ == "__main__":
    main()

