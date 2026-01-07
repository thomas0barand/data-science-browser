#!/usr/bin/env python3
"""Compute validation metrics for all methods (baseline, tfidf, bigram)"""
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.generate_predictions import load_predictions, METHODS

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"

def compute_metrics_for_method(method: str, optimal_threshold=None):
    """Compute precision, recall, f1, and AUC for a given method"""
    config = METHODS[method]
    predictions_csv = config["output"]
    valid_tsv = DATA_DIR / "valid.tsv"
    
    if not predictions_csv.exists():
        print(f"✗ Predictions not found for {method}: {predictions_csv}")
        print(f"  Run: poetry run python scripts/generate_predictions.py --method {method}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Computing metrics for: {config['name']}")
    print(f"{'='*70}")
    
    # Load predictions and calculate AUC
    predictions, auc = load_predictions(predictions_csv, valid_tsv)
    
    # Find optimal threshold if not provided
    if optimal_threshold is None:
        print("Finding optimal threshold...")
        best_f1 = 0
        best_threshold = 0.1
        best_metrics = None
        
        for thresh in tqdm([0.01 * i for i in range(100)], desc="Testing thresholds"):
            metrics = calculate_metrics_at_threshold(predictions, valid_tsv, thresh, auc)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = thresh
                best_metrics = metrics
        
        optimal_threshold = best_threshold
        print(f"✓ Optimal threshold: {optimal_threshold:.3f} (F1={best_f1:.4f})")
    else:
        best_metrics = calculate_metrics_at_threshold(predictions, valid_tsv, optimal_threshold, auc)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"METRICS FOR {config['name'].upper()}")
    print(f"{'='*70}")
    print(f"Threshold:  {optimal_threshold:.3f}")
    print(f"Precision:  {best_metrics['precision']:.4f}")
    print(f"Recall:     {best_metrics['recall']:.4f}")
    print(f"F1-Score:   {best_metrics['f1']:.4f}")
    print(f"AUC:        {best_metrics['auc']:.4f}")
    print(f"Accuracy:   {best_metrics['accuracy']:.4f}")
    print(f"{'='*70}")
    
    return {
        'method': method,
        'name': config['name'],
        'threshold': optimal_threshold,
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'auc': best_metrics['auc'],
        'accuracy': best_metrics['accuracy'],
        'confusion_matrix': {
            'tp': best_metrics['true_positives'],
            'fp': best_metrics['false_positives'],
            'tn': best_metrics['true_negatives'],
            'fn': best_metrics['false_negatives']
        }
    }

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
    parser = argparse.ArgumentParser(description="Compute validation metrics for all methods")
    parser.add_argument('--method', choices=['baseline', 'tfidf', 'bigram', 'all'], default='all',
                       help='Method to compute metrics for (default: all)')
    parser.add_argument('--threshold', type=float, help='Fixed threshold to use (default: find optimal)')
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    methods_to_compute = ['baseline', 'tfidf', 'bigram'] if args.method == 'all' else [args.method]
    
    all_results = {}
    for method in methods_to_compute:
        result = compute_metrics_for_method(method, args.threshold)
        if result:
            all_results[method] = result
    
    # Save summary
    summary_file = OUTPUT_DIR / "metrics_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✓ Summary saved to: {summary_file}")
    
    # Create comparison table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
        print(f"{'-'*70}")
        for method_key, result in all_results.items():
            print(f"{result['name']:<20} {result['precision']:<12.4f} {result['recall']:<12.4f} "
                  f"{result['f1']:<12.4f} {result['auc']:<12.4f}")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()

