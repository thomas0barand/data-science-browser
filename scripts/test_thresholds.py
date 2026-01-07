#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.generate_predictions import load_predictions, evaluate_predictions, METHODS

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"

def main():
    parser = argparse.ArgumentParser(description="Test different thresholds for binary classification")
    parser.add_argument('--method', choices=['baseline', 'tfidf', 'bigram'], default='baseline',
                       help='Embedding method to test (default: baseline)')
    args = parser.parse_args()
    
    config = METHODS[args.method]
    predictions_csv = config["output"]
    valid_tsv = DATA_DIR / "valid.tsv"
    thresholds = [0.01 * i for i in range(100)]
    
    if not predictions_csv.exists():
        print(f"✗ Predictions file not found: {predictions_csv}")
        print(f"  Run: python scripts/generate_predictions.py --method {args.method}")
        return
    
    print(f"Testing {len(thresholds)} thresholds on {config['name']}")
    
    predictions, auc = load_predictions(predictions_csv, valid_tsv)
    results_summary = {}
    
    for threshold in thresholds:
        output_csv = OUTPUT_DIR / f"evaluation_{args.method}_threshold_{threshold:.2f}.csv"
        results = evaluate_predictions(predictions, valid_tsv, output_csv, threshold, auc)
        results_summary[threshold] = results
        
        summary_file = OUTPUT_DIR / f"results_summary_{args.method}.json"
        with open(summary_file, "w") as f:
            json.dump(results_summary, f, indent=4)
    
    print(f"\nResults in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

