#!/usr/bin/env python3
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.generate_predictions_sparse import load_predictions, evaluate_predictions

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"

def main():
    predictions_csv = OUTPUT_DIR / "preds_sparse_embedding_without_corpus_text_decimated.csv"
    valid_tsv = DATA_DIR / "valid.tsv"
    thresholds = [0.01 * i for i in range(100)]
    
    print(f"Testing {len(thresholds)} thresholds on {predictions_csv.name}")
    
    predictions, auc = load_predictions(predictions_csv, valid_tsv)
    results_summary = {}
    
    for threshold in thresholds:
        output_csv = OUTPUT_DIR / f"evaluation_threshold_{threshold:.2f}.csv"
        results = evaluate_predictions(predictions, valid_tsv, output_csv, threshold, auc)
        results_summary[threshold] = results
        
        with open(OUTPUT_DIR / "results_summary.json", "w") as f:
            json.dump(results_summary, f, indent=4)
    
    print(f"\nResults in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

