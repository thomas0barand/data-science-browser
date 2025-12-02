#!/usr/bin/env python3
"""Test different thresholds for binary classification."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.generate_predictions_sparse import evaluate_predictions

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"

def main():
    predictions_csv = OUTPUT_DIR / "preds_sparse_embedding_without_corpus_text_decimated.csv"
    valid_tsv = DATA_DIR / "valid.tsv"
    
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print("="*70)
    print("TESTING DIFFERENT THRESHOLDS")
    print("="*70)
    print(f"\nPredictions: {predictions_csv.name}")
    print(f"Validation: {valid_tsv.name}\n")
    
    results_summary = []
    
    for threshold in thresholds:
        output_csv = OUTPUT_DIR / f"evaluation_threshold_{threshold:.2f}.csv"
        
        print(f"\n{'='*70}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'='*70}")
        
        evaluate_predictions(
            predictions_csv=predictions_csv,
            valid_tsv=valid_tsv,
            output_csv=output_csv,
            threshold=threshold
        )
        
        # Parse the metrics from output (we'd need to modify evaluate_predictions to return them)
        # For now, the function prints them
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nCheck the outputs/results/ directory for detailed results.")

if __name__ == "__main__":
    main()

