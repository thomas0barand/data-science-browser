#!/usr/bin/env python3
"""Quick status check for predictions and metrics"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

def check_file(path, description):
    exists = path.exists()
    status = "✓" if exists else "✗"
    size = f"({path.stat().st_size / 1024 / 1024:.1f} MB)" if exists else ""
    print(f"{status} {description:<40} {size}")
    return exists

print("\n" + "="*70)
print("STATUS CHECK - Data Science Browser Project")
print("="*70)

print("\n📦 EMBEDDINGS (data/):")
check_file(DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl", "Baseline embeddings")
check_file(DATA_DIR / "sparses_embedding_tfidf.pkl", "TF-IDF embeddings")
check_file(DATA_DIR / "sparses_embedding_bigram_tfidf.pkl", "Bigram embeddings")

print("\n📊 PREDICTIONS (outputs/results/):")
baseline_preds = check_file(RESULTS_DIR / "preds_sparse_embedding_without_corpus_text_decimated.csv", "Baseline predictions")
tfidf_preds = check_file(RESULTS_DIR / "preds_tfidf.csv", "TF-IDF predictions")
bigram_preds = check_file(RESULTS_DIR / "preds_bigram.csv", "Bigram predictions")

print("\n📈 METRICS (outputs/results/):")
baseline_metrics = check_file(RESULTS_DIR / "metrics_preds_sparse_embedding_without_corpus_text_decimated.json", "Baseline metrics")
summary = check_file(RESULTS_DIR / "metrics_summary.json", "All methods summary")

print("\n" + "="*70)
print("TODO:")
print("="*70)

todos = []
if not bigram_preds:
    todos.append("1. Generate bigram predictions:")
    todos.append("   poetry run python scripts/generate_predictions.py --method bigram")

if not summary or not (baseline_metrics and tfidf_preds and bigram_preds):
    todos.append("2. Compute metrics for all methods:")
    todos.append("   poetry run python scripts/compute_metrics.py --method all")

if not todos:
    print("✓ All predictions and metrics are ready!")
    print("\nYou can now:")
    print("  - View comparison: cat outputs/results/metrics_summary.json")
    print("  - Create submissions: poetry run python scripts/create_submissions.py")
    print("  - Generate visualizations: poetry run python scripts/visualizations.py")
else:
    for todo in todos:
        print(todo)

print("="*70 + "\n")

