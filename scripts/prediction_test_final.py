#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results"

def load_predictions_simple(predictions_csv: Path):
    print(f"Loading {predictions_csv.name}...")
    
    with predictions_csv.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        corpus_ids = header[1:]
        
        predictions = {}
        for row in tqdm(reader, desc="Loading"):
            if row:
                predictions[row[0]] = {cid: float(val) for cid, val in zip(corpus_ids, row[1:])}
    
    print(f"✓ Loaded {len(predictions)} queries\n")
    return predictions

def generate_test_predictions(predictions, test_tsv: Path, output_csv: Path):
    print(f"Processing {test_tsv.name}...")
    results, missing = [], 0
    
    with test_tsv.open('r', encoding='utf-8') as f:
        next(f)
        for line in tqdm(f, desc="Processing"):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid, cid = parts[0], parts[1]
                score = predictions.get(qid, {}).get(cid, 0.0)
                if score == 0.0:
                    missing += 1
                results.append({'query-id': qid, 'corpus-id': cid, 'score': score})
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query-id', 'corpus-id', 'score'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Saved {len(results)} predictions ({missing} missing)")
    return len(results)

def main():
    predictions_csv = OUTPUT_DIR / "preds_sparse_embedding_without_corpus_text_decimated.csv"
    test_tsv = DATA_DIR / "test_final.tsv"
    output_csv = PROJECT_ROOT / "outputs" / "submissions" / "test_final_predictions.csv"
    
    predictions = load_predictions_simple(predictions_csv)
    num_rows = generate_test_predictions(predictions, test_tsv, output_csv)
    
    print(f"\n✓ Done! {num_rows} rows saved to {output_csv.name}")

if __name__ == "__main__":
    main()
