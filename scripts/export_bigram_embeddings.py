#!/usr/bin/env python3
"""Generate bigram embeddings with TF-IDF."""

import json
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bigrams import bigram_embedding, tfidf_weights_bigram
from utils import concat_sparse

DATA_DIR = PROJECT_ROOT / "data"

def load_jsonl_dict(path: Path, id_field: str = "_id") -> dict[str, dict]:
    items = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=f"Loading {path.name}"):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            key = data.get(id_field)
            if key is not None:
                items[str(key)] = data
    return items

def collect_texts_bigram(use_tfidf=True):
    queries = load_jsonl_dict(DATA_DIR / "queries.jsonl")
    corpus = load_jsonl_dict(DATA_DIR / "corpus.jsonl")
    
    ids = []
    embeddings = []
    seen_ids = set()
    
    print("Processing queries...")
    for qid, payload in tqdm(queries.items(), desc="Queries"):
        text = payload.get("text", "").strip()
        if not text or qid in seen_ids:
            continue
        ids.append(qid)
        embeddings.append(bigram_embedding(text))
        seen_ids.add(qid)
    
    print("Processing corpus...")
    for cid, payload in tqdm(corpus.items(), desc="Corpus"):
        if cid in seen_ids:
            continue
        title = (payload.get("title") or "").strip()
        combined_text = title
        if not combined_text:
            continue
        ids.append(cid)
        embeddings.append(bigram_embedding(combined_text))
        seen_ids.add(cid)
    
    if use_tfidf:
        print("\nApplying TF-IDF...")
        embeddings, _ = tfidf_weights_bigram(embeddings)
    
    return ids, embeddings

def main():
    OUTPUT_PATH = DATA_DIR / "sparses_embedding_bigram_tfidf.pkl"
    
    ids, embeddings = collect_texts_bigram(use_tfidf=True)
    ids, vocab, matrix = concat_sparse(ids, embeddings)
    
    print(f"\nBigram embeddings: {len(ids)} docs with vocab size {len(vocab)}")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)
    
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

