#!/usr/bin/env python3
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

def load_jsonl(path: Path):
    items = {}
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            if line.strip():
                data = json.loads(line)
                if "_id" in data:
                    items[data["_id"]] = data
    return items

def collect_texts_bigram():
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    
    ids, embeddings, seen = [], [], set()
    
    for qid, payload in tqdm(queries.items(), desc="Queries"):
        text = payload.get("text", "").strip()
        if text and qid not in seen:
            ids.append(qid)
            embeddings.append(bigram_embedding(text))
            seen.add(qid)
    
    for cid, payload in tqdm(corpus.items(), desc="Corpus"):
        if cid not in seen:
            title = (payload.get("title") or "").strip()
            if title:
                ids.append(cid)
                embeddings.append(bigram_embedding(title))
                seen.add(cid)
    
    print("Applying TF-IDF...")
    embeddings, _ = tfidf_weights_bigram(embeddings)
    return ids, embeddings

def main():
    OUTPUT_PATH = DATA_DIR / "sparses_embedding_bigram_tfidf.pkl"
    
    ids, embeddings = collect_texts_bigram()
    ids, vocab, matrix = concat_sparse(ids, embeddings)
    
    print(f"\n{len(ids)} docs, {len(vocab)} terms")
    
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

