#!/usr/bin/env python3

import json
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

# Set up project root and path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bigrams import bigram_embedding, tfidf_weights_bigram
from utils import concat_sparse

DATA_DIR = PROJECT_ROOT / "data"

def load_jsonl(path: Path):
    """Load items (dict) from a .jsonl file given a Path"""
    items = {}
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            if line.strip():
                data = json.loads(line)
                if "_id" in data:
                    items[data["_id"]] = data  # Use "_id" as key
    return items

def collect_texts_bigram():
    """Generate bigram embeddings for queries and corpus titles"""
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    
    ids, embeddings, seen = [], [], set()
    
    # Process queries
    for qid, payload in tqdm(queries.items(), desc="Queries"):
        text = payload.get("text", "").strip()
        if text and qid not in seen:
            ids.append(qid)
            embeddings.append(bigram_embedding(text))  # bigram for query text
            seen.add(qid)
    
    # Process corpus (titles only)
    for cid, payload in tqdm(corpus.items(), desc="Corpus"):
        if cid not in seen:
            title = (payload.get("title") or "").strip()
            if title:
                ids.append(cid)
                embeddings.append(bigram_embedding(title))  # bigram for corpus title
                seen.add(cid)
    
    print("Applying TF-IDF...")
    embeddings, _ = tfidf_weights_bigram(embeddings)  # Apply TF-IDF weights to bigram embeddings
    return ids, embeddings

def main():
    OUTPUT_PATH = DATA_DIR / "sparses_embedding_bigram_tfidf.pkl"
    
    # Create bigram embeddings and apply TF-IDF
    ids, embeddings = collect_texts_bigram()
    ids, vocab, matrix = concat_sparse(ids, embeddings)  # Combine into matrix format
    
    print(f"\n{len(ids)} docs, {len(vocab)} terms")
    
    # Save all as pickle
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

