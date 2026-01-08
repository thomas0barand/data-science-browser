#!/usr/bin/env python3
"""
Unified script to export all types of embeddings:
- Baseline (raw frequencies)
- TF-IDF corrected (TF-IDF after decimation)
- Bigrams with TF-IDF
- Bigrams RAW (without TF-IDF)
"""
import json
import pickle
import sys
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import concat_sparse, sparse_embedding, tfidf_weights
from bigrams import bigram_embedding, tfidf_weights_bigram

DATA_DIR = PROJECT_ROOT / "data"

STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'have', 'this', 'which', 'or', 'but', 'not', 'can', 'all', 'we', 'been', 'their'
}

STOP_BIGRAMS = {
    'of the', 'in the', 'to the', 'and the', 'for the', 'on the', 'at the',
    'is the', 'to be', 'of a', 'in a', 'and a', 'it is', 'that is',
    'this is', 'as a', 'by the', 'from the', 'with the', 'that the'
}

def load_jsonl(path: Path):
    items = {}
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            if line.strip():
                data = json.loads(line)
                if "_id" in data:
                    items[str(data["_id"])] = data
    return items

def simple_stem(word: str):
    suffixes = ['ing', 'ed', 'ly', 'es', 's', 'tion', 'ment', 'ness', 'ity', 'er']
    for suffix in suffixes:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def decimate(ids, vocab, matrix, max_vocab_size=None, remove_stop_words=False, 
             min_freq=None, max_freq=None, use_stemming=False):
    term_freqs = [sum(doc[i] for doc in matrix) for i in range(len(vocab))]
    kept_indices = list(range(len(vocab)))
    
    if remove_stop_words:
        kept_indices = [i for i in kept_indices if vocab[i].lower() not in STOP_WORDS]
        print(f"After stop words: {len(kept_indices)} terms")
    
    if min_freq:
        kept_indices = [i for i in kept_indices if term_freqs[i] >= min_freq]
        print(f"After min_freq={min_freq}: {len(kept_indices)} terms")
    
    if max_freq:
        kept_indices = [i for i in kept_indices if term_freqs[i] <= max_freq]
        print(f"After max_freq={max_freq}: {len(kept_indices)} terms")
    
    if max_vocab_size and len(kept_indices) > max_vocab_size:
        kept_with_freq = [(i, term_freqs[i]) for i in kept_indices]
        kept_with_freq.sort(key=lambda x: x[1], reverse=True)
        kept_indices = [i for i, _ in kept_with_freq[:max_vocab_size]]
        print(f"After max_vocab_size={max_vocab_size}: {len(kept_indices)} terms")
    
    if use_stemming:
        print("Applying stemming...")
        stem_map = {}
        for i in kept_indices:
            stem = simple_stem(vocab[i])
            if stem not in stem_map:
                stem_map[stem] = []
            stem_map[stem].append(i)
        
        new_vocab = sorted(stem_map.keys())
        new_matrix = []
        for doc in tqdm(matrix, desc="Stemming"):
            new_doc = [sum(doc[idx] for idx in stem_map[stem]) for stem in new_vocab]
            new_matrix.append(new_doc)
        print(f"After stemming: {len(new_vocab)} terms")
    else:
        kept_indices.sort()
        new_vocab = [vocab[i] for i in kept_indices]
        new_matrix = [[doc[i] for i in kept_indices] for doc in matrix]
    
    print(f"Final: {len(matrix)} docs x {len(new_vocab)} terms")
    print(f"Reduction: {len(vocab)} -> {len(new_vocab)} ({100*(1-len(new_vocab)/len(vocab)):.1f}%)")
    
    return ids, new_vocab, new_matrix

def decimate_bigrams(vocab, matrix, max_vocab=5000, min_freq=2, remove_stops=True):
    term_freqs = [sum(doc[i] for doc in matrix) for i in range(len(vocab))]
    kept_indices = list(range(len(vocab)))
    
    if remove_stops:
        kept_indices = [i for i in kept_indices if vocab[i].lower() not in STOP_BIGRAMS]
        print(f"After removing stop bigrams: {len(kept_indices)} terms")
    
    if min_freq:
        kept_indices = [i for i in kept_indices if term_freqs[i] >= min_freq]
        print(f"After min_freq={min_freq}: {len(kept_indices)} terms")
    
    if max_vocab and len(kept_indices) > max_vocab:
        kept_with_freq = [(i, term_freqs[i]) for i in kept_indices]
        kept_with_freq.sort(key=lambda x: x[1], reverse=True)
        kept_indices = [i for i, _ in kept_with_freq[:max_vocab]]
        print(f"After max_vocab={max_vocab}: {len(kept_indices)} terms")
    
    kept_indices.sort()
    new_vocab = [vocab[i] for i in kept_indices]
    new_matrix = [[doc[i] for i in kept_indices] for doc in tqdm(matrix, desc="Rebuilding")]
    
    print(f"Final: {len(new_vocab)} terms ({100*(1-len(new_vocab)/len(vocab)):.1f}% reduction)")
    
    return new_vocab, new_matrix

def export_baseline():
    """Export baseline embeddings (raw frequencies with decimation)"""
    print("\n=== BASELINE: Raw Frequencies ===")
    OUTPUT_PATH = DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl"
    
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    
    ids, embeddings, seen = [], [], set()
    
    for qid, payload in tqdm(queries.items(), desc="Queries"):
        text = payload.get("text", "").strip()
        if text and qid not in seen:
            ids.append(qid)
            embeddings.append(sparse_embedding(text))
            seen.add(qid)
    
    for cid, payload in tqdm(corpus.items(), desc="Corpus"):
        if cid not in seen:
            title = (payload.get("title") or "").strip()
            if title:
                ids.append(cid)
                embeddings.append(sparse_embedding(title))
                seen.add(cid)
    
    ids, vocab, matrix = concat_sparse(ids, embeddings)
    print(f"\nOriginal: {len(ids)} docs, {len(vocab)} terms")
    
    ids, vocab, matrix = decimate(ids, vocab, matrix, max_vocab_size=3000, 
                                   remove_stop_words=True, min_freq=3, max_freq=350, use_stemming=True)
    
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)
    
    print(f"✓ Saved to {OUTPUT_PATH}")

def export_tfidf_corrected():
    """Export TF-IDF corrected (TF-IDF applied AFTER decimation)"""
    print("\n=== TF-IDF CORRECTED: TF-IDF After Decimation ===")
    OUTPUT_PATH = DATA_DIR / "sparses_embedding_tfidf_corrected.pkl"
    
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    
    ids, embeddings, seen = [], [], set()
    
    for qid, payload in tqdm(queries.items(), desc="Queries"):
        text = payload.get("text", "").strip()
        if text and qid not in seen:
            ids.append(qid)
            embeddings.append(sparse_embedding(text))
            seen.add(qid)
    
    for cid, payload in tqdm(corpus.items(), desc="Corpus"):
        if cid not in seen:
            title = (payload.get("title") or "").strip()
            if title:
                ids.append(cid)
                embeddings.append(sparse_embedding(title))
                seen.add(cid)
    
    ids, vocab, matrix = concat_sparse(ids, embeddings)
    print(f"\nOriginal: {len(ids)} docs, {len(vocab)} terms")
    
    ids, vocab, matrix = decimate(ids, vocab, matrix, max_vocab_size=3000,
                                   remove_stop_words=True, min_freq=3, max_freq=350, use_stemming=True)
    
    print("\n🔧 Applying TF-IDF on decimated vocabulary...")
    embeddings_decimated = [
        {vocab[i]: matrix[doc_idx][i] for i in range(len(vocab)) if matrix[doc_idx][i] > 0}
        for doc_idx in tqdm(range(len(matrix)), desc="Rebuilding embeddings")
    ]
    
    embeddings_tfidf, _ = tfidf_weights(embeddings_decimated)
    matrix = [[emb.get(word, 0) for word in vocab] for emb in tqdm(embeddings_tfidf, desc="TF-IDF matrix")]
    
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)
    
    print(f"✓ Saved to {OUTPUT_PATH}")

def export_bigram_tfidf():
    """Export bigram embeddings with TF-IDF"""
    print("\n=== BIGRAM TF-IDF ===")
    OUTPUT_PATH = DATA_DIR / "sparses_embedding_bigram_tfidf_decimated.pkl"
    
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
    ids, vocab, matrix = concat_sparse(ids, embeddings)
    
    print(f"\nOriginal: {len(ids)} docs, {len(vocab)} bigram terms")
    
    vocab, matrix = decimate_bigrams(vocab, matrix, max_vocab=5000, min_freq=2, remove_stops=True)
    
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)
    
    print(f"✓ Saved to {OUTPUT_PATH}")

def export_bigram_raw():
    """Export bigram embeddings without TF-IDF"""
    print("\n=== BIGRAM RAW: Without TF-IDF ===")
    OUTPUT_PATH = DATA_DIR / "sparses_embedding_bigram_raw_decimated.pkl"
    
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
    
    ids, vocab, matrix = concat_sparse(ids, embeddings)
    print(f"\nOriginal: {len(ids)} docs, {len(vocab)} bigram terms")
    
    vocab, matrix = decimate_bigrams(vocab, matrix, max_vocab=5000, min_freq=2, remove_stops=True)
    
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)
    
    print(f"✓ Saved to {OUTPUT_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Export embeddings for different methods")
    parser.add_argument('--method', choices=['baseline', 'tfidf', 'bigram', 'bigram_raw', 'all'],
                       default='all', help='Method to export (default: all)')
    args = parser.parse_args()
    
    methods = {
        'baseline': export_baseline,
        'tfidf': export_tfidf_corrected,
        'bigram': export_bigram_tfidf,
        'bigram_raw': export_bigram_raw
    }
    
    if args.method == 'all':
        for name, func in methods.items():
            print(f"\n{'='*60}")
            print(f"Exporting: {name}")
            print(f"{'='*60}")
            func()
    else:
        methods[args.method]()
    
    print("\n✅ All exports completed!")

if __name__ == "__main__":
    main()

