import json
import pickle
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import concat_sparse, sparse_embedding, tfidf_weights
DATA_DIR = PROJECT_ROOT / "data"

STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'have', 'this', 'which', 'or', 'but', 'not', 'can', 'all', 'we', 'been', 'their'
}

def load_jsonl(path: Path):
    items = {}
    print(f"Loading {path.name}...")
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading"):
            if line.strip():
                data = json.loads(line)
                if "_id" in data:
                    items[str(data["_id"])] = data
    return items

def collect_texts(queries="queries.jsonl", corpus="corpus.jsonl", get_corpus_text=True, use_tfidf=False):
    queries = load_jsonl(DATA_DIR / queries)
    corpus = load_jsonl(DATA_DIR / corpus)

    ids, embeddings, seen = [], [], set()

    print("Processing queries...")
    for qid, payload in tqdm(queries.items(), desc="Queries"):
        text = payload.get("text", "").strip()
        if text and qid not in seen:
            ids.append(qid)
            embeddings.append(sparse_embedding(text))
            seen.add(qid)

    print("Processing corpus...")
    for cid, payload in tqdm(corpus.items(), desc="Corpus"):
        if cid not in seen:
            title = (payload.get("title") or "").strip()
            text = payload.get("text", "").strip() if get_corpus_text else ""
            combined = f"{title} {text}".strip()
            
            if combined:
                ids.append(cid)
                embeddings.append(sparse_embedding(combined))
                seen.add(cid)

    if use_tfidf:
        print("Applying TF-IDF...")
        embeddings, _ = tfidf_weights(embeddings)

    return ids, embeddings

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

def main():
    USE_TFIDF = True
    OUTPUT_PATH = DATA_DIR / ("sparses_embedding_tfidf.pkl" if USE_TFIDF else "sparses_embedding_without_corpus_text_decimated.pkl")
    
    ids, embeddings = collect_texts("queries.jsonl", "corpus.jsonl", get_corpus_text=False, use_tfidf=USE_TFIDF)
    ids, vocab, matrix = concat_sparse(ids, embeddings)
    
    print(f"\nOriginal: {len(ids)} docs, {len(vocab)} terms")
    
    ids, vocab, matrix = decimate(
        ids, vocab, matrix,
        max_vocab_size=3000,
        remove_stop_words=True,
        min_freq=3,
        max_freq=350,
        use_stemming=True
    )

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)

    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
