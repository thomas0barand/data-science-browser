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

# Mots-outils français et anglais communs
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'have', 'this', 'which', 'or', 'but', 'not', 'can', 'all', 'we', 'been', 'their'
}


def load_jsonl_dict(path: Path, id_field: str = "_id") -> dict[str, dict]:
    items: dict[str, dict] = {}
    print(f"Loading {path.name}...")
    with path.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=f"Loading {path.name}", unit="line"):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            key = data.get(id_field)
            if key is not None:
                items[str(key)] = data
    return items


def collect_texts(queries = "queries_small.jsonl", corpus = "corpus_small.jsonl", get_corpus_text = True, use_tfidf = False):
    queries = load_jsonl_dict(DATA_DIR / queries)
    corpus = load_jsonl_dict(DATA_DIR / corpus)

    ids: list[str] = []
    embeddings: list[dict[str, int]] = []
    seen_ids: set[str] = set()

    print("Processing queries...")
    for qid, payload in tqdm(queries.items(), desc="Queries", unit="query"):
        text = payload.get("text", "").strip()
        if not text or qid in seen_ids:
            continue
        ids.append(qid)
        embeddings.append(sparse_embedding(text))
        seen_ids.add(qid)

    print("Processing corpus...")
    for cid, payload in tqdm(corpus.items(), desc="Corpus", unit="doc"):
        if cid in seen_ids:
            continue
        
        # Combine title and text for corpus entries
        title = (payload.get("title") or "").strip()
        if get_corpus_text:
            text = payload.get("text", "").strip()
        else:
            text = ""
        combined_text = f"{title} {text}".strip()
        
        if not combined_text:
            continue
        
        ids.append(cid)
        embeddings.append(sparse_embedding(combined_text))
        seen_ids.add(cid)

    if not ids:
        raise RuntimeError("No texts available to embed.")

    # Apply TF-IDF if requested
    if use_tfidf:
        print("\nApplying TF-IDF weighting...")
        embeddings, _ = tfidf_weights(embeddings)
        print(f"TF-IDF applied to {len(embeddings)} documents")

    return ids, embeddings


def concat_sparse_with_progress(
    ids: list[str], embeddings: list[dict]
) -> tuple[list[str], list[str], list[list[float]]]:
    """
    Wrapper around concat_sparse with progress bar.
    Handles both int (frequency) and float (TF-IDF) embeddings.
    """
    print("Building vocabulary...")
    from src.utils import build_vocabulary
    vocab = build_vocabulary(embeddings)
    
    print("Building matrix...")
    matrix = []
    for emb in tqdm(embeddings, desc="Matrix", unit="doc"):
        matrix.append([emb.get(word, 0) for word in vocab])
    
    return list(ids), vocab, matrix


def simple_stem(word: str) -> str:
    """Simple stemming: remove common suffixes."""
    suffixes = ['ing', 'ed', 'ly', 'es', 's', 'tion', 'ment', 'ness', 'ity', 'er']
    for suffix in suffixes:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def decimate(
    ids: list[str],
    vocab: list[str],
    matrix: list[list[int]],
    max_vocab_size: int = None,
    remove_stop_words: bool = False,
    min_freq: int = None,
    max_freq: int = None,
    use_stemming: bool = False
) -> tuple[list[str], list[str], list[list[int]]]:
    """
    Decimate the vocabulary and matrix using various strategies.
    
    Args:
        ids: List of document IDs
        vocab: List of vocabulary terms
        matrix: Document-term matrix (list of lists)
        max_vocab_size: Keep only the top N most frequent terms
        remove_stop_words: Remove common stop words
        min_freq: Remove terms appearing less than this many times
        max_freq: Remove terms appearing more than this many times
        use_stemming: Apply simple stemming to reduce vocabulary
    
    Returns:
        (ids, new_vocab, new_matrix) with reduced vocabulary
    """
    
    # Calculate term frequencies
    term_freqs = [sum(doc[i] for doc in matrix) for i in tqdm(range(len(vocab)), desc="Calculating term frequencies", unit="term")]
    
    # Step 1: Remove stop words
    kept_indices = list(range(len(vocab)))
    if remove_stop_words:
        kept_indices = [
            i for i in kept_indices 
            if vocab[i].lower() not in STOP_WORDS
        ]
        print(f"\nAfter removing stop words: {len(kept_indices)} terms kept")
    
    # Step 2: Filter by frequency
    if min_freq is not None:
        kept_indices = [
            i for i in kept_indices 
            if term_freqs[i] >= min_freq
        ]
        print(f"After min_freq={min_freq}: {len(kept_indices)} terms kept")
    
    if max_freq is not None:
        kept_indices = [
            i for i in kept_indices 
            if term_freqs[i] <= max_freq
        ]
        print(f"After max_freq={max_freq}: {len(kept_indices)} terms kept")
    
    # Step 3: Keep only top N most frequent
    if max_vocab_size is not None and len(kept_indices) > max_vocab_size:
        # Sort by frequency
        kept_indices_with_freq = [(i, term_freqs[i]) for i in kept_indices]
        kept_indices_with_freq.sort(key=lambda x: x[1], reverse=True)
        kept_indices = [i for i, _ in kept_indices_with_freq[:max_vocab_size]]
        print(f"After max_vocab_size={max_vocab_size}: {len(kept_indices)} terms kept")
    
    # Step 4: Apply stemming if requested
    if use_stemming:
        print("\nApplying stemming...")
        # Create stemmed vocab
        stemmed_vocab_map = {}  # stem -> list of original indices
        for i in kept_indices:
            stem = simple_stem(vocab[i])
            if stem not in stemmed_vocab_map:
                stemmed_vocab_map[stem] = []
            stemmed_vocab_map[stem].append(i)
        
        # Build new vocab and matrix
        new_vocab = sorted(stemmed_vocab_map.keys())
        new_matrix = []
        
        for doc in tqdm(matrix, desc="Stemming", unit="doc"):
            new_doc = []
            for stem in new_vocab:
                # Sum counts for all terms that map to this stem
                count = sum(doc[orig_idx] for orig_idx in stemmed_vocab_map[stem])
                new_doc.append(count)
            new_matrix.append(new_doc)
        
        print(f"After stemming: {len(new_vocab)} stems")
    else:
        # Sort kept indices to maintain order
        kept_indices.sort()
        new_vocab = [vocab[i] for i in kept_indices]
        new_matrix = [
            [doc[i] for i in kept_indices]
            for doc in matrix
        ]
    
    print(f"\n{'='*60}")
    print(f"FINAL: {len(matrix)} docs x {len(new_vocab)} terms")
    print(f"Reduction: {len(vocab)} -> {len(new_vocab)} ({100*(1-len(new_vocab)/len(vocab)):.1f}% reduction)")
    print(f"{'='*60}\n")
    
    return ids, new_vocab, new_matrix


def main():

    USE_TFIDF = True  # Set to True to use TF-IDF, False for raw frequencies
    OUTPUT_PATH = DATA_DIR / ("sparses_embedding_tfidf.pkl" if USE_TFIDF else "sparses_embedding_without_corpus_text_decimated.pkl")
    corpus_name = "corpus.jsonl"
    queries_name = "queries.jsonl"

    ids, embeddings = collect_texts(queries=queries_name, corpus=corpus_name, get_corpus_text=False, use_tfidf=USE_TFIDF)
    ids, vocab, matrix = concat_sparse(ids, embeddings)
    
    print(f"\nOriginal embeddings: {len(ids)} docs with vocab size {len(vocab)}")
    

    # Uncomment and modify parameters as needed
    ids, vocab, matrix = decimate(
        ids, vocab, matrix,
        max_vocab_size=3000,        # Keep top 7000 most frequent terms
        remove_stop_words=True,     # Remove common stop words
        min_freq=3,                 # Remove terms appearing less than 3 times
        max_freq=350,              # upper frequency limit
        use_stemming=True          # Apply simple stemming
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)

    print(f"\nSaved {len(ids)} embeddings with vocab size {len(vocab)} to {OUTPUT_PATH}")
    print(f"Sample vocabulary: {vocab[:20]}")

if __name__ == "__main__":
    main()
