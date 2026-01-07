#!/usr/bin/env python3
"""Decimate bigram embeddings to reduce vocabulary size"""
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

STOP_BIGRAMS = {
    # Common bigrams with stop words
    'of the', 'in the', 'to the', 'and the', 'for the', 'on the', 'at the',
    'is the', 'to be', 'of a', 'in a', 'and a', 'it is', 'that is',
    'this is', 'as a', 'by the', 'from the', 'with the', 'that the'
}

def decimate_bigram(vocab, matrix, max_vocab=5000, min_freq=2, remove_stops=True):
    """Decimate bigram vocabulary"""
    
    print(f"Original: {len(vocab)} bigram terms")
    
    # Calculate term frequencies
    term_freqs = [sum(doc[i] for doc in matrix) for i in range(len(vocab))]
    
    # Filter indices
    kept_indices = list(range(len(vocab)))
    
    # Remove stop bigrams
    if remove_stops:
        kept_indices = [i for i in kept_indices if vocab[i].lower() not in STOP_BIGRAMS]
        print(f"After removing stop bigrams: {len(kept_indices)} terms")
    
    # Remove low frequency
    if min_freq:
        kept_indices = [i for i in kept_indices if term_freqs[i] >= min_freq]
        print(f"After min_freq={min_freq}: {len(kept_indices)} terms")
    
    # Keep only top N
    if max_vocab and len(kept_indices) > max_vocab:
        kept_with_freq = [(i, term_freqs[i]) for i in kept_indices]
        kept_with_freq.sort(key=lambda x: x[1], reverse=True)
        kept_indices = [i for i, _ in kept_with_freq[:max_vocab]]
        print(f"After max_vocab={max_vocab}: {len(kept_indices)} terms")
    
    # Build new vocab and matrix
    kept_indices.sort()
    new_vocab = [vocab[i] for i in kept_indices]
    new_matrix = [[doc[i] for i in kept_indices] for doc in tqdm(matrix, desc="Rebuilding matrix")]
    
    print(f"Final: {len(new_vocab)} terms ({100*(1-len(new_vocab)/len(vocab)):.1f}% reduction)")
    
    return new_vocab, new_matrix

def main():
    INPUT_PATH = DATA_DIR / "sparses_embedding_bigram_tfidf.pkl"
    OUTPUT_PATH = DATA_DIR / "sparses_embedding_bigram_tfidf_decimated.pkl"
    
    print("Loading bigram embeddings...")
    with INPUT_PATH.open("rb") as f:
        data = pickle.load(f)
    
    ids = data["ids"]
    vocab = data["vocab"]
    matrix = data["matrix"]
    
    print(f"Loaded: {len(ids)} docs, {len(vocab)} terms")
    
    # Decimate (similar to baseline settings)
    new_vocab, new_matrix = decimate_bigram(
        vocab, matrix,
        max_vocab=5000,      # Keep top 5000 bigrams
        min_freq=2,          # At least 2 occurrences
        remove_stops=True    # Remove common stop bigrams
    )
    
    # Save decimated version
    print(f"\nSaving to {OUTPUT_PATH}...")
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": new_vocab, "matrix": new_matrix}, f)
    
    # Show file sizes
    input_size = INPUT_PATH.stat().st_size / (1024**3)
    output_size = OUTPUT_PATH.stat().st_size / (1024**3)
    print(f"Input size:  {input_size:.2f} GB")
    print(f"Output size: {output_size:.2f} GB ({100*output_size/input_size:.1f}% of original)")
    print(f"\n✓ Done! Now update generate_predictions.py to use decimated bigram")

if __name__ == "__main__":
    main()

