import re
import math
from collections import Counter
from typing import Sequence
from tqdm import tqdm


def sparse_embedding(text: str) -> dict[str, int]:
    clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = [w for w in clean.split() if w]
    return Counter(words)


def tfidf_weights(
    embeddings: Sequence[dict[str, int]]
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """
    Convert frequency-based embeddings to TF-IDF weights.
    
    Args:
        embeddings: List of word frequency dictionaries
    
    Returns:
        (tfidf_embeddings, idf_dict)
        - tfidf_embeddings: List of TF-IDF weighted dictionaries
        - idf_dict: Dictionary of IDF values for each term
    """
    n_docs = len(embeddings)
    
    # Calculate document frequency (DF) for each term
    df = Counter()
    for emb in embeddings:
        for word in emb.keys():
            df[word] += 1
    
    # Calculate IDF: log(N / DF)
    idf = {word: math.log(n_docs / freq) for word, freq in df.items()}
    
    # Calculate TF-IDF for each document
    tfidf_embeddings = []
    for emb in embeddings:
        doc_length = sum(emb.values())
        if doc_length == 0:
            tfidf_embeddings.append({})
            continue
        
        # TF: normalized frequency, IDF: inverse document frequency
        tfidf_doc = {
            word: (freq / doc_length) * idf[word]
            for word, freq in emb.items()
        }
        tfidf_embeddings.append(tfidf_doc)
    
    return tfidf_embeddings, idf


def build_vocabulary(embeddings: Sequence[dict[str, int]]) -> list[str]:
    """Return sorted list of unique tokens across embeddings."""
    return sorted({word for emb in embeddings for word in emb})


def concat_sparse(
    ids: Sequence[str], embeddings: Sequence[dict[str, int]]
) -> tuple[list[str], list[str], list[list[int]]]:
    """
    Build a dense matrix from several sparse embeddings.
    Returns (row_ids, vocab, matrix).
    """
    if len(ids) != len(embeddings):
        raise ValueError("ids and embeddings must have the same length")

    vocab = build_vocabulary(embeddings)
    matrix = [[emb.get(word, 0) for word in vocab] for emb in tqdm(embeddings, desc="Building matrix", unit="doc")]
    return list(ids), vocab, matrix


if __name__ == "__main__":
    txt1 = "Hello, world! Hello again. I'm Thomas learning to split words into a dictionary."
    txt2 = "Hello again, world. Words split simply into frequencies."
    txt3 = "Hello, world! Hello again. I'm Thomas learning to split words words and frequencies into a dictionary."
    emb1 = sparse_embedding(txt1)
    emb2 = sparse_embedding(txt2)
    emb3 = sparse_embedding(txt3)
    ids, vocab, matrix = concat_sparse(["txt1", "txt2", "txt3"], [emb1, emb2, emb3])
    print(ids)
    print(vocab)
    print(matrix)
