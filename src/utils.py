import re
from collections import Counter
from typing import Sequence


def sparse_embedding(text: str) -> dict[str, int]:
    clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = [w for w in clean.split() if w]
    return Counter(words)


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
    matrix = [[emb.get(word, 0) for word in vocab] for emb in embeddings]
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
