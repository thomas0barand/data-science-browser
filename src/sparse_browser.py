#!/usr/bin/env python3
"""
Optimized sparse embedding search engine using cosine similarity.

Ce script implémente un moteur de recherche basé sur des embeddings creux (sparse).
Il utilise la mesure du cosinus pour comparer des vecteurs dans une matrice Documents x Termes.

Fonctionnalités:
1. Comparaison de paires de documents (calcul du cosinus et identification des mots communs)
2. Moteur de recherche par mots-clés (scoring de tous les documents et classement)

La similarité cosinus mesure l'angle entre deux vecteurs, donnant un score entre 0 et 1:
- 1.0 = documents identiques
- 0.0 = aucun mot en commun

OPTIMIZATIONS:
- Sparse cosine similarity: Only computes dot product for non-zero elements
- Pre-computed norms: Calculates document norms once for reuse across queries
- Embedding reuse: Can use document IDs as queries to avoid recalculating embeddings
- Memory efficiency: Works with sparse representations (mostly zeros)
"""



import json
import pickle
import math
from pathlib import Path
from typing import Optional
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

from utils import sparse_embedding




def cosine_similarity_manual(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def cosine_similarity_sparse(vec1: list, vec2: list, norm1: float = None, norm2: float = None) -> float:
    """
    Optimized cosine similarity for sparse vectors.
    Only computes dot product for non-zero elements.
    
    Args:
        vec1: First vector
        vec2: Second vector  
        norm1: Pre-computed norm of vec1 (optional)
        norm2: Pre-computed norm of vec2 (optional)
    """
    # Compute dot product only for non-zero elements
    dot_product = 0.0
    for i in range(len(vec1)):
        if vec1[i] != 0 and vec2[i] != 0:
            dot_product += vec1[i] * vec2[i]
    
    # Compute norms if not provided
    if norm1 is None:
        norm1 = math.sqrt(sum(a * a for a in vec1 if a != 0))
    if norm2 is None:
        norm2 = math.sqrt(sum(b * b for b in vec2 if b != 0))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def precompute_norms(matrix: list[list[int]]) -> list[float]:
    """Pre-compute norms for all document vectors."""
    return [math.sqrt(sum(val * val for val in vec if val != 0)) for vec in matrix]


def load_embeddings(pkl_path: Path = DATA_DIR / "sparses_embedding_without_corpus_text.pkl"):
    """Load sparse embeddings from pickle file."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    return data["ids"], data["vocab"], data["matrix"]


def load_documents(corpus_path: Path = DATA_DIR / "corpus.jsonl"):
    """Load document metadata from JSONL."""
    docs = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docs[doc["_id"]] = doc
    return docs


def query_to_vector(query_text: str, vocab: list[str]) -> list[int]:
    """Convert query text to sparse vector using existing vocabulary."""
    query_embedding = sparse_embedding(query_text)
    return [query_embedding.get(word, 0) for word in vocab]


def search(
    query: str,
    ids: list[str],
    vocab: list[str],
    matrix: list[list[int]],
    top_k: int = 10,
    precomputed_norms: list[float] = None
) -> list[tuple[str, float]]:
    """
    Search documents using cosine similarity (optimized for sparse vectors).
    
    Args:
        query: Either a text query OR a document ID from the matrix
        ids: List of document IDs
        vocab: Vocabulary list
        matrix: Document-term matrix
        top_k: Number of results to return
        precomputed_norms: Pre-computed norms for all documents (optional, for speed)
    
    Returns:
        List of (doc_id, score) tuples sorted by score
    """
    # Check if query is an existing document ID
    if query in ids:
        # Reuse embedding from matrix
        query_idx = ids.index(query)
        query_vec = matrix[query_idx]
        print(f"Using existing embedding for document ID: {query}")
    else:
        # Create new embedding from text
        query_vec = query_to_vector(query, vocab)
    
    # Pre-compute document norms if not provided
    if precomputed_norms is None:
        precomputed_norms = precompute_norms(matrix)
    
    # Compute query norm once
    query_norm = math.sqrt(sum(val * val for val in query_vec if val != 0))
    
    # Calculate similarities using optimized sparse function
    scores = [
        cosine_similarity_sparse(query_vec, doc_vec, query_norm, doc_norm)
        for doc_vec, doc_norm in tqdm(zip(matrix, precomputed_norms), desc="Calculating similarities", unit="doc")
    ]
    
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def compare_documents(
    doc_id1: str,
    doc_id2: str,
    ids: list[str],
    vocab: list[str],
    matrix: list[list[int]],
    precomputed_norms: list[float] = None
) -> tuple[float, list[str]]:
    """
    Compare two documents using optimized cosine similarity.
    Returns (similarity_score, common_words).
    """
    try:
        idx1 = ids.index(doc_id1)
        idx2 = ids.index(doc_id2)
    except ValueError as e:
        raise ValueError(f"Document ID not found: {e}")
    
    vec1 = matrix[idx1]
    vec2 = matrix[idx2]
    
    # Use optimized sparse similarity
    if precomputed_norms is not None:
        similarity = cosine_similarity_sparse(vec1, vec2, precomputed_norms[idx1], precomputed_norms[idx2])
    else:
        similarity = cosine_similarity_sparse(vec1, vec2)
    
    # Find common words (only check non-zero positions)
    common_words = [
        vocab[i] for i in range(len(vocab))
        if vec1[i] > 0 and vec2[i] > 0
    ]
    
    return similarity, common_words


def display_document(doc_id: str, docs: dict, score: Optional[float] = None):
    """Display document information."""
    if doc_id not in docs:
        print(f"  ID: {doc_id}")
        if score is not None:
            print(f"  Score: {score:.4f}")
        print()
        return
    
    doc = docs[doc_id]
    print(f"  ID: {doc_id}")
    if score is not None:
        print(f"  Score: {score:.4f}")
    print(f"  Text: {doc.get('text', 'N/A')[:200]}...")
    if 'title' in doc:
        print(f"  Title: {doc.get('title', 'N/A')[:150]}...")
    print(f"  Year: {doc.get('metadata', {}).get('year', 'N/A')}")
    print()


def print_vocabulary_stats(vocab: list[str], matrix: list[list[int]]):
    """Print statistics about the vocabulary."""
    print("\nVocabulary Statistics:")
    print(f"  Total unique terms: {len(vocab)}")
    print(f"  Sample terms: {', '.join(vocab[:15])}...")
    
    term_frequencies = [sum(doc[i] for doc in matrix) for i in range(len(vocab))]
    top_terms_idx = sorted(range(len(term_frequencies)), 
                           key=lambda i: term_frequencies[i], 
                           reverse=True)[:10]
    
    print("\n  Top 10 most frequent terms:")
    for idx in top_terms_idx:
        print(f"    '{vocab[idx]}': {term_frequencies[idx]} occurrences")


def main():
    print("Loading embeddings...")
    ids, vocab, matrix = load_embeddings(pkl_path=DATA_DIR / "sparses_embedding_without_corpus_text.pkl")
    print(f"Loaded {len(ids)} embeddings with vocab size {len(vocab)}")
    
    print("\nLoading documents...")
    docs = load_documents(corpus_path=DATA_DIR / "corpus.jsonl")
    queries = load_documents(corpus_path=DATA_DIR / "queries.jsonl")
    print(f"Loaded {len(docs)} documents and {len(queries)} queries")
    
    # Pre-compute norms for optimization
    print("\nPre-computing document norms for optimization...")
    precomputed_norms = precompute_norms(matrix)
    print(f"Pre-computed {len(precomputed_norms)} norms")
    
    # print_vocabulary_stats(vocab, matrix)
    
    # print("\n" + "="*80)
    # print("TEXT QUERY EXAMPLES")
    # print("="*80)
    
    # queries = [
    #     "machine learning neural network",
    #     "economic dispatch optimization",
    #     "security attacks network"
    # ]
    
    # for query in queries:
    #     print(f"\nQuery: '{query}'")
    #     print("-" * 80)
        
    #     results = search(query, ids, vocab, matrix, top_k=5, precomputed_norms=precomputed_norms)
        
    #     print(f"Top {min(5, len(results))} results:")
    #     for i, (doc_id, score) in enumerate(results, 1):
    #         print(f"\n{i}. Score: {score:.4f} - ID: {doc_id[:16]}...")
    #         if doc_id in docs:
    #             print(f"   {docs[doc_id].get('text', '')[:100]}...")
    
    # Demonstrate using document ID as query
    print("\n" + "="*80)
    print("DOCUMENT ID QUERY EXAMPLE (Reuses existing embedding)")
    print("="*80)
    
    if len(ids) > 0:
        example_id = ids[10]
        print(f"\nUsing document as query: {example_id}, title: {queries[example_id].get('text', 'N/A')}")
        print("-" * 80)
        
        results = search(example_id, ids, vocab, matrix, top_k=5, precomputed_norms=precomputed_norms)
        
        print(f"Top {min(5, len(results))} similar documents:")
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f} - ID: {doc_id[:16]}...")
            if doc_id in docs:
                print(f"   {docs[doc_id].get('text', '')[:100]}...")


if __name__ == "__main__":
    main()

