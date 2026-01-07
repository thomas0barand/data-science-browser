import re
import math
from collections import Counter
from typing import Sequence

def extract_bigrams(text: str) -> list[str]:
    """Extract bigrams from text."""
    clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = [w for w in clean.split() if w]
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    return bigrams

def bigram_embedding(text: str) -> dict[str, int]:
    """Create embedding with unigrams + bigrams."""
    clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = [w for w in clean.split() if w]
    
    # Unigrams
    tokens = words.copy()
    
    # Add bigrams
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    tokens.extend(bigrams)
    
    return Counter(tokens)

def tfidf_weights_bigram(
    embeddings: Sequence[dict[str, int]]
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Calculate TF-IDF for bigram embeddings."""
    n_docs = len(embeddings)
    
    # Document frequency
    df = Counter()
    for emb in embeddings:
        for term in emb.keys():
            df[term] += 1
    
    # IDF
    idf = {term: math.log(n_docs / freq) for term, freq in df.items()}
    
    # TF-IDF
    tfidf_embeddings = []
    for emb in embeddings:
        doc_length = sum(emb.values())
        if doc_length == 0:
            tfidf_embeddings.append({})
            continue
        
        tfidf_doc = {
            term: (freq / doc_length) * idf[term]
            for term, freq in emb.items()
        }
        tfidf_embeddings.append(tfidf_doc)
    
    return tfidf_embeddings, idf

