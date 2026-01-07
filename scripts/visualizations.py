#!/usr/bin/env python3
import sys
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

from sparse_browser import load_embeddings, load_documents, search, precompute_norms

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

def plot_vocabulary_distribution(vocab, matrix, output_path, title_suffix=""):
    """Plot the vocabulary distribution of the matrix."""
    print(f"\nGenerating vocabulary distribution: {title_suffix}")
    
    term_frequencies = [sum(doc[i] for doc in matrix) for i in range(len(vocab))]
    sorted_frequencies = sorted(term_frequencies, reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full distribution
    ax1.plot(range(1, len(sorted_frequencies) + 1), sorted_frequencies, 
             linewidth=2, color='#E63946', alpha=0.8)
    ax1.fill_between(range(1, len(sorted_frequencies) + 1), sorted_frequencies, 
                     alpha=0.3, color='#E63946')
    
    ax1.set_xlabel('Term rank', fontweight='bold')
    ax1.set_ylabel('Total frequency', fontweight='bold')
    ax1.set_title(f'Vocabulary distribution {title_suffix}', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax1.annotate('Few very frequent\nterms', 
                xy=(50, sorted_frequencies[49]), 
                xytext=(len(vocab)//8, max(sorted_frequencies)*0.8),
                fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.5, edgecolor='black', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='red', lw=2))
    
    ax1.annotate('Many rare\nterms', 
                xy=(len(sorted_frequencies)*0.7, sorted_frequencies[int(len(sorted_frequencies)*0.7)]), 
                xytext=(len(vocab)*0.7, max(sorted_frequencies)*0.3),
                fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.5, edgecolor='black', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', color='blue', lw=2))
    
    # Top N terms
    top_n = 30
    top_indices = sorted(range(len(term_frequencies)), key=lambda i: term_frequencies[i], reverse=True)[:top_n]
    top_terms = [vocab[i] for i in top_indices]
    top_freqs = [term_frequencies[i] for i in top_indices]
    
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = ax2.barh(range(top_n), top_freqs, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_terms, fontsize=9)
    ax2.set_xlabel('Total frequency', fontweight='bold')
    ax2.set_title(f'Top {top_n} Most Frequent Terms', fontweight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    for i, (bar, freq) in enumerate(zip(bars, top_freqs)):
        ax2.text(freq + max(top_freqs)*0.01, i, f'{freq}', 
                va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Max freq: {max(term_frequencies)} ('{vocab[top_indices[0]]}')")
    print(f"  Top 10%: {sum(sorted_frequencies[:len(vocab)//10]) / sum(sorted_frequencies) * 100:.1f}% of occurrences")
    
    return fig

def plot_query_similarity_scores(query_text, results, docs, output_path):
    """Plot the similarity scores for a query."""
    print("\nGenerating query similarity example")
    print(f"Query: '{query_text}'")
    
    doc_ids = [doc_id for doc_id, score in results]
    scores = [score for doc_id, score in results]
    
    labels = []
    for doc_id in doc_ids:
        if doc_id in docs:
            title = docs[doc_id].get('title', docs[doc_id].get('text', doc_id))
            labels.append(title[:57] + "..." if len(title) > 60 else title)
        else:
            labels.append(doc_id[:30])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
    bars = ax.barh(range(len(scores)), scores, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Cosine Similarity Score', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {len(results)} Similar Documents\nQuery: "{query_text}"', fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    for i, (bar, score) in enumerate(zip(bars, scores)):
        x_pos = score + 0.02 if score < 0.8 else score - 0.02
        ha = 'left' if score < 0.8 else 'right'
        color = 'black' if score < 0.8 else 'white'
        ax.text(x_pos, i, f'{score:.4f}', va='center', ha=ha, fontsize=10, fontweight='bold', color=color)
    
    ax.axvspan(0.7, 1.0, alpha=0.1, color='green', zorder=0)
    ax.axvspan(0.5, 0.7, alpha=0.1, color='yellow', zorder=0)
    ax.axvspan(0.3, 0.5, alpha=0.1, color='orange', zorder=0)
    ax.axvspan(0.0, 0.3, alpha=0.1, color='red', zorder=0)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Very similar (0.7-1.0)'),
        Patch(facecolor='yellow', alpha=0.3, label='Similar (0.5-0.7)'),
        Patch(facecolor='orange', alpha=0.3, label='Moderately similar (0.3-0.5)'),
        Patch(facecolor='red', alpha=0.3, label='Weakly similar (0.0-0.3)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    print(f"\nTop {len(results)} results:")
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] {labels[i-1]}")
    
    return fig

def main():
    print("Generating visualizations")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading embeddings BEFORE decimation...")
    ids_before, vocab_before, matrix_before = load_embeddings(DATA_DIR / "sparses_embedding_without_corpus_text.pkl")
    print(f"✓ {len(ids_before)} docs, {len(vocab_before)} terms")
    
    print("\nLoading embeddings AFTER decimation...")
    ids_after, vocab_after, matrix_after = load_embeddings(DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl")
    print(f"✓ {len(ids_after)} docs, {len(vocab_after)} terms")
    
    print("\nLoading metadata...")
    corpus_docs = load_documents(DATA_DIR / "corpus.jsonl")
    query_docs = load_documents(DATA_DIR / "queries.jsonl")
    all_docs = {**corpus_docs, **query_docs}
    print(f"✓ {len(all_docs)} docs with metadata")
    
    plot_vocabulary_distribution(vocab_before, matrix_before, OUTPUT_DIR / "vocab_distribution_before.png", "before decimation")
    plot_vocabulary_distribution(vocab_after, matrix_after, OUTPUT_DIR / "vocab_distribution_after.png", "after decimation")
    
    print("\nGenerating query example...")
    precomputed_norms = precompute_norms(matrix_after)
    
    example_query_id = ids_after[10] if ids_after[10] in query_docs else ids_after[0]
    
    if example_query_id in query_docs:
        query_text = query_docs[example_query_id].get('text', '')
        query_display = query_text[:77] + "..." if len(query_text) > 80 else query_text
    else:
        query_display = "Document example"
    
    print(f"  Query: {example_query_id}")
    print(f"  Text: {query_display}")
    
    results = search(example_query_id, ids_after, vocab_after, matrix_after, top_k=15, precomputed_norms=precomputed_norms)
    results_filtered = [(doc_id, score) for doc_id, score in results if doc_id != example_query_id][:10]
    
    plot_query_similarity_scores(query_display, results_filtered, all_docs, OUTPUT_DIR / "query_similarity_example.png")
    
    print(f"\n✓ All visualizations generated!")
    print(f"  Files in: {OUTPUT_DIR}/")
    print(f"  - vocab_distribution_before.png")
    print(f"  - vocab_distribution_after.png")
    print(f"  - query_similarity_example.png")
    print(f"\nComparison:")
    print(f"  BEFORE: {len(vocab_before):,} terms")
    print(f"  AFTER:  {len(vocab_after):,} terms")
    print(f"  Reduction: {100*(1-len(vocab_after)/len(vocab_before)):.1f}%")

if __name__ == "__main__":
    main()
