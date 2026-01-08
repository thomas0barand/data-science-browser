#!/usr/bin/env python3
"""
Generate t-SNE visualization for LDA topics
(Alternative to UMAP that doesn't require LLVM)
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def main():
    print("="*80)
    print("t-SNE VISUALIZATION FOR LDA TOPICS")
    print("="*80)
    
    # Load LDA results
    print("\nLoading LDA model...")
    model_path = OUTPUT_DIR / "lda_model.pkl"
    
    if not model_path.exists():
        print(f"❌ Error: {model_path} not found!")
        print("Please run topic_modeling_lda.py first.")
        return
    
    with open(model_path, 'rb') as f:
        results = pickle.load(f)
    
    doc_topics = results['doc_topics']
    doc_ids = results['doc_ids']
    n_topics = results['n_topics']
    
    print(f"✓ Loaded model")
    print(f"  Documents: {len(doc_ids):,}")
    print(f"  Topics: {n_topics}")
    print(f"  Topic distribution shape: {doc_topics.shape}")
    
    # t-SNE projection
    print("\nApplying t-SNE dimensionality reduction...")
    print("  This may take a few minutes...")
    
    reducer = TSNE(
        n_components=2,
        metric='cosine',
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    
    embedding_2d = reducer.fit_transform(doc_topics)
    print(f"✓ Reduced to 2D: {embedding_2d.shape}")
    
    # Get dominant topics
    dominant_topics = doc_topics.argmax(axis=1)
    
    # Count documents per topic
    from collections import Counter
    topic_counts = Counter(dominant_topics)
    
    # Plot
    print("\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_topics))
    
    for topic_id in range(n_topics):
        mask = dominant_topics == topic_id
        count = mask.sum()
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[topic_id]],
            label=f'Topic {topic_id} ({count} docs, {100*count/len(doc_ids):.1f}%)',
            alpha=0.6,
            s=25,
            edgecolors='black',
            linewidth=0.4
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontweight='bold', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=12)
    ax.set_title('Document Distribution in Topic Space\n(t-SNE Projection of LDA Topic Distributions)', 
                 fontweight='bold', fontsize=14, pad=20)
    
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2, 
              edgecolor='black', bbox_to_anchor=(0, 1))
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Add text annotation
    ax.text(0.98, 0.02, 
            f'Total documents: {len(doc_ids):,}\nTopics: {n_topics}',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "lda_tsne_projection.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Statistics
    print("\n" + "="*80)
    print("t-SNE PROJECTION STATISTICS")
    print("="*80)
    print(f"Embedding shape: {embedding_2d.shape}")
    print(f"X range: [{embedding_2d[:, 0].min():.2f}, {embedding_2d[:, 0].max():.2f}]")
    print(f"Y range: [{embedding_2d[:, 1].min():.2f}, {embedding_2d[:, 1].max():.2f}]")
    
    print("\nDocuments per topic:")
    for topic_id in sorted(topic_counts.keys()):
        count = topic_counts[topic_id]
        pct = 100 * count / len(doc_ids)
        bar = '█' * int(pct / 2)
        print(f"  Topic {topic_id:2d}: {count:5d} docs ({pct:5.1f}%) {bar}")
    
    print("\n✓ t-SNE visualization completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()

