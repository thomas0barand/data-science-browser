#!/usr/bin/env python3
"""
Topic Modeling with Latent Dirichlet Allocation (LDA)
Explores thematic structure of the scientific articles corpus
"""
import sys
import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_jsonl(path: Path):
    """Load JSONL file into dictionary."""
    items = {}
    print(f"Loading {path.name}...")
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading"):
            if line.strip():
                data = json.loads(line)
                if "_id" in data:
                    items[str(data["_id"])] = data
    print(f"✓ Loaded {len(items)} documents")
    return items

def prepare_texts(corpus):
    """Extract and combine text fields from corpus."""
    texts = []
    doc_ids = []
    
    print("\nPreparing texts from corpus...")
    for doc_id, doc in tqdm(corpus.items(), desc="Processing"):
        title = doc.get('title', '').strip()
        abstract = doc.get('text', '').strip()
        
        combined = f"{title} {abstract}".strip()
        
        if combined:
            texts.append(combined)
            doc_ids.append(doc_id)
    
    print(f"✓ Prepared {len(texts)} texts")
    return texts, doc_ids

def vectorize_texts(texts, max_features=5000, min_df=5, max_df=0.7):
    """Vectorize texts using CountVectorizer."""
    print(f"\nVectorizing with parameters:")
    print(f"  max_features: {max_features}")
    print(f"  min_df: {min_df}")
    print(f"  max_df: {max_df}")
    
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    
    print(f"✓ Matrix shape: {X.shape[0]:,} documents × {X.shape[1]:,} terms")
    print(f"  Sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
    
    return X, vocab, vectorizer

def fit_lda(X, n_topics=10, max_iter=20, random_state=42):
    """Fit LDA model."""
    print(f"\nFitting LDA with {n_topics} topics...")
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method='online',
        batch_size=128,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    doc_topics = lda.fit_transform(X)
    
    perplexity = lda.perplexity(X)
    print(f"✓ Model trained")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Log-likelihood: {lda.score(X):.2f}")
    
    return lda, doc_topics, perplexity

def display_topics(lda, vocab, n_words=15):
    """Extract and display top words for each topic."""
    print(f"\n{'='*80}")
    print(f"DISCOVERED TOPICS (top {n_words} words per topic)")
    print(f"{'='*80}\n")
    
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-n_words:][::-1]
        top_words = [vocab[i] for i in top_words_idx]
        top_weights = topic[top_words_idx]
        
        topics.append({
            'id': topic_idx,
            'words': top_words,
            'weights': top_weights
        })
        
        print(f"Topic {topic_idx:2d}: {', '.join(top_words[:10])}")
    
    print(f"\n{'='*80}\n")
    return topics

def analyze_document_topics(doc_topics, doc_ids, corpus, n_examples=3):
    """Analyze topic distribution across documents."""
    print(f"DOCUMENT-TOPIC DISTRIBUTION ANALYSIS\n")
    
    dominant_topics = doc_topics.argmax(axis=1)
    topic_counts = Counter(dominant_topics)
    
    print("Distribution of dominant topics:")
    for topic_id in sorted(topic_counts.keys()):
        count = topic_counts[topic_id]
        pct = 100 * count / len(dominant_topics)
        bar = '█' * int(pct / 2)
        print(f"  Topic {topic_id:2d}: {count:5d} docs ({pct:5.1f}%) {bar}")
    
    print(f"\nTop {n_examples} representative documents per topic:")
    for topic_id in range(doc_topics.shape[1]):
        doc_scores = doc_topics[:, topic_id]
        top_docs_idx = doc_scores.argsort()[-n_examples:][::-1]
        
        print(f"\n  Topic {topic_id}:")
        for rank, idx in enumerate(top_docs_idx, 1):
            doc_id = doc_ids[idx]
            score = doc_scores[idx]
            title = corpus[doc_id].get('title', 'N/A')
            if len(title) > 70:
                title = title[:67] + "..."
            print(f"    {rank}. [{score:.3f}] {title}")

def plot_topic_distribution(topic_counts, n_topics, output_path):
    """Plot the distribution of documents across topics."""
    print("\nGenerating topic distribution plot...")
    
    topics = list(range(n_topics))
    counts = [topic_counts.get(i, 0) for i in topics]
    percentages = [100 * c / sum(counts) for c in counts]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_topics))
    bars = ax.bar(topics, counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_xlabel('Topic ID', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Documents', fontweight='bold', fontsize=12)
    ax.set_title('Document Distribution Across Topics\n(based on dominant topic)', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(topics)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_topic_words(topics, output_path, n_topics_to_plot=6):
    """Plot top words for selected topics."""
    print("\nGenerating topic words visualization...")
    
    n_topics_to_plot = min(n_topics_to_plot, len(topics))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, topic in enumerate(topics[:n_topics_to_plot]):
        ax = axes[i]
        
        words = topic['words'][:10]
        weights = topic['weights'][:10]
        
        colors = plt.cm.RdYlGn(weights / weights.max())
        bars = ax.barh(range(len(words)), weights, color=colors, 
                      edgecolor='black', linewidth=0.8, alpha=0.85)
        
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Weight', fontweight='bold', fontsize=10)
        ax.set_title(f'Topic {topic["id"]}', fontweight='bold', fontsize=12, pad=10)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        for bar, weight in zip(bars, weights):
            width = bar.get_width()
            ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2,
                   f'{weight:.1f}',
                   ha='left', va='center', fontsize=8, fontweight='bold')
    
    plt.suptitle('Top Words per Topic (by weight)', 
                 fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def save_results(lda, doc_topics, doc_ids, vocab, vectorizer, topics, perplexity, output_path):
    """Save all results to pickle file."""
    print(f"\nSaving results to {output_path}...")
    
    results = {
        'model': lda,
        'doc_topics': doc_topics,
        'doc_ids': doc_ids,
        'vocab': vocab,
        'vectorizer': vectorizer,
        'topics': topics,
        'perplexity': perplexity,
        'n_topics': lda.n_components,
        'n_documents': len(doc_ids)
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ Saved model and results")

def save_topics_json(topics, output_path):
    """Save topics to JSON for easy reading."""
    print(f"Saving topics to {output_path}...")
    
    topics_data = []
    for topic in topics:
        topics_data.append({
            'id': int(topic['id']),
            'top_words': list(topic['words'][:15]),
            'weights': [float(w) for w in topic['weights'][:15]]
        })
    
    with output_path.open('w') as f:
        json.dump(topics_data, f, indent=2)
    
    print(f"✓ Saved topics to JSON")

def main():
    print("="*80)
    print("TOPIC MODELING WITH LDA")
    print("="*80)
    
    # Configuration
    N_TOPICS = 10
    MAX_FEATURES = 5000
    MIN_DF = 5
    MAX_DF = 0.7
    MAX_ITER = 20
    
    # Load data
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    
    # Prepare texts
    texts, doc_ids = prepare_texts(corpus)
    
    # Vectorize
    X, vocab, vectorizer = vectorize_texts(texts, MAX_FEATURES, MIN_DF, MAX_DF)
    
    # Fit LDA
    lda, doc_topics, perplexity = fit_lda(X, n_topics=N_TOPICS, max_iter=MAX_ITER)
    
    # Display topics
    topics = display_topics(lda, vocab, n_words=15)
    
    # Analyze document-topic distribution
    dominant_topics = doc_topics.argmax(axis=1)
    topic_counts = Counter(dominant_topics)
    analyze_document_topics(doc_topics, doc_ids, corpus, n_examples=3)
    
    # Generate visualizations
    plot_topic_distribution(topic_counts, N_TOPICS, FIGURES_DIR / "lda_topic_distribution.png")
    plot_topic_words(topics, FIGURES_DIR / "lda_top_words.png", n_topics_to_plot=6)
    
    # Save results
    save_results(lda, doc_topics, doc_ids, vocab, vectorizer, topics, perplexity,
                OUTPUT_DIR / "lda_model.pkl")
    save_topics_json(topics, OUTPUT_DIR / "lda_topics.json")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Documents processed:  {len(doc_ids):,}")
    print(f"  Vocabulary size:      {len(vocab):,}")
    print(f"  Number of topics:     {N_TOPICS}")
    print(f"  Perplexity:           {perplexity:.2f}")
    print(f"\nOutputs saved:")
    print(f"  - {OUTPUT_DIR / 'lda_model.pkl'}")
    print(f"  - {OUTPUT_DIR / 'lda_topics.json'}")
    print(f"  - {FIGURES_DIR / 'lda_topic_distribution.png'}")
    print(f"  - {FIGURES_DIR / 'lda_top_words.png'}")
    print("\n✓ Topic modeling completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()

