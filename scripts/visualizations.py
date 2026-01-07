#!/usr/bin/env python3
"""
Script de visualisation pour le rapport Data Science Browser.

Génère :
1. Distribution du vocabulaire (loi de puissance)
2. Exemple de requête avec scores de similarité cosinus
"""

import sys
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configuration des chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

# Import des fonctions du sparse browser
from sparse_browser import (
    load_embeddings,
    load_documents,
    search,
    precompute_norms
)

# Configuration matplotlib pour de beaux graphiques
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'


def plot_vocabulary_distribution(vocab, matrix, output_path, title_suffix=""):
    """
    Visualisation: Distribution du vocabulaire.
    
    Montre comment les fréquences des termes suivent une loi de puissance:
    - Quelques termes très fréquents
    - Beaucoup de termes rares
    """
    print(f"\nGénération de la visualisation: {title_suffix}")
    
    # Calculer les fréquences totales de chaque terme
    print("Calcul des fréquences des termes...")
    term_frequencies = [sum(doc[i] for doc in matrix) for i in range(len(vocab))]
    
    # Trier par fréquence décroissante
    sorted_frequencies = sorted(term_frequencies, reverse=True)
    
    # Créer la figure avec deux subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== SUBPLOT 1: Distribution complète en échelle linéaire ==========
    ax1.plot(range(1, len(sorted_frequencies) + 1), sorted_frequencies, 
             linewidth=2, color='#E63946', alpha=0.8)
    ax1.fill_between(range(1, len(sorted_frequencies) + 1), sorted_frequencies, 
                     alpha=0.3, color='#E63946')
    
    ax1.set_xlabel('Rang du terme', fontweight='bold')
    ax1.set_ylabel('Fréquence totale', fontweight='bold')
    ax1.set_title(f'Distribution du vocabulaire {title_suffix}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Annotation "Quelques termes très fréquents"
    ax1.annotate('Quelques termes\ntrès fréquents', 
                xy=(50, sorted_frequencies[49]), 
                xytext=(len(vocab)//8, max(sorted_frequencies)*0.8),
                fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.5, edgecolor='black', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                              color='red', lw=2))
    
    # Annotation "Beaucoup de termes rares"
    ax1.annotate('Beaucoup de\ntermes rares', 
                xy=(len(sorted_frequencies)*0.7, sorted_frequencies[int(len(sorted_frequencies)*0.7)]), 
                xytext=(len(vocab)*0.7, max(sorted_frequencies)*0.3),
                fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.5, edgecolor='black', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', 
                              color='blue', lw=2))
    
    # ========== SUBPLOT 2: Top 30 termes les plus fréquents ==========
    top_n = 30
    top_indices = sorted(range(len(term_frequencies)), 
                        key=lambda i: term_frequencies[i], 
                        reverse=True)[:top_n]
    top_terms = [vocab[i] for i in top_indices]
    top_freqs = [term_frequencies[i] for i in top_indices]
    
    # Créer un gradient de couleurs
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    
    bars = ax2.barh(range(top_n), top_freqs, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_terms, fontsize=9)
    ax2.set_xlabel('Fréquence totale', fontweight='bold')
    ax2.set_title(f'Top {top_n} Termes les Plus Fréquents', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Ajouter les valeurs sur les barres
    for i, (bar, freq) in enumerate(zip(bars, top_freqs)):
        ax2.text(freq + max(top_freqs)*0.01, i, f'{freq}', 
                va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_path}")
    
    # Statistiques
    print(f"\nStatistiques du vocabulaire {title_suffix}:")
    print(f"  - Taille totale: {len(vocab)} termes")
    print(f"  - Fréquence max: {max(term_frequencies)} (terme: '{vocab[top_indices[0]]}')")
    print(f"  - Fréquence min: {min(term_frequencies)} ")
    print(f"  - Fréquence moyenne: {np.mean(term_frequencies):.1f}")
    print(f"  - Fréquence médiane: {np.median(term_frequencies):.1f}")
    print(f"  - Top 10% des termes représentent {sum(sorted_frequencies[:len(vocab)//10]) / sum(sorted_frequencies) * 100:.1f}% des occurrences")
    
    return fig


def plot_query_similarity_scores(query_text, results, docs, output_path):
    """
    Visualisation 2: Scores de similarité cosinus pour une requête exemple.
    
    Montre les top K documents les plus similaires avec leurs scores.
    """
    print("\n" + "="*80)
    print("VISUALISATION 2: Exemple de Requête - Scores de Similarité")
    print("="*80)
    print(f"Requête: '{query_text}'")
    
    # Extraire les données
    doc_ids = [doc_id for doc_id, score in results]
    scores = [score for doc_id, score in results]
    
    # Créer les labels (titres tronqués)
    labels = []
    for doc_id in doc_ids:
        if doc_id in docs:
            title = docs[doc_id].get('title', docs[doc_id].get('text', doc_id))
            # Tronquer le titre
            if len(title) > 60:
                title = title[:57] + "..."
            labels.append(title)
        else:
            labels.append(doc_id[:30])
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Gradient de couleurs du rouge (faible) au vert (fort)
    colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
    
    # Créer le graphique en barres horizontales
    bars = ax.barh(range(len(scores)), scores, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Configuration des axes
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Score de Similarité Cosinus', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {len(results)} Documents Similaires\nRequête: "{query_text}"', 
                fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Ajouter les valeurs sur les barres
    for i, (bar, score) in enumerate(zip(bars, scores)):
        # Positionner le texte à droite de la barre
        x_pos = score + 0.02 if score < 0.8 else score - 0.02
        ha = 'left' if score < 0.8 else 'right'
        color = 'black' if score < 0.8 else 'white'
        
        ax.text(x_pos, i, f'{score:.4f}', 
               va='center', ha=ha, fontsize=10, 
               fontweight='bold', color=color)
    
    # Ajouter des zones de couleur pour l'interprétation
    ax.axvspan(0.7, 1.0, alpha=0.1, color='green', zorder=0)
    ax.axvspan(0.5, 0.7, alpha=0.1, color='yellow', zorder=0)
    ax.axvspan(0.3, 0.5, alpha=0.1, color='orange', zorder=0)
    ax.axvspan(0.0, 0.3, alpha=0.1, color='red', zorder=0)
    
    # Légende pour les zones
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Très similaire (0.7-1.0)'),
        Patch(facecolor='yellow', alpha=0.3, label='Similaire (0.5-0.7)'),
        Patch(facecolor='orange', alpha=0.3, label='Modérément similaire (0.3-0.5)'),
        Patch(facecolor='red', alpha=0.3, label='Faiblement similaire (0.0-0.3)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
             framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_path}")
    
    # Afficher les résultats
    print(f"\nTop {len(results)} résultats:")
    for i, (doc_id, score) in enumerate(results, 1):
        title = labels[i-1]
        print(f"  {i}. [{score:.4f}] {title}")
    
    return fig


def main():
    """Fonction principale pour générer toutes les visualisations."""
    
    print("="*80)
    print("GÉNÉRATION DES VISUALISATIONS POUR LE RAPPORT")
    print("="*80)
    
    # Créer le dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Charger les données AVANT décimation
    print("\n[1/5] Chargement des embeddings AVANT décimation...")
    ids_before, vocab_before, matrix_before = load_embeddings(
        pkl_path=DATA_DIR / "sparses_embedding_without_corpus_text.pkl"
    )
    print(f"  ✓ Chargé: {len(ids_before)} documents, {len(vocab_before)} termes")
    
    # 2. Charger les données APRÈS décimation
    print("\n[2/5] Chargement des embeddings APRÈS décimation...")
    ids_after, vocab_after, matrix_after = load_embeddings(
        pkl_path=DATA_DIR / "sparses_embedding_without_corpus_text_decimated.pkl"
    )
    print(f"  ✓ Chargé: {len(ids_after)} documents, {len(vocab_after)} termes")
    
    print("\n[3/5] Chargement des métadonnées...")
    corpus_docs = load_documents(DATA_DIR / "corpus.jsonl")
    query_docs = load_documents(DATA_DIR / "queries.jsonl")
    all_docs = {**corpus_docs, **query_docs}
    print(f"  ✓ Chargé: {len(all_docs)} documents avec métadonnées")
    
    # 3. Générer la visualisation AVANT décimation
    print("\n[4/5] Génération de la visualisation du vocabulaire...")
    print("\n" + "="*80)
    print("VISUALISATION 1A: Vocabulaire AVANT Décimation")
    print("="*80)
    plot_vocabulary_distribution(
        vocab_before, 
        matrix_before, 
        OUTPUT_DIR / "vocab_distribution_before.png",
        title_suffix="avant décimation"
    )
    
    # 4. Générer la visualisation APRÈS décimation
    print("\n" + "="*80)
    print("VISUALISATION 1B: Vocabulaire APRÈS Décimation")
    print("="*80)
    plot_vocabulary_distribution(
        vocab_after, 
        matrix_after, 
        OUTPUT_DIR / "vocab_distribution_after.png",
        title_suffix="après décimation"
    )
    
    # 5. Générer la visualisation d'exemple de requête (utilise les embeddings après décimation)
    print("\n[5/5] Génération de la visualisation de recherche...")
    print("\n" + "="*80)
    print("VISUALISATION 2: Exemple de Requête - Scores de Similarité")
    print("="*80)
    
    # Pré-calculer les normes pour l'optimisation
    print("  Pré-calcul des normes...")
    precomputed_norms = precompute_norms(matrix_after)
    
    # Choisir une requête intéressante
    # Option 1: Utiliser une vraie requête du dataset
    example_query_id = ids_after[10] if ids_after[10] in query_docs else ids_after[0]
    
    # Récupérer le texte de la requête
    if example_query_id in query_docs:
        query_text = query_docs[example_query_id].get('text', '')
        # Tronquer pour l'affichage
        if len(query_text) > 80:
            query_display = query_text[:77] + "..."
        else:
            query_display = query_text
    else:
        query_display = "Document example"
    
    print(f"  Requête exemple: {example_query_id}")
    print(f"  Texte: {query_display}")
    
    # Effectuer la recherche
    print("  Recherche des documents similaires...")
    results = search(
        example_query_id,
        ids_after,
        vocab_after,
        matrix_after,
        top_k=15,
        precomputed_norms=precomputed_norms
    )
    
    # Exclure le document requête lui-même (score = 1.0)
    results_filtered = [(doc_id, score) for doc_id, score in results 
                       if doc_id != example_query_id][:10]
    
    # Générer le graphique
    plot_query_similarity_scores(
        query_display,
        results_filtered,
        all_docs,
        OUTPUT_DIR / "query_similarity_example.png"
    )
    
    print("\n" + "="*80)
    print("✓ TOUTES LES VISUALISATIONS ONT ÉTÉ GÉNÉRÉES AVEC SUCCÈS!")
    print("="*80)
    print(f"\nFichiers générés dans: {OUTPUT_DIR}/")
    print("  1. vocab_distribution_before.png - Vocabulaire AVANT décimation")
    print("  2. vocab_distribution_after.png - Vocabulaire APRÈS décimation")
    print("  3. query_similarity_example.png - Exemple de requête")
    print(f"\n📊 Comparaison AVANT/APRÈS décimation:")
    print(f"  • AVANT: {len(vocab_before):,} termes")
    print(f"  • APRÈS: {len(vocab_after):,} termes")
    print(f"  • Réduction: {100*(1-len(vocab_after)/len(vocab_before)):.1f}%")
    print("\nCes images sont prêtes à être intégrées dans votre rapport LaTeX.")


if __name__ == "__main__":
    main()

