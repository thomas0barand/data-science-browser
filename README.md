# Data Science Browser

Moteur de recherche d'information dans la littérature scientifique - MOD 7.2 (Introduction à la science des données)

## Description du Projet

Construction d'un moteur de recherche qui, étant donnée une publication scientifique (requête), retourne les articles sémantiquement les plus proches.

**Données**:
- Corpus: 25,000+ articles scientifiques
- Requêtes: 1,000 articles
- Validation: ~21,000 paires (requête, candidat) annotées

**Objectif**: Pour chaque requête, identifier les 5 articles pertinents parmi ~30 candidats.

## Approches Implémentées

Le projet explore trois familles de méthodes :

### 1. Approche Creuse (Sparse Embeddings) + Exploration Thématique (LDA)
- Matrice Documents × Termes (bag-of-words)
- TF-IDF et variantes
- Bigrammes
- **→ Branche `thomas`**

### 2. Approche Dense (Dense Embeddings)
- all-MiniLM-L6-v2
- FAISS
- **→ Branche `aya`**

### 3. Approche Structurelle (Graphe de Citations)
- À compléter
- **→ Branche `oumaima`**

## Détail des Branches

### Branch `thomas` - Sparse Embeddings ✅

**Implémentation complète de l'approche creuse**:
- Sparse embeddings (fréquences, TF-IDF)
- Bigrammes avec/sans TF-IDF
- Décimation du vocabulaire (stop words, stemming, filtrage fréquentiel)
- Optimisations numpy (50x speedup)

**Résultats**:
- Best model: Bigram TF-IDF
- AUC: **0.720**
- F1: 0.540

### Branch `aya` - Dense Embeddings
- Exploration & statistiques des données  
    
**Approches creuses – Représentations lexicales**:  
- Encodage Bag-of-Words avec CountVectorizer  
    
**Implémentation complète de l'approche dense**:  
- Sentence-BERT all-MiniLM-L6-v2 (384D)
- Similarité cosinus + FAISS
- Embeddings pré-calculés et indexés  
  
**Résultats**:    
- Précision : 0.8097
- Rappel    : 0.8214
- F-mesure  : 0.8155
- AUC       : **0.9586**  
### Branch `oumaima` - Graph Structure

*À compléter*

## 🚀 Quick Start

```bash
# Cloner le repo
git clone https://github.com/thomas0barand/data-science-browser.git
cd data-science-browser

# Choisir une branche
git checkout thomas    # Sparse embeddings
git checkout aya       # Dense embeddings
git checkout oumaima   # Graph structure

# Installer les dépendances
poetry install
```

## 📊 Métriques d'Évaluation

- **Précision, Recall, F1-Score**
- **AUC-ROC**
- **Confusion Matrix**


## The Best Team

- **Thomas**: Sparse embeddings & bigrammes
- **Aya**: Dense embeddings
- **Oumaima**: Graphe de citations

---

**École Centrale de Lyon** - MOD 7.2  
**Enseignants**: Julien Velcin, Erwan Versmée
