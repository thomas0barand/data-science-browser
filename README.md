# Data Science Browser

Moteur de recherche d'information dans la littérature scientifique - MOD 7.2

## 📋 Projet

Construction d'un moteur de recherche qui, pour une publication scientifique (requête), retourne les articles sémantiquement proches.

**Données**: 25,000+ articles, 1,000 requêtes, ~21,000 paires annotées  
**Objectif**: Identifier les 5 articles pertinents parmi ~30 candidats par requête

## 🌿 Branches

### `thomas` - Sparse Embeddings ✅
Approche creuse: matrice Documents × Termes, TF-IDF, bigrammes  
**Résultats**: AUC 0.720, F1 0.540

### `aya` - Dense Embeddings ✅
Approche dense: Sentence-BERT (MiniLM-L6), similarité cosinus, FAISS  
**Résultats**: AUC 0.9586, F1 0.8155

### `oumaima` - Graph Structure ✅
Approche structurelle: graphe de citations, enrichissement MiniLM par voisinage citationnel  
**Résultats**: AUC 0.967 (amélioration +0.9% vs MiniLM seul)

## 🚀 Utilisation

```bash
git clone https://github.com/thomas0barand/data-science-browser.git
cd data-science-browser
git checkout <branch>  # thomas, aya, ou oumaima
poetry install
```

## 👥 Équipe

Thomas, Aya, Oumaima - École Centrale de Lyon
