# Data Science Browser - Branch Aya

Approche dense utilisant des embeddings pré-entraînés pour la recherche d'information scientifique.

## 🎯 Approche

**Dense Embeddings**: Représentations vectorielles denses utilisant Sentence-BERT pour capturer la sémantique profonde.

## 📊 Méthodes

- **Modèle**: Sentence-BERT (all-MiniLM-L6-v2, 384 dimensions)
- **Indexation**: FAISS pour la recherche rapide par similarité cosinus
- **Baseline**: Bag-of-Words avec CountVectorizer pour comparaison

## 🚀 Quick Start

```bash
git clone https://github.com/thomas0barand/data-science-browser.git
cd data-science-browser
git checkout aya
poetry install
```

## 📊 Résultats

| Métrique | Score |
|----------|-------|
| Précision | 0.8097 |
| Rappel | 0.8214 |
| F1-Score | 0.8155 |
| **AUC** | **0.9586** ✅ |

## 📁 Contenu

- Notebooks d'exploration et expérimentation
- Scripts de génération d'embeddings
- Résultats et visualisations

---

**École Centrale de Lyon** - MOD 7.2
