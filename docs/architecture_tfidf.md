# Architecture du système avec TF-IDF

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
└─────────────────────────────────────────────────────────────┘

Input Data
├── queries.jsonl (1,000 queries)
├── corpus.jsonl (25,657 documents)
└── valid.tsv (ground truth pairs)

                    ↓

┌─────────────────────────────────────────────────────────────┐
│             PREPROCESSING (export_sparse_embeddings.py)     │
├─────────────────────────────────────────────────────────────┤
│ 1. Text cleaning: lowercase, remove punctuation            │
│ 2. Tokenization: split into words                          │
│ 3. Stop words removal: 37 common words                     │
│ 4. Frequency filtering: min=3, max=350 occurrences         │
│ 5. Vocabulary selection: top 3,000 terms                   │
│ 6. Stemming: remove suffixes (ing, ed, ly, etc.)           │
│ 7. TF-IDF weighting: importance-based scoring ✨ NEW       │
└─────────────────────────────────────────────────────────────┘

                    ↓

Output Embeddings
├── sparses_embedding_tfidf.pkl (TF-IDF, 138 MB) ✨ NEW
├── sparses_embedding_without_corpus_text_decimated.pkl (Raw, 137 MB)
└── Structure: {ids: [...], vocab: [...], matrix: [[...]]}

                    ↓

┌─────────────────────────────────────────────────────────────┐
│              SEARCH ENGINE (sparse_browser.py)              │
├─────────────────────────────────────────────────────────────┤
│ 1. Load embeddings from .pkl                               │
│ 2. Pre-compute document norms                              │
│ 3. Calculate cosine similarity (sparse optimized)          │
│ 4. Rank documents by similarity score                      │
└─────────────────────────────────────────────────────────────┘

                    ↓

┌─────────────────────────────────────────────────────────────┐
│        PREDICTION GENERATION (generate_predictions_*.py)    │
├─────────────────────────────────────────────────────────────┤
│ For each query:                                             │
│   - Compare against all 25,657 corpus documents            │
│   - Store similarity scores in CSV matrix                  │
│ Output: preds_tfidf.csv (1,000 × 25,657 matrix) ✨ NEW     │
└─────────────────────────────────────────────────────────────┘

                    ↓

┌─────────────────────────────────────────────────────────────┐
│               EVALUATION (evaluate mode)                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Load predictions CSV                                     │
│ 2. Compare with valid.tsv ground truth                     │
│ 3. Apply threshold for binary classification               │
│ 4. Calculate metrics: Accuracy, Precision, Recall, F1      │
└─────────────────────────────────────────────────────────────┘

                    ↓

Final Metrics
├── Accuracy
├── Precision, Recall, F1
└── Confusion Matrix
```

---

## Comparaison TF-IDF vs Raw Frequencies

### Pipeline Raw Frequencies

```
Text → Tokenize → Count → Sparse Vector
"hello world hello" → {"hello": 2, "world": 1}
```

**Problème** : Tous les mots ont le même poids relatif

### Pipeline TF-IDF ✨

```
Text → Tokenize → Count → TF Normalization → IDF Weighting → TF-IDF Vector
"hello world hello" → {"hello": 2, "world": 1}
                   → {"hello": 0.67, "world": 0.33}  (TF)
                   → {"hello": 0.27, "world": 0.36}  (TF-IDF)
```

**Avantage** : Mots rares ont plus de poids

---

## Formule TF-IDF Implémentée

### 1. Term Frequency (TF)

```
TF(mot, doc) = fréquence(mot, doc) / longueur_totale(doc)
```

Normalise par la longueur du document (documents longs vs courts)

### 2. Inverse Document Frequency (IDF)

```
IDF(mot) = log(N / DF(mot))

où:
  N = nombre total de documents
  DF(mot) = nombre de documents contenant le mot
```

Favorise les termes rares

### 3. TF-IDF Final

```
TF-IDF(mot, doc) = TF(mot, doc) × IDF(mot)
```

---

## Flux de données

### 1. Génération d'embeddings

```bash
poetry run python scripts/export_sparse_embeddings.py
```

**Input** :
- `data/queries.jsonl`
- `data/corpus.jsonl`

**Processing** :
- Tokenization + cleaning
- Stop words removal
- Stemming
- TF-IDF weighting ✨

**Output** :
- `data/sparses_embedding_tfidf.pkl`

**Structure du fichier** :
```python
{
  "ids": ["query1_id", "doc1_id", ...],        # 26,657 IDs
  "vocab": ["algorithm", "network", ...],      # 2,663 terms
  "matrix": [[0.5, 0.0, 0.3, ...], ...]       # 26,657 × 2,663 matrix
}
```

### 2. Génération de prédictions

```bash
poetry run python scripts/generate_predictions_tfidf.py
```

**Input** :
- `data/sparses_embedding_tfidf.pkl`

**Processing** :
- For each query (1,000)
  - Compare with each corpus doc (25,657)
  - Calculate cosine similarity
  - Store score

**Output** :
- `outputs/results/preds_tfidf.csv`
- Size: 1,000 rows × 25,658 columns (1 query_id + 25,657 scores)

### 3. Évaluation

```bash
poetry run python scripts/generate_predictions_sparse.py \
    --mode evaluate \
    --predictions-csv outputs/results/preds_tfidf.csv
```

**Input** :
- `outputs/results/preds_tfidf.csv`
- `data/valid.tsv`

**Processing** :
- For each (query_id, corpus_id, label) in valid.tsv
  - Lookup similarity score
  - Apply threshold (default: 0.1)
  - Compare prediction vs ground truth

**Output** :
- `outputs/results/evaluation_results.csv`
- Metrics: Accuracy, Precision, Recall, F1

---

## Fichiers du projet

### Code source

| Fichier | Rôle |
|---------|------|
| `src/utils.py` | Fonctions de base : `sparse_embedding()`, `tfidf_weights()` ✨ |
| `src/sparse_browser.py` | Moteur de recherche : cosine similarity, search |
| `scripts/export_sparse_embeddings.py` | Génération d'embeddings TF-IDF ✨ |
| `scripts/generate_predictions_tfidf.py` | Génération de prédictions TF-IDF ✨ |
| `scripts/compare_tfidf.py` | Comparaison TF-IDF vs raw ✨ |

### Données

| Fichier | Taille | Description |
|---------|--------|-------------|
| `data/queries.jsonl` | - | 1,000 requêtes |
| `data/corpus.jsonl` | - | 25,657 documents |
| `data/valid.tsv` | - | Ground truth (train set) |
| `data/test_final.tsv` | - | Test set (submission) |
| `data/sparses_embedding_tfidf.pkl` ✨ | 138 MB | Embeddings TF-IDF |
| `outputs/results/preds_tfidf.csv` ✨ | ~200 MB | Matrice de prédictions |

### Documentation

| Fichier | Contenu |
|---------|---------|
| `TFIDF_GUIDE.md` ✨ | Guide utilisateur rapide |
| `docs/tfidf_implementation.md` ✨ | Détails techniques TF-IDF |
| `docs/ameliorations_implementees.md` ✨ | Bilan des améliorations |
| `docs/architecture_tfidf.md` ✨ | Ce document |
| `docs/evaluation_guide.md` | Guide d'évaluation original |

---

## Performance attendue

### Sans TF-IDF (fréquences brutes)

- ✅ Simple et rapide
- ❌ Scores peu discriminants (beaucoup de 0.707...)
- ❌ Mots communs surpondérés

### Avec TF-IDF ✨

- ✅ Scores discriminants (variance élevée)
- ✅ Termes rares favorisés
- ✅ Meilleure pertinence attendue
- ⚠️ Légèrement plus lent (calculs TF-IDF)

### Exemple concret

**Requête** : "Social engineering attack framework"

| Méthode | Top-1 Score | Top-5 Variance | Pertinence |
|---------|-------------|----------------|------------|
| Raw frequencies | 0.707 | 0.000 (tous identiques) | Moyenne |
| TF-IDF ✨ | 0.668 | 0.159 (scores variés) | Élevée |

---

## Étapes suivantes

### Immédiat
1. ✅ TF-IDF implémenté
2. ⏭️ Évaluer sur `valid.tsv` pour mesurer l'impact réel
3. ⏭️ Optimiser le seuil de classification

### Court terme
4. ⏭️ Générer soumission pour `test_final.tsv` avec TF-IDF
5. ⏭️ Comparer les métriques TF-IDF vs raw sur valid.tsv

### Moyen terme
6. ❌ BM25 (amélioration de TF-IDF)
7. ❌ Bigrammes sélectifs (ex: "neural network")
8. ❌ Embeddings denses (BERT, Sentence-BERT)

