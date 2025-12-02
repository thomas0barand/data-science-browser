# Sparse Browser - Guide d'utilisation

## Description

`sparse_browser.py` est un moteur de recherche basé sur des embeddings creux (sparse) qui utilise la **similarité cosinus** pour comparer des documents dans une matrice Documents × Termes.

## Architecture

### 1. Embeddings Creux
- Chaque document est représenté comme un vecteur de fréquences de mots
- La matrice contient 20 documents (10 queries + 10 corpus) × 138 termes
- Format: liste de listes Python (compatible pickle)

### 2. Similarité Cosinus
Mesure l'angle entre deux vecteurs:
```
cosine(A, B) = (A · B) / (||A|| × ||B||)
```

**Interprétation des scores:**
- `1.0` : Documents identiques (même vocabulaire, mêmes proportions)
- `0.5-0.7` : Forte similarité (beaucoup de mots en commun)
- `0.2-0.4` : Similarité modérée (quelques mots en commun)
- `0.0` : Aucun mot en commun

### 3. Fonctionnalités

#### a) Comparaison de paires de documents
```python
from sparse_browser import compare_documents

similarity, common_words = compare_documents(doc_id1, doc_id2, ids, vocab, matrix)
print(f"Similarité: {similarity:.4f}")
print(f"Mots communs: {', '.join(common_words)}")
```

**Exemple de résultat:**
- Documents sur réseaux neuronaux: similarité = 0.45
- Documents non liés: similarité = 0.0

#### b) Moteur de recherche
```python
from sparse_browser import search

results = search("neural network optimization", ids, vocab, matrix, top_k=10)
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

**Étapes du moteur:**
1. **Construction du vecteur requête**: transformation du texte en vecteur sparse
2. **Calcul des scores**: similarité cosinus entre requête et chaque document
3. **Tri et affichage**: classement décroissant des top-K résultats

## Utilisation

### Mode 1: Script principal
```bash
poetry run python3 src/sparse_browser.py
```

Exécute automatiquement:
- Statistiques sur le vocabulaire
- Tests de comparaison de paires
- Exemples de recherche avec 3 requêtes

### Mode 2: Script de test
```bash
poetry run python3 scripts/test_search.py
```

Exemples interactifs:
- Recherche personnalisée
- Comparaison de documents spécifiques
- Tests multiples requêtes

### Mode 3: Import dans vos scripts
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparse_browser import load_embeddings, search

# Charger les données
ids, vocab, matrix = load_embeddings()

# Rechercher
results = search("votre requête ici", ids, vocab, matrix, top_k=5)
```

## Résultats Observés

### Tests de similarité
```
Pair 11 vs 12:
- Doc 1: "evolutionary recurrent network... neural/fuzzy networks..."
- Doc 2: "Dynamic economic dispatch... optimization..."
- Similarité: 0.2965
- Mots communs (4): a, and, for, hybrid
```

### Recherche par requête
```
Query: 'machine learning neural network'
Top résultat: Score 0.2425
→ Document sur "detection of distributed denial of service attacks"

Query: 'economic dispatch optimization'
Top résultat: Score 0.3203
→ Document sur "Direct Search Method to solve Economic Dispatch Problem"
```

## Limites et Améliorations Possibles

### Limites actuelles
- Vocabulaire simple (pas de stemming/lemmatization)
- Pas de pondération TF-IDF
- Vecteurs non normalisés au stockage
- Corpus de test très petit (20 documents)

### Améliorations possibles
1. **TF-IDF**: pondérer les termes par leur importance
2. **N-grams**: capturer les expressions multi-mots
3. **Stop words**: filtrer les mots trop communs
4. **Stemming**: normaliser les formes des mots
5. **Sparse matrix**: utiliser scipy.sparse pour l'efficacité

## Fichiers

- `src/sparse_browser.py` : Module principal
- `scripts/test_search.py` : Script d'exemples
- `data/sparses_embedding_small.pkl` : Embeddings (20 docs)
- `data/corpus_small.jsonl` : Corpus simplifié (10 docs)
- `data/queries_small.jsonl` : Requêtes simplifiées (10 docs)

