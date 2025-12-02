# Guide de Décimation du Vocabulaire

## Vue d'ensemble

La fonction `decimate()` permet de réduire la taille du vocabulaire et de la matrice Documents×Termes en utilisant plusieurs stratégies.

## Fonction

```python
def decimate(
    ids: list[str],
    vocab: list[str],
    matrix: list[list[int]],
    max_vocab_size: int = None,
    remove_stop_words: bool = False,
    min_freq: int = None,
    max_freq: int = None,
    use_stemming: bool = False
) -> tuple[list[str], list[str], list[list[int]]]
```

## Paramètres

### 1. `max_vocab_size` (int, optionnel)
**Limite la taille du vocabulaire aux N termes les plus fréquents.**

Exemple:
```python
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    max_vocab_size=200  # Garde seulement les 200 termes les plus fréquents
)
```

**Résultat:** 728 → 200 termes (72.5% de réduction)

---

### 2. `remove_stop_words` (bool, défaut=False)
**Supprime les mots-outils courants (anglais et français).**

Mots supprimés: `a`, `an`, `and`, `the`, `is`, `of`, `to`, `le`, `la`, `les`, `et`, etc.

Exemple:
```python
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    remove_stop_words=True
)
```

**Résultat:** 728 → 694 termes (4.7% de réduction)

---

### 3. `min_freq` (int, optionnel)
**Supprime les termes trop rares (apparaissant moins de N fois).**

Exemple:
```python
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    min_freq=3  # Supprime les termes apparaissant < 3 fois
)
```

**Résultat:** 728 → 168 termes (76.9% de réduction)

---

### 4. `max_freq` (int, optionnel)
**Supprime les termes trop fréquents (apparaissant plus de N fois).**

Utile pour éliminer les termes omniprésents qui n'apportent pas d'information discriminante.

Exemple:
```python
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    max_freq=100  # Supprime les termes apparaissant > 100 fois
)
```

---

### 5. `use_stemming` (bool, défaut=False)
**Applique un stemming simple pour regrouper les formes dérivées.**

Regroupe: `learning`, `learned`, `learns` → `learn`

Exemple:
```python
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    use_stemming=True
)
```

**Résultat:** 
- Sans stemming: `['applications', 'applied', 'applying', ...]`
- Avec stemming: `['appli', ...]` (regroupe toutes les formes)

---

## Stratégies Combinées

### Stratégie 1: Nettoyage Léger
```python
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    remove_stop_words=True,
    min_freq=2
)
```
**Usage:** Supprime le bruit sans trop réduire le vocabulaire.

---

### Stratégie 2: Réduction Modérée
```python
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    remove_stop_words=True,
    min_freq=3,
    max_vocab_size=200
)
```
**Usage:** Bon équilibre entre taille et qualité.
**Résultat:** ~200 termes (réduction de 70-80%)

---

### Stratégie 3: Réduction Aggressive
```python
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    remove_stop_words=True,
    min_freq=5,
    max_vocab_size=100,
    use_stemming=True
)
```
**Usage:** Vocabulaire très compact pour calculs rapides.
**Résultat:** ~100 termes (réduction de 85-90%)

---

### Stratégie 4: Filtrage par Fréquence
```python
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    min_freq=2,      # Pas trop rares
    max_freq=50,     # Pas trop fréquents
    remove_stop_words=True
)
```
**Usage:** Garde les termes discriminants (ni trop rares, ni trop communs).

---

## Ordre d'Application

La fonction applique les filtres dans cet ordre:

1. **Suppression des mots-outils** (`remove_stop_words`)
2. **Filtrage par fréquence minimale** (`min_freq`)
3. **Filtrage par fréquence maximale** (`max_freq`)
4. **Limitation de taille** (`max_vocab_size`)
5. **Stemming** (`use_stemming`)

---

## Exemples Pratiques

### Exemple 1: Usage dans export_sparse_embeddings.py

```python
from scripts.export_sparse_embeddings import collect_texts, decimate
from src.utils import concat_sparse

# Collecter les textes
ids, embeddings = collect_texts()
ids, vocab, matrix = concat_sparse(ids, embeddings)

# Appliquer la décimation
ids, vocab, matrix = decimate(
    ids, vocab, matrix,
    max_vocab_size=200,
    remove_stop_words=True,
    min_freq=2
)

# Sauvegarder
import pickle
with open("data/embeddings_decimated.pkl", "wb") as f:
    pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)
```

### Exemple 2: Tester Différentes Configurations

```python
# Configuration originale
original_size = len(vocab)

# Test 1
ids1, vocab1, matrix1 = decimate(ids, vocab, matrix, max_vocab_size=300)
print(f"Config 1: {len(vocab1)} termes ({100*len(vocab1)/original_size:.1f}%)")

# Test 2
ids2, vocab2, matrix2 = decimate(
    ids, vocab, matrix,
    remove_stop_words=True,
    min_freq=3,
    max_vocab_size=200
)
print(f"Config 2: {len(vocab2)} termes ({100*len(vocab2)/original_size:.1f}%)")
```

---

## Impact sur les Performances

### Avantages de la Décimation

✅ **Calculs plus rapides** - Moins de dimensions à traiter  
✅ **Moins de mémoire** - Matrices plus petites  
✅ **Meilleure généralisation** - Réduit le sur-apprentissage  
✅ **Focus sur termes importants** - Élimine le bruit

### Inconvénients Potentiels

⚠️ **Perte d'information** - Certains termes utiles peuvent disparaître  
⚠️ **Précision réduite** - Sur des requêtes très spécifiques  
⚠️ **Stemming imparfait** - Le stemming simple peut être trop agressif

---

## Recommandations

1. **Commencez léger**: Test avec `remove_stop_words=True` et `min_freq=2`
2. **Testez progressivement**: Augmentez graduellement les restrictions
3. **Évaluez l'impact**: Comparez les résultats de recherche avant/après
4. **Adaptez au contexte**:
   - Corpus technique → Gardez plus de termes spécifiques
   - Corpus général → Décimation plus agressive acceptable

---

## Résumé des Tests

Sur le corpus `corpus_small.jsonl` (728 termes initiaux):

| Configuration | Termes finaux | Réduction |
|--------------|---------------|-----------|
| Stop words seulement | 694 | 4.7% |
| Top 100 fréquents | 100 | 86.3% |
| min_freq=3 | 168 | 76.9% |
| Combiné (stop + min=2 + top 150) | 150 | 79.4% |
| Avec stemming | 96 | 86.8% |

---

## Script de Test

Utilisez `scripts/test_decimation.py` pour tester différentes configurations:

```bash
poetry run python3 scripts/test_decimation.py
```

