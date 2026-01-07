# Implémentation TF-IDF

## Vue d'ensemble

Cette implémentation ajoute la pondération **TF-IDF** (Term Frequency - Inverse Document Frequency) au système d'embeddings sparse existant.

## Principe TF-IDF

TF-IDF pondère les termes selon leur importance relative :
- **Termes fréquents partout** (ex: "the", "and") → poids faible
- **Termes rares et spécifiques** (ex: "convolutional", "optimization") → poids élevé

### Formule

```
TF-IDF(mot, doc) = TF(mot, doc) × IDF(mot)

où:
  TF(mot, doc) = fréquence(mot, doc) / longueur(doc)
  IDF(mot) = log(nombre_total_docs / nombre_docs_contenant_mot)
```

## Modifications apportées

### 1. `src/utils.py`

Ajout de la fonction `tfidf_weights()` :
- Calcule la fréquence des documents (DF) pour chaque terme
- Calcule l'IDF : log(N / DF)
- Normalise les fréquences (TF) par la longueur du document
- Retourne les embeddings pondérés TF-IDF

### 2. `scripts/export_sparse_embeddings.py`

Modifications de `collect_texts()` :
- Nouveau paramètre `use_tfidf` (bool)
- Applique automatiquement la pondération TF-IDF si activé
- Génère le fichier `sparses_embedding_tfidf.pkl`

Variable de contrôle dans `main()` :
```python
USE_TFIDF = True  # Activer/désactiver TF-IDF
```

## Fichiers générés

### Embeddings TF-IDF
- **Fichier**: `data/sparses_embedding_tfidf.pkl`
- **Taille**: 138 MB
- **Documents**: 26,657
- **Vocabulaire**: 2,663 termes (après décimation)
- **Réduction**: 87% par rapport au vocabulaire initial

### Caractéristiques
- Stop words supprimés
- Stemming appliqué
- Fréquence min: 3 occurrences
- Fréquence max: 350 occurrences
- Vocabulaire limité aux 3,000 termes les plus fréquents

## Utilisation

### Générer des embeddings TF-IDF

```bash
poetry run python scripts/export_sparse_embeddings.py
```

Modifier `USE_TFIDF = True/False` dans le script selon le besoin.

### Générer des prédictions avec TF-IDF

```bash
poetry run python scripts/generate_predictions_sparse.py
```

Modifier le chemin du fichier pickle pour utiliser `sparses_embedding_tfidf.pkl`.

## Avantages attendus

1. **Meilleure discrimination** : Les termes rares et spécifiques ont plus de poids
2. **Réduction du bruit** : Les mots communs ont moins d'influence
3. **Standard établi** : TF-IDF est un standard dans l'IR (Information Retrieval)

## Comparaison avec fréquences brutes

| Méthode | Poids "the" | Poids "convolutional" | Discrimination |
|---------|-------------|----------------------|----------------|
| Fréquence brute | Élevé | Variable | Faible |
| TF-IDF | Faible | Élevé | Élevée |

## Prochaines étapes possibles

1. **BM25** : Variante améliorée de TF-IDF avec saturation de fréquence
2. **N-grams** : Capturer des séquences de mots ("neural network")
3. **Lemmatisation** : Analyse grammaticale plus précise que le stemming

