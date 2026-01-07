# Partie 6 - Exploration Thématique avec LDA

## Texte pour le Rapport

### 6.1 Introduction et Motivation

Dans cette partie exploratoire, nous avons appliqué un modèle de découverte de thématiques (topic modeling) pour mieux comprendre la structure latente du corpus d'articles scientifiques. L'objectif est double : d'une part, obtenir une vision d'ensemble des domaines couverts par les 25,657 articles, et d'autre part, explorer une dimension complémentaire aux approches creuses (TF-IDF) et denses (sentence transformers) utilisées précédemment.

Nous avons choisi le modèle **LDA (Latent Dirichlet Allocation)**, un algorithme probabiliste largement utilisé en traitement automatique des langues. Le principe fondamental du LDA repose sur deux hypothèses :
1. Chaque document est un mélange de plusieurs thématiques
2. Chaque thématique est une distribution de probabilité sur les mots du vocabulaire

### 6.2 Méthodologie

#### 6.2.1 Prétraitement des Textes

Les textes ont été préparés en combinant le titre et le résumé de chaque article. Un pipeline de prétraitement a été appliqué via `CountVectorizer` de scikit-learn :

- **Limitation du vocabulaire** : 5,000 termes maximum
- **Filtrage fréquentiel** :
  - `min_df=5` : élimination des mots trop rares (< 5 occurrences)
  - `max_df=0.7` : élimination des mots trop fréquents (> 70% des documents)
- **Stop words** : suppression des mots vides anglais
- **Normalisation** : conversion en minuscules, tokens d'au moins 3 caractères

Cette étape a produit une matrice documents-termes de dimensions 25,657 × 5,000 avec une sparsité de 98.76%.

#### 6.2.2 Configuration du Modèle LDA

Le modèle a été configuré avec les paramètres suivants :
- **n_components** : 10 thématiques
- **learning_method** : 'online' (plus rapide que 'batch')
- **max_iter** : 20 itérations
- **batch_size** : 128 documents par batch

Le choix de 10 thématiques résulte d'un compromis entre granularité (trop peu de topics = thèmes trop généraux) et interprétabilité (trop de topics = redondance, difficulté d'interprétation).

### 6.3 Résultats

#### 6.3.1 Thématiques Découvertes

Le modèle a identifié 10 thématiques distinctes et interprétables :

| ID | Thématique | Mots-clés principaux | Distribution |
|----|-----------|---------------------|--------------|
| 0 | Antennes & Télécommunications | antenna, frequency, signal, ghz, array | 5.8% |
| 1 | Programmation & Décision | learning, problems, decision, code, programming | 1.7% |
| 2 | **Machine Learning** | learning, neural, classification, deep, training | **14.1%** |
| 3 | Contrôle & Énergie | power, control, performance, energy, robot | 9.8% |
| 4 | Vision par Ordinateur | image, recognition, object, visual, detection | 10.4% |
| 5 | **Systèmes Logiciels** | research, systems, software, development, design | **16.3%** |
| 6 | Études Médicales | study, patients, health, effects, factors | 11.0% |
| 7 | Réseaux & Sécurité | network, security, sensor, traffic, attacks | 4.6% |
| 8 | Web & Réseaux Sociaux | user, social, web, internet, online | 9.7% |
| 9 | **Algorithmes** | algorithm, method, time, problem, approach | **16.4%** |

**Figure 1** : Distribution des documents par thématique dominante (voir `lda_topic_distribution.png`)

**Figure 2** : Mots les plus représentatifs par thématique (voir `lda_top_words.png`)

#### 6.3.2 Distribution Thématique

La répartition des documents montre trois thématiques dominantes :
- **Algorithmes** (16.4%) et **Systèmes Logiciels** (16.3%) : reflètent le caractère informatique du corpus
- **Machine Learning** (14.1%) : confirme l'importance de l'IA dans la recherche actuelle

À l'opposé, la thématique **Programmation** (1.7%) est la moins représentée, suggérant que les articles de programmation pure sont minoritaires par rapport aux applications.

#### 6.3.3 Exemples de Documents Représentatifs

**Topic 2 (Machine Learning)** - Documents avec score > 0.93 :
- "An Association Thesaurus for Information Retrieval" (0.983)
- "Using Clustering for Categorization of Support Tickets" (0.936)
- "Bigrams of Syntactic Labels for Authorship Discrimination" (0.931)

**Topic 4 (Vision par Ordinateur)** :
- "Multiple View Geometry in Computer Vision" (0.850)
- "Robust Real-Time Face Detection" (0.850)

Ces exemples montrent que le modèle capture bien la sémantique des articles.

#### 6.3.4 Qualité du Modèle

La **perplexité** du modèle sur le corpus d'entraînement est de **1761.80**. Cette métrique, bien que difficile à interpréter en valeur absolue, indique la capacité du modèle à prédire des mots dans de nouveaux documents. Une perplexité plus basse est meilleure, et notre valeur est cohérente pour un corpus de cette taille et diversité.

Le **log-vraisemblance** de -17,231,949 confirme la convergence du modèle après 20 itérations.

### 6.4 Discussion

#### 6.4.1 Insights sur le Corpus

L'analyse thématique révèle plusieurs observations importantes :

1. **Diversité scientifique** : Le corpus couvre un large spectre allant de l'informatique théorique (algorithmes) aux sciences humaines (santé, psychologie)

2. **Dominance de l'IA** : Les topics liés à l'apprentissage automatique représentent collectivement ~30% du corpus

3. **Hétérogénéité** : Cette diversité explique la difficulté de la tâche de recherche d'information : des articles pertinents peuvent appartenir à des domaines très différents

#### 6.4.2 Utilité pour le Moteur de Recherche

Bien que non directement intégré dans notre système final, cette exploration offre plusieurs perspectives :

**Option 1 - Features additionnelles** : Les distributions thématiques (`doc_topics`) pourraient être utilisées comme features supplémentaires, concaténées aux embeddings denses.

**Option 2 - Filtrage thématique** : Permettre aux utilisateurs de filtrer les résultats par thématique dominante.

**Option 3 - Réordonnancement** : Utiliser la similarité thématique (distance de Jensen-Shannon) pour réordonner les résultats.

Dans notre cas, nous avons choisi de ne pas intégrer ces features dans le système final pour plusieurs raisons :
- La perplexité élevée suggère que les thématiques sont assez générales
- Les embeddings denses (sentence transformers) capturent déjà implicitement la sémantique
- L'ajout de features LDA n'a pas montré d'amélioration significative lors de tests préliminaires

#### 6.4.3 Limites du LDA

Plusieurs limitations du LDA doivent être mentionnées :

1. **Nombre de topics fixe** : Nécessite de définir K à l'avance (ici 10), alors que le nombre optimal de thématiques peut être différent

2. **Hypothèse du "sac de mots"** : Le LDA ignore l'ordre des mots, perdant des informations contextuelles

3. **Distribution de Dirichlet** : L'hypothèse de distribution de Dirichlet n'est pas toujours adaptée à tous les corpus

4. **Interprétabilité variable** : Certains topics peuvent être redondants ou difficiles à interpréter

#### 6.4.4 Alternatives Modernes

Des approches plus récentes pourraient être explorées :

- **NMF (Non-negative Matrix Factorization)** : Souvent plus interprétable que LDA
- **BERTopic** : Utilise des embeddings BERT pour créer des clusters de documents, puis extrait des topics via TF-IDF
- **Top2Vec** : Approche entièrement basée sur des embeddings denses
- **LDA dynamique** : Pour corpus évoluant dans le temps

### 6.5 Conclusion

L'analyse LDA a permis de confirmer l'hétérogénéité du corpus d'articles scientifiques et d'identifier 10 thématiques distinctes couvrant différents domaines de l'informatique et au-delà. 

Bien que cette exploration n'ait pas été intégrée directement dans le moteur de recherche final, elle offre une compréhension précieuse de la structure du corpus et pourrait servir de base à des développements futurs, notamment pour des systèmes hybrides combinant similarité textuelle et similarité thématique.

Les résultats montrent que les thématiques liées à l'informatique et l'IA dominent le corpus (>60%), validant le choix d'un moteur de recherche spécialisé en littérature scientifique informatique.

---

## Fichiers et Code

### Script Complet
```bash
poetry run python scripts/topic_modeling_lda.py
```

### Code pour le Notebook (Cellule 47)
Le code minimal se trouve dans `notebooks/lda_cell.py`

### Résultats Générés
- `outputs/models/lda_model.pkl` : Modèle complet
- `outputs/models/lda_topics.json` : Topics en JSON
- `outputs/figures/lda_topic_distribution.png` : Distribution
- `outputs/figures/lda_top_words.png` : Visualisation des mots

### Documentation Complète
- `docs/lda_topic_modeling.md` : Guide complet
- `docs/lda_results_summary.md` : Résumé des résultats
- `docs/PARTIE_6_RAPPORT.md` : Ce document

---

## Références

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993-1022.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.

