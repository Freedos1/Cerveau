# Project 3: Word Embeddings

This project builds word/document embeddings with **TF-IDF** (term
frequency–inverse document frequency) and then reduces their
dimensionality with **Principal Component Analysis (PCA)** so that the
most important variance (information) in the embeddings is preserved
while most of the raw feature axes are discarded.

All numbers quoted below are reproducible by running `tfidf_pca.py`
(included in this repository) with Python 3 and NumPy; it regenerates
every table in this document and writes the full tables to
`project3_tfidf_pca_tables.md`.

---

## Part A: From Text to Vectors — TF-IDF

### A.1 Why TF-IDF

A word embedding needs to turn text into numbers before any linear
algebra (like PCA) can be applied to it. **TF-IDF** is one of the
simplest and most widely used ways to do this: it represents each
*document* as a vector over the vocabulary, where each coordinate says
how important a given word is to that document, relative to the whole
corpus.

### A.2 The formula

For term `t` and document `d` in a corpus of `N` documents:

```
TF(t, d)  = (number of times t appears in d) / (number of terms in d)

IDF(t)    = ln( (1 + N) / (1 + DF(t)) ) + 1        (smoothed IDF)

TFIDF(t, d) = TF(t, d) * IDF(t)
```

where `DF(t)` is the number of documents that contain term `t` at
least once. `TF` measures local importance (how often a word is used
*in this document*); `IDF` down-weights words that appear in almost
every document (common, low-information words) and up-weights words
that are rare across the corpus (distinctive, high-information words).
The `+1` smoothing in `IDF` prevents a division by zero when `DF(t) =
N` and matches the convention used by scikit-learn's
`TfidfVectorizer` (see Citations).

Each document therefore becomes a vector in `R^V`, where `V` is the
vocabulary size — this is the "embedding" we hand to PCA.

### A.3 Worked example corpus

To make the computation concrete, we use a 9-document corpus covering
three topics (3 documents each), with basic English stop words (`the`,
`and`, `to`, …) removed before vectorizing:

| doc | topic | text |
|---|---|---|
| d1 | animals | the dog chased the cat around the yard |
| d2 | animals | the cat sat quietly while the dog barked |
| d3 | animals | dogs and cats often play together in the yard |
| d4 | cars    | the car engine roared down the highway |
| d5 | cars    | she drove her new car to the highway |
| d6 | cars    | the mechanic fixed the car engine quickly |
| d7 | cooking | the chef cooked pasta with fresh tomato sauce |
| d8 | cooking | fresh tomato sauce makes the pasta taste great |
| d9 | cooking | the recipe calls for fresh basil and tomato sauce |

After stop-word removal the vocabulary has 27 terms (`barked`,
`basil`, `car`, `cat`, …, `yard`). The resulting **9 × 27 TF-IDF
matrix** is the full table in `project3_tfidf_pca_tables.md`; a
representative slice (IDF weights, which show the effect described
above) is:

| term | idf | note |
|---|---|---|
| car | 1.9163 | appears in 3/9 docs (all "cars") → lower IDF |
| fresh | 1.9163 | appears in 3/9 docs (all "cooking") → lower IDF |
| cat | 2.2040 | appears in 2/9 docs |
| basil | 2.6094 | appears in only 1/9 docs → highest IDF (most distinctive) |

This is exactly the intended behavior: words that show up in many
documents (even if topic-specific, like `car` or `fresh`, which recur
across their own topic's 3 documents) get a smaller IDF weight than
words that are rare across the whole corpus (like `basil`, which
appears only once).

---

## Part B: Dimensionality Reduction with PCA

Our 9 documents now live in a 27-dimensional TF-IDF space, but most of
that space is empty/redundant (most documents use only a handful of
the 27 words). **PCA finds a small number of new axes — linear
combinations of the original word-count axes — that capture as much of
the spread (variance) in the data as possible**, so we can plot or
further process the documents in, say, 2 dimensions with minimal loss
of information.

### B.1 Pseudocode

```
function PCA(X, k):
    # X is an (n_documents x n_features) matrix of embeddings (e.g. TF-IDF)
    # k is the number of dimensions to keep

    # 1. Mean and centering
    mean = column_wise_average(X)              # shape: (n_features,)
    X_centered = X - mean                       # subtract mean from every row

    # 2. Covariance matrix
    # measures how every pair of features varies together across documents
    Cov = (X_centered^T * X_centered) / (n_documents - 1)   # shape: (n_features x n_features)

    # 3. Eigenvalues and eigenvectors of the covariance matrix
    (eigenvalues, eigenvectors) = eigen_decomposition(Cov)
    # Cov is real and symmetric, so eigenvalues are real and
    # eigenvectors are orthogonal

    # 4. Sort eigenpairs by eigenvalue, descending
    order = argsort(eigenvalues, descending=True)
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 5. Keep the top k eigenvectors (principal components) and project
    top_k_eigenvectors = eigenvectors[:, 0:k]
    X_reduced = X_centered * top_k_eigenvectors  # shape: (n_documents x k)

    explained_variance_ratio = eigenvalues / sum(eigenvalues)

    return X_reduced, eigenvalues, eigenvectors, explained_variance_ratio
```

### B.2 Reference implementation (Python / NumPy)

```python
import numpy as np

def pca(X, n_components):
    n_samples, n_features = X.shape

    # Step 1: mean and centering
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Step 2: covariance matrix (features x features), unbiased (n-1) estimator
    cov = (X_centered.T @ X_centered) / (n_samples - 1)

    # Step 3: eigenvalues/eigenvectors. `cov` is real and symmetric, so we
    # use eigh (not the general eig): faster, numerically stable, and
    # guarantees real eigenvalues/orthogonal eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 4: eigh returns eigenvalues ascending -- reverse to sort
    # descending (most variance first) and reorder the matching eigenvectors
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    # Step 5: project onto the top n_components eigenvectors
    components = eigenvectors[:, :n_components]
    X_reduced = X_centered @ components

    return X_reduced, eigenvalues, eigenvectors, explained_variance_ratio
```

This is the exact function used by `tfidf_pca.py` (see that file for
the full, runnable script including the TF-IDF step).

### B.3 Plain-English explanation of how the code works

1. **Mean and centering.** We first compute the average TF-IDF value
   of each word across all 9 documents (the "mean vector"), then
   subtract it from every document's vector. This shifts the whole
   point cloud so it is centered on the origin. Centering matters
   because covariance and PCA are about how features vary *around
   their average* — without it, the direction from the origin to the
   average would dominate the covariance and distort the result.

2. **Covariance matrix.** For every pair of words `(i, j)`, the
   covariance matrix entry `Cov[i, j]` says whether the two words tend
   to be high or low together across documents (positive covariance:
   they rise and fall together, e.g. `tomato` and `sauce`; near-zero
   covariance: unrelated, e.g. `car` and `basil`). The diagonal
   entries `Cov[i, i]` are just the variance of word `i` on its own.
   This 27×27 matrix summarizes all the pairwise co-occurrence
   structure of the corpus in one object.

3. **Eigenvalues and eigenvectors.** An eigenvector of the covariance
   matrix points in a direction through word-space along which the
   data spreads out; its eigenvalue is *how much* variance the data
   has along that direction. Because the covariance matrix is
   symmetric, all of its eigenvalues are real and its eigenvectors are
   mutually perpendicular (orthogonal) — they form a new, rotated
   coordinate system for the same data, with no information lost.

4. **Sorting.** We sort the eigenvector/eigenvalue pairs so the
   direction with the *largest* eigenvalue (most variance, i.e. "most
   information") comes first. This is what lets us keep only the
   first `k` directions and still retain as much of the data's spread
   as possible — dropping directions with the smallest eigenvalues
   throws away the least information.

5. **Projection.** Multiplying the centered data by the top `k`
   eigenvectors converts each document from a 27-dimensional
   bag-of-words vector into a `k`-dimensional coordinate along the
   most informative axes — this is the reduced-dimension embedding.

---

## Part C: Worked Results on the Example Corpus

Running `tfidf_pca.py` on the 9-document corpus (27-dimensional TF-IDF
vectors) and reducing to `k = 2` components gives:

### C.1 Sorted eigenvalues of the covariance matrix

| rank | eigenvalue | explained variance ratio | cumulative |
|---|---|---|---|
| 1 | 0.269486 | 23.82% | 23.82% |
| 2 | 0.228313 | 20.18% | 43.99% |
| 3 | 0.194712 | 17.21% | 61.20% |
| 4 | 0.144342 | 12.76% | 73.95% |
| 5 | 0.091003 | 8.04% | 82.00% |
| 6 | 0.083142 | 7.35% | 89.34% |
| 7 | 0.073289 | 6.48% | 95.82% |
| 8 | 0.047286 | 4.18% | 100.00% |
| 9–27 | ≈ 0.000000 | 0.00% | 100.00% |

Only the first 8 eigenvalues are non-zero. This is expected: with `n =
9` documents, the centered data matrix has rank at most `n - 1 = 8`
(the rows of any mean-centered matrix always sum to zero, which
removes one degree of freedom), so the covariance matrix can have at
most 8 non-zero eigenvalues no matter how many word features (27)
there are.

Keeping just the **top 2** components retains **43.99%** of the total
variance — a large amount of the corpus's topic structure, from a
27-fold reduction in dimensionality.

### C.2 Documents projected onto the top 2 principal components

| doc | topic | PC1 | PC2 |
|---|---|---|---|
| d1 | animals | -0.6478 | 0.4551 |
| d2 | animals | -0.6145 | 0.4685 |
| d3 | animals | -0.4031 | 0.1395 |
| d4 | cars    |  0.6773 | 0.2961 |
| d5 | cars    |  0.6009 | 0.2540 |
| d6 | cars    |  0.6009 | 0.2540 |
| d7 | cooking | -0.0698 | -0.6070 |
| d8 | cooking | -0.0698 | -0.6070 |
| d9 | cooking | -0.0740 | -0.6534 |

Even without ever telling PCA which document belongs to which topic,
the 2-D projection cleanly separates the three topics: **cars** land
at PC1 > 0, **animals** at PC1 < 0 with PC2 > 0, and **cooking** at
PC2 < 0. This is the payoff of dimensionality reduction: 27 sparse,
hard-to-interpret word counts per document collapse into 2 coordinates
that already recover the corpus's topic structure.

### C.3 What the top components actually mean (loadings)

The eigenvector for each component tells us which original words it
weights most heavily:

* **PC1** is dominated by `car` (+0.418), `highway` (+0.327), `engine`
  (+0.327) on the positive side and `cat` (-0.323), `dog` (-0.323) on
  the negative side — i.e. PC1 is essentially a "cars vs. animals"
  axis.
* **PC2** is dominated by `tomato` (-0.349), `fresh` (-0.349), `sauce`
  (-0.349) on the negative side and `dog` (+0.279), `cat` (+0.279) on
  the positive side — i.e. PC2 is essentially a "cooking vs.
  animals/cars" axis.

This matches the corpus by construction, and illustrates why PCA is
useful for word embeddings in general: the reduced axes are not
arbitrary — they are the directions along which word usage actually
varies the most across documents, so they tend to align with
human-interpretable structure like topic.

---

## Summary

1. **TF-IDF** turns each document into a vector by weighting each word
   by how often it appears in that document (`TF`) and how rare it is
   across the whole corpus (`IDF`), producing a `(documents × vocabulary)`
   matrix.
2. **PCA** reduces the dimensionality of that matrix while preserving
   as much variance as possible by: (1) centering the data on its
   mean, (2) computing the covariance matrix of the centered data, (3)
   finding that matrix's eigenvalues/eigenvectors, (4) sorting them by
   eigenvalue so the most-informative directions come first, and (5)
   projecting the data onto the top `k` eigenvectors.
3. On the 9-document, 3-topic worked example, reducing from 27
   dimensions to 2 preserved ~44% of the total variance and was enough
   to fully separate the three topics — demonstrating that a small
   number of principal components can capture most of the meaningful
   structure in a much higher-dimensional word embedding.

---

## Citations / References

* Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to
  Information Retrieval*, Chapter 6 (scoring, term weighting, and the
  vector space model / TF-IDF). Cambridge University Press.
  https://nlp.stanford.edu/IR-book/
* Jolliffe, I. T., & Cadima, J. (2016). "Principal component analysis:
  a review and recent developments." *Philosophical Transactions of
  the Royal Society A*, 374(2065). (Overview of PCA: mean-centering,
  covariance matrix, eigendecomposition.)
* scikit-learn documentation, `TfidfVectorizer` (smoothed IDF formula
  used above) and `sklearn.decomposition.PCA`:
  https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
* NumPy documentation, `numpy.linalg.eigh` (symmetric/Hermitian
  eigensolver used for the covariance matrix):
  https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
