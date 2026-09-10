"""
Project 3: Word Embeddings -- TF-IDF + PCA from scratch.

This script builds TF-IDF document vectors for a small corpus and then
reduces their dimensionality with Principal Component Analysis (PCA),
implementing every PCA step by hand (mean/centering, covariance matrix,
eigen-decomposition, sorting) rather than calling a library PCA function.
Only numpy is used, and only for basic linear algebra (matrix products,
logarithms, and the symmetric eigensolver `eigh`).

Run with: python3 tfidf_pca.py
Regenerates every number quoted in Project3_Word_Embeddings.md and writes
the full tables to project3_tfidf_pca_tables.md.
"""

import numpy as np

np.set_printoptions(suppress=True, linewidth=200)

# ---------------------------------------------------------------------------
# Part 1: Corpus -- three topics (animals, cars, cooking), 3 documents each
# ---------------------------------------------------------------------------

CORPUS = [
    "the dog chased the cat around the yard",           # d1 - animals
    "the cat sat quietly while the dog barked",         # d2 - animals
    "dogs and cats often play together in the yard",    # d3 - animals
    "the car engine roared down the highway",           # d4 - cars
    "she drove her new car to the highway",             # d5 - cars
    "the mechanic fixed the car engine quickly",        # d6 - cars
    "the chef cooked pasta with fresh tomato sauce",    # d7 - cooking
    "fresh tomato sauce makes the pasta taste great",   # d8 - cooking
    "the recipe calls for fresh basil and tomato sauce",  # d9 - cooking
]

LABELS = ["animals"] * 3 + ["cars"] * 3 + ["cooking"] * 3

STOPWORDS = {
    "the", "and", "in", "with", "to", "her", "on", "while", "often",
    "together", "for", "makes", "quickly", "down", "around", "near",
    "quietly", "calls", "she",
}


def tokenize(doc):
    return [w for w in doc.lower().split() if w not in STOPWORDS]


# ---------------------------------------------------------------------------
# Part 2: TF-IDF vectorization (built from scratch)
# ---------------------------------------------------------------------------

def build_tfidf_matrix(corpus):
    tokenized = [tokenize(d) for d in corpus]
    vocab = sorted({w for doc in tokenized for w in doc})
    vocab_index = {w: i for i, w in enumerate(vocab)}
    n_docs, n_terms = len(corpus), len(vocab)

    # Term frequency: count of term t in document d, divided by doc length
    tf = np.zeros((n_docs, n_terms))
    for d, doc in enumerate(tokenized):
        for w in doc:
            tf[d, vocab_index[w]] += 1.0
        if len(doc) > 0:
            tf[d, :] /= len(doc)

    # Document frequency: number of documents containing term t
    df = np.count_nonzero(tf > 0, axis=0)

    # Inverse document frequency (smoothed, matches scikit-learn's default):
    # idf(t) = ln((1 + N) / (1 + df(t))) + 1
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    tfidf = tf * idf
    return tfidf, vocab, idf


# ---------------------------------------------------------------------------
# Part 3: PCA from scratch
#   Step 1: compute the mean of the data and center it
#   Step 2: compute the covariance matrix of the centered data
#   Step 3: compute eigenvalues/eigenvectors of the covariance matrix
#   Step 4: sort eigenpairs by eigenvalue, descending
#   Step 5: project the centered data onto the top-k eigenvectors
# ---------------------------------------------------------------------------

def pca(X, n_components):
    n_samples, n_features = X.shape

    # Step 1: mean and centering
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Step 2: covariance matrix (features x features), unbiased (n-1) estimator
    cov = (X_centered.T @ X_centered) / (n_samples - 1)

    # Step 3: eigenvalues/eigenvectors. `cov` is real and symmetric, so we
    # use eigh (not the general eig), which is faster and numerically more
    # stable and guarantees real eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 4: eigh returns eigenvalues in ascending order -- reverse to sort
    # descending (most variance first) and reorder the matching eigenvectors
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    # Step 5: project onto the top n_components eigenvectors
    components = eigenvectors[:, :n_components]
    X_reduced = X_centered @ components

    return {
        "mean": mean,
        "covariance": cov,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance_ratio": explained_variance_ratio,
        "X_reduced": X_reduced,
    }


def main():
    tfidf, vocab, idf = build_tfidf_matrix(CORPUS)
    result = pca(tfidf, n_components=2)

    lines = []
    lines.append("# Project 3 -- Generated Tables (from tfidf_pca.py)\n")

    lines.append("## Vocabulary and IDF weights\n")
    lines.append("| term | idf |")
    lines.append("|---|---|")
    for w, v in zip(vocab, idf):
        lines.append(f"| {w} | {v:.4f} |")
    lines.append("")

    lines.append("## TF-IDF matrix (documents x terms)\n")
    header = "| doc | topic | " + " | ".join(vocab) + " |"
    sep = "|---|---|" + "---|" * len(vocab)
    lines.append(header)
    lines.append(sep)
    for i, row in enumerate(tfidf):
        vals = " | ".join(f"{v:.3f}" for v in row)
        lines.append(f"| d{i+1} | {LABELS[i]} | {vals} |")
    lines.append("")

    lines.append("## Feature means (mean vector used for centering)\n")
    lines.append("| term | mean tf-idf |")
    lines.append("|---|---|")
    for w, m in zip(vocab, result["mean"]):
        lines.append(f"| {w} | {m:.4f} |")
    lines.append("")

    lines.append("## Sorted eigenvalues of the covariance matrix\n")
    lines.append("| rank | eigenvalue | explained variance ratio | cumulative |")
    lines.append("|---|---|---|---|")
    cum = 0.0
    for i, (val, ratio) in enumerate(zip(result["eigenvalues"], result["explained_variance_ratio"])):
        cum += ratio
        lines.append(f"| {i+1} | {val:.6f} | {ratio:.4%} | {cum:.4%} |")
    lines.append("")

    lines.append("## Documents projected onto the top 2 principal components\n")
    lines.append("| doc | topic | PC1 | PC2 |")
    lines.append("|---|---|---|---|")
    for i, (pc1, pc2) in enumerate(result["X_reduced"]):
        lines.append(f"| d{i+1} | {LABELS[i]} | {pc1:.4f} | {pc2:.4f} |")
    lines.append("")

    lines.append("## Top loadings (largest |weight|) of PC1 and PC2\n")
    for pc_idx in range(2):
        loadings = result["eigenvectors"][:, pc_idx]
        order = np.argsort(-np.abs(loadings))[:5]
        lines.append(f"**PC{pc_idx+1}:** " + ", ".join(
            f"{vocab[j]} ({loadings[j]:+.3f})" for j in order
        ))
    lines.append("")

    with open("project3_tfidf_pca_tables.md", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
