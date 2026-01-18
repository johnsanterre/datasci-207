"""
DATASCI 207: Module 8 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def kmeans_step(X, centroids):
    """Perform one iteration of k-means."""
    k = len(centroids)
    n_samples = len(X)
    
    # Compute distances
    distances = np.zeros((n_samples, k))
    for j in range(k):
        distances[:, j] = np.linalg.norm(X - centroids[j], axis=1)
    
    # Assign labels
    labels = np.argmin(distances, axis=1)
    
    # Update centroids
    new_centroids = np.zeros_like(centroids)
    for j in range(k):
        if np.sum(labels == j) > 0:
            new_centroids[j] = X[labels == j].mean(axis=0)
        else:
            new_centroids[j] = centroids[j]
    
    return new_centroids, labels


def silhouette_one(X, labels, i):
    """Compute silhouette score for point i."""
    cluster_i = labels[i]
    same = X[labels == cluster_i]
    
    # a: avg distance within cluster
    if len(same) > 1:
        a = np.mean(np.linalg.norm(same - X[i], axis=1))
    else:
        a = 0
    
    # b: min avg distance to other clusters
    b = np.inf
    for k in np.unique(labels):
        if k != cluster_i:
            other = X[labels == k]
            b = min(b, np.mean(np.linalg.norm(other - X[i], axis=1)))
    
    if b == np.inf:
        return 0
    return (b - a) / max(a, b)


def pca_step1(X):
    """Center data and compute covariance."""
    X_centered = X - X.mean(axis=0)
    cov_matrix = np.cov(X_centered.T)
    return X_centered, cov_matrix


if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(20, 2) + [0, 0],
        np.random.randn(20, 2) + [5, 5]
    ])
    
    # Verify kmeans step
    centroids = np.array([[0, 0], [5, 5]])
    new_c, labels = kmeans_step(X, centroids)
    assert len(np.bincount(labels)) == 2
    print("K-means step: VERIFIED")
    
    # Verify silhouette
    labels = np.array([0]*20 + [1]*20)
    s = silhouette_one(X, labels, 0)
    assert s > 0
    print("Silhouette: VERIFIED")
    
    # Verify PCA
    Xc, cov = pca_step1(X)
    assert np.allclose(Xc.mean(axis=0), 0, atol=0.01)
    print("PCA prep: VERIFIED")
