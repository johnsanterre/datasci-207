"""
DATASCI 207: Module 8 - Exercise: Clustering and PCA
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Implement K-Means one iteration
# =============================================================================

def kmeans_step(X, centroids):
    """
    Perform one iteration of k-means.
    
    Args:
        X: Data array, shape (n_samples, n_features)
        centroids: Current centroids, shape (k, n_features)
    
    Returns:
        new_centroids: Updated centroids
        labels: Cluster assignments for each point
    """
    k = len(centroids)
    n_samples = len(X)
    
    # TODO: 
    # 1. Compute distance from each point to each centroid
    # 2. Assign each point to nearest centroid (labels)
    # 3. Update centroids as mean of assigned points
    
    labels = None  # Replace
    new_centroids = None  # Replace
    
    return new_centroids, labels


# =============================================================================
# EXERCISE 2: Compute Silhouette for One Point
# =============================================================================

def silhouette_one(X, labels, i):
    """
    Compute silhouette score for point i.
    
    s(i) = (b - a) / max(a, b)
    a = average distance to same cluster
    b = average distance to nearest other cluster
    """
    # TODO: Implement silhouette for point i
    score = None  # Replace
    return score


# =============================================================================
# EXERCISE 3: PCA - Center and Covariance
# =============================================================================

def pca_step1(X):
    """
    First steps of PCA: center data and compute covariance.
    
    Returns:
        X_centered: Data with mean subtracted
        cov_matrix: Covariance matrix
    """
    # TODO: 
    # 1. Subtract mean from each feature
    # 2. Compute covariance matrix
    
    X_centered = None  # Replace
    cov_matrix = None  # Replace
    
    return X_centered, cov_matrix


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Unsupervised Learning Concepts\n")
    
    # Test data
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(20, 2) + [0, 0],
        np.random.randn(20, 2) + [5, 5]
    ])
    
    # Test 1: K-means step
    print("=" * 50)
    print("Test 1: K-Means Step")
    print("=" * 50)
    initial_centroids = np.array([[0, 0], [5, 5]])
    new_centroids, labels = kmeans_step(X, initial_centroids)
    
    if new_centroids is not None and labels is not None:
        print(f"Initial centroids:\n{initial_centroids}")
        print(f"New centroids:\n{new_centroids}")
        print(f"Cluster sizes: {np.bincount(labels)}")
        # Should be roughly 20, 20
        if len(np.bincount(labels)) == 2:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: Silhouette
    print("\n" + "=" * 50)
    print("Test 2: Silhouette Score")
    print("=" * 50)
    labels = np.array([0]*20 + [1]*20)
    score = silhouette_one(X, labels, 0)
    
    if score is not None:
        print(f"Silhouette for point 0: {score:.4f}")
        print("(Should be positive for well-separated clusters)")
        if score > 0:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 3: PCA prep
    print("\n" + "=" * 50)
    print("Test 3: PCA Centering and Covariance")
    print("=" * 50)
    X_centered, cov = pca_step1(X)
    
    if X_centered is not None and cov is not None:
        print(f"Original mean: {X.mean(axis=0)}")
        print(f"Centered mean: {X_centered.mean(axis=0)}")
        print(f"Covariance shape: {cov.shape}")
        if np.allclose(X_centered.mean(axis=0), 0, atol=0.01):
            print("PASS")
    else:
        print("Not implemented yet")
