"""
DATASCI 207: Applied Machine Learning
Module 8: Unsupervised Learning

This module covers:
- K-means clustering
- Hierarchical clustering
- Principal Component Analysis (PCA)
- Cluster evaluation

Using NumPy and scikit-learn - no pandas.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: K-MEANS CLUSTERING FROM SCRATCH
# =============================================================================

print("=== Part 1: K-Means Clustering ===\n")


def kmeans(X, k, max_iters=100):
    """
    K-means clustering from scratch.
    
    Args:
        X: Data array, shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum iterations
    
    Returns:
        centroids: Final cluster centers
        labels: Cluster assignment for each point
    """
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()
    
    for iteration in range(max_iters):
        # Assign points to nearest centroid
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.linalg.norm(X - centroids[j], axis=1)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.zeros((k, n_features))
        for j in range(k):
            if np.sum(labels == j) > 0:
                new_centroids[j] = X[labels == j].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {iteration}")
            break
        
        centroids = new_centroids
    
    return centroids, labels


# Generate sample data with 3 clusters
np.random.seed(42)
cluster1 = np.random.randn(30, 2) + [0, 0]
cluster2 = np.random.randn(30, 2) + [5, 5]
cluster3 = np.random.randn(30, 2) + [5, 0]
X = np.vstack([cluster1, cluster2, cluster3])

centroids, labels = kmeans(X, k=3)

print(f"Data shape: {X.shape}")
print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Centroids:\n{centroids}")


# =============================================================================
# PART 2: CHOOSING K (ELBOW METHOD)
# =============================================================================

print("\n=== Part 2: Choosing k ===\n")


def compute_inertia(X, centroids, labels):
    """Sum of squared distances to centroids."""
    inertia = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        inertia += np.sum((cluster_points - centroids[k]) ** 2)
    return inertia


print("Elbow method: compute inertia for different k values")
inertias = []
for k in range(1, 7):
    centroids, labels = kmeans(X.copy(), k, max_iters=50)
    inertia = compute_inertia(X, centroids, labels)
    inertias.append(inertia)
    print(f"  k={k}: inertia = {inertia:.2f}")

print("\nLook for the 'elbow' where adding more clusters doesn't help much.")


# =============================================================================
# PART 3: SILHOUETTE SCORE
# =============================================================================

print("\n=== Part 3: Silhouette Score ===\n")


def silhouette_sample(X, labels, i):
    """
    Compute silhouette score for a single sample.
    
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    a(i) = avg distance to same cluster
    b(i) = avg distance to nearest other cluster
    """
    cluster_i = labels[i]
    same_cluster = X[labels == cluster_i]
    
    # a(i): average distance within cluster
    if len(same_cluster) > 1:
        a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
    else:
        a = 0
    
    # b(i): min average distance to other clusters
    b = np.inf
    for k in np.unique(labels):
        if k != cluster_i:
            other_cluster = X[labels == k]
            avg_dist = np.mean(np.linalg.norm(other_cluster - X[i], axis=1))
            b = min(b, avg_dist)
    
    if b == np.inf:
        return 0
    
    return (b - a) / max(a, b)


def silhouette_score(X, labels):
    """Average silhouette score over all samples."""
    scores = [silhouette_sample(X, labels, i) for i in range(len(X))]
    return np.mean(scores)


centroids, labels = kmeans(X, k=3, max_iters=50)
sil_score = silhouette_score(X, labels)
print(f"Silhouette score for k=3: {sil_score:.4f}")
print("(Range: -1 to 1, higher is better)")


# =============================================================================
# PART 4: PCA FROM SCRATCH
# =============================================================================

print("\n=== Part 4: Principal Component Analysis ===\n")


def pca(X, n_components):
    """
    Principal Component Analysis from scratch.
    
    Args:
        X: Data array, shape (n_samples, n_features)
        n_components: Number of components to keep
    
    Returns:
        X_transformed: Projected data
        components: Principal component directions
        explained_variance: Variance explained by each component
    """
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    
    # Project data
    X_transformed = X_centered @ components
    
    # Explained variance ratio
    explained_variance = eigenvalues[:n_components] / eigenvalues.sum()
    
    return X_transformed, components, explained_variance


# Create higher-dimensional data
np.random.seed(42)
X_high = np.random.randn(100, 5)
X_high[:, 1] = X_high[:, 0] + np.random.randn(100) * 0.1  # Correlated
X_high[:, 2] = X_high[:, 0] * 2 + np.random.randn(100) * 0.1  # Correlated

X_pca, components, explained = pca(X_high, n_components=2)

print(f"Original shape: {X_high.shape}")
print(f"PCA shape: {X_pca.shape}")
print(f"Explained variance ratio: {explained}")
print(f"Total variance explained: {sum(explained):.2%}")


# =============================================================================
# PART 5: SCIKIT-LEARN COMPARISON
# =============================================================================

print("\n=== Part 5: scikit-learn Comparison ===\n")

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA as SKLearnPCA
    from sklearn.metrics import silhouette_score as sklearn_silhouette
    
    # K-means
    kmeans_sklearn = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_sklearn = kmeans_sklearn.fit_predict(X)
    
    print("K-Means (scikit-learn):")
    print(f"  Cluster sizes: {np.bincount(labels_sklearn)}")
    print(f"  Inertia: {kmeans_sklearn.inertia_:.2f}")
    print(f"  Silhouette: {sklearn_silhouette(X, labels_sklearn):.4f}")
    
    # PCA
    pca_sklearn = SKLearnPCA(n_components=2)
    X_pca_sklearn = pca_sklearn.fit_transform(X_high)
    
    print(f"\nPCA (scikit-learn):")
    print(f"  Explained variance: {pca_sklearn.explained_variance_ratio_}")
    print(f"  Total: {sum(pca_sklearn.explained_variance_ratio_):.2%}")

except ImportError:
    print("(scikit-learn not available)")


# =============================================================================
# PART 6: HIERARCHICAL CLUSTERING
# =============================================================================

print("\n=== Part 6: Hierarchical Clustering ===\n")

print("""
Hierarchical Clustering builds a tree (dendrogram):

1. Start: each point is its own cluster
2. Find closest pair of clusters
3. Merge them
4. Repeat until all in one cluster

Linkage methods for cluster distance:
- Single: min distance between any two points
- Complete: max distance between any two points
- Average: mean distance between all pairs
- Ward: minimize variance increase (often best)
""")

try:
    from sklearn.cluster import AgglomerativeClustering
    
    agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels_agg = agg.fit_predict(X)
    
    print("Agglomerative Clustering (Ward linkage):")
    print(f"  Cluster sizes: {np.bincount(labels_agg)}")
    print(f"  Silhouette: {sklearn_silhouette(X, labels_agg):.4f}")

except ImportError:
    print("(scikit-learn not available)")



# =============================================================================
# PART 7: GAUSSIAN MIXTURE MODELS AND EM
# =============================================================================

print("\n=== Part 7: Gaussian Mixture Models and EM ===\n")

np.random.seed(7)
data = np.concatenate([np.random.normal(-2.0, 0.6, 150),
                       np.random.normal(1.5, 1.0, 100)])

mu = np.array([-1.0, 1.0])      # initial guesses, deliberately mediocre
sd = np.array([1.0, 1.0])
pi = np.array([0.5, 0.5])       # mixture weights

def normal_pdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

for it in range(100):
    # E-step: responsibility of each component for each point
    dens = np.stack([pi[k] * normal_pdf(data, mu[k], sd[k]) for k in range(2)])
    resp = dens / dens.sum(axis=0)
    # M-step: re-estimate each component from its weighted points
    nk = resp.sum(axis=1)
    new_mu = (resp * data).sum(axis=1) / nk
    sd = np.sqrt((resp * (data - new_mu[:, None]) ** 2).sum(axis=1) / nk)
    pi = nk / len(data)
    moved = np.abs(new_mu - mu).max()
    mu = new_mu
    if moved < 1e-6:
        break

print("converged after", it + 1, "iterations")
print("means:", mu.round(2), "  (true: -2.0, 1.5)")
print("stds:", sd.round(2), "   (true: 0.6, 1.0)")
print("weights:", pi.round(2), " (true: 0.6, 0.4)")

# Soft assignment: a point in the overlap gets a probability, not a verdict
x0 = -0.3
d = np.array([pi[k] * normal_pdf(x0, mu[k], sd[k]) for k in range(2)])
d /= d.sum()
print("p(component | x = %.1f):" % x0, d.round(2), " <- no hard border")

# Generative: sample new points from the fitted mixture
comp = np.random.choice(2, size=5, p=pi)
print("5 invented points:", np.random.normal(mu[comp], sd[comp]).round(2))


# =============================================================================
# PART 8: DBSCAN AND t-SNE
# =============================================================================

print("\n=== Part 8: DBSCAN and t-SNE ===\n")

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.manifold import TSNE

# DBSCAN on two interleaved moons - a shape k-means cannot cut correctly
X_moons, _ = make_moons(n_samples=300, noise=0.07, random_state=3)
db = DBSCAN(eps=0.2, min_samples=5).fit(X_moons)
n_found = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print("DBSCAN found", n_found, "clusters and", int((db.labels_ == -1).sum()), "noise points")
print("k was never chosen - density decided.\n")

# t-SNE: 10-D blobs down to 2-D; neighborhoods survive the trip
X10, y10 = make_blobs(n_samples=300, n_features=10, centers=4, random_state=5)
emb = TSNE(n_components=2, random_state=5, perplexity=30).fit_transform(X10)
D = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
same = y10[:, None] == y10[None, :]
print("mean 2-D distance within a true cluster: %.1f" % D[same].mean())
print("mean 2-D distance between clusters:      %.1f" % D[~same].mean())
print("neighborhoods preserved - but the axes mean nothing. Look, do not measure.")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 8")
print("=" * 60)
print("""
1. UNSUPERVISED LEARNING: No labels, find structure
   - Clustering: group similar points
   - Dimensionality reduction: find compact representations

2. K-MEANS:
   - Initialize centroids, assign points, update centroids, repeat
   - Must choose k (use elbow or silhouette)
   - Assumes spherical clusters

3. HIERARCHICAL CLUSTERING:
   - Builds dendrogram (tree of merges)
   - No need to specify k upfront
   - Cut tree at desired level

4. PCA:
   - Find directions of maximum variance
   - Eigenvectors of covariance matrix
   - Explained variance shows how much info kept

5. EVALUATION:
   - Silhouette score: compactness vs separation
   - Elbow method: diminishing returns on k
   - Always scale data first!
""")

print("""
6. GAUSSIAN MIXTURES + EM:
   - k overlapping normals; every point gets a probability per component
   - E-step soft-assigns, M-step re-estimates; k-means minus the certainty
   - A fitted mixture is generative: it can sample new points

7. DBSCAN AND t-SNE:
   - DBSCAN: clusters where points are dense, noise elsewhere; k falls out
   - t-SNE: neighborhood-faithful 2-D pictures - for looking, not measuring
""")
