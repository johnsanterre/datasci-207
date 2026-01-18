"""
DATASCI 207: Applied Machine Learning
Module 7: KNN, Decision Trees, and Ensembles

This module covers:
- K-Nearest Neighbors
- Decision Trees (entropy, information gain)
- Random Forests
- Gradient Boosting

Using NumPy and scikit-learn - no pandas.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: K-NEAREST NEIGHBORS
# =============================================================================

print("=== Part 1: K-Nearest Neighbors ===\n")


def euclidean_distance(x1, x2):
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_predict(X_train, y_train, x_query, k=3):
    """
    Predict class for x_query using k-nearest neighbors.
    
    Args:
        X_train: Training features
        y_train: Training labels
        x_query: Single query point
        k: Number of neighbors
    
    Returns:
        Predicted class (majority vote)
    """
    # Compute distances to all training points
    distances = [euclidean_distance(x_query, x) for x in X_train]
    
    # Find k nearest neighbors
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    
    # Majority vote
    unique, counts = np.unique(k_labels, return_counts=True)
    return unique[np.argmax(counts)]


# Example data
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

x_new = np.array([5, 5])
pred = knn_predict(X_train, y_train, x_new, k=3)

print(f"Training data:\n{X_train}")
print(f"Labels: {y_train}")
print(f"\nQuery point: {x_new}")
print(f"Prediction (k=3): class {pred}")


# =============================================================================
# PART 2: ENTROPY AND INFORMATION GAIN
# =============================================================================

print("\n=== Part 2: Entropy and Information Gain ===\n")


def entropy(y):
    """
    Compute entropy: H(S) = -sum(p * log2(p))
    
    Measures disorder/impurity. 0 = pure, 1 = max (for binary).
    """
    if len(y) == 0:
        return 0
    
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    
    # Avoid log(0)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


# Demonstrate entropy
pure = np.array([0, 0, 0, 0])        # All same class
mixed = np.array([0, 0, 1, 1])        # 50-50 split
mostly_a = np.array([0, 0, 0, 1])     # 75-25 split

print("Entropy examples (binary):")
print(f"  Pure [0,0,0,0]: {entropy(pure):.4f}")
print(f"  50-50 [0,0,1,1]: {entropy(mixed):.4f}")
print(f"  75-25 [0,0,0,1]: {entropy(mostly_a):.4f}")


def information_gain(y_parent, y_left, y_right):
    """
    Information gain from a split.
    
    IG = H(parent) - weighted average of H(children)
    """
    n = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)
    
    weighted_child_entropy = (n_left/n) * entropy(y_left) + (n_right/n) * entropy(y_right)
    
    return entropy(y_parent) - weighted_child_entropy


# Example split
y_before = np.array([0, 0, 0, 1, 1, 1])
y_left = np.array([0, 0, 0])
y_right = np.array([1, 1, 1])

ig = information_gain(y_before, y_left, y_right)
print(f"\nInformation Gain example:")
print(f"  Before split: {y_before}")
print(f"  Left:  {y_left}")
print(f"  Right: {y_right}")
print(f"  Information Gain: {ig:.4f} (perfect split!)")


# =============================================================================
# PART 3: DECISION TREE FROM SCRATCH (SIMPLIFIED)
# =============================================================================

print("\n=== Part 3: Simple Decision Tree ===\n")


def find_best_split(X, y):
    """Find the best feature and threshold for splitting."""
    best_gain = 0
    best_feature = None
    best_threshold = None
    
    n_features = X.shape[1]
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            gain = information_gain(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain


# Simple 2D data
X_tree = np.array([
    [1, 1], [1, 2], [2, 1],  # Class 0
    [4, 4], [4, 5], [5, 4]   # Class 1
])
y_tree = np.array([0, 0, 0, 1, 1, 1])

feature, threshold, gain = find_best_split(X_tree, y_tree)
print(f"Data:\n{X_tree}")
print(f"Labels: {y_tree}")
print(f"\nBest split: feature {feature} <= {threshold}")
print(f"Information gain: {gain:.4f}")


# =============================================================================
# PART 4: SCIKIT-LEARN DECISION TREES
# =============================================================================

print("\n=== Part 4: scikit-learn Decision Trees ===\n")

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=4, n_informative=2,
                                n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train decision tree
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    
    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)
    
    print(f"Decision Tree (max_depth=3):")
    print(f"  Training accuracy: {train_acc:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")
    
    # Feature importance
    print(f"\nFeature importances:")
    for i, imp in enumerate(tree.feature_importances_):
        print(f"  Feature {i}: {imp:.4f}")

except ImportError:
    print("(scikit-learn not available)")


# =============================================================================
# PART 5: RANDOM FORESTS
# =============================================================================

print("\n=== Part 5: Random Forests ===\n")

print("""
Random Forests combine multiple decision trees:
1. Bootstrap sample for each tree
2. Random feature subset at each split
3. Average predictions (regression) or vote (classification)

This decorrelates trees, reducing variance.
""")

try:
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    rf_train_acc = rf.score(X_train, y_train)
    rf_test_acc = rf.score(X_test, y_test)
    
    print(f"Random Forest (100 trees, max_depth=5):")
    print(f"  Training accuracy: {rf_train_acc:.2%}")
    print(f"  Test accuracy: {rf_test_acc:.2%}")
    
    print(f"\nFeature importances (averaged over trees):")
    for i, imp in enumerate(rf.feature_importances_):
        print(f"  Feature {i}: {imp:.4f}")

except ImportError:
    print("(scikit-learn not available)")


# =============================================================================
# PART 6: GRADIENT BOOSTING
# =============================================================================

print("\n=== Part 6: Gradient Boosting ===\n")

print("""
Gradient Boosting builds trees sequentially:
1. Fit tree to data
2. Compute residuals (errors)
3. Fit next tree to residuals
4. Add new tree's predictions (with shrinkage)
5. Repeat

Each tree corrects errors of the previous ensemble.
""")

try:
    from sklearn.ensemble import GradientBoostingClassifier
    
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, 
                                     learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    
    gb_train_acc = gb.score(X_train, y_train)
    gb_test_acc = gb.score(X_test, y_test)
    
    print(f"Gradient Boosting (100 iterations):")
    print(f"  Training accuracy: {gb_train_acc:.2%}")
    print(f"  Test accuracy: {gb_test_acc:.2%}")

except ImportError:
    print("(scikit-learn not available)")


# =============================================================================
# PART 7: COMPARING METHODS
# =============================================================================

print("\n=== Part 7: Method Comparison ===\n")

try:
    print("Performance comparison on test set:")
    print(f"  Decision Tree:     {test_acc:.2%}")
    print(f"  Random Forest:     {rf_test_acc:.2%}")
    print(f"  Gradient Boosting: {gb_test_acc:.2%}")
    
    print("""
Typical patterns:
- Single tree: Fast, interpretable, may overfit
- Random Forest: More accurate, handles overfitting well
- Gradient Boosting: Often best accuracy, needs tuning
""")

except:
    pass


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 7")
print("=" * 60)
print("""
1. KNN: Predict by finding k nearest neighbors
   - Simple but slow (no training, slow prediction)
   - Requires feature scaling

2. DECISION TREES: Partition space with if-then rules
   - Entropy measures impurity
   - Information gain guides splits
   - Prone to overfitting (control with max_depth)

3. RANDOM FORESTS: Ensemble of decorrelated trees
   - Bootstrap samples + random features
   - Reduces variance, robust

4. GRADIENT BOOSTING: Sequential error correction
   - Each tree fits residuals of previous
   - Often best performance on tabular data
   - Key hyperparameters: n_estimators, learning_rate, max_depth

5. FEATURE IMPORTANCE: Trees naturally rank features
   - Sum of information gains
   - Helps interpretability
""")
