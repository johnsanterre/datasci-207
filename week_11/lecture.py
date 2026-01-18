"""
DATASCI 207: Applied Machine Learning
Module 11: Regularization and Hyperparameter Tuning

This module covers:
- L1 and L2 regularization
- Dropout and early stopping
- Cross-validation
- Grid search and random search
- Pipelines

Using NumPy and scikit-learn.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: REGULARIZED LOSS FUNCTIONS
# =============================================================================

print("=== Part 1: Regularized Loss Functions ===\n")


def mse_loss(y_true, y_pred):
    """Mean Squared Error loss."""
    return np.mean((y_true - y_pred) ** 2)


def l2_penalty(weights, lambda_):
    """L2 regularization penalty: λ * Σ(w²)"""
    return lambda_ * np.sum(weights ** 2)


def l1_penalty(weights, lambda_):
    """L1 regularization penalty: λ * Σ|w|"""
    return lambda_ * np.sum(np.abs(weights))


# Example
weights = np.array([0.5, -2.0, 0.1, 3.0])
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 1.8, 3.2])

base_loss = mse_loss(y_true, y_pred)
l2_reg = l2_penalty(weights, lambda_=0.1)
l1_reg = l1_penalty(weights, lambda_=0.1)

print(f"Weights: {weights}")
print(f"Base MSE Loss: {base_loss:.4f}")
print(f"L2 Penalty (λ=0.1): {l2_reg:.4f}")
print(f"L1 Penalty (λ=0.1): {l1_reg:.4f}")
print(f"Total L2 Loss: {base_loss + l2_reg:.4f}")
print(f"Total L1 Loss: {base_loss + l1_reg:.4f}")


# =============================================================================
# PART 2: L1 VS L2 - SPARSITY
# =============================================================================

print("\n=== Part 2: L1 vs L2 Sparsity ===\n")

print("""
L1 (Lasso) encourages sparsity - pushes weights to exactly zero.
L2 (Ridge) shrinks weights but rarely to exactly zero.

Example with scikit-learn:
""")

try:
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.datasets import make_regression
    
    # Generate data with many irrelevant features
    X, y = make_regression(n_samples=100, n_features=20, 
                           n_informative=5, noise=10, random_state=42)
    
    # L2 (Ridge)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    
    # L1 (Lasso)
    lasso = Lasso(alpha=1.0)
    lasso.fit(X, y)
    
    print(f"Ridge (L2) - Non-zero weights: {np.sum(ridge.coef_ != 0)}")
    print(f"Lasso (L1) - Non-zero weights: {np.sum(lasso.coef_ != 0)}")
    
    print("\nLasso performs automatic feature selection!")

except ImportError:
    print("(scikit-learn not available)")


# =============================================================================
# PART 3: CROSS-VALIDATION
# =============================================================================

print("\n=== Part 3: Cross-Validation ===\n")


def k_fold_split(n_samples, k=5):
    """Generate k-fold indices."""
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_size = n_samples // k
    
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k-1 else n_samples
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        yield train_indices, val_indices


# Demonstrate folds
print("5-Fold Cross-Validation on 20 samples:")
for fold, (train_idx, val_idx) in enumerate(k_fold_split(20, k=5)):
    print(f"  Fold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}")

try:
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X, y, cv=5)
    
    print(f"\nscikit-learn 5-fold CV scores: {np.round(scores, 3)}")
    print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

except ImportError:
    pass


# =============================================================================
# PART 4: GRID SEARCH
# =============================================================================

print("\n=== Part 4: Grid Search ===\n")

try:
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    
    grid = GridSearchCV(SVC(), param_grid, cv=3, verbose=0)
    grid.fit(X, y)
    
    print("Grid Search Results:")
    print(f"  Best parameters: {grid.best_params_}")
    print(f"  Best CV score: {grid.best_score_:.3f}")
    
    print(f"\n  Total combinations tried: {len(grid.cv_results_['params'])}")

except ImportError:
    print("(scikit-learn not available)")


# =============================================================================
# PART 5: RANDOM SEARCH
# =============================================================================

print("\n=== Part 5: Random Search ===\n")

try:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import loguniform, uniform
    
    param_dist = {
        'C': loguniform(0.01, 100),  # Log-uniform distribution
        'gamma': loguniform(0.001, 1)
    }
    
    random_search = RandomizedSearchCV(
        SVC(kernel='rbf'), param_dist, 
        n_iter=10, cv=3, random_state=42
    )
    random_search.fit(X, y)
    
    print("Random Search Results:")
    print(f"  Best parameters: {random_search.best_params_}")
    print(f"  Best CV score: {random_search.best_score_:.3f}")
    print(f"  Iterations: 10 (vs. potentially infinite grid)")

except ImportError:
    print("(scikit-learn or scipy not available)")


# =============================================================================
# PART 6: PIPELINES
# =============================================================================

print("\n=== Part 6: Pipelines ===\n")

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=5)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Cross-validate the entire pipeline
    scores = cross_val_score(pipeline, X, y, cv=5)
    
    print("Pipeline: Scaler -> PCA -> LogisticRegression")
    print(f"CV Scores: {np.round(scores, 3)}")
    print(f"Mean: {scores.mean():.3f}")
    
    print("\nBenefit: Preprocessing is done correctly inside each CV fold!")

except ImportError:
    print("(scikit-learn not available)")


# =============================================================================
# PART 7: EARLY STOPPING (CONCEPT)
# =============================================================================

print("\n=== Part 7: Early Stopping ===\n")

print("""
Early Stopping for Neural Networks:

```python
from tensorflow import keras

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',    # What to monitor
    patience=10,           # Epochs to wait
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=1000,           # Set high
    callbacks=[early_stop]  # Will stop early
)
```

Benefits:
- Automatic regularization
- Saves best model
- No need to tune epochs
""")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 11")
print("=" * 60)
print("""
1. REGULARIZATION prevents overfitting:
   - L2 (Ridge): shrinks weights, smooth
   - L1 (Lasso): sparse weights, feature selection
   - Dropout: random zeroing, ensemble effect
   - Early stopping: stop when validation worsens

2. CROSS-VALIDATION:
   - Split data into k folds
   - Train on k-1, validate on 1
   - Average scores for robust estimate

3. HYPERPARAMETER TUNING:
   - Grid Search: exhaustive, expensive
   - Random Search: efficient, continuous params
   - Both use cross-validation internally

4. PIPELINES:
   - Chain preprocessing + modeling
   - Ensures correct handling in CV
   - Use pipeline__param syntax in grid search
""")
