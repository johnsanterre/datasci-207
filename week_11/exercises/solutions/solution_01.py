"""
DATASCI 207: Module 11 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def k_fold_split(n_samples, k=5, shuffle=True):
    """Generate k-fold train/validation indices."""
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    fold_size = n_samples // k
    
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        yield train_indices, val_indices


def cross_val_mse(X, y, model_fn, k=5):
    """Compute k-fold cross-validation MSE."""
    scores = []
    for train_idx, val_idx in k_fold_split(len(X), k):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        fit, predict = model_fn()
        fit(X_train, y_train)
        y_pred = predict(X_val)
        
        mse = np.mean((y_val - y_pred) ** 2)
        scores.append(mse)
    
    return np.array(scores)


def ridge_closed_form(X, y, lambda_):
    """Ridge regression closed-form solution."""
    n_features = X.shape[1]
    I = np.eye(n_features)
    weights = np.linalg.solve(X.T @ X + lambda_ * I, X.T @ y)
    return weights


if __name__ == "__main__":
    # Test k-fold
    folds = list(k_fold_split(20, k=5))
    assert len(folds) == 5
    print("K-Fold Split: VERIFIED")
    
    # Test ridge
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    y = np.array([2, 4, 5, 4])
    w = ridge_closed_form(X, y, lambda_=1.0)
    assert len(w) == 2
    print("Ridge Regression: VERIFIED")
