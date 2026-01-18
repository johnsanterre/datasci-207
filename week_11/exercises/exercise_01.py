"""
DATASCI 207: Module 11 - Exercise: Regularization and CV
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Implement K-Fold Split
# =============================================================================

def k_fold_split(n_samples, k=5, shuffle=True):
    """
    Generate k-fold train/validation indices.
    
    Args:
        n_samples: Total number of samples
        k: Number of folds
        shuffle: Whether to shuffle before splitting
    
    Yields:
        (train_indices, val_indices) for each fold
    """
    # TODO: Implement k-fold splitting
    # 1. Create array of indices
    # 2. Optionally shuffle
    # 3. Split into k folds
    # 4. Yield train/val indices for each fold
    
    pass  # Replace with implementation


# =============================================================================
# EXERCISE 2: Cross-Validation Score
# =============================================================================

def cross_val_mse(X, y, model_fn, k=5):
    """
    Compute k-fold cross-validation MSE.
    
    Args:
        X: Features
        y: Targets
        model_fn: Function that returns (fit, predict) functions
        k: Number of folds
    
    Returns:
        Array of k MSE scores
    """
    # TODO: Implement cross-validation
    # 1. For each fold:
    #    - Split data
    #    - Fit model on train
    #    - Predict on val
    #    - Compute MSE
    # 2. Return array of scores
    
    scores = None  # Replace
    return scores


# =============================================================================
# EXERCISE 3: L2 Regularized Linear Regression
# =============================================================================

def ridge_closed_form(X, y, lambda_):
    """
    Ridge regression closed-form solution.
    
    w = (X'X + λI)^(-1) X'y
    
    Args:
        X: Features with bias column
        y: Targets
        lambda_: Regularization strength
    
    Returns:
        weights
    """
    # TODO: Implement ridge regression
    weights = None  # Replace
    return weights


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Regularization and CV\n")
    
    # Test 1: K-Fold Split
    print("=" * 50)
    print("Test 1: K-Fold Split")
    print("=" * 50)
    
    folds = list(k_fold_split(20, k=5))
    if folds:
        print(f"Generated {len(folds)} folds")
        for i, (train, val) in enumerate(folds):
            print(f"  Fold {i+1}: train={len(train)}, val={len(val)}")
        if len(folds) == 5 and all(len(v) == 4 for _, v in folds):
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: Ridge Regression
    print("\n" + "=" * 50)
    print("Test 2: Ridge Regression")
    print("=" * 50)
    
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    y = np.array([2, 4, 5, 4])
    
    w = ridge_closed_form(X, y, lambda_=1.0)
    if w is not None:
        print(f"Weights: {w}")
        y_pred = X @ w
        mse = np.mean((y - y_pred)**2)
        print(f"MSE: {mse:.4f}")
        if len(w) == 2:
            print("PASS")
    else:
        print("Not implemented yet")
