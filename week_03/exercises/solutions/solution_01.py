"""
DATASCI 207: Module 3 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def z_score_standardize(X):
    """Standardize features to have mean=0 and std=1."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / (std + 1e-8)
    return X_standardized


def one_hot_encode(categories):
    """Convert categorical list to one-hot encoded numpy array."""
    unique_cats = sorted(set(categories))
    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
    
    n_samples = len(categories)
    n_cats = len(unique_cats)
    encoded = np.zeros((n_samples, n_cats))
    
    for i, cat in enumerate(categories):
        encoded[i, cat_to_idx[cat]] = 1
    
    return encoded, unique_cats


def impute_with_mean(X):
    """Replace NaN values with the mean of non-NaN values."""
    X_imputed = X.copy()
    mean_val = np.nanmean(X)
    X_imputed[np.isnan(X_imputed)] = mean_val
    return X_imputed


if __name__ == "__main__":
    # Verify solutions
    data = np.array([10, 20, 30, 40, 50])
    result = z_score_standardize(data)
    assert abs(np.mean(result)) < 0.01
    assert abs(np.std(result) - 1) < 0.01
    print("Z-score: VERIFIED")
    
    categories = ["cat", "dog", "cat", "bird"]
    encoded, unique = one_hot_encode(categories)
    assert encoded.shape == (4, 3)
    print("One-hot: VERIFIED")
    
    data_missing = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    imputed = impute_with_mean(data_missing)
    assert not np.any(np.isnan(imputed))
    print("Imputation: VERIFIED")
