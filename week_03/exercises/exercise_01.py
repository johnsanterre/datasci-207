"""
DATASCI 207: Module 3 - Exercise: Feature Engineering Practice

Complete the TODO sections to practice feature engineering techniques.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Implement Z-Score Standardization
# =============================================================================

def z_score_standardize(X):
    """
    Standardize features to have mean=0 and std=1.
    
    Formula: X_std = (X - mean) / std
    
    Args:
        X: numpy array of shape (n_samples,) or (n_samples, n_features)
    
    Returns:
        Standardized array of same shape
    """
    # TODO: Calculate mean and std, then standardize
    # Hint: Use np.mean() and np.std()
    X_standardized = None  # Replace
    
    return X_standardized


# =============================================================================
# EXERCISE 2: Implement One-Hot Encoding
# =============================================================================

def one_hot_encode(categories):
    """
    Convert categorical list to one-hot encoded numpy array.
    
    Example:
        ["a", "b", "a"] -> [[1, 0], [0, 1], [1, 0]]
    
    Args:
        categories: list of category strings
    
    Returns:
        encoded: numpy array of shape (n_samples, n_unique_categories)
        unique_cats: list of unique category names
    """
    # TODO: Implement one-hot encoding
    # Step 1: Find unique categories (sorted)
    # Step 2: Create a mapping from category to index
    # Step 3: Create the one-hot matrix
    
    unique_cats = None  # Replace
    encoded = None  # Replace
    
    return encoded, unique_cats


# =============================================================================
# EXERCISE 3: Implement Missing Value Imputation
# =============================================================================

def impute_with_mean(X):
    """
    Replace NaN values with the mean of non-NaN values.
    
    Args:
        X: numpy array that may contain np.nan values
    
    Returns:
        Array with NaN replaced by mean
    """
    # TODO: Calculate mean of non-NaN values and fill NaNs
    # Hint: np.nanmean() ignores NaN, np.isnan() finds NaN locations
    
    X_imputed = None  # Replace
    
    return X_imputed


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Feature Engineering Implementations\n")
    
    # Test 1: Z-Score
    print("=" * 50)
    print("Test 1: Z-Score Standardization")
    print("=" * 50)
    data = np.array([10, 20, 30, 40, 50])
    result = z_score_standardize(data)
    # mean=30, std=14.14, so 10 -> (10-30)/14.14 = -1.41
    expected_mean = 0
    expected_std = 1
    if result is not None:
        print(f"Input: {data}")
        print(f"Output: {result}")
        print(f"Mean: {np.mean(result):.4f} (expected ~0)")
        print(f"Std: {np.std(result):.4f} (expected ~1)")
        if abs(np.mean(result)) < 0.01 and abs(np.std(result) - 1) < 0.01:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: One-Hot
    print("\n" + "=" * 50)
    print("Test 2: One-Hot Encoding")
    print("=" * 50)
    categories = ["cat", "dog", "cat", "bird"]
    encoded, unique = one_hot_encode(categories)
    if encoded is not None:
        print(f"Input: {categories}")
        print(f"Unique categories: {unique}")
        print(f"Encoded:\n{encoded}")
        # Expected: bird=0, cat=1, dog=2
        # ["cat", "dog", "cat", "bird"] -> [[0,1,0], [0,0,1], [0,1,0], [1,0,0]]
        if encoded.shape == (4, 3):
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 3: Imputation
    print("\n" + "=" * 50)
    print("Test 3: Mean Imputation")
    print("=" * 50)
    data_missing = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    imputed = impute_with_mean(data_missing)
    # Mean of [1, 2, 4] = 7/3 = 2.33
    if imputed is not None:
        print(f"Input: {data_missing}")
        print(f"Output: {imputed}")
        expected_fill = np.nanmean(data_missing)
        if abs(imputed[2] - expected_fill) < 0.01:
            print("PASS")
    else:
        print("Not implemented yet")
