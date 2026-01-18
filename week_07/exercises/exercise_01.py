"""
DATASCI 207: Module 7 - Exercise: Tree-Based Models
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Implement Entropy
# =============================================================================

def entropy(y):
    """
    Compute entropy: H(S) = -sum(p * log2(p))
    
    Args:
        y: Array of class labels
    
    Returns:
        Entropy value (float)
    
    Handle empty arrays and avoid log(0).
    """
    # TODO: Implement entropy
    result = None  # Replace
    return result


# =============================================================================
# EXERCISE 2: Implement Information Gain
# =============================================================================

def information_gain(y_parent, y_left, y_right):
    """
    Compute information gain from a split.
    
    IG = H(parent) - (n_left/n * H(left) + n_right/n * H(right))
    
    Args:
        y_parent: Labels before split
        y_left: Labels in left child
        y_right: Labels in right child
    
    Returns:
        Information gain (float)
    """
    # TODO: Implement information gain
    # Use your entropy function
    result = None  # Replace
    return result


# =============================================================================
# EXERCISE 3: Find Best Threshold
# =============================================================================

def find_best_threshold(feature, y):
    """
    Find the threshold that maximizes information gain for a single feature.
    
    Args:
        feature: Array of feature values
        y: Array of labels
    
    Returns:
        (best_threshold, best_gain)
    """
    # TODO: Try all unique thresholds and find the one with max IG
    best_threshold = None
    best_gain = 0
    
    # Your code here
    
    return best_threshold, best_gain


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Tree Concepts\n")
    
    # Test 1: Entropy
    print("=" * 50)
    print("Test 1: Entropy")
    print("=" * 50)
    
    pure = np.array([0, 0, 0, 0])
    mixed = np.array([0, 0, 1, 1])
    
    if entropy(pure) is not None:
        print(f"Pure [0,0,0,0]: {entropy(pure):.4f} (expected: 0)")
        print(f"50-50 [0,0,1,1]: {entropy(mixed):.4f} (expected: 1)")
        if abs(entropy(pure)) < 0.001 and abs(entropy(mixed) - 1) < 0.001:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: Information Gain
    print("\n" + "=" * 50)
    print("Test 2: Information Gain")
    print("=" * 50)
    
    y_before = np.array([0, 0, 0, 1, 1, 1])
    y_left = np.array([0, 0, 0])  # Pure class 0
    y_right = np.array([1, 1, 1])  # Pure class 1
    
    ig = information_gain(y_before, y_left, y_right)
    if ig is not None:
        print(f"Perfect split:")
        print(f"  Before: {y_before}")
        print(f"  Left: {y_left}, Right: {y_right}")
        print(f"  IG: {ig:.4f} (expected: 1.0)")
        if abs(ig - 1.0) < 0.001:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 3: Best Threshold
    print("\n" + "=" * 50)
    print("Test 3: Find Best Threshold")
    print("=" * 50)
    
    feature = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    thresh, gain = find_best_threshold(feature, y)
    if thresh is not None:
        print(f"Feature: {feature}")
        print(f"Labels: {y}")
        print(f"Best threshold: {thresh} (expected: 3 or 3.5)")
        print(f"Info gain: {gain:.4f} (expected: 1.0)")
        if gain > 0.9:
            print("PASS")
    else:
        print("Not implemented yet")
