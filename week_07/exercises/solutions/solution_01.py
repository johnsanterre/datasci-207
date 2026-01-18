"""
DATASCI 207: Module 7 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def entropy(y):
    """Compute entropy."""
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def information_gain(y_parent, y_left, y_right):
    """Compute information gain from a split."""
    n = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)
    
    child_entropy = (n_left/n) * entropy(y_left) + (n_right/n) * entropy(y_right)
    return entropy(y_parent) - child_entropy


def find_best_threshold(feature, y):
    """Find threshold that maximizes information gain."""
    best_threshold = None
    best_gain = 0
    
    thresholds = np.unique(feature)
    
    for thresh in thresholds:
        left_mask = feature <= thresh
        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
            continue
        
        gain = information_gain(y, y[left_mask], y[~left_mask])
        if gain > best_gain:
            best_gain = gain
            best_threshold = thresh
    
    return best_threshold, best_gain


if __name__ == "__main__":
    # Verify
    assert abs(entropy(np.array([0, 0, 0, 0]))) < 0.001
    assert abs(entropy(np.array([0, 0, 1, 1])) - 1) < 0.001
    print("Entropy: VERIFIED")
    
    ig = information_gain(np.array([0,0,0,1,1,1]), np.array([0,0,0]), np.array([1,1,1]))
    assert abs(ig - 1.0) < 0.001
    print("Information Gain: VERIFIED")
    
    thresh, gain = find_best_threshold(np.array([1,2,3,4,5,6]), np.array([0,0,0,1,1,1]))
    assert gain > 0.9
    print("Best Threshold: VERIFIED")
