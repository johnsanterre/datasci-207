"""
DATASCI 207: Module 5 - Exercise: Classification Metrics

Implement key classification metrics from scratch.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Implement Softmax
# =============================================================================

def softmax(z):
    """
    Compute softmax probabilities.
    
    softmax(z)_i = exp(z_i) / sum(exp(z_j))
    
    Args:
        z: Logits, shape (n_classes,) or (n_samples, n_classes)
    
    Returns:
        Probabilities, same shape as z
    
    Hint: Subtract max for numerical stability before exp()
    """
    # TODO: Implement softmax
    probs = None  # Replace
    
    return probs


# =============================================================================
# EXERCISE 2: Implement Precision and Recall
# =============================================================================

def precision_recall(y_true, y_pred):
    """
    Compute precision and recall for binary classification.
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
    
    Returns:
        precision, recall (both floats)
    """
    # TODO: Count TP, FP, FN and compute metrics
    # TP: y_true=1 and y_pred=1
    # FP: y_true=0 and y_pred=1
    # FN: y_true=1 and y_pred=0
    
    precision = None  # Replace
    recall = None     # Replace
    
    return precision, recall


# =============================================================================
# EXERCISE 3: Implement F1 Score
# =============================================================================

def f1_score(precision, recall):
    """
    Compute F1 score as harmonic mean of precision and recall.
    
    F1 = 2 * (P * R) / (P + R)
    
    Handle the case where P + R = 0 by returning 0.
    """
    # TODO: Implement F1
    f1 = None  # Replace
    
    return f1


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Classification Metrics\n")
    
    # Test 1: Softmax
    print("=" * 50)
    print("Test 1: Softmax")
    print("=" * 50)
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    if probs is not None:
        print(f"Logits: {logits}")
        print(f"Softmax: {probs}")
        print(f"Sum: {np.sum(probs):.4f} (expected: 1.0)")
        if abs(np.sum(probs) - 1.0) < 0.001:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: Precision and Recall
    print("\n" + "=" * 50)
    print("Test 2: Precision and Recall")
    print("=" * 50)
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    # TP=3, FP=1, FN=2
    # Precision = 3/(3+1) = 0.75
    # Recall = 3/(3+2) = 0.60
    
    p, r = precision_recall(y_true, y_pred)
    if p is not None:
        print(f"True:       {y_true}")
        print(f"Predicted:  {y_pred}")
        print(f"Precision: {p:.2f} (expected: 0.75)")
        print(f"Recall: {r:.2f} (expected: 0.60)")
        if abs(p - 0.75) < 0.01 and abs(r - 0.60) < 0.01:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 3: F1 Score
    print("\n" + "=" * 50)
    print("Test 3: F1 Score")
    print("=" * 50)
    f1 = f1_score(0.75, 0.60)
    # F1 = 2 * 0.75 * 0.60 / (0.75 + 0.60) = 0.90 / 1.35 = 0.667
    if f1 is not None:
        print(f"F1(0.75, 0.60) = {f1:.3f} (expected: 0.667)")
        if abs(f1 - 0.667) < 0.01:
            print("PASS")
    else:
        print("Not implemented yet")
