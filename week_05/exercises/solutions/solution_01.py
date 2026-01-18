"""
DATASCI 207: Module 5 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def softmax(z):
    """Compute softmax probabilities."""
    z_stable = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def precision_recall(y_true, y_pred):
    """Compute precision and recall."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall


def f1_score(precision, recall):
    """Compute F1 score."""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


if __name__ == "__main__":
    # Verify softmax
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    assert abs(np.sum(probs) - 1.0) < 0.001
    print("Softmax: VERIFIED")
    
    # Verify precision/recall
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    p, r = precision_recall(y_true, y_pred)
    assert abs(p - 0.75) < 0.01
    assert abs(r - 0.60) < 0.01
    print("Precision/Recall: VERIFIED")
    
    # Verify F1
    f1 = f1_score(0.75, 0.60)
    assert abs(f1 - 0.667) < 0.01
    print("F1 Score: VERIFIED")
