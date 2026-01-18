"""
DATASCI 207: Module 4 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))


def logistic_loss(y_true, y_pred_proba):
    """Compute binary cross-entropy loss."""
    eps = 1e-15
    p = np.clip(y_pred_proba, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return loss


def compute_accuracy(y_true, y_pred_proba, threshold=0.5):
    """Compute classification accuracy."""
    predictions = (y_pred_proba >= threshold).astype(int)
    accuracy = np.mean(predictions == y_true)
    return accuracy


if __name__ == "__main__":
    # Verify sigmoid
    assert abs(sigmoid(0) - 0.5) < 0.001
    print("Sigmoid: VERIFIED")
    
    # Verify loss
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.3])
    loss = logistic_loss(y, p)
    assert loss < 0.5
    print("Logistic loss: VERIFIED")
    
    # Verify accuracy
    y_true = np.array([1, 0, 1, 0, 1])
    y_probs = np.array([0.9, 0.2, 0.8, 0.4, 0.3])
    acc = compute_accuracy(y_true, y_probs)
    assert abs(acc - 0.8) < 0.001
    print("Accuracy: VERIFIED")
