"""
DATASCI 207: Module 4 - Exercise: Logistic Regression

Implement the core components of logistic regression.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Implement the Sigmoid Function
# =============================================================================

def sigmoid(z):
    """
    Compute the sigmoid function.
    
    sigmoid(z) = 1 / (1 + exp(-z))
    
    Args:
        z: Input value(s) - scalar or numpy array
    
    Returns:
        Sigmoid of z, same shape as input
    """
    # TODO: Implement sigmoid
    result = None  # Replace
    
    return result


# =============================================================================
# EXERCISE 2: Implement Logistic Loss
# =============================================================================

def logistic_loss(y_true, y_pred_proba):
    """
    Compute binary cross-entropy loss.
    
    L = -(1/n) * sum[y*log(p) + (1-y)*log(1-p)]
    
    Args:
        y_true: True labels (0 or 1), shape (n,)
        y_pred_proba: Predicted probabilities, shape (n,)
    
    Returns:
        Average loss (scalar)
    """
    # TODO: Implement logistic loss
    # Hint: Clip probabilities to avoid log(0)
    #       Use np.clip(p, 1e-15, 1 - 1e-15)
    
    loss = None  # Replace
    
    return loss


# =============================================================================
# EXERCISE 3: Compute Accuracy
# =============================================================================

def compute_accuracy(y_true, y_pred_proba, threshold=0.5):
    """
    Compute classification accuracy.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Accuracy as a float
    """
    # TODO: Convert probabilities to predictions and compute accuracy
    # Step 1: predictions = 1 if p >= threshold else 0
    # Step 2: accuracy = fraction of correct predictions
    
    accuracy = None  # Replace
    
    return accuracy


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Logistic Regression Implementations\n")
    
    # Test 1: Sigmoid
    print("=" * 50)
    print("Test 1: Sigmoid Function")
    print("=" * 50)
    
    if sigmoid(0) is not None:
        print(f"sigmoid(0) = {sigmoid(0):.4f} (expected: 0.5)")
        print(f"sigmoid(10) = {sigmoid(10):.6f} (expected: ~1)")
        print(f"sigmoid(-10) = {sigmoid(-10):.6f} (expected: ~0)")
        
        if abs(sigmoid(0) - 0.5) < 0.001:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: Logistic Loss
    print("\n" + "=" * 50)
    print("Test 2: Logistic Loss")
    print("=" * 50)
    
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.3])
    loss = logistic_loss(y, p)
    
    if loss is not None:
        print(f"y = {y}")
        print(f"p = {p}")
        print(f"Loss = {loss:.4f} (expected: ~0.20)")
        
        # Also test extreme case
        loss_bad = logistic_loss(np.array([1]), np.array([0.1]))
        loss_good = logistic_loss(np.array([1]), np.array([0.9]))
        print(f"\nBad prediction (y=1, p=0.1): {loss_bad:.4f}")
        print(f"Good prediction (y=1, p=0.9): {loss_good:.4f}")
        
        if loss_bad > loss_good:
            print("Bad predictions have higher loss - PASS")
    else:
        print("Not implemented yet")
    
    # Test 3: Accuracy
    print("\n" + "=" * 50)
    print("Test 3: Accuracy")
    print("=" * 50)
    
    y_true = np.array([1, 0, 1, 0, 1])
    y_probs = np.array([0.9, 0.2, 0.8, 0.4, 0.3])
    acc = compute_accuracy(y_true, y_probs)
    
    if acc is not None:
        print(f"True labels: {y_true}")
        print(f"Probabilities: {y_probs}")
        print(f"Predictions (threshold=0.5): {(y_probs >= 0.5).astype(int)}")
        print(f"Accuracy = {acc:.0%} (expected: 80%)")
        
        if abs(acc - 0.8) < 0.001:
            print("PASS")
    else:
        print("Not implemented yet")
