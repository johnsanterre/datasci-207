"""
DATASCI 207: Applied Machine Learning
Module 2 - Exercise 1: Implement Gradient Descent

OBJECTIVE:
Implement gradient descent for linear regression from scratch.

INSTRUCTIONS:
1. Complete the functions marked with TODO
2. Run the file to check your answers
3. Experiment with different learning rates
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# SETUP: Generate synthetic data
# =============================================================================

# True relationship: y = 2*x + 5 + noise
n_samples = 50
X = np.random.uniform(0, 10, size=(n_samples, 1))
y_true = 2 * X + 5 + np.random.normal(0, 0.5, size=(n_samples, 1))


# =============================================================================
# EXERCISE 1: Implement the prediction function
# =============================================================================

def predict(X, weight, bias):
    """
    Compute predictions for linear regression.
    
    y_pred = X * weight + bias
    
    Args:
        X: Input features, shape (n_samples, 1)
        weight: The weight parameter (scalar)
        bias: The bias parameter (scalar)
    
    Returns:
        Predictions, shape (n_samples, 1)
    """
    # TODO: Implement the prediction formula
    y_pred = None  # Replace with your code
    
    return y_pred


# =============================================================================
# EXERCISE 2: Implement Mean Squared Error
# =============================================================================

def compute_loss(y_pred, y_true):
    """
    Compute Mean Squared Error loss.
    
    MSE = (1/n) * sum((y_pred - y_true)^2)
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        MSE loss (scalar)
    """
    # TODO: Implement MSE
    # Hint: Use np.mean() and squaring
    loss = None  # Replace with your code
    
    return loss


# =============================================================================
# EXERCISE 3: Implement gradient computation
# =============================================================================

def compute_gradients(X, y_true, weight, bias):
    """
    Compute gradients of MSE with respect to weight and bias.
    
    For MSE loss, the gradients are:
        dL/dw = (2/n) * sum((y_pred - y_true) * X)
        dL/db = (2/n) * sum(y_pred - y_true)
    
    Args:
        X: Input features
        y_true: True values
        weight: Current weight
        bias: Current bias
    
    Returns:
        grad_weight, grad_bias (both scalars)
    """
    n = len(X)
    y_pred = predict(X, weight, bias)
    error = y_pred - y_true
    
    # TODO: Compute the gradients
    # Hint: Use np.sum()
    grad_weight = None  # Replace with your code
    grad_bias = None    # Replace with your code
    
    return grad_weight, grad_bias


# =============================================================================
# EXERCISE 4: Implement gradient descent training loop
# =============================================================================

def train_linear_regression(X, y_true, learning_rate=0.01, n_iterations=500):
    """
    Train linear regression using gradient descent.
    
    Args:
        X: Input features
        y_true: True values
        learning_rate: Step size for updates
        n_iterations: Number of training steps
    
    Returns:
        weight, bias, loss_history
    """
    # Initialize parameters to zero
    weight = 0.0
    bias = 0.0
    loss_history = []
    
    for i in range(n_iterations):
        # TODO: Complete the training loop
        # Step 1: Compute predictions (use predict function)
        y_pred = None  # Replace with your code
        
        # Step 2: Compute loss (use compute_loss function)
        loss = None  # Replace with your code
        loss_history.append(loss)
        
        # Step 3: Compute gradients (use compute_gradients function)
        grad_w, grad_b = None, None  # Replace with your code
        
        # Step 4: Update parameters using gradient descent rule
        # weight = weight - learning_rate * grad_weight
        weight = None  # Replace with your code
        bias = None    # Replace with your code
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: loss = {loss:.4f}, weight = {weight:.4f}, bias = {bias:.4f}")
    
    return weight, bias, loss_history


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

if __name__ == "__main__":
    print("Testing your gradient descent implementation...\n")
    print("True parameters: weight = 2.0, bias = 5.0\n")
    
    # Test prediction function
    print("=" * 50)
    print("Test 1: Prediction function")
    print("=" * 50)
    test_pred = predict(np.array([[1.0]]), 2.0, 3.0)
    expected_pred = 5.0  # 2*1 + 3 = 5
    if test_pred is not None and abs(test_pred[0,0] - expected_pred) < 0.001:
        print(f"predict([[1.0]], 2.0, 3.0) = {test_pred[0,0]} (expected 5.0) - PASS")
    else:
        print(f"predict([[1.0]], 2.0, 3.0) = {test_pred} (expected 5.0) - Check implementation")
    
    # Test loss function
    print("\n" + "=" * 50)
    print("Test 2: Loss function")
    print("=" * 50)
    test_loss = compute_loss(np.array([[3.0], [5.0]]), np.array([[2.0], [4.0]]))
    expected_loss = 1.0  # ((3-2)^2 + (5-4)^2) / 2 = (1 + 1) / 2 = 1
    if test_loss is not None and abs(test_loss - expected_loss) < 0.001:
        print(f"Loss = {test_loss} (expected 1.0) - PASS")
    else:
        print(f"Loss = {test_loss} (expected 1.0) - Check implementation")
    
    # Test gradients
    print("\n" + "=" * 50)
    print("Test 3: Gradient computation")
    print("=" * 50)
    test_X = np.array([[1.0], [2.0]])
    test_y = np.array([[3.0], [5.0]])
    gw, gb = compute_gradients(test_X, test_y, 1.0, 1.0)
    # With w=1, b=1: preds = [2, 3], errors = [-1, -2]
    # grad_w = (2/2) * ((-1)*1 + (-2)*2) = -5
    # grad_b = (2/2) * (-1 + -2) = -3
    if gw is not None and abs(gw - (-5.0)) < 0.001 and abs(gb - (-3.0)) < 0.001:
        print(f"grad_weight = {gw} (expected -5.0) - PASS")
        print(f"grad_bias = {gb} (expected -3.0) - PASS")
    else:
        print(f"grad_weight = {gw} (expected -5.0)")
        print(f"grad_bias = {gb} (expected -3.0)")
        print("Check implementation")
    
    # Test full training
    print("\n" + "=" * 50)
    print("Test 4: Full gradient descent training")
    print("=" * 50)
    learned_w, learned_b, losses = train_linear_regression(X, y_true, learning_rate=0.01, n_iterations=500)
    
    if learned_w is not None:
        print(f"\nFinal: weight = {learned_w:.4f} (true: 2.0), bias = {learned_b:.4f} (true: 5.0)")
        if abs(learned_w - 2.0) < 0.2 and abs(learned_b - 5.0) < 0.5:
            print("Training converged close to true parameters - PASS")
        else:
            print("Parameters not close enough - check your update rule")
