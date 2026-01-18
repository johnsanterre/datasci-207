"""
DATASCI 207: Applied Machine Learning
Module 2 - Exercise 1: SOLUTIONS
"""

import numpy as np
np.random.seed(42)

# Setup
n_samples = 50
X = np.random.uniform(0, 10, size=(n_samples, 1))
y_true = 2 * X + 5 + np.random.normal(0, 0.5, size=(n_samples, 1))


def predict(X, weight, bias):
    """Compute predictions for linear regression."""
    y_pred = X * weight + bias
    return y_pred


def compute_loss(y_pred, y_true):
    """Compute Mean Squared Error loss."""
    loss = np.mean((y_pred - y_true) ** 2)
    return loss


def compute_gradients(X, y_true, weight, bias):
    """Compute gradients of MSE with respect to weight and bias."""
    n = len(X)
    y_pred = predict(X, weight, bias)
    error = y_pred - y_true
    
    grad_weight = (2 / n) * np.sum(error * X)
    grad_bias = (2 / n) * np.sum(error)
    
    return grad_weight, grad_bias


def train_linear_regression(X, y_true, learning_rate=0.01, n_iterations=500):
    """Train linear regression using gradient descent."""
    weight = 0.0
    bias = 0.0
    loss_history = []
    
    for i in range(n_iterations):
        # Step 1: Compute predictions
        y_pred = predict(X, weight, bias)
        
        # Step 2: Compute loss
        loss = compute_loss(y_pred, y_true)
        loss_history.append(loss)
        
        # Step 3: Compute gradients
        grad_w, grad_b = compute_gradients(X, y_true, weight, bias)
        
        # Step 4: Update parameters
        weight = weight - learning_rate * grad_w
        bias = bias - learning_rate * grad_b
        
        if i % 100 == 0:
            print(f"Iteration {i}: loss = {loss:.4f}, weight = {weight:.4f}, bias = {bias:.4f}")
    
    return weight, bias, loss_history


if __name__ == "__main__":
    print("Solution verification:\n")
    learned_w, learned_b, losses = train_linear_regression(X, y_true, 0.01, 500)
    print(f"\nFinal: weight = {learned_w:.4f}, bias = {learned_b:.4f}")
    print(f"True:  weight = 2.0, bias = 5.0")
