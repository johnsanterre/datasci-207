"""
DATASCI 207: Applied Machine Learning
Module 2: Linear Regression and Gradient Descent

This module covers:
- Linear regression model
- Loss functions (MSE)
- Gradient descent optimization
- Batch vs Stochastic vs Mini-batch gradient descent
- Learning rate and convergence

We use only NumPy - no pandas, no sklearn for the core concepts.
"""

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# PART 1: LINEAR REGRESSION MODEL
# =============================================================================

# Linear regression predicts output as a weighted sum of inputs plus a bias:
#   y_pred = w1*x1 + w2*x2 + ... + wn*xn + b
# 
# In vector form:
#   y_pred = w · x + b
#
# where w is the weight vector and b is the bias (intercept)

print("=== Part 1: Linear Regression Model ===\n")

# Let's create some synthetic data
# True relationship: y = 3*x + 2 + noise
n_samples = 100
X = np.random.uniform(0, 10, size=(n_samples, 1))  # Shape: (100, 1)
true_weight = 3.0
true_bias = 2.0
noise = np.random.normal(0, 1, size=(n_samples, 1))  # Small random noise
y = true_weight * X + true_bias + noise  # Shape: (100, 1)

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"True parameters: weight={true_weight}, bias={true_bias}")
print(f"First 5 examples:")
for i in range(5):
    print(f"  x={X[i,0]:.2f} -> y={y[i,0]:.2f}")


def predict(X, weight, bias):
    """
    Make predictions using linear regression.
    
    y_pred = X * weight + bias
    
    Args:
        X: Input features, shape (n_samples, 1)
        weight: The weight parameter (scalar for 1D input)
        bias: The bias parameter (scalar)
    
    Returns:
        Predictions, shape (n_samples, 1)
    """
    return X * weight + bias


# Test with random initial parameters
initial_weight = 0.0
initial_bias = 0.0
y_pred_initial = predict(X, initial_weight, initial_bias)

print(f"\nInitial parameters: weight={initial_weight}, bias={initial_bias}")
print(f"Initial predictions (first 5): {y_pred_initial[:5, 0]}")


# =============================================================================
# PART 2: LOSS FUNCTION - MEAN SQUARED ERROR
# =============================================================================

# The loss function measures how wrong our predictions are.
# For regression, Mean Squared Error (MSE) is common:
#
#   MSE = (1/n) * sum((y_pred - y_true)^2)
#
# We want to MINIMIZE this loss.

print("\n=== Part 2: Loss Function (MSE) ===\n")


def mean_squared_error(y_pred, y_true):
    """
    Calculate Mean Squared Error.
    
    Args:
        y_pred: Predicted values, shape (n_samples, 1)
        y_true: True values, shape (n_samples, 1)
    
    Returns:
        MSE (scalar)
    """
    n = len(y_pred)
    squared_errors = (y_pred - y_true) ** 2
    mse = np.sum(squared_errors) / n
    return mse


# Calculate initial loss
initial_loss = mean_squared_error(y_pred_initial, y)
print(f"Initial MSE (with weight=0, bias=0): {initial_loss:.2f}")

# Calculate loss with true parameters
y_pred_true = predict(X, true_weight, true_bias)
true_params_loss = mean_squared_error(y_pred_true, y)
print(f"MSE with true parameters: {true_params_loss:.2f}")
print("(Not zero due to noise in the data)")


# =============================================================================
# PART 3: GRADIENT DESCENT - THE KEY IDEA
# =============================================================================

# Gradient descent finds parameters that minimize loss by:
# 1. Calculate the gradient (derivative) of loss with respect to parameters
# 2. Update parameters in the direction that decreases loss
# 3. Repeat until convergence
#
# Update rule:
#   parameter_new = parameter_old - learning_rate * gradient

print("\n=== Part 3: Gradient Descent ===\n")


def compute_gradients(X, y_true, weight, bias):
    """
    Compute gradients of MSE with respect to weight and bias.
    
    For MSE = (1/n) * sum((y_pred - y_true)^2)
    where y_pred = X * weight + bias
    
    The gradients are:
        dL/dw = (2/n) * sum((y_pred - y_true) * X)
        dL/db = (2/n) * sum(y_pred - y_true)
    
    Args:
        X: Input features, shape (n_samples, 1)
        y_true: True values, shape (n_samples, 1)
        weight: Current weight
        bias: Current bias
    
    Returns:
        grad_weight, grad_bias (scalars)
    """
    n = len(X)
    y_pred = predict(X, weight, bias)
    error = y_pred - y_true  # Shape: (n_samples, 1)
    
    # Gradient with respect to weight
    grad_weight = (2 / n) * np.sum(error * X)
    
    # Gradient with respect to bias
    grad_bias = (2 / n) * np.sum(error)
    
    return grad_weight, grad_bias


# Test gradients at initial point
grad_w, grad_b = compute_gradients(X, y, initial_weight, initial_bias)
print(f"Gradients at (weight=0, bias=0):")
print(f"  dL/dw = {grad_w:.4f}")
print(f"  dL/db = {grad_b:.4f}")
print("\nNegative gradient points toward the minimum (true parameters).")


# =============================================================================
# PART 4: BATCH GRADIENT DESCENT
# =============================================================================

# Batch gradient descent uses ALL data points to compute each gradient update.
# - Pro: Stable convergence
# - Con: Slow for large datasets

print("\n=== Part 4: Batch Gradient Descent ===\n")


def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, verbose=True):
    """
    Train linear regression using batch gradient descent.
    
    Args:
        X: Input features
        y: True values
        learning_rate: Step size (eta)
        n_iterations: Number of update steps
        verbose: Print progress
    
    Returns:
        weight, bias, loss_history
    """
    # Initialize parameters
    weight = 0.0
    bias = 0.0
    loss_history = []
    
    for i in range(n_iterations):
        # Compute loss
        y_pred = predict(X, weight, bias)
        loss = mean_squared_error(y_pred, y)
        loss_history.append(loss)
        
        # Compute gradients
        grad_w, grad_b = compute_gradients(X, y, weight, bias)
        
        # Update parameters (move opposite to gradient)
        weight = weight - learning_rate * grad_w
        bias = bias - learning_rate * grad_b
        
        # Print progress
        if verbose and (i % 200 == 0 or i == n_iterations - 1):
            print(f"Iteration {i:4d}: loss={loss:.4f}, weight={weight:.4f}, bias={bias:.4f}")
    
    return weight, bias, loss_history


# Train the model
learned_weight, learned_bias, loss_history = batch_gradient_descent(
    X, y, learning_rate=0.01, n_iterations=1000
)

print(f"\nLearned parameters: weight={learned_weight:.4f}, bias={learned_bias:.4f}")
print(f"True parameters:    weight={true_weight:.4f}, bias={true_bias:.4f}")


# =============================================================================
# PART 5: LEARNING RATE - TOO HIGH vs TOO LOW
# =============================================================================

# The learning rate controls step size:
# - Too low: Slow convergence, may not reach minimum in time
# - Too high: May overshoot and diverge (loss goes to infinity)
# - Just right: Smooth convergence to minimum

print("\n=== Part 5: Learning Rate Effects ===\n")

# Too low
print("Learning rate = 0.001 (too low):")
_, _, loss_low = batch_gradient_descent(X, y, learning_rate=0.001, n_iterations=100, verbose=False)
print(f"  After 100 iterations: loss = {loss_low[-1]:.4f}")

# Good
print("\nLearning rate = 0.01 (good):")
_, _, loss_good = batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=100, verbose=False)
print(f"  After 100 iterations: loss = {loss_good[-1]:.4f}")

# Higher but still ok
print("\nLearning rate = 0.05 (faster):")
_, _, loss_fast = batch_gradient_descent(X, y, learning_rate=0.05, n_iterations=100, verbose=False)
print(f"  After 100 iterations: loss = {loss_fast[-1]:.4f}")

print("\nNote: Too high a learning rate (e.g., 1.0) would cause loss to explode!")


# =============================================================================
# PART 6: STOCHASTIC GRADIENT DESCENT (SGD)
# =============================================================================

# SGD updates parameters using ONE random sample at a time:
# - Pro: Faster updates, can escape local minima, works with large data
# - Con: Noisy updates, may not converge exactly

print("\n=== Part 6: Stochastic Gradient Descent ===\n")


def stochastic_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, verbose=True):
    """
    Train using stochastic gradient descent (one sample at a time).
    """
    weight = 0.0
    bias = 0.0
    n_samples = len(X)
    loss_history = []
    
    for i in range(n_iterations):
        # Pick a random sample
        idx = np.random.randint(0, n_samples)
        x_i = X[idx:idx+1]  # Keep 2D shape
        y_i = y[idx:idx+1]
        
        # Compute gradient on single sample
        y_pred_i = predict(x_i, weight, bias)
        error_i = y_pred_i - y_i
        
        grad_w = 2 * np.sum(error_i * x_i)  # No division by n (single sample)
        grad_b = 2 * np.sum(error_i)
        
        # Update parameters
        weight = weight - learning_rate * grad_w
        bias = bias - learning_rate * grad_b
        
        # Track full dataset loss periodically
        if i % 100 == 0:
            y_pred_all = predict(X, weight, bias)
            loss = mean_squared_error(y_pred_all, y)
            loss_history.append(loss)
            if verbose and i % 500 == 0:
                print(f"Iteration {i:4d}: loss={loss:.4f}")
    
    return weight, bias, loss_history


print("Training with SGD:")
sgd_weight, sgd_bias, sgd_loss = stochastic_gradient_descent(
    X, y, learning_rate=0.01, n_iterations=2000
)
print(f"\nSGD result: weight={sgd_weight:.4f}, bias={sgd_bias:.4f}")


# =============================================================================
# PART 7: MINI-BATCH GRADIENT DESCENT
# =============================================================================

# Mini-batch is a compromise: use a small batch of samples (e.g., 32)
# - More stable than SGD
# - Faster than batch gradient descent
# - Most common in practice (especially deep learning)

print("\n=== Part 7: Mini-Batch Gradient Descent ===\n")


def minibatch_gradient_descent(X, y, batch_size=16, learning_rate=0.01, n_epochs=100, verbose=True):
    """
    Train using mini-batch gradient descent.
    
    An epoch is one full pass through the data.
    """
    weight = 0.0
    bias = 0.0
    n_samples = len(X)
    loss_history = []
    
    for epoch in range(n_epochs):
        # Shuffle data at the start of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Compute gradients on batch
            grad_w, grad_b = compute_gradients(X_batch, y_batch, weight, bias)
            
            # Update parameters
            weight = weight - learning_rate * grad_w
            bias = bias - learning_rate * grad_b
        
        # Track loss after each epoch
        y_pred = predict(X, weight, bias)
        loss = mean_squared_error(y_pred, y)
        loss_history.append(loss)
        
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss={loss:.4f}")
    
    return weight, bias, loss_history


print("Training with Mini-Batch GD (batch_size=16):")
mb_weight, mb_bias, mb_loss = minibatch_gradient_descent(
    X, y, batch_size=16, learning_rate=0.01, n_epochs=100
)
print(f"\nMini-batch result: weight={mb_weight:.4f}, bias={mb_bias:.4f}")


# =============================================================================
# PART 8: COMPARISON OF METHODS
# =============================================================================

print("\n" + "=" * 60)
print("COMPARISON: Gradient Descent Variants")
print("=" * 60)

print(f"\nTrue parameters:       weight={true_weight:.4f}, bias={true_bias:.4f}")
print(f"Batch GD result:       weight={learned_weight:.4f}, bias={learned_bias:.4f}")
print(f"SGD result:            weight={sgd_weight:.4f}, bias={sgd_bias:.4f}")
print(f"Mini-Batch GD result:  weight={mb_weight:.4f}, bias={mb_bias:.4f}")

print("""
Summary of Methods:
-------------------
| Method      | Uses           | Pros                    | Cons                  |
|-------------|----------------|-------------------------|----------------------|
| Batch GD    | All data       | Stable, smooth          | Slow for large data  |
| SGD         | 1 sample       | Fast, memory efficient  | Noisy, may not conv. |
| Mini-Batch  | Small batch    | Good balance            | Need to tune batch   |

In practice, mini-batch gradient descent (or variants like Adam) is most common.
""")


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("SUMMARY: Key Takeaways from Module 2")
print("=" * 60)
print("""
1. LINEAR REGRESSION: y = w*x + b
   - Weights (w) and bias (b) are learned parameters

2. LOSS FUNCTION: MSE = (1/n) * sum((y_pred - y_true)^2)
   - Measures how wrong predictions are
   - We minimize this during training

3. GRADIENT DESCENT: parameter = parameter - lr * gradient
   - Gradient tells us the direction of steepest increase
   - We move opposite to gradient to decrease loss
   - Learning rate (lr) controls step size

4. GRADIENT DESCENT VARIANTS:
   - Batch: Use all data (stable but slow)
   - Stochastic: Use 1 sample (fast but noisy)
   - Mini-Batch: Use small batch (best of both)

5. LEARNING RATE:
   - Too low = slow convergence
   - Too high = divergence
   - Just right = smooth convergence
""")
