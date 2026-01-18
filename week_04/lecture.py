"""
DATASCI 207: Applied Machine Learning
Module 4: Logistic Regression

This module covers:
- Binary classification vs regression
- The sigmoid function
- Logistic regression model
- Decision boundaries and thresholds
- Logistic loss (cross-entropy)
- Training with gradient descent

Using NumPy only - no pandas.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: BINARY CLASSIFICATION
# =============================================================================

# Classification: predict discrete class labels, not continuous values
# Binary classification: exactly two classes (0 or 1, yes or no, spam or not)

print("=== Part 1: Binary Classification ===\n")

# Example: predict if a student passes (1) or fails (0) based on study hours
# Let's create some data

study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
passed = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 = fail, 1 = pass

print("Study Hours -> Passed?")
for hours, result in zip(study_hours.flatten(), passed):
    print(f"  {hours} hours -> {'Pass' if result else 'Fail'}")

# Problem: Linear regression outputs any value, but we need 0 or 1
# We need a function that maps to the range (0, 1) -> probability


# =============================================================================
# PART 2: THE SIGMOID FUNCTION
# =============================================================================

# The sigmoid function squashes any real number to the range (0, 1):
#
#   sigmoid(z) = 1 / (1 + exp(-z))
#
# Properties:
# - Output is always between 0 and 1
# - sigmoid(0) = 0.5
# - sigmoid(large positive) ≈ 1
# - sigmoid(large negative) ≈ 0

print("\n=== Part 2: The Sigmoid Function ===\n")


def sigmoid(z):
    """
    The sigmoid activation function.
    
    sigmoid(z) = 1 / (1 + exp(-z))
    
    Args:
        z: Input value(s) - can be scalar or array
    
    Returns:
        Values squashed to range (0, 1)
    """
    return 1 / (1 + np.exp(-z))


# Test sigmoid at various points
test_values = [-10, -5, -1, 0, 1, 5, 10]
print("Sigmoid function values:")
for z in test_values:
    print(f"  sigmoid({z:3d}) = {sigmoid(z):.6f}")

# Key observations:
# - sigmoid(0) = 0.5 (the midpoint)
# - Large positive values -> close to 1
# - Large negative values -> close to 0


# =============================================================================
# PART 3: LOGISTIC REGRESSION MODEL
# =============================================================================

# Logistic regression combines linear regression with sigmoid:
#
#   z = w * x + b           (linear part)
#   p = sigmoid(z)          (probability)
#   prediction = 1 if p >= 0.5 else 0
#
# The output p is interpreted as P(y=1 | x)

print("\n=== Part 3: Logistic Regression Model ===\n")


def logistic_regression_predict_proba(X, weights, bias):
    """
    Predict probabilities using logistic regression.
    
    Args:
        X: Input features, shape (n_samples, n_features)
        weights: Model weights, shape (n_features,)
        bias: Model bias (scalar)
    
    Returns:
        Probabilities, shape (n_samples,)
    """
    z = X @ weights + bias  # Linear combination
    probabilities = sigmoid(z)  # Squash to (0, 1)
    return probabilities


def logistic_regression_predict(X, weights, bias, threshold=0.5):
    """
    Predict class labels (0 or 1).
    
    Args:
        X, weights, bias: Model parameters
        threshold: Probability threshold for class 1 (default 0.5)
    
    Returns:
        Class predictions (0 or 1), shape (n_samples,)
    """
    probabilities = logistic_regression_predict_proba(X, weights, bias)
    return (probabilities >= threshold).astype(int)


# Example with made-up parameters
example_weight = np.array([0.5])
example_bias = -2.5

print("Example predictions with weight=0.5, bias=-2.5:")
probs = logistic_regression_predict_proba(study_hours, example_weight, example_bias)
preds = logistic_regression_predict(study_hours, example_weight, example_bias)

for hours, prob, pred, actual in zip(study_hours.flatten(), probs, preds, passed):
    status = "correct" if pred == actual else "WRONG"
    print(f"  {hours} hours: P(pass)={prob:.3f}, Predict={pred}, Actual={actual} [{status}]")


# =============================================================================
# PART 4: DECISION BOUNDARY
# =============================================================================

# The decision boundary is where P(y=1) = 0.5
# This happens when z = w*x + b = 0, so x = -b/w

print("\n=== Part 4: Decision Boundary ===\n")

# With weight=0.5 and bias=-2.5:
# Decision boundary: 0.5*x - 2.5 = 0  =>  x = 5

decision_boundary = -example_bias / example_weight[0]
print(f"Decision boundary: x = {decision_boundary}")
print(f"Students with >= {decision_boundary} study hours are predicted to pass")


# =============================================================================
# PART 5: CLASSIFICATION THRESHOLD
# =============================================================================

# By default, we predict class 1 if P >= 0.5
# But we can change this threshold!

print("\n=== Part 5: Classification Threshold ===\n")

# Lower threshold = more likely to predict positive (more false positives)
# Higher threshold = less likely to predict positive (more false negatives)

thresholds = [0.3, 0.5, 0.7]
for thresh in thresholds:
    preds = logistic_regression_predict(study_hours, example_weight, example_bias, threshold=thresh)
    accuracy = np.mean(preds == passed)
    n_positive = np.sum(preds)
    print(f"Threshold {thresh}: {n_positive} predicted positive, accuracy = {accuracy:.0%}")

print("\nChoice of threshold depends on the cost of false positives vs false negatives.")


# =============================================================================
# PART 6: LOGISTIC LOSS (CROSS-ENTROPY)
# =============================================================================

# We can't use MSE for classification (bad gradients!)
# Instead, use logistic loss (binary cross-entropy):
#
#   L = -[y * log(p) + (1-y) * log(1-p)]
#
# This loss is low when predictions match labels:
# - If y=1 and p is high, loss is low (good)
# - If y=1 and p is low, loss is high (bad)
# - If y=0 and p is low, loss is low (good)
# - If y=0 and p is high, loss is high (bad)

print("\n=== Part 6: Logistic Loss (Cross-Entropy) ===\n")


def logistic_loss(y_true, y_pred_proba):
    """
    Compute binary cross-entropy loss.
    
    L = -(1/n) * sum[y*log(p) + (1-y)*log(1-p)]
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities
    
    Returns:
        Average loss (scalar)
    """
    # Clip predictions to avoid log(0)
    eps = 1e-15
    p = np.clip(y_pred_proba, eps, 1 - eps)
    
    # Binary cross-entropy
    loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return loss


# Demonstrate loss behavior
print("Loss for different prediction scenarios:")
print(f"  y=1, p=0.9: loss = {logistic_loss(np.array([1]), np.array([0.9])):.4f} (good prediction)")
print(f"  y=1, p=0.1: loss = {logistic_loss(np.array([1]), np.array([0.1])):.4f} (bad prediction)")
print(f"  y=0, p=0.1: loss = {logistic_loss(np.array([0]), np.array([0.1])):.4f} (good prediction)")
print(f"  y=0, p=0.9: loss = {logistic_loss(np.array([0]), np.array([0.9])):.4f} (bad prediction)")


# =============================================================================
# PART 7: GRADIENT DESCENT FOR LOGISTIC REGRESSION
# =============================================================================

# The gradient of logistic loss with respect to weights has a nice form:
#
#   dL/dw = (1/n) * X.T @ (predictions - true_labels)
#   dL/db = (1/n) * sum(predictions - true_labels)
#
# This is the same form as linear regression! The sigmoid "absorbs" into the math.

print("\n=== Part 7: Training with Gradient Descent ===\n")


def train_logistic_regression(X, y, learning_rate=0.1, n_iterations=1000, verbose=True):
    """
    Train logistic regression using gradient descent.
    
    Args:
        X: Features, shape (n_samples, n_features)
        y: Labels (0 or 1), shape (n_samples,)
        learning_rate: Step size
        n_iterations: Number of training steps
        verbose: Print progress
    
    Returns:
        weights, bias, loss_history
    """
    n_samples, n_features = X.shape
    
    # Initialize parameters
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    
    for i in range(n_iterations):
        # Forward pass: compute predictions
        z = X @ weights + bias
        predictions = sigmoid(z)
        
        # Compute loss
        loss = logistic_loss(y, predictions)
        loss_history.append(loss)
        
        # Compute gradients
        errors = predictions - y  # (n_samples,)
        grad_weights = (1 / n_samples) * (X.T @ errors)
        grad_bias = (1 / n_samples) * np.sum(errors)
        
        # Update parameters
        weights = weights - learning_rate * grad_weights
        bias = bias - learning_rate * grad_bias
        
        if verbose and i % 200 == 0:
            accuracy = np.mean((predictions >= 0.5) == y)
            print(f"Iteration {i:4d}: loss={loss:.4f}, accuracy={accuracy:.0%}")
    
    return weights, bias, loss_history


# Train on our data
print("Training logistic regression on study hours data...\n")
learned_weights, learned_bias, losses = train_logistic_regression(
    study_hours, passed, learning_rate=0.5, n_iterations=1000
)

print(f"\nLearned parameters:")
print(f"  weight = {learned_weights[0]:.4f}")
print(f"  bias = {learned_bias:.4f}")
print(f"  decision boundary = {-learned_bias / learned_weights[0]:.2f} hours")


# Final predictions
print("\nFinal predictions:")
final_probs = logistic_regression_predict_proba(study_hours, learned_weights, learned_bias)
final_preds = logistic_regression_predict(study_hours, learned_weights, learned_bias)

for hours, prob, pred, actual in zip(study_hours.flatten(), final_probs, final_preds, passed):
    status = "correct" if pred == actual else "WRONG"
    print(f"  {hours} hours: P(pass)={prob:.3f}, Predict={pred}, Actual={actual} [{status}]")

accuracy = np.mean(final_preds == passed)
print(f"\nFinal accuracy: {accuracy:.0%}")


# =============================================================================
# PART 8: SIGMOID DERIVATIVE
# =============================================================================

# For reference, the derivative of sigmoid is:
#   sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
# This is used in backpropagation for neural networks.

print("\n=== Part 8: Sigmoid Derivative (Bonus) ===\n")


def sigmoid_derivative(z):
    """
    Derivative of the sigmoid function.
    
    sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
    """
    s = sigmoid(z)
    return s * (1 - s)


print("Sigmoid derivative at various points:")
for z in [-2, -1, 0, 1, 2]:
    print(f"  sigmoid'({z:2d}) = {sigmoid_derivative(z):.4f}")

print("\nNote: Derivative is highest at z=0, near 0 for large |z|.")
print("This can cause vanishing gradients in deep networks.")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 4")
print("=" * 60)
print("""
1. CLASSIFICATION vs REGRESSION:
   - Regression: predict continuous values
   - Classification: predict discrete class labels

2. SIGMOID FUNCTION: sigmoid(z) = 1 / (1 + exp(-z))
   - Maps any real number to (0, 1)
   - Used to output probabilities

3. LOGISTIC REGRESSION:
   - z = w*x + b (linear part)
   - p = sigmoid(z) (probability of class 1)
   - Predict class 1 if p >= threshold

4. DECISION BOUNDARY:
   - Where p = 0.5, i.e., w*x + b = 0
   - Linear boundary in feature space

5. CLASSIFICATION THRESHOLD:
   - Default 0.5, but adjustable
   - Lower = more positive predictions
   - Choose based on cost of errors

6. LOGISTIC LOSS (Cross-Entropy):
   - L = -[y*log(p) + (1-y)*log(1-p)]
   - High when prediction is wrong
   - Low when prediction is correct

7. TRAINING:
   - Gradient descent works the same way
   - Gradient has same form as linear regression
""")
