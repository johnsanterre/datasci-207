"""
DATASCI 207: Applied Machine Learning
Module 5: Multiclass Classification and Metrics

This module covers:
- Softmax function for multiclass
- Categorical cross-entropy loss
- Precision, recall, F1 score
- Confusion matrices
- Limitations of linear models

Using NumPy and sklearn for metrics - no pandas.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: MULTICLASS CLASSIFICATION
# =============================================================================

print("=== Part 1: Multiclass Classification ===\n")

# Binary: 2 classes (cat vs dog)
# Multiclass: k classes (cat vs dog vs bird vs fish...)

# Create synthetic 3-class data
n_samples = 100
n_classes = 3
n_features = 4

# Random features and labels
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

print(f"Data shape: X = {X.shape}")
print(f"Classes: {n_classes}")
print(f"Class distribution: {np.bincount(y)}")


# =============================================================================
# PART 2: SOFTMAX FUNCTION
# =============================================================================

# Softmax converts a vector of arbitrary values (logits) to probabilities.
# softmax(z)_i = exp(z_i) / sum(exp(z_j))
#
# Properties:
# - All outputs are positive
# - Outputs sum to 1
# - Higher logit -> higher probability

print("\n=== Part 2: Softmax Function ===\n")


def softmax(z):
    """
    Compute softmax probabilities.
    
    Args:
        z: Logits, shape (n_samples, n_classes) or (n_classes,)
    
    Returns:
        Probabilities, same shape as z
    """
    # Subtract max for numerical stability (prevents overflow)
    z_stable = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


# Example
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(f"Logits: {logits}")
print(f"Softmax probabilities: {probs}")
print(f"Sum of probabilities: {np.sum(probs):.4f}")


# Batch example
batch_logits = np.array([
    [2.0, 1.0, 0.0],
    [0.0, 3.0, 1.0],
    [1.0, 1.0, 1.0]
])
batch_probs = softmax(batch_logits)

print(f"\nBatch softmax:")
for i in range(3):
    print(f"  {batch_logits[i]} -> {batch_probs[i]}")


# =============================================================================
# PART 3: CATEGORICAL CROSS-ENTROPY LOSS
# =============================================================================

# Cross-entropy loss for multiclass:
# L = -sum(y_true * log(y_pred))
# where y_true is one-hot encoded

print("\n=== Part 3: Categorical Cross-Entropy Loss ===\n")


def one_hot_encode(labels, n_classes):
    """Convert class labels to one-hot encoding."""
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot


def categorical_cross_entropy(y_true_onehot, y_pred_proba):
    """
    Compute categorical cross-entropy loss.
    
    Args:
        y_true_onehot: One-hot encoded labels, shape (n_samples, n_classes)
        y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)
    
    Returns:
        Average loss (scalar)
    """
    eps = 1e-15  # Avoid log(0)
    y_pred_clipped = np.clip(y_pred_proba, eps, 1 - eps)
    
    # Sum over classes, average over samples
    loss = -np.mean(np.sum(y_true_onehot * np.log(y_pred_clipped), axis=1))
    return loss


# Example
y_true = np.array([0, 1, 2])
y_true_onehot = one_hot_encode(y_true, n_classes=3)
y_pred_good = np.array([
    [0.9, 0.05, 0.05],  # High prob for class 0 (correct)
    [0.1, 0.8, 0.1],    # High prob for class 1 (correct)
    [0.05, 0.05, 0.9]   # High prob for class 2 (correct)
])
y_pred_bad = np.array([
    [0.1, 0.8, 0.1],    # Wrong prediction
    [0.9, 0.05, 0.05],  # Wrong prediction
    [0.1, 0.8, 0.1]     # Wrong prediction
])

loss_good = categorical_cross_entropy(y_true_onehot, y_pred_good)
loss_bad = categorical_cross_entropy(y_true_onehot, y_pred_bad)

print(f"Good predictions loss: {loss_good:.4f}")
print(f"Bad predictions loss: {loss_bad:.4f}")
print("(Lower is better)")


# =============================================================================
# PART 4: CONFUSION MATRIX
# =============================================================================

print("\n=== Part 4: Confusion Matrix ===\n")


def confusion_matrix(y_true, y_pred, n_classes):
    """
    Build confusion matrix.
    
    Returns:
        Matrix of shape (n_classes, n_classes)
        cm[i, j] = number of samples with true class i predicted as class j
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


# Example predictions
y_true_example = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred_example = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])

cm = confusion_matrix(y_true_example, y_pred_example, n_classes=3)

print("Confusion Matrix:")
print(f"True labels: {y_true_example}")
print(f"Predictions: {y_pred_example}")
print()
print("         Predicted")
print("          0   1   2")
print(f"True  0 [ {cm[0,0]}   {cm[0,1]}   {cm[0,2]} ]")
print(f"      1 [ {cm[1,0]}   {cm[1,1]}   {cm[1,2]} ]")
print(f"      2 [ {cm[2,0]}   {cm[2,1]}   {cm[2,2]} ]")
print()
print("Diagonal = correct predictions")
print("Off-diagonal = errors")


# =============================================================================
# PART 5: PRECISION AND RECALL
# =============================================================================

print("\n=== Part 5: Precision and Recall ===\n")

# For binary classification:
# Precision = TP / (TP + FP)  -- Of predicted positive, how many are correct?
# Recall = TP / (TP + FN)     -- Of actual positive, how many did we find?


def compute_precision_recall(y_true, y_pred, positive_class=1):
    """
    Compute precision and recall for binary classification.
    """
    tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
    fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
    fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall


# Imbalanced example: disease detection
# 100 samples, 10 have disease (class 1)
y_true_disease = np.array([0]*90 + [1]*10)

# Model A: Conservative (high threshold)
y_pred_conservative = np.array([0]*95 + [1]*5)  # Predicts 5 as positive

# Model B: Aggressive (low threshold)
y_pred_aggressive = np.array([0]*70 + [1]*30)  # Predicts 30 as positive

# Assume: of the 10 actual positives, conservative catches 4, aggressive catches 9
# We need to craft the predictions to match
np.random.seed(42)
y_pred_conservative = np.zeros(100, dtype=int)
y_pred_conservative[90:94] = 1  # Catches 4 of 10 diseased
y_pred_conservative[50] = 1     # 1 false positive

y_pred_aggressive = np.zeros(100, dtype=int)
y_pred_aggressive[90:99] = 1    # Catches 9 of 10 diseased
y_pred_aggressive[30:50] = 1    # 20 false positives

prec_cons, rec_cons = compute_precision_recall(y_true_disease, y_pred_conservative)
prec_aggr, rec_aggr = compute_precision_recall(y_true_disease, y_pred_aggressive)

print("Disease Detection Example:")
print(f"Conservative model: Precision={prec_cons:.2f}, Recall={rec_cons:.2f}")
print(f"Aggressive model:   Precision={prec_aggr:.2f}, Recall={rec_aggr:.2f}")
print()
print("Conservative: High precision (reliable), low recall (misses cases)")
print("Aggressive: Low precision (many false alarms), high recall (finds most cases)")


# =============================================================================
# PART 6: F1 SCORE
# =============================================================================

print("\n=== Part 6: F1 Score ===\n")

# F1 = 2 * (precision * recall) / (precision + recall)
# Harmonic mean - low if either P or R is low


def f1_score(precision, recall):
    """Compute F1 score as harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


f1_cons = f1_score(prec_cons, rec_cons)
f1_aggr = f1_score(prec_aggr, rec_aggr)

print(f"Conservative model F1: {f1_cons:.2f}")
print(f"Aggressive model F1: {f1_aggr:.2f}")
print()

# Show harmonic mean behavior
print("F1 penalizes imbalance:")
print(f"  P=0.9, R=0.9 -> F1={f1_score(0.9, 0.9):.2f}")
print(f"  P=0.9, R=0.1 -> F1={f1_score(0.9, 0.1):.2f}")
print(f"  P=0.5, R=0.5 -> F1={f1_score(0.5, 0.5):.2f}")


# =============================================================================
# PART 7: MULTICLASS METRICS
# =============================================================================

print("\n=== Part 7: Multiclass Metrics ===\n")

# For multiclass, we compute per-class precision/recall, then average

def multiclass_precision_recall(y_true, y_pred, n_classes):
    """Compute per-class and averaged metrics."""
    precisions = []
    recalls = []
    
    for c in range(n_classes):
        # Treat class c as positive, all others as negative
        y_true_binary = (y_true == c).astype(int)
        y_pred_binary = (y_pred == c).astype(int)
        
        p, r = compute_precision_recall(y_true_binary, y_pred_binary, positive_class=1)
        precisions.append(p)
        recalls.append(r)
    
    return precisions, recalls


prec_list, rec_list = multiclass_precision_recall(y_true_example, y_pred_example, 3)

print("Per-class metrics:")
for c in range(3):
    print(f"  Class {c}: Precision={prec_list[c]:.2f}, Recall={rec_list[c]:.2f}")

print(f"\nMacro-average Precision: {np.mean(prec_list):.2f}")
print(f"Macro-average Recall: {np.mean(rec_list):.2f}")


# =============================================================================
# PART 8: LIMITATIONS OF LINEAR MODELS
# =============================================================================

print("\n=== Part 8: Limitations of Linear Models ===\n")

# Linear models create linear decision boundaries (hyperplanes)
# This fails for non-linearly separable data

# XOR Problem - classic example
print("The XOR Problem:")
print("(0,0) -> 0")
print("(1,1) -> 0")
print("(0,1) -> 1")
print("(1,0) -> 1")
print()
print("No straight line can separate these classes!")

# Create XOR data
X_xor = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y_xor = np.array([0, 0, 1, 1])


def train_logistic_regression(X, y, lr=0.1, n_iter=1000):
    """Simple logistic regression for demonstration."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    
    for _ in range(n_iter):
        z = X @ weights + bias
        preds = 1 / (1 + np.exp(-z))
        errors = preds - y
        weights -= lr * (X.T @ errors) / n_samples
        bias -= lr * np.mean(errors)
    
    return weights, bias


weights_xor, bias_xor = train_logistic_regression(X_xor, y_xor)
z_xor = X_xor @ weights_xor + bias_xor
preds_xor = (1 / (1 + np.exp(-z_xor)) >= 0.5).astype(int)

print(f"\nLogistic regression on XOR:")
print(f"Predictions: {preds_xor}")
print(f"True labels: {y_xor}")
print(f"Accuracy: {np.mean(preds_xor == y_xor):.0%}")
print()
print("Linear models CANNOT solve XOR!")
print("Solutions: Neural networks (Module 6) or feature crosses")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 5")
print("=" * 60)
print("""
1. SOFTMAX: Converts logits to probability distribution
   - exp(z_i) / sum(exp(z_j))
   - Generalizes sigmoid to k classes

2. CROSS-ENTROPY LOSS: L = -sum(y * log(p))
   - Penalizes confident wrong predictions heavily
   - Standard loss for classification

3. CONFUSION MATRIX: Full picture of predictions
   - Rows = true class, Columns = predicted class
   - Diagonal = correct, Off-diagonal = errors

4. PRECISION: TP / (TP + FP)
   - "When I predict positive, am I right?"
   - High precision = few false positives

5. RECALL: TP / (TP + FN)
   - "Do I find all the positives?"
   - High recall = few missed positives

6. F1 SCORE: Harmonic mean of P and R
   - Balanced single metric
   - Low if either P or R is low

7. LINEAR LIMITATIONS:
   - Can only create linear boundaries
   - Fails on XOR and complex patterns
   - Need neural networks or feature engineering
""")
