"""
DATASCI 207: Applied Machine Learning
Module 6: Feedforward Neural Networks

This module covers:
- Why hidden layers are necessary (XOR problem)
- Activation functions (ReLU, sigmoid, tanh)
- Forward propagation
- Backpropagation basics
- Building networks with TensorFlow/Keras

Using NumPy for concepts, TensorFlow/Keras for practical training.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: XOR PROBLEM - WHY WE NEED HIDDEN LAYERS
# =============================================================================

print("=== Part 1: The XOR Problem ===\n")

# XOR cannot be solved with a linear model
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

print("XOR Data:")
for x, y in zip(X_xor, y_xor):
    print(f"  {x} -> {y}")

print("\nNo straight line can separate class 0 from class 1!")
print("Solution: Add a hidden layer to transform the space.")


# =============================================================================
# PART 2: ACTIVATION FUNCTIONS
# =============================================================================

print("\n=== Part 2: Activation Functions ===\n")


def relu(z):
    """ReLU: max(0, z). Most common for hidden layers."""
    return np.maximum(0, z)


def sigmoid(z):
    """Sigmoid: 1 / (1 + exp(-z)). For binary output."""
    return 1 / (1 + np.exp(-z))


def tanh_activation(z):
    """Tanh: output in (-1, 1). Zero-centered."""
    return np.tanh(z)


# Demonstrate activations
test_values = np.array([-2, -1, 0, 1, 2])
print(f"Input values: {test_values}")
print(f"ReLU:    {relu(test_values)}")
print(f"Sigmoid: {np.round(sigmoid(test_values), 3)}")
print(f"Tanh:    {np.round(tanh_activation(test_values), 3)}")

print("""
Activation Summary:
- ReLU: Simple, fast, no vanishing gradient (for positive values)
- Sigmoid: Output (0,1), good for binary output, vanishing gradients
- Tanh: Output (-1,1), zero-centered, vanishing gradients
""")


# =============================================================================
# PART 3: FORWARD PROPAGATION - FROM SCRATCH
# =============================================================================

print("\n=== Part 3: Forward Propagation ===\n")

# Let's build a simple network: 2 inputs -> 2 hidden -> 1 output
# This can solve XOR!


def forward_pass(X, W1, b1, W2, b2):
    """
    Forward propagation through a 2-layer network.
    
    Layer 1: hidden = ReLU(X @ W1 + b1)
    Layer 2: output = sigmoid(hidden @ W2 + b2)
    
    Returns: predictions, hidden activations
    """
    # Layer 1 (hidden layer)
    z1 = X @ W1 + b1
    a1 = relu(z1)
    
    # Layer 2 (output layer)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    
    return a2, a1, z1  # Return intermediate values for backprop


# Initialize weights (small random values)
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5  # 2 inputs -> 4 hidden
b1 = np.zeros(4)
W2 = np.random.randn(4, 1) * 0.5  # 4 hidden -> 1 output
b2 = np.zeros(1)

# Forward pass on XOR data
y_pred, hidden, z_hidden = forward_pass(X_xor, W1, b1, W2, b2)

print("Forward pass on XOR with random weights:")
for x, y_true, pred in zip(X_xor, y_xor, y_pred.flatten()):
    print(f"  Input: {x}, True: {y_true}, Pred: {pred:.3f}")

print("\nWith random weights, predictions are poor. We need training!")


# =============================================================================
# PART 4: BACKWARD PROPAGATION (CONCEPTUAL)
# =============================================================================

print("\n=== Part 4: Backpropagation Concept ===\n")


def backward_pass(X, y, y_pred, hidden, z_hidden, W1, b1, W2, b2, lr=0.1):
    """
    Backpropagation: compute gradients and update weights.
    
    Uses chain rule to propagate gradients from output to input.
    """
    n = len(X)
    y = y.reshape(-1, 1)
    
    # Output layer gradient
    dz2 = y_pred - y  # Derivative of BCE loss with sigmoid
    dW2 = hidden.T @ dz2 / n
    db2 = np.mean(dz2, axis=0)
    
    # Hidden layer gradient (chain rule)
    da1 = dz2 @ W2.T
    dz1 = da1 * (z_hidden > 0)  # ReLU derivative
    dW1 = X.T @ dz1 / n
    db1 = np.mean(dz1, axis=0)
    
    # Update weights
    W2_new = W2 - lr * dW2
    b2_new = b2 - lr * db2
    W1_new = W1 - lr * dW1
    b1_new = b1 - lr * db1
    
    return W1_new, b1_new, W2_new, b2_new


print("Backpropagation:")
print("1. Compute output error: (prediction - target)")
print("2. Compute gradients for output layer weights")
print("3. Propagate error backwards through hidden layer")
print("4. Compute gradients for hidden layer weights")
print("5. Update all weights using gradient descent")


# =============================================================================
# PART 5: TRAINING A NETWORK FROM SCRATCH
# =============================================================================

print("\n=== Part 5: Training from Scratch ===\n")


def train_network(X, y, hidden_size=8, lr=1.0, epochs=1000):
    """Train a 2-layer neural network on XOR."""
    n_features = X.shape[1]
    n_samples = len(X)
    y = y.reshape(-1, 1)
    
    # Initialize weights
    np.random.seed(42)
    W1 = np.random.randn(n_features, hidden_size) * 0.5
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, 1) * 0.5
    b2 = np.zeros(1)
    
    for epoch in range(epochs):
        # Forward pass
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        y_pred = sigmoid(z2)
        
        # Compute loss (binary cross-entropy)
        eps = 1e-15
        loss = -np.mean(y * np.log(y_pred + eps) + (1-y) * np.log(1-y_pred + eps))
        
        # Backward pass
        dz2 = y_pred - y
        dW2 = a1.T @ dz2 / n_samples
        db2 = np.mean(dz2, axis=0)
        
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1 / n_samples
        db1 = np.mean(dz1, axis=0)
        
        # Update weights
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")
    
    return W1, b1, W2, b2


print("Training neural network on XOR...")
W1_trained, b1_trained, W2_trained, b2_trained = train_network(
    X_xor, y_xor, hidden_size=8, lr=2.0, epochs=1000
)

# Test trained network
y_final, _, _ = forward_pass(X_xor, W1_trained, b1_trained, W2_trained, b2_trained)
print("\nFinal predictions:")
for x, y_true, pred in zip(X_xor, y_xor, y_final.flatten()):
    pred_class = 1 if pred > 0.5 else 0
    status = "correct" if pred_class == y_true else "wrong"
    print(f"  {x} -> pred: {pred:.3f} (class {pred_class}), true: {y_true} [{status}]")


# =============================================================================
# PART 6: BUILDING WITH TENSORFLOW/KERAS
# =============================================================================

print("\n=== Part 6: TensorFlow/Keras ===\n")

print("""
With Keras, the same network is just a few lines:

```python
import tensorflow as tf
from tensorflow import keras

# Build model
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_xor, y_xor, epochs=1000, verbose=0)

# Predict
predictions = model.predict(X_xor)
```

Keras handles:
- Weight initialization
- Backpropagation (automatic differentiation)
- Optimization (Adam, SGD, etc.)
- Batching and shuffling
""")

# Actually run Keras if available
try:
    import tensorflow as tf
    tf.random.set_seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_xor, y_xor, epochs=500, verbose=0)
    
    preds = model.predict(X_xor, verbose=0).flatten()
    print("Keras model predictions on XOR:")
    for x, y_true, p in zip(X_xor, y_xor, preds):
        print(f"  {x} -> {p:.3f} (class {int(p > 0.5)}), true: {y_true}")
    
    print(f"\nTest accuracy: {np.mean((preds > 0.5) == y_xor):.0%}")
    
except ImportError:
    print("(TensorFlow not available - install with: pip install tensorflow)")


# =============================================================================
# PART 7: DEEPER NETWORKS
# =============================================================================

print("\n=== Part 7: Going Deeper ===\n")

print("""
More layers = more representational power

Benefits of depth:
- Learn hierarchical representations
- Features at each layer build on previous
- Often more efficient than wide but shallow

Challenges of depth:
- Vanishing/exploding gradients
- Harder to train
- Need more data

Solutions:
- ReLU activation (reduces vanishing gradient)
- Batch normalization (stabilizes training)
- Skip connections (ResNets)
- Careful initialization
""")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 6")
print("=" * 60)
print("""
1. HIDDEN LAYERS enable non-linear decision boundaries
   - Transform inputs to new representation
   - XOR becomes solvable!

2. ACTIVATION FUNCTIONS introduce non-linearity:
   - ReLU: max(0, z) - default for hidden layers
   - Sigmoid: for binary output
   - Softmax: for multiclass output

3. FORWARD PROPAGATION:
   - Compute layer by layer: z = Wx + b, a = activation(z)
   - Save intermediate values for backprop

4. BACKPROPAGATION:
   - Apply chain rule backwards through network
   - Compute gradient of loss w.r.t. each parameter
   - Modern frameworks do this automatically

5. PRACTICAL TRAINING:
   - Use Keras/TensorFlow for real applications
   - Adam optimizer is robust default
   - Monitor training and validation loss
""")
