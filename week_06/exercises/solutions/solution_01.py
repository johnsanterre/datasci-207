"""
DATASCI 207: Module 6 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def relu(z):
    """ReLU activation: max(0, z)"""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0, else 0"""
    return (z > 0).astype(float)


def forward_layer(a_prev, W, b, activation='relu'):
    """Forward pass through a single layer."""
    z = a_prev @ W + b
    
    if activation == 'relu':
        a = relu(z)
    elif activation == 'sigmoid':
        a = 1 / (1 + np.exp(-z))
    else:
        a = z
    
    return a, z


def build_network(X, W1, b1, W2, b2):
    """Forward pass through a 2-layer network."""
    a1, z1 = forward_layer(X, W1, b1, 'relu')
    a2, z2 = forward_layer(a1, W2, b2, 'sigmoid')
    return a2


if __name__ == "__main__":
    # Verify ReLU
    test_input = np.array([-2, -1, 0, 1, 2])
    assert np.allclose(relu(test_input), [0, 0, 0, 1, 2])
    print("ReLU: VERIFIED")
    
    # Verify derivative
    assert np.allclose(relu_derivative(test_input), [0, 0, 0, 1, 1])
    print("ReLU derivative: VERIFIED")
    
    # Verify forward layer
    X = np.array([[1, 2], [3, 4]])
    W = np.array([[0.5, -0.5], [0.5, 0.5]])
    b = np.array([0, 0])
    a, z = forward_layer(X, W, b, 'relu')
    assert np.allclose(z, [[1.5, 0.5], [3.5, 0.5]])
    print("Forward layer: VERIFIED")
    
    # Test XOR network
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    W1 = np.array([[1, 1], [1, 1]])
    b1 = np.array([0, -1])
    W2 = np.array([[1], [-2]])
    b2 = np.array([0])
    
    y_pred = build_network(X_xor, W1, b1, W2, b2)
    print("\nXOR with pre-trained weights:")
    for x, p in zip(X_xor, y_pred.flatten()):
        print(f"  {x} -> {p:.3f}")
