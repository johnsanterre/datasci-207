"""
DATASCI 207: Module 6 - Exercise: Build a Neural Network

Practice building and training neural networks.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Implement ReLU and its Derivative
# =============================================================================

def relu(z):
    """
    ReLU activation: max(0, z)
    
    Args:
        z: Input array
    Returns:
        ReLU applied element-wise
    """
    # TODO: Implement ReLU
    result = None  # Replace
    return result


def relu_derivative(z):
    """
    Derivative of ReLU: 1 if z > 0, else 0
    
    Args:
        z: Input array
    Returns:
        Derivative applied element-wise
    """
    # TODO: Implement ReLU derivative
    result = None  # Replace
    return result


# =============================================================================
# EXERCISE 2: Forward Pass Through One Layer
# =============================================================================

def forward_layer(a_prev, W, b, activation='relu'):
    """
    Forward pass through a single layer.
    
    z = a_prev @ W + b
    a = activation(z)
    
    Args:
        a_prev: Activations from previous layer, shape (n_samples, n_in)
        W: Weights, shape (n_in, n_out)
        b: Biases, shape (n_out,)
        activation: 'relu' or 'sigmoid'
    
    Returns:
        a: Activations, shape (n_samples, n_out)
        z: Pre-activation values (for backprop)
    """
    # TODO: Compute z and apply activation
    z = None  # Replace
    
    if activation == 'relu':
        a = None  # Replace
    elif activation == 'sigmoid':
        a = None  # Replace (sigmoid formula)
    else:
        a = z  # Linear
    
    return a, z


# =============================================================================
# EXERCISE 3: Build a Simple Network
# =============================================================================

def build_network(X, W1, b1, W2, b2):
    """
    Forward pass through a 2-layer network.
    
    Layer 1: ReLU
    Layer 2: Sigmoid (output)
    
    Args:
        X: Input, shape (n_samples, n_features)
        W1, b1: Layer 1 parameters
        W2, b2: Layer 2 parameters
    
    Returns:
        y_pred: Predictions, shape (n_samples, 1)
    """
    # TODO: Implement 2-layer forward pass
    # Hint: Use forward_layer twice with appropriate activations
    
    y_pred = None  # Replace
    
    return y_pred


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Neural Network Components\n")
    
    # Test 1: ReLU
    print("=" * 50)
    print("Test 1: ReLU")
    print("=" * 50)
    test_input = np.array([-2, -1, 0, 1, 2])
    relu_out = relu(test_input)
    expected = np.array([0, 0, 0, 1, 2])
    if relu_out is not None:
        print(f"Input: {test_input}")
        print(f"ReLU:  {relu_out}")
        print(f"Expected: {expected}")
        if np.allclose(relu_out, expected):
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: ReLU Derivative
    print("\n" + "=" * 50)
    print("Test 2: ReLU Derivative")
    print("=" * 50)
    relu_deriv = relu_derivative(test_input)
    expected_deriv = np.array([0, 0, 0, 1, 1])
    if relu_deriv is not None:
        print(f"Input: {test_input}")
        print(f"ReLU': {relu_deriv}")
        print(f"Expected: {expected_deriv}")
        if np.allclose(relu_deriv, expected_deriv):
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 3: Forward Layer
    print("\n" + "=" * 50)
    print("Test 3: Forward Layer")
    print("=" * 50)
    X = np.array([[1, 2], [3, 4]])
    W = np.array([[0.5, -0.5], [0.5, 0.5]])
    b = np.array([0, 0])
    a, z = forward_layer(X, W, b, 'relu')
    if a is not None and z is not None:
        print(f"Input X:\n{X}")
        print(f"z = X @ W + b:\n{z}")
        print(f"a = ReLU(z):\n{a}")
        # X @ W = [[1.5, 0.5], [3.5, 0.5]]
        # ReLU keeps same (all positive)
        if np.allclose(z, np.array([[1.5, 0.5], [3.5, 0.5]])):
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 4: Full Network
    print("\n" + "=" * 50)
    print("Test 4: Full Network on XOR")
    print("=" * 50)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Pre-trained weights that solve XOR
    W1 = np.array([[1, 1], [1, 1]])
    b1 = np.array([0, -1])
    W2 = np.array([[1], [-2]])
    b2 = np.array([0])
    
    y_pred = build_network(X_xor, W1, b1, W2, b2)
    if y_pred is not None:
        print("XOR predictions with pre-trained weights:")
        for x, p in zip(X_xor, y_pred.flatten()):
            print(f"  {x} -> {p:.3f}")
    else:
        print("Not implemented yet")
