"""
DATASCI 207: Module 13 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention(Q, K, V):
    """Scaled dot-product attention."""
    d_k = K.shape[-1]
    
    # Scores
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    # Weights
    weights = softmax(scores, axis=-1)
    
    # Output
    output = np.matmul(weights, V)
    
    return output, weights


def create_causal_mask(seq_len):
    """Create lower triangular mask."""
    return np.tril(np.ones((seq_len, seq_len)))


if __name__ == "__main__":
    # Test softmax
    x = np.array([1.0, 2.0, 3.0])
    s = softmax(x)
    assert abs(s.sum() - 1.0) < 0.01
    print("Softmax: VERIFIED")
    
    # Test attention
    Q = np.random.randn(4, 8)
    K = np.random.randn(4, 8)
    V = np.random.randn(4, 8)
    out, w = attention(Q, K, V)
    assert out.shape == V.shape
    print("Attention: VERIFIED")
    
    # Test causal mask
    mask = create_causal_mask(4)
    expected = np.array([[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]])
    assert np.allclose(mask, expected)
    print("Causal Mask: VERIFIED")
