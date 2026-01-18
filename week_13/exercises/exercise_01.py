"""
DATASCI 207: Module 13 - Exercise: Attention Mechanisms
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Softmax Implementation
# =============================================================================

def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    
    Subtract max before exp for numerical stability.
    """
    # TODO: Implement softmax
    # 1. Subtract max for stability
    # 2. Compute exp
    # 3. Normalize
    
    result = None  # Replace
    return result


# =============================================================================
# EXERCISE 2: Scaled Dot-Product Attention
# =============================================================================

def attention(Q, K, V):
    """
    Scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
    
    Args:
        Q: Queries, shape (seq_len, d_k)
        K: Keys, shape (seq_len, d_k)
        V: Values, shape (seq_len, d_v)
    
    Returns:
        output: Attention output
        weights: Attention weights
    """
    # TODO: Implement attention
    # 1. Compute scores = Q @ K^T
    # 2. Scale by sqrt(d_k)
    # 3. Apply softmax
    # 4. Weighted sum of V
    
    output = None
    weights = None
    return output, weights


# =============================================================================
# EXERCISE 3: Causal Mask
# =============================================================================

def create_causal_mask(seq_len):
    """
    Create lower triangular mask for causal attention.
    
    Position i can only attend to positions 0, 1, ..., i
    
    Returns:
        mask: Shape (seq_len, seq_len), 1 where attention allowed, 0 otherwise
    """
    # TODO: Create lower triangular mask
    mask = None  # Replace
    return mask


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Attention Mechanisms\n")
    
    # Test 1: Softmax
    print("=" * 50)
    print("Test 1: Softmax")
    print("=" * 50)
    
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    
    if result is not None:
        print(f"Input: {x}")
        print(f"Softmax: {np.round(result, 4)}")
        print(f"Sum: {result.sum():.4f}")
        if abs(result.sum() - 1.0) < 0.01:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: Attention
    print("\n" + "=" * 50)
    print("Test 2: Scaled Dot-Product Attention")
    print("=" * 50)
    
    seq_len, d_k = 4, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    output, weights = attention(Q, K, V)
    
    if output is not None:
        print(f"Q shape: {Q.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Weights shape: {weights.shape}")
        print(f"Weights sum per row: {weights.sum(axis=-1)}")
        if output.shape == V.shape:
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 3: Causal Mask
    print("\n" + "=" * 50)
    print("Test 3: Causal Mask")
    print("=" * 50)
    
    mask = create_causal_mask(4)
    
    if mask is not None:
        print(f"Causal mask (4x4):\n{mask.astype(int)}")
        expected = np.array([[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]])
        if np.allclose(mask, expected):
            print("PASS")
    else:
        print("Not implemented yet")
