"""
DATASCI 207: Applied Machine Learning
Module 13: Transformers and Attention

This module covers:
- Attention mechanisms
- Self-attention and multi-head attention
- Transformer architecture
- BERT and GPT overview

Using NumPy for scratch implementations.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: BASIC ATTENTION
# =============================================================================

print("=== Part 1: Basic Attention ===\n")


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention(query, keys, values):
    """
    Simple attention mechanism.
    
    Args:
        query: Shape (d,)
        keys: Shape (n, d)
        values: Shape (n, d_v)
    
    Returns:
        Weighted sum of values
    """
    # Compute similarity scores (dot product)
    scores = np.dot(keys, query)  # Shape (n,)
    
    # Convert to probabilities
    weights = softmax(scores)
    
    # Weighted sum of values
    output = np.dot(weights, values)  # Shape (d_v,)
    
    return output, weights


# Example
d = 4
n = 5

query = np.random.randn(d)
keys = np.random.randn(n, d)
values = np.random.randn(n, d)

output, weights = attention(query, keys, values)

print(f"Query shape: {query.shape}")
print(f"Keys shape: {keys.shape}")
print(f"Attention weights: {np.round(weights, 3)}")
print(f"Output shape: {output.shape}")


# =============================================================================
# PART 2: SCALED DOT-PRODUCT ATTENTION
# =============================================================================

print("\n=== Part 2: Scaled Dot-Product Attention ===\n")


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention (batched).
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Queries, shape (..., seq_len_q, d_k)
        K: Keys, shape (..., seq_len_k, d_k)
        V: Values, shape (..., seq_len_k, d_v)
        mask: Optional mask
    
    Returns:
        Output and attention weights
    """
    d_k = K.shape[-1]
    
    # Compute attention scores
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)
    
    # Apply mask (for causal attention)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Softmax over keys
    weights = softmax(scores, axis=-1)
    
    # Weighted sum
    output = np.matmul(weights, V)
    
    return output, weights


# Batch example
batch_size = 2
seq_len = 4
d_k = 8
d_v = 8

Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Q shape: {Q.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nSample attention weights (batch 0):")
print(np.round(weights[0], 2))


# =============================================================================
# PART 3: CAUSAL (MASKED) ATTENTION
# =============================================================================

print("\n=== Part 3: Causal Attention ===\n")

# Create causal mask (lower triangular)
def create_causal_mask(seq_len):
    """Create mask where position i can only attend to positions <= i."""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


causal_mask = create_causal_mask(seq_len)
print("Causal mask (1 = can attend, 0 = cannot):")
print(causal_mask.astype(int))

output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, causal_mask)
print(f"\nCausal attention weights (batch 0):")
print(np.round(weights_causal[0], 2))
print("\nNote: Each row only has non-zero weights for positions at or before it.")


# =============================================================================
# PART 4: MULTI-HEAD ATTENTION CONCEPT
# =============================================================================

print("\n=== Part 4: Multi-Head Attention ===\n")


def multi_head_attention(Q, K, V, num_heads, d_model):
    """
    Simplified multi-head attention.
    
    Note: This is conceptual; real implementations use better projections.
    """
    batch_size, seq_len, _ = Q.shape
    d_k = d_model // num_heads
    
    # Initialize projection matrices
    W_Q = np.random.randn(num_heads, d_model, d_k) * 0.1
    W_K = np.random.randn(num_heads, d_model, d_k) * 0.1
    W_V = np.random.randn(num_heads, d_model, d_k) * 0.1
    W_O = np.random.randn(num_heads * d_k, d_model) * 0.1
    
    heads = []
    
    for h in range(num_heads):
        # Project
        Q_h = np.matmul(Q, W_Q[h])
        K_h = np.matmul(K, W_K[h])
        V_h = np.matmul(V, W_V[h])
        
        # Attention
        head_output, _ = scaled_dot_product_attention(Q_h, K_h, V_h)
        heads.append(head_output)
    
    # Concatenate heads
    concat = np.concatenate(heads, axis=-1)
    
    # Final projection
    output = np.matmul(concat, W_O)
    
    return output


d_model = 16
num_heads = 4

Q_mh = np.random.randn(batch_size, seq_len, d_model)
K_mh = np.random.randn(batch_size, seq_len, d_model)
V_mh = np.random.randn(batch_size, seq_len, d_model)

mh_output = multi_head_attention(Q_mh, K_mh, V_mh, num_heads, d_model)
print(f"Multi-head attention output shape: {mh_output.shape}")
print(f"Number of heads: {num_heads}")


# =============================================================================
# PART 5: POSITIONAL ENCODING
# =============================================================================

print("\n=== Part 5: Positional Encoding ===\n")


def positional_encoding(seq_len, d_model):
    """
    Sinusoidal positional encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]
    
    angles = positions / np.power(10000, (2 * (dimensions // 2)) / d_model)
    
    # Apply sin to even indices, cos to odd
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe


pe = positional_encoding(10, 16)
print("Positional encoding captures position information:")
print(f"Shape: {pe.shape}")
print(f"First 4 positions, first 4 dims:")
print(np.round(pe[:4, :4], 3))


# =============================================================================
# PART 6: TRANSFORMER ENCODER BLOCK (CONCEPTUAL)
# =============================================================================

print("\n=== Part 6: Transformer Block ===\n")

print("""
Transformer Encoder Block:

Input (seq_len × d_model)
    ↓
┌───────────────────────────────────────┐
│  Multi-Head Self-Attention            │
│  (Each position attends to all)       │
└───────────────────────────────────────┘
    ↓
Add & Layer Norm (residual connection)
    ↓
┌───────────────────────────────────────┐
│  Feed-Forward Network                 │
│  FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂       │
│  Applies to each position separately  │
└───────────────────────────────────────┘
    ↓
Add & Layer Norm
    ↓
Output (seq_len × d_model)

Key innovations:
1. Self-attention: O(1) path length between any positions
2. Residual connections: Better gradient flow
3. Layer norm: Stabilizes training
4. Parallelizable: Unlike RNNs
""")


# =============================================================================
# PART 7: BERT VS GPT
# =============================================================================

print("\n=== Part 7: BERT vs GPT ===\n")

print("""
BERT (Bidirectional Encoder):
- Uses ENCODER architecture
- Bidirectional: each token sees all tokens
- Pre-training: Masked Language Model (MLM)
  - Mask 15% of tokens, predict them
- Use cases: Classification, NER, QA
- Example: bert-base-uncased (110M params)

GPT (Generative Pre-Training):
- Uses DECODER architecture  
- Autoregressive: each token only sees previous tokens
- Pre-training: Next token prediction
- Use cases: Text generation, completion
- Example: GPT-3 (175B params)

Key difference:
- BERT: Good at UNDERSTANDING text
- GPT: Good at GENERATING text
""")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 13")
print("=" * 60)
print("""
1. ATTENTION allows modeling long-range dependencies
   - Query: what to look for
   - Key: what's available
   - Value: what to retrieve

2. SELF-ATTENTION: token attends to all tokens
   Attention(Q,K,V) = softmax(QK^T/√d_k)V

3. MULTI-HEAD: multiple attention patterns in parallel
   - Different heads learn different relationships
   
4. TRANSFORMER ARCHITECTURE:
   - Self-attention + Feed-Forward + Residual + LayerNorm
   - Parallelizable (unlike RNNs)
   - Positional encoding for sequence order

5. BERT (Encoder):
   - Bidirectional, good for understanding
   - Pre-trained with masked language modeling

6. GPT (Decoder):
   - Autoregressive, good for generation
   - Pre-trained with next-token prediction
   - Causal masking

7. TRANSFER LEARNING:
   - Pre-train on large corpus
   - Fine-tune on specific task
   - Hugging Face makes it easy!
""")
