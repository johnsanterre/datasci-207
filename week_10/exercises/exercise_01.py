"""
DATASCI 207: Module 10 - Exercise: CNN Operations
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Implement 2D Convolution
# =============================================================================

def convolve2d(image, kernel):
    """
    Perform 2D convolution (no padding, stride 1).
    
    Args:
        image: 2D array (height, width)
        kernel: 2D array (kh, kw)
    
    Returns:
        2D array of convolved output
    """
    # TODO: Implement convolution
    # 1. Calculate output dimensions
    # 2. Slide kernel across image
    # 3. At each position, compute dot product
    
    output = None  # Replace
    
    return output


# =============================================================================
# EXERCISE 2: Implement Max Pooling
# =============================================================================

def max_pool2d(image, pool_size=2):
    """
    Perform 2x2 max pooling with stride 2.
    
    Args:
        image: 2D array
        pool_size: Size of pooling window
    
    Returns:
        Pooled 2D array
    """
    # TODO: Implement max pooling
    # 1. Calculate output dimensions
    # 2. For each pool_size x pool_size region
    # 3. Take the maximum value
    
    output = None  # Replace
    
    return output


# =============================================================================
# EXERCISE 3: Calculate Output Size
# =============================================================================

def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    """
    Calculate output size after convolution.
    
    Formula: (input - kernel + 2*padding) / stride + 1
    """
    # TODO: Implement the formula
    output_size = None  # Replace
    
    return output_size


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing CNN Operations\n")
    
    # Test 1: Convolution
    print("=" * 50)
    print("Test 1: 2D Convolution")
    print("=" * 50)
    
    image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=float)
    
    identity_kernel = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=float)
    
    conv_result = convolve2d(image, identity_kernel)
    if conv_result is not None:
        print(f"Input:\n{image}")
        print(f"\nIdentity kernel convolution:\n{conv_result}")
        # With identity kernel, center values should be preserved
        expected = np.array([[6, 7], [10, 11]])
        if conv_result.shape == (2, 2) and np.allclose(conv_result, expected):
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: Max Pooling
    print("\n" + "=" * 50)
    print("Test 2: Max Pooling")
    print("=" * 50)
    
    pool_result = max_pool2d(image, pool_size=2)
    if pool_result is not None:
        print(f"Input:\n{image}")
        print(f"\n2x2 Max Pooling:\n{pool_result}")
        expected = np.array([[6, 8], [14, 16]])
        if np.allclose(pool_result, expected):
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 3: Output Size
    print("\n" + "=" * 50)
    print("Test 3: Output Size Calculation")
    print("=" * 50)
    
    size = conv_output_size(28, 3, 1, 0)
    if size is not None:
        print(f"28x28 input, 3x3 kernel, stride=1, no padding: {size}")
        print(f"Expected: 26")
        if size == 26:
            print("PASS")
    else:
        print("Not implemented yet")
