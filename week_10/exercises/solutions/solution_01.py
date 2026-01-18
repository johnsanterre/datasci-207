"""
DATASCI 207: Module 10 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def convolve2d(image, kernel):
    """Perform 2D convolution."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(patch * kernel)
    return output


def max_pool2d(image, pool_size=2):
    """Perform max pooling."""
    h, w = image.shape
    oh, ow = h // pool_size, w // pool_size
    
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = image[i*pool_size:(i+1)*pool_size,
                         j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(patch)
    return output


def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    """Calculate output size."""
    return (input_size - kernel_size + 2*padding) // stride + 1


if __name__ == "__main__":
    image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=float)
    
    identity = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float)
    conv = convolve2d(image, identity)
    assert np.allclose(conv, [[6,7],[10,11]])
    print("Convolution: VERIFIED")
    
    pool = max_pool2d(image, 2)
    assert np.allclose(pool, [[6,8],[14,16]])
    print("Max Pooling: VERIFIED")
    
    assert conv_output_size(28, 3, 1, 0) == 26
    print("Output Size: VERIFIED")
