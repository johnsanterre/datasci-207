"""
DATASCI 207: Applied Machine Learning
Module 10: Convolutional Neural Networks

This module covers:
- Convolution operations
- Pooling layers
- CNN architectures
- Transfer learning

Using NumPy and TensorFlow/Keras.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: CONVOLUTION FROM SCRATCH
# =============================================================================

print("=== Part 1: Convolution Operation ===\n")


def convolve2d(image, kernel):
    """
    Simple 2D convolution (no padding, stride 1).
    
    Args:
        image: 2D array (height, width)
        kernel: 2D array (kh, kw)
    
    Returns:
        2D array of convolved output
    """
    ih, iw = image.shape
    kh, kw = kernel.shape
    
    # Output dimensions
    oh = ih - kh + 1
    ow = iw - kw + 1
    
    output = np.zeros((oh, ow))
    
    for i in range(oh):
        for j in range(ow):
            # Extract patch
            patch = image[i:i+kh, j:j+kw]
            # Element-wise multiply and sum
            output[i, j] = np.sum(patch * kernel)
    
    return output


# Example: Edge detection
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=float)

# Horizontal edge detector
horizontal_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=float)

# Vertical edge detector
vertical_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

print("Input image:")
print(image)
print("\nHorizontal edge detection:")
print(convolve2d(image, horizontal_kernel))
print("\nVertical edge detection:")
print(convolve2d(image, vertical_kernel))


# =============================================================================
# PART 2: MAX POOLING
# =============================================================================

print("\n=== Part 2: Max Pooling ===\n")


def max_pool2d(image, pool_size=2):
    """
    Max pooling with non-overlapping regions.
    
    Args:
        image: 2D array
        pool_size: Size of pooling region
    
    Returns:
        Pooled array
    """
    h, w = image.shape
    oh = h // pool_size
    ow = w // pool_size
    
    output = np.zeros((oh, ow))
    
    for i in range(oh):
        for j in range(ow):
            patch = image[i*pool_size:(i+1)*pool_size, 
                         j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(patch)
    
    return output


feature_map = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=float)

print("Feature map:")
print(feature_map)
print("\nAfter 2x2 max pooling:")
print(max_pool2d(feature_map, 2))


# =============================================================================
# PART 3: CNN OUTPUT SIZE CALCULATION
# =============================================================================

print("\n=== Part 3: Output Size Calculation ===\n")


def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    """Calculate output size after convolution."""
    return (input_size - kernel_size + 2 * padding) // stride + 1


print("Output size formula: (input - kernel + 2*padding) / stride + 1")
print()

# Examples
print("Example 1: 28x28 input, 3x3 kernel, stride=1, no padding")
print(f"  Output: {conv_output_size(28, 3, 1, 0)}x{conv_output_size(28, 3, 1, 0)}")

print("\nExample 2: 28x28 input, 3x3 kernel, stride=1, padding=1 (same)")
print(f"  Output: {conv_output_size(28, 3, 1, 1)}x{conv_output_size(28, 3, 1, 1)}")

print("\nExample 3: 28x28 input, 2x2 max pool, stride=2")
print(f"  Output: {conv_output_size(28, 2, 2, 0)}x{conv_output_size(28, 2, 2, 0)}")


# =============================================================================
# PART 4: CNN IN KERAS
# =============================================================================

print("\n=== Part 4: CNN with Keras ===\n")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Build a simple CNN for MNIST-like data (28x28 grayscale)
    model = keras.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third conv block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Classification head
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    print("CNN Architecture:")
    model.summary()
    
    # Count parameters
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    print(f"\nTotal trainable parameters: {trainable:,}")
    
except ImportError:
    print("(TensorFlow not available)")


# =============================================================================
# PART 5: TRANSFER LEARNING CONCEPT
# =============================================================================

print("\n=== Part 5: Transfer Learning ===\n")

print("""
Transfer Learning with Pre-trained Models:

1. FEATURE EXTRACTION
   - Load pre-trained model (VGG16, ResNet, etc.)
   - Remove classification head
   - Freeze all layers
   - Add new classification layers
   - Train only new layers

2. FINE-TUNING
   - After feature extraction works
   - Unfreeze top layers of base model
   - Train with very low learning rate
   - Fine-tune features for your task

Why it works:
- Early layers: edges, colors, textures (universal)
- Middle layers: shapes, parts (somewhat universal)
- Late layers: task-specific patterns
""")

try:
    # Example: Loading VGG16 for transfer learning
    base_model = keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    transfer_model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    print("Transfer Learning Model:")
    print(f"  Base model params: {base_model.count_params():,}")
    print(f"  Trainable params: {sum([np.prod(v.shape) for v in transfer_model.trainable_weights]):,}")
    print("  (Only training the new layers!)")
    
except Exception as e:
    print(f"(Could not load VGG16: {e})")


# =============================================================================
# PART 6: DATA AUGMENTATION
# =============================================================================

print("\n=== Part 6: Data Augmentation ===\n")

print("""
Data Augmentation creates variations of training images:

Common augmentations:
- RandomFlip (horizontal/vertical)
- RandomRotation
- RandomZoom
- RandomTranslation
- RandomContrast
- RandomBrightness

Benefits:
1. More effective training data
2. Reduces overfitting
3. Invariance to real-world variations
""")

try:
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ])
    
    print("Augmentation pipeline created successfully!")
    print("Apply during training: model.fit(augmented_data, ...)")
    
except:
    pass


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 10")
print("=" * 60)
print("""
1. CONVOLUTION: Slide filter across image
   - Detects local patterns (edges, textures)
   - Parameters shared across positions
   - Multiple filters = multiple features

2. POOLING: Downsample feature maps
   - Max pooling most common
   - Provides translation invariance
   - Reduces computation

3. CNN ARCHITECTURE:
   - [Conv -> ReLU -> Pool] blocks
   - Then Dense layers for classification
   - Deep layers = higher-level features

4. TRANSFER LEARNING:
   - Reuse pre-trained models (VGG, ResNet)
   - Feature extraction: freeze base, train head
   - Fine-tuning: unfreeze top layers

5. DATA AUGMENTATION:
   - Create image variations
   - Reduces overfitting
   - Use Keras preprocessing layers
""")
