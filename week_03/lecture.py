"""
DATASCI 207: Applied Machine Learning
Module 3: Feature Engineering

This module covers:
- Multivariate linear regression with matrix notation
- Feature scaling (normalization, standardization)
- Handling missing data
- Encoding categorical variables
- Feature crosses
- Feature selection basics
- The curse of dimensionality

Using NumPy only - no pandas.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: MULTIVARIATE LINEAR REGRESSION
# =============================================================================

# With multiple features, our model becomes:
#   y = w1*x1 + w2*x2 + ... + wn*xn + b
#
# In matrix notation:
#   y = X @ w + b
# 
# Where X is (n_samples, n_features) and w is (n_features,)

print("=== Part 1: Multivariate Linear Regression ===\n")

# Create data with 3 features
n_samples = 100
n_features = 3

# Random features
X = np.random.randn(n_samples, n_features)

# True weights and bias
true_weights = np.array([2.0, -1.5, 0.5])
true_bias = 3.0

# Generate targets: y = X @ w + b + noise
y = X @ true_weights + true_bias + np.random.normal(0, 0.5, n_samples)

print(f"Data shape: X = {X.shape}, y = {y.shape}")
print(f"True weights: {true_weights}")
print(f"True bias: {true_bias}")


def predict_multivariate(X, weights, bias):
    """
    Predictions for multivariate linear regression.
    
    y = X @ weights + bias
    
    Args:
        X: shape (n_samples, n_features)
        weights: shape (n_features,)
        bias: scalar
    
    Returns:
        predictions: shape (n_samples,)
    """
    return X @ weights + bias


def train_multivariate(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Train multivariate linear regression using gradient descent.
    """
    n_samples, n_features = X.shape
    
    # Initialize
    weights = np.zeros(n_features)
    bias = 0.0
    
    for i in range(n_iterations):
        # Predictions
        y_pred = predict_multivariate(X, weights, bias)
        
        # Errors
        errors = y_pred - y
        
        # Gradients
        # dL/dw = (2/n) * X.T @ errors
        # dL/db = (2/n) * sum(errors)
        grad_weights = (2 / n_samples) * (X.T @ errors)
        grad_bias = (2 / n_samples) * np.sum(errors)
        
        # Update
        weights = weights - learning_rate * grad_weights
        bias = bias - learning_rate * grad_bias
        
        if i % 200 == 0:
            mse = np.mean(errors ** 2)
            print(f"Iteration {i}: MSE = {mse:.4f}")
    
    return weights, bias


learned_weights, learned_bias = train_multivariate(X, y)
print(f"\nLearned weights: {learned_weights}")
print(f"Learned bias: {learned_bias:.4f}")


# =============================================================================
# PART 2: FEATURE SCALING
# =============================================================================

# Features on different scales can cause problems:
# - Gradient descent converges slowly
# - Features with large values dominate

print("\n=== Part 2: Feature Scaling ===\n")

# Example: Features on very different scales
feature_1 = np.array([1000, 2000, 3000, 4000, 5000])    # Square feet
feature_2 = np.array([1, 2, 3, 4, 5])                    # Bedrooms
feature_3 = np.array([50000, 60000, 70000, 80000, 90000])  # Price

print("Original features:")
print(f"  Square feet: {feature_1}")
print(f"  Bedrooms: {feature_2}")
print(f"  Price: {feature_3}")


# --- Min-Max Normalization (scales to [0, 1]) ---

def min_max_normalize(X):
    """
    Scale features to range [0, 1].
    
    X_norm = (X - X_min) / (X_max - X_min)
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)  # Add small value to avoid division by zero


print("\nMin-Max Normalized:")
print(f"  Square feet: {min_max_normalize(feature_1)}")
print(f"  Bedrooms: {min_max_normalize(feature_2)}")
print(f"  Price: {min_max_normalize(feature_3)}")


# --- Z-Score Standardization (mean=0, std=1) ---

def z_score_standardize(X):
    """
    Standardize features to have mean=0 and std=1.
    
    X_std = (X - mean) / std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)


print("\nZ-Score Standardized:")
print(f"  Square feet: {z_score_standardize(feature_1)}")
print(f"  Bedrooms: {z_score_standardize(feature_2)}")


# --- Log Scaling (for skewed distributions) ---

def log_scale(X):
    """
    Apply log transformation.
    Useful for skewed data with a long tail.
    """
    return np.log1p(X)  # log(1 + X) to handle zeros


# Example: highly skewed data
skewed_data = np.array([1, 2, 5, 10, 100, 1000, 10000])
print(f"\nLog scaling skewed data:")
print(f"  Original: {skewed_data}")
print(f"  Log scaled: {log_scale(skewed_data)}")


# =============================================================================
# PART 3: HANDLING MISSING DATA
# =============================================================================

print("\n=== Part 3: Handling Missing Data ===\n")

# In numpy, we can use np.nan for missing values
data_with_missing = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0])
print(f"Data with missing values: {data_with_missing}")

# Strategy 1: Remove rows with missing values
def remove_missing(X):
    """Remove rows containing NaN."""
    return X[~np.isnan(X)]

print(f"After removal: {remove_missing(data_with_missing)}")

# Strategy 2: Fill with mean
def fill_with_mean(X):
    """Replace NaN with the mean of non-NaN values."""
    mean_val = np.nanmean(X)  # Mean ignoring NaN
    filled = X.copy()
    filled[np.isnan(filled)] = mean_val
    return filled

print(f"Filled with mean: {fill_with_mean(data_with_missing)}")

# Strategy 3: Fill with median (more robust to outliers)
def fill_with_median(X):
    """Replace NaN with the median."""
    median_val = np.nanmedian(X)
    filled = X.copy()
    filled[np.isnan(filled)] = median_val
    return filled

print(f"Filled with median: {fill_with_median(data_with_missing)}")


# =============================================================================
# PART 4: ENCODING CATEGORICAL VARIABLES
# =============================================================================

print("\n=== Part 4: Encoding Categorical Variables ===\n")

# Categorical data cannot be used directly in most ML models.
# We need to convert them to numbers.

# Example: Color categories
colors = ["red", "blue", "green", "red", "blue", "green", "red"]


# --- One-Hot Encoding ---
# Each category becomes a binary column

def one_hot_encode(categories):
    """
    Convert categorical variable to one-hot encoding.
    
    Returns:
        encoded: numpy array of shape (n_samples, n_categories)
        category_names: list of category names
    """
    unique_categories = sorted(set(categories))
    n_samples = len(categories)
    n_categories = len(unique_categories)
    
    # Create mapping
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    
    # Create one-hot matrix
    encoded = np.zeros((n_samples, n_categories))
    for i, cat in enumerate(categories):
        encoded[i, category_to_idx[cat]] = 1
    
    return encoded, unique_categories


encoded, category_names = one_hot_encode(colors)
print(f"One-hot encoding:")
print(f"Categories: {category_names}")
print(f"Encoded:\n{encoded}")


# --- Label Encoding ---
# Each category becomes an integer (use with caution - implies ordering)

def label_encode(categories):
    """Convert categories to integer labels."""
    unique_categories = sorted(set(categories))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    return np.array([category_to_idx[cat] for cat in categories])


print(f"\nLabel encoding:")
print(f"Original: {colors}")
print(f"Encoded: {label_encode(colors)}")
print("Warning: Label encoding implies an ordering that may not exist!")


# =============================================================================
# PART 5: FEATURE CROSSES
# =============================================================================

print("\n=== Part 5: Feature Crosses ===\n")

# Feature crosses capture interactions between features.
# Example: The effect of location might depend on size.

# Original features
sizes = np.array([1000, 1500, 2000, 1000, 1500, 2000])  # Square feet
locations = np.array(["urban", "urban", "urban", "rural", "rural", "rural"])

print(f"Sizes: {sizes}")
print(f"Locations: {locations}")

# Create feature cross: size_location
# This captures that a 1000 sqft urban house is different from 1000 sqft rural

def create_feature_cross(feature1, feature2):
    """Create crossed features by combining two features."""
    # For categorical x numerical: encode categorical, then multiply
    # For simplicity, we'll create string combinations
    return [f"{f1}_{f2}" for f1, f2 in zip(feature1, feature2)]


crossed = create_feature_cross(locations, sizes)
print(f"Feature cross (location_size): {crossed}")

# Then one-hot encode the crossed feature
encoded_cross, cross_names = one_hot_encode(crossed)
print(f"Crossed feature categories: {cross_names}")

# For numerical features, we can simply multiply
# This captures interaction: if x1 and x2 are both high, the product is very high

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([5, 4, 3, 2, 1])
interaction = x1 * x2
print(f"\nNumerical interaction x1 * x2: {interaction}")


# =============================================================================
# PART 6: BUCKETING (BINNING)
# =============================================================================

print("\n=== Part 6: Bucketing / Binning ===\n")

# Convert continuous values into discrete buckets.
# Useful when the relationship is non-linear or for categorical treatment.

ages = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85])

def bucketize(values, bin_edges):
    """
    Assign values to buckets based on bin edges.
    
    Returns bucket index for each value.
    """
    return np.digitize(values, bin_edges)

# Define age buckets: [0-18), [18-35), [35-55), [55+)
age_bins = [18, 35, 55]
bucketed_ages = bucketize(ages, age_bins)

print(f"Ages: {ages}")
print(f"Bucket edges: {age_bins}")
print(f"Bucket assignments: {bucketed_ages}")
print("(0=child, 1=young adult, 2=middle age, 3=senior)")


# =============================================================================
# PART 7: CURSE OF DIMENSIONALITY
# =============================================================================

print("\n=== Part 7: Curse of Dimensionality ===\n")

# As the number of features increases:
# 1. Data becomes sparse (points are far apart)
# 2. More data is needed to cover the space
# 3. Distance metrics become less meaningful
# 4. Overfitting becomes more likely

# Demonstration: Volume of hypersphere vs hypercube

def hypersphere_volume_ratio(n_dimensions):
    """
    Ratio of inscribed hypersphere volume to hypercube volume.
    Shows how data concentrates in corners as dimensions increase.
    """
    # Volume of hypersphere with radius 0.5 (inscribed in unit hypercube)
    from math import pi, gamma
    r = 0.5
    sphere_vol = (pi ** (n_dimensions / 2)) / gamma(n_dimensions / 2 + 1) * (r ** n_dimensions)
    cube_vol = 1.0  # Unit hypercube
    return sphere_vol / cube_vol


print("Hypersphere to hypercube volume ratio:")
for d in [1, 2, 3, 5, 10, 20, 50]:
    ratio = hypersphere_volume_ratio(d)
    print(f"  {d:2d} dimensions: {ratio:.6f}")

print("\nAs dimensions increase, most of the volume is in the corners!")
print("This makes distance-based methods less effective.")


# =============================================================================
# PART 8: FEATURE SELECTION BASICS
# =============================================================================

print("\n=== Part 8: Feature Selection ===\n")

# Not all features are useful. Some may:
# - Be irrelevant to the target
# - Be redundant with other features
# - Add noise

# Simple method: Correlation with target

def correlation_with_target(X, y):
    """Calculate correlation of each feature with target."""
    n_features = X.shape[1]
    correlations = []
    for i in range(n_features):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(corr)
    return np.array(correlations)


# Create data with one useful feature and one random feature
n = 100
useful_feature = np.random.randn(n)
random_feature = np.random.randn(n)
target = 3 * useful_feature + np.random.randn(n) * 0.5  # Only depends on useful_feature

X_selection = np.column_stack([useful_feature, random_feature])
correlations = correlation_with_target(X_selection, target)

print("Feature correlations with target:")
print(f"  Useful feature: {correlations[0]:.3f}")
print(f"  Random feature: {correlations[1]:.3f}")
print("\nHigher absolute correlation suggests more predictive power.")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 3")
print("=" * 60)
print("""
1. MULTIVARIATE REGRESSION: y = X @ w + b
   - Matrix notation generalizes to any number of features

2. FEATURE SCALING:
   - Min-Max: Scale to [0, 1]
   - Z-Score: Scale to mean=0, std=1
   - Log: For skewed distributions

3. MISSING DATA:
   - Remove, fill with mean/median, or use indicator

4. CATEGORICAL ENCODING:
   - One-hot: Safe, no implied ordering
   - Label: Compact, but implies ordering

5. FEATURE CROSSES:
   - Capture interactions between features
   - Can dramatically improve model capacity

6. BUCKETING:
   - Convert continuous to discrete
   - Captures non-linear relationships

7. CURSE OF DIMENSIONALITY:
   - More features = sparser data
   - Need exponentially more data
   - Important to select relevant features
""")
