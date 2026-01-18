"""
DATASCI 207: Applied Machine Learning
Module 1 - Exercise 1: SOLUTIONS

These are the solutions to the exercises. 
Compare with your own implementations after attempting them.
"""


# =============================================================================
# SOLUTION 1: Train/Test Split
# =============================================================================

def train_test_split(data, labels, test_fraction=0.2):
    """
    Split data into training and test sets.
    """
    # Calculate where to split
    # We want (1 - test_fraction) of data for training
    split_index = int(len(data) * (1 - test_fraction))
    
    # Split using list slicing
    train_data = data[:split_index]
    train_labels = labels[:split_index]
    test_data = data[split_index:]
    test_labels = labels[split_index:]
    
    return train_data, train_labels, test_data, test_labels


# =============================================================================
# SOLUTION 2: Mean Squared Error
# =============================================================================

def mean_squared_error(predictions, actuals):
    """
    Calculate the mean squared error between predictions and actual values.
    """
    total = 0
    n = len(predictions)
    
    for pred, actual in zip(predictions, actuals):
        diff = pred - actual
        total += diff ** 2
    
    mse = total / n
    return mse


# =============================================================================
# SOLUTION 3: Accuracy
# =============================================================================

def accuracy(predictions, actuals):
    """
    Calculate classification accuracy.
    """
    correct = 0
    total = len(predictions)
    
    for pred, actual in zip(predictions, actuals):
        if pred == actual:
            correct += 1
    
    return correct / total


# =============================================================================
# VERIFICATION
# =============================================================================

if __name__ == "__main__":
    # Verify solutions work
    
    # Test 1
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    train_d, train_l, test_d, test_l = train_test_split(data, labels, 0.2)
    assert train_d == [1, 2, 3, 4, 5, 6, 7, 8]
    assert test_d == [9, 10]
    print("Solution 1: VERIFIED")
    
    # Test 2
    preds = [2.5, 0.0, 2.0, 8.0]
    actuals = [3.0, -0.5, 2.0, 7.0]
    mse = mean_squared_error(preds, actuals)
    assert abs(mse - 0.375) < 0.001
    print("Solution 2: VERIFIED")
    
    # Test 3
    pred_labels = ["cat", "dog", "cat", "dog", "cat"]
    actual_labels = ["cat", "dog", "dog", "dog", "cat"]
    acc = accuracy(pred_labels, actual_labels)
    assert abs(acc - 0.8) < 0.001
    print("Solution 3: VERIFIED")
    
    print("\nAll solutions verified!")
