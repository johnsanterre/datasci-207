"""
DATASCI 207: Applied Machine Learning
Module 1 - Exercise 1: Train/Test Split and Evaluation

OBJECTIVE:
Practice splitting data and evaluating model performance.

INSTRUCTIONS:
1. Complete the functions marked with TODO
2. Run the file to check your answers
3. Compare your results with the expected output
"""


# =============================================================================
# EXERCISE 1: Implement a Train/Test Split
# =============================================================================

def train_test_split(data, labels, test_fraction=0.2):
    """
    Split data into training and test sets.
    
    Args:
        data: list of input values
        labels: list of corresponding output values
        test_fraction: fraction of data to use for testing (default 0.2 = 20%)
    
    Returns:
        train_data, train_labels, test_data, test_labels
    
    Example:
        If data has 10 items and test_fraction=0.2, 
        then 8 items go to training and 2 to testing.
    """
    # TODO: Calculate the split index
    # Hint: Use int() to convert to integer, and len() to get length
    split_index = None  # Replace with your code
    
    # TODO: Split the data and labels
    train_data = None    # Replace with your code
    train_labels = None  # Replace with your code
    test_data = None     # Replace with your code
    test_labels = None   # Replace with your code
    
    return train_data, train_labels, test_data, test_labels


# =============================================================================
# EXERCISE 2: Implement Mean Squared Error
# =============================================================================

def mean_squared_error(predictions, actuals):
    """
    Calculate the mean squared error between predictions and actual values.
    
    MSE = (1/n) * sum((pred - actual)^2)
    
    Args:
        predictions: list of predicted values
        actuals: list of actual values
    
    Returns:
        The mean squared error (a single number)
    """
    # TODO: Implement MSE
    # Hint: Loop through both lists, calculate squared differences, sum them, divide by n
    
    mse = None  # Replace with your code
    
    return mse


# =============================================================================
# EXERCISE 3: Implement Accuracy
# =============================================================================

def accuracy(predictions, actuals):
    """
    Calculate classification accuracy.
    
    Accuracy = (number of correct predictions) / (total predictions)
    
    Args:
        predictions: list of predicted labels
        actuals: list of actual labels
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    # TODO: Implement accuracy
    # Hint: Count how many predictions match actuals, divide by total
    
    acc = None  # Replace with your code
    
    return acc


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================

if __name__ == "__main__":
    print("Testing your implementations...\n")
    
    # Test 1: Train/Test Split
    print("=" * 50)
    print("Test 1: Train/Test Split")
    print("=" * 50)
    
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    train_d, train_l, test_d, test_l = train_test_split(data, labels, test_fraction=0.2)
    
    print(f"Original data length: {len(data)}")
    print(f"Train data: {train_d} (expected: [1, 2, 3, 4, 5, 6, 7, 8])")
    print(f"Test data: {test_d} (expected: [9, 10])")
    
    if train_d == [1, 2, 3, 4, 5, 6, 7, 8] and test_d == [9, 10]:
        print("PASS!")
    else:
        print("Not quite - check your implementation")
    
    # Test 2: Mean Squared Error
    print("\n" + "=" * 50)
    print("Test 2: Mean Squared Error")
    print("=" * 50)
    
    preds = [2.5, 0.0, 2.0, 8.0]
    actuals = [3.0, -0.5, 2.0, 7.0]
    
    mse = mean_squared_error(preds, actuals)
    expected_mse = 0.375  # ((0.5)^2 + (0.5)^2 + (0)^2 + (1)^2) / 4 = 1.5 / 4 = 0.375
    
    print(f"Predictions: {preds}")
    print(f"Actuals: {actuals}")
    print(f"Your MSE: {mse} (expected: {expected_mse})")
    
    if mse is not None and abs(mse - expected_mse) < 0.001:
        print("PASS!")
    else:
        print("Not quite - check your implementation")
    
    # Test 3: Accuracy
    print("\n" + "=" * 50)
    print("Test 3: Accuracy")
    print("=" * 50)
    
    pred_labels = ["cat", "dog", "cat", "dog", "cat"]
    actual_labels = ["cat", "dog", "dog", "dog", "cat"]
    
    acc = accuracy(pred_labels, actual_labels)
    expected_acc = 0.8  # 4 out of 5 correct
    
    print(f"Predictions: {pred_labels}")
    print(f"Actuals: {actual_labels}")
    print(f"Your accuracy: {acc} (expected: {expected_acc})")
    
    if acc is not None and abs(acc - expected_acc) < 0.001:
        print("PASS!")
    else:
        print("Not quite - check your implementation")
    
    print("\n" + "=" * 50)
    print("All tests complete!")
    print("=" * 50)
