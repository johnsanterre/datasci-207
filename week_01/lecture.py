"""
DATASCI 207: Applied Machine Learning
Module 1: Introduction and Framing

This module introduces the core concepts of machine learning:
- What is a function and how do we represent it?
- What is a model and how does it learn?
- How do we evaluate if a model is good?
- What is generalization and why does it matter?

No external libraries needed for this module - just Python basics.
"""

# =============================================================================
# PART 1: FUNCTIONS - THE FOUNDATION OF ML
# =============================================================================

# In machine learning, we think of problems as functions.
# A function maps inputs to outputs: f(x) -> y

# Example: A simple function that predicts if a number is positive
def is_positive(x):
    """
    A simple deterministic function.
    Input: a number
    Output: True if positive, False otherwise
    """
    return x > 0

# Let's test it
print("=== Part 1: Functions ===")
print(f"is_positive(5) = {is_positive(5)}")    # True
print(f"is_positive(-3) = {is_positive(-3)}")  # False
print(f"is_positive(0) = {is_positive(0)}")    # False

# In ML, we often don't know the exact function.
# Instead, we have data (input-output pairs) and want to learn the function.


# =============================================================================
# PART 2: DATA AND INPUT/OUTPUT RELATIONSHIPS
# =============================================================================

# Let's create some simple data.
# Imagine we're predicting house prices based on square footage.

# Our "data" - pairs of (square_feet, price)
# In real ML, we'd load this from a file, but we'll keep it simple.
square_feet = [1000, 1500, 2000, 2500, 3000]
prices = [200000, 300000, 400000, 500000, 600000]

print("\n=== Part 2: Data ===")
print("Square Feet -> Price")
for sqft, price in zip(square_feet, prices):
    print(f"  {sqft} -> ${price:,}")

# Notice the pattern: price = 200 * square_feet
# The "true" function is: f(sqft) = 200 * sqft
# But in real problems, we don't know this - we have to learn it!


# =============================================================================
# PART 3: MODELS - LEARNED FUNCTIONS
# =============================================================================

# A model is our attempt to approximate the true function.
# Let's create a simple linear model: prediction = weight * input

def linear_model(x, weight):
    """
    A simple linear model with one parameter (weight).
    This is our hypothesis for what the function might be.
    """
    return weight * x

# Let's try different weights and see which works best
print("\n=== Part 3: Models ===")

# Try weight = 150 (wrong)
weight_guess_1 = 150
print(f"With weight = {weight_guess_1}:")
for sqft, actual_price in zip(square_feet, prices):
    predicted = linear_model(sqft, weight_guess_1)
    error = actual_price - predicted
    print(f"  {sqft} sqft: predicted ${predicted:,}, actual ${actual_price:,}, error ${error:,}")

# Try weight = 200 (correct!)
weight_guess_2 = 200
print(f"\nWith weight = {weight_guess_2}:")
for sqft, actual_price in zip(square_feet, prices):
    predicted = linear_model(sqft, weight_guess_2)
    error = actual_price - predicted
    print(f"  {sqft} sqft: predicted ${predicted:,}, actual ${actual_price:,}, error ${error:,}")


# =============================================================================
# PART 4: EVALUATION - HOW GOOD IS OUR MODEL?
# =============================================================================

# We need a way to measure how wrong our predictions are.
# This is called a "loss" or "error" metric.

def mean_absolute_error(predictions, actuals):
    """
    Calculate the average absolute difference between predictions and actuals.
    Lower is better. Zero means perfect predictions.
    """
    total_error = 0
    for pred, actual in zip(predictions, actuals):
        total_error += abs(pred - actual)
    return total_error / len(predictions)

def mean_squared_error(predictions, actuals):
    """
    Calculate the average squared difference.
    Squaring penalizes large errors more heavily.
    """
    total_error = 0
    for pred, actual in zip(predictions, actuals):
        total_error += (pred - actual) ** 2
    return total_error / len(predictions)

print("\n=== Part 4: Evaluation ===")

# Evaluate weight = 150
preds_1 = [linear_model(x, 150) for x in square_feet]
mae_1 = mean_absolute_error(preds_1, prices)
mse_1 = mean_squared_error(preds_1, prices)
print(f"Weight = 150: MAE = ${mae_1:,.0f}, MSE = {mse_1:,.0f}")

# Evaluate weight = 200
preds_2 = [linear_model(x, 200) for x in square_feet]
mae_2 = mean_absolute_error(preds_2, prices)
mse_2 = mean_squared_error(preds_2, prices)
print(f"Weight = 200: MAE = ${mae_2:,.0f}, MSE = {mse_2:,.0f}")

# Weight = 200 has zero error - it's the correct model!


# =============================================================================
# PART 5: TRAIN/TEST SPLIT - MEASURING GENERALIZATION
# =============================================================================

# The real question: will our model work on NEW data it hasn't seen?
# This is called "generalization."

# To test this, we split our data:
# - Training set: used to learn the model
# - Test set: held out to evaluate generalization

# Here's a larger dataset
all_sqft = [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500]
all_prices = [200000, 240000, 300000, 360000, 400000, 440000, 500000, 560000, 600000, 700000]

# Split: first 7 for training, last 3 for testing
train_sqft = all_sqft[:7]
train_prices = all_prices[:7]
test_sqft = all_sqft[7:]
test_prices = all_prices[7:]

print("\n=== Part 5: Train/Test Split ===")
print(f"Training data: {len(train_sqft)} examples")
print(f"Test data: {len(test_sqft)} examples")

# "Train" our model (in this case, we just pick weight = 200)
learned_weight = 200

# Evaluate on training data
train_preds = [linear_model(x, learned_weight) for x in train_sqft]
train_mae = mean_absolute_error(train_preds, train_prices)
print(f"\nTraining MAE: ${train_mae:,.0f}")

# Evaluate on test data (data the model hasn't "seen")
test_preds = [linear_model(x, learned_weight) for x in test_sqft]
test_mae = mean_absolute_error(test_preds, test_prices)
print(f"Test MAE: ${test_mae:,.0f}")

# If test error is similar to training error, the model generalizes well!


# =============================================================================
# PART 6: OVERFITTING - WHEN MODELS MEMORIZE
# =============================================================================

# Overfitting happens when a model learns the training data too well,
# including its noise and peculiarities, and fails on new data.

# Let's simulate this with a "memorizing" model
class MemorizingModel:
    """
    A model that just memorizes the training data exactly.
    It looks up answers instead of learning patterns.
    """
    def __init__(self):
        self.memory = {}
    
    def train(self, inputs, outputs):
        # Just store each input-output pair
        for x, y in zip(inputs, outputs):
            self.memory[x] = y
    
    def predict(self, x):
        # If we've seen this input, return the memorized answer
        if x in self.memory:
            return self.memory[x]
        else:
            # For unseen inputs, just guess the average
            return sum(self.memory.values()) / len(self.memory)

print("\n=== Part 6: Overfitting ===")

memorizer = MemorizingModel()
memorizer.train(train_sqft, train_prices)

# Perfect on training data!
train_preds_mem = [memorizer.predict(x) for x in train_sqft]
train_mae_mem = mean_absolute_error(train_preds_mem, train_prices)
print(f"Memorizing model - Training MAE: ${train_mae_mem:,.0f}")

# But poor on test data (it hasn't seen these exact values)
test_preds_mem = [memorizer.predict(x) for x in test_sqft]
test_mae_mem = mean_absolute_error(test_preds_mem, test_prices)
print(f"Memorizing model - Test MAE: ${test_mae_mem:,.0f}")

# The memorizing model overfits! It has zero training error but high test error.
# Our simple linear model generalizes better.


# =============================================================================
# PART 7: BASELINES - SETTING EXPECTATIONS
# =============================================================================

# A baseline is a simple, often naive model that sets a minimum bar.
# Your real model should beat the baseline, or why use ML at all?

# Common baselines:
# - Predict the mean (for regression)
# - Predict the most common class (for classification)
# - Predict the previous value (for time series)

def mean_baseline(train_outputs):
    """
    A baseline that always predicts the average of training outputs.
    """
    return sum(train_outputs) / len(train_outputs)

print("\n=== Part 7: Baselines ===")

# Calculate baseline prediction (mean of training prices)
baseline_pred = mean_baseline(train_prices)
print(f"Baseline prediction (mean): ${baseline_pred:,.0f}")

# Evaluate baseline on test data
baseline_preds = [baseline_pred] * len(test_prices)
baseline_mae = mean_absolute_error(baseline_preds, test_prices)
print(f"Baseline Test MAE: ${baseline_mae:,.0f}")

# Compare to our linear model
print(f"Linear Model Test MAE: ${test_mae:,.0f}")
print(f"\nOur model improves over baseline by ${baseline_mae - test_mae:,.0f}")


# =============================================================================
# PART 8: ACCURACY FOR CLASSIFICATION
# =============================================================================

# For classification problems, we often use accuracy instead of MAE.
# Accuracy = (correct predictions) / (total predictions)

def accuracy(predictions, actuals):
    """
    Calculate what fraction of predictions are correct.
    Returns a value between 0 and 1. Higher is better.
    """
    correct = 0
    for pred, actual in zip(predictions, actuals):
        if pred == actual:
            correct += 1
    return correct / len(predictions)

print("\n=== Part 8: Classification Accuracy ===")

# Example: predicting if a house is "expensive" (price > 400000)
actual_labels = ["cheap", "cheap", "cheap", "expensive", "expensive", "expensive"]
predicted_labels = ["cheap", "cheap", "expensive", "cheap", "expensive", "expensive"]

acc = accuracy(predicted_labels, actual_labels)
print(f"Predictions: {predicted_labels}")
print(f"Actual:      {actual_labels}")
print(f"Accuracy: {acc:.2%}")  # 66.67%


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 1")
print("=" * 60)
print("""
1. FUNCTION: ML problems are about learning functions f(x) -> y

2. MODEL: A model is our approximation of the true function

3. EVALUATION: We measure model quality with metrics (MAE, MSE, accuracy)

4. TRAIN/TEST SPLIT: 
   - Train on some data
   - Test on held-out data to check generalization

5. OVERFITTING: When a model memorizes training data but fails on new data

6. BASELINE: A simple reference model your real model should beat

7. GENERALIZATION: The ultimate goal - perform well on unseen data
""")
