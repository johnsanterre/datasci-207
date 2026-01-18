"""
DATASCI 207: Module 12 - Exercise: Fairness Metrics
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Compute Group-Level Metrics
# =============================================================================

def compute_group_metrics(y_true, y_pred, group, group_val):
    """
    Compute metrics for a specific group.
    
    Args:
        y_true: True labels (0/1)
        y_pred: Predicted labels (0/1)
        group: Group membership array
        group_val: Which group to compute for
    
    Returns:
        Dictionary with:
        - positive_rate: P(Y_pred=1)
        - tpr: True positive rate
        - fpr: False positive rate
    """
    # TODO: Implement group-level metrics
    # 1. Filter to only this group
    # 2. Compute positive_rate, tpr, fpr
    
    metrics = {
        'positive_rate': None,
        'tpr': None,
        'fpr': None
    }
    return metrics


# =============================================================================
# EXERCISE 2: Demographic Parity Difference
# =============================================================================

def demographic_parity_difference(y_pred, group):
    """
    Compute demographic parity difference.
    
    DP difference = |P(Y_pred=1|A=0) - P(Y_pred=1|A=1)|
    
    Returns:
        Absolute difference in positive rates
    """
    # TODO: Compute DP difference
    dp_diff = None
    return dp_diff


# =============================================================================
# EXERCISE 3: Equalized Odds Difference
# =============================================================================

def equalized_odds_difference(y_true, y_pred, group):
    """
    Compute equalized odds difference.
    
    Average of |TPR_0 - TPR_1| and |FPR_0 - FPR_1|
    
    Returns:
        Average absolute difference
    """
    # TODO: Compute EO difference
    eo_diff = None
    return eo_diff


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Fairness Metrics\n")
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    group = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
    y_true = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    
    # Biased predictions
    y_pred = y_true.copy()
    # Add bias against group 0
    bias_mask = (group == 0) & (y_true == 1)
    y_pred[bias_mask] = np.random.choice([0, 1], size=np.sum(bias_mask), p=[0.3, 0.7])
    
    # Test 1: Group Metrics
    print("=" * 50)
    print("Test 1: Group Metrics")
    print("=" * 50)
    
    m0 = compute_group_metrics(y_true, y_pred, group, 0)
    m1 = compute_group_metrics(y_true, y_pred, group, 1)
    
    if m0['positive_rate'] is not None:
        print(f"Group 0 positive rate: {m0['positive_rate']:.3f}")
        print(f"Group 1 positive rate: {m1['positive_rate']:.3f}")
        print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: Demographic Parity
    print("\n" + "=" * 50)
    print("Test 2: Demographic Parity Difference")
    print("=" * 50)
    
    dp = demographic_parity_difference(y_pred, group)
    if dp is not None:
        print(f"DP Difference: {dp:.3f}")
        print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 3: Equalized Odds
    print("\n" + "=" * 50)
    print("Test 3: Equalized Odds Difference")
    print("=" * 50)
    
    eo = equalized_odds_difference(y_true, y_pred, group)
    if eo is not None:
        print(f"EO Difference: {eo:.3f}")
        print("PASS")
    else:
        print("Not implemented yet")
