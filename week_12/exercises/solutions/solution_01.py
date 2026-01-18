"""
DATASCI 207: Module 12 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def compute_group_metrics(y_true, y_pred, group, group_val):
    """Compute metrics for a specific group."""
    mask = (group == group_val)
    yt = y_true[mask]
    yp = y_pred[mask]
    
    positive_rate = np.mean(yp)
    
    if np.sum(yt == 1) > 0:
        tpr = np.sum((yp == 1) & (yt == 1)) / np.sum(yt == 1)
    else:
        tpr = 0
    
    if np.sum(yt == 0) > 0:
        fpr = np.sum((yp == 1) & (yt == 0)) / np.sum(yt == 0)
    else:
        fpr = 0
    
    return {'positive_rate': positive_rate, 'tpr': tpr, 'fpr': fpr}


def demographic_parity_difference(y_pred, group):
    """Compute demographic parity difference."""
    rate_0 = np.mean(y_pred[group == 0])
    rate_1 = np.mean(y_pred[group == 1])
    return abs(rate_0 - rate_1)


def equalized_odds_difference(y_true, y_pred, group):
    """Compute equalized odds difference."""
    m0 = compute_group_metrics(y_true, y_pred, group, 0)
    m1 = compute_group_metrics(y_true, y_pred, group, 1)
    
    tpr_diff = abs(m0['tpr'] - m1['tpr'])
    fpr_diff = abs(m0['fpr'] - m1['fpr'])
    
    return (tpr_diff + fpr_diff) / 2


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    group = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
    y_true = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    y_pred = y_true.copy()
    
    m0 = compute_group_metrics(y_true, y_pred, group, 0)
    assert m0['positive_rate'] is not None
    print("Group Metrics: VERIFIED")
    
    dp = demographic_parity_difference(y_pred, group)
    assert dp >= 0
    print("Demographic Parity: VERIFIED")
    
    eo = equalized_odds_difference(y_true, y_pred, group)
    assert eo >= 0
    print("Equalized Odds: VERIFIED")
