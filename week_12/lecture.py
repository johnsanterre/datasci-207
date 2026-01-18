"""
DATASCI 207: Applied Machine Learning
Module 12: Fairness and Responsible AI

This module covers:
- Sources of algorithmic bias
- Fairness definitions (demographic parity, equalized odds)
- Measuring fairness metrics
- Bias mitigation strategies

Using NumPy for computations.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: SIMULATED EXAMPLE - HIRING CLASSIFIER
# =============================================================================

print("=== Part 1: Simulated Hiring Scenario ===\n")

# Simulate a hiring classifier with biased outcomes
# Group A=0: underrepresented group, A=1: majority group

n_samples = 1000
group = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])

# True "qualification" - should be similar across groups in ideal world
# But historical data has bias
qualification = np.random.randn(n_samples)

# Model's prediction - biased against group 0
threshold_group0 = 0.5   # Harder threshold for group 0
threshold_group1 = 0.0   # Easier threshold for group 1

y_pred = np.zeros(n_samples)
y_pred[(group == 0) & (qualification > threshold_group0)] = 1
y_pred[(group == 1) & (qualification > threshold_group1)] = 1
y_pred = y_pred.astype(int)

# Ground truth (actual capability)
y_true = (qualification > 0).astype(int)

print(f"Total samples: {n_samples}")
print(f"Group 0 (underrepresented): {np.sum(group == 0)}")
print(f"Group 1 (majority): {np.sum(group == 1)}")


# =============================================================================
# PART 2: MEASURING FAIRNESS METRICS
# =============================================================================

print("\n=== Part 2: Fairness Metrics ===\n")


def group_metrics(y_true, y_pred, group, group_val):
    """Compute metrics for a specific group."""
    mask = (group == group_val)
    yt = y_true[mask]
    yp = y_pred[mask]
    
    n = len(yt)
    pos_rate = np.mean(yp)
    
    # True positive rate (recall)
    if np.sum(yt == 1) > 0:
        tpr = np.sum((yp == 1) & (yt == 1)) / np.sum(yt == 1)
    else:
        tpr = 0
    
    # False positive rate
    if np.sum(yt == 0) > 0:
        fpr = np.sum((yp == 1) & (yt == 0)) / np.sum(yt == 0)
    else:
        fpr = 0
    
    return {
        'n': n,
        'positive_rate': pos_rate,
        'tpr': tpr,
        'fpr': fpr
    }


metrics_0 = group_metrics(y_true, y_pred, group, 0)
metrics_1 = group_metrics(y_true, y_pred, group, 1)

print("Group 0 (underrepresented):")
print(f"  Positive rate: {metrics_0['positive_rate']:.3f}")
print(f"  True positive rate: {metrics_0['tpr']:.3f}")
print(f"  False positive rate: {metrics_0['fpr']:.3f}")

print("\nGroup 1 (majority):")
print(f"  Positive rate: {metrics_1['positive_rate']:.3f}")
print(f"  True positive rate: {metrics_1['tpr']:.3f}")
print(f"  False positive rate: {metrics_1['fpr']:.3f}")


# =============================================================================
# PART 3: DEMOGRAPHIC PARITY
# =============================================================================

print("\n=== Part 3: Demographic Parity ===\n")

dp_ratio = metrics_0['positive_rate'] / metrics_1['positive_rate']

print("Demographic Parity: Equal positive prediction rates across groups")
print(f"  Group 0 positive rate: {metrics_0['positive_rate']:.3f}")
print(f"  Group 1 positive rate: {metrics_1['positive_rate']:.3f}")
print(f"  Ratio (0/1): {dp_ratio:.3f}")

if dp_ratio < 0.8:
    print(f"  WARNING: Ratio < 0.8 indicates potential disparate impact!")
else:
    print(f"  Ratio is within acceptable range (>= 0.8)")


# =============================================================================
# PART 4: EQUALIZED ODDS
# =============================================================================

print("\n=== Part 4: Equalized Odds ===\n")

tpr_diff = abs(metrics_0['tpr'] - metrics_1['tpr'])
fpr_diff = abs(metrics_0['fpr'] - metrics_1['fpr'])

print("Equalized Odds: Equal TPR and FPR across groups")
print(f"  TPR difference: |{metrics_0['tpr']:.3f} - {metrics_1['tpr']:.3f}| = {tpr_diff:.3f}")
print(f"  FPR difference: |{metrics_0['fpr']:.3f} - {metrics_1['fpr']:.3f}| = {fpr_diff:.3f}")

if tpr_diff > 0.1 or fpr_diff > 0.1:
    print("  WARNING: Large differences indicate equalized odds violation!")


# =============================================================================
# PART 5: POST-PROCESSING MITIGATION
# =============================================================================

print("\n=== Part 5: Bias Mitigation (Post-processing) ===\n")

# Simple mitigation: Adjust thresholds to achieve demographic parity
# Lower threshold for group 0

y_pred_fair = y_pred.copy()

# Group 0: be more lenient
group0_mask = group == 0
# Accept more from group 0 by using original qualification scores
additional_accepts = (group == 0) & (qualification > 0.25) & (y_pred == 0)
y_pred_fair[additional_accepts] = 1

# Recalculate metrics
metrics_0_fair = group_metrics(y_true, y_pred_fair, group, 0)
metrics_1_fair = group_metrics(y_true, y_pred_fair, group, 1)

print("After threshold adjustment for Group 0:")
print(f"  Group 0 positive rate: {metrics_0_fair['positive_rate']:.3f}")
print(f"  Group 1 positive rate: {metrics_1_fair['positive_rate']:.3f}")

dp_ratio_fair = metrics_0_fair['positive_rate'] / metrics_1_fair['positive_rate']
print(f"  New DP ratio: {dp_ratio_fair:.3f}")


# =============================================================================
# PART 6: IMPOSSIBILITY THEOREM ILLUSTRATION
# =============================================================================

print("\n=== Part 6: Fairness Tradeoffs ===\n")

print("""
IMPOSSIBILITY RESULT:

When base rates differ between groups, you CANNOT simultaneously achieve:
  1. Calibration (P(Y=1|score) same for both groups)
  2. Equal False Positive Rates
  3. Equal False Negative Rates

This is a mathematical impossibility, not a technical limitation.

Implications:
- Must choose which fairness criterion matters most
- Context and values determine the choice
- Tradeoffs should be documented explicitly
""")


# =============================================================================
# PART 7: SOURCES OF BIAS
# =============================================================================

print("\n=== Part 7: Common Sources of Bias ===\n")

print("""
DATA-LEVEL BIAS:
- Historical bias: Data reflects past discrimination
- Representation bias: Groups under/overrepresented
- Measurement bias: Features measured differently
- Label bias: Biased human annotations

ALGORITHM-LEVEL BIAS:
- Objective function focuses only on accuracy
- Features correlate with protected attributes
- Model complexity varies across groups

DEPLOYMENT-LEVEL BIAS:
- Distribution shift between training and deployment
- Feedback loops amplifying bias
- Different usage patterns across groups

MITIGATION STRATEGIES:
- Pre-processing: Resampling, reweighting, fair representations
- In-processing: Constraints, adversarial debiasing
- Post-processing: Threshold adjustment, calibration
""")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 12")
print("=" * 60)
print("""
1. ALGORITHMIC BIAS is systematic unfairness in ML predictions
   - Can arise from data, algorithms, or deployment

2. PROTECTED ATTRIBUTES: race, gender, age, etc.
   - Models can learn these from proxies (ZIP code, name)

3. FAIRNESS DEFINITIONS:
   - Demographic Parity: Equal positive rates
   - Equalized Odds: Equal TPR and FPR
   - Calibration: Accurate probability estimates

4. IMPOSSIBILITY: Different fairness criteria conflict
   - Must choose based on context and values

5. MITIGATION:
   - Pre-processing: Fix data
   - In-processing: Constrained learning
   - Post-processing: Adjust outputs

6. ONGOING RESPONSIBILITY:
   - Monitor deployed systems
   - Engage stakeholders
   - Document tradeoffs
""")
