import numpy as np

def bonferroni_correction(p_values, alpha):
    if not p_values:
        return np.array([], dtype=bool)
    m = len(p_values)
    p_values_np = np.asarray(p_values)
    adjusted_alpha = alpha/m
    is_significant = p_values_np <= adjusted_alpha
    return is_significant
    

def benjamini_hochberg_correction(p_values, alpha):
    m = len(p_values)
    p_values_np = np.asarray(p_values)
    order = np.argsort(p_values_np)
    p_sorted = p_values_np[order]
    ranks = np.arange(1, m + 1)
    thresholds = ranks * (alpha / m)

    passes = p_sorted <= thresholds

    last_index = -1

    for i in reversed(range(m)):
        if passes[i]:
            last_index = i
            break

    sorted_mask = np.zeros(m, dtype = bool)
    sorted_mask[:last_index + 1] = True

    result = np.empty(m, dtype = bool)
    result[order] = sorted_mask
    return result
