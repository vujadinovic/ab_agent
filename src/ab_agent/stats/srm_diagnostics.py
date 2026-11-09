import math
import numpy as np

def chi_square_statistic(observed_counts, expected_counts):
    np_observed_counts = np.asarray(observed_counts, dtype = float)
    np_expected_counts = np.asarray(expected_counts, dtype = float)
    np_difference = np_observed_counts - np_expected_counts
    return float(np.dot(np_difference, np_difference / np_expected_counts) )



def sample_ratio_mismatch_check(observed_counts, expected_ratios, alpha):
    observed_counts_np = np.asarray(observed_counts)
    expected_ratios_np = np.asarray(expected_ratios)

    total_count = np.sum(observed_counts_np)
    expected_counts = total_count * expected_ratios_np

    chi_square = chi_square_statistic(observed_counts, expected_counts)
    p = 2*(1 - standard_normal_cdf(math.sqrt(chi_square)))
    srm_detected = p < alpha

    return {'chi_square': chi_square, 'p_value': p, 'srm_detected': srm_detected}    