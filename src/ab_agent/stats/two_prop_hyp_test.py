import math
from ab_agent.stats.normal_dist import standard_normal_cdf, standard_normal_ppf

def pooled_proportion(successes_a, total_a, successes_b, total_b):
    return (successes_a + successes_b) / (total_a + total_b)

def pooled_standard_error(pooled_p, total_a, total_b):
    return math.sqrt(pooled_p*(1-pooled_p)* (1/total_a + 1/total_b))

def two_proportion_z_statistic(p_a, p_b, pooled_se):
    return (p_b - p_a) / (pooled_se)

def two_sided_p_value(z):
    return math.erfc(abs(z) / math.sqrt(2.0))

def unpooled_standard_error(successes_a, total_a, successes_b, total_b):
    p_a = successes_a / total_a
    p_b = successes_b / total_b
    var_a = p_a*(1-p_a) / total_a
    var_b = p_b*(1-p_b) / total_b
    return math.sqrt(var_a + var_b)

def confidence_interval_from_se(point_estimate, standard_error, confidence_level):
    alpha = 1 - confidence_level
    left = point_estimate - standard_error * standard_normal_ppf(1 - alpha/2) 
    right = point_estimate + standard_error * standard_normal_ppf(1 - alpha/2)
    return (left, right)

def two_prop_z_test(successes_a, total_a, successes_b, total_b, confidence_level):
    p_a = successes_a / total_a
    p_b = successes_b / total_b
    effect = p_b - p_a

    p_pool = pooled_proportion(successes_a, total_a, successes_b, total_b)
    se_test = pooled_standard_error(p_pool, total_a, total_b)

    if se_test == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = two_proportion_z_statistic(p_a, p_b, se_test)
        p_value = two_sided_p_value(z)

    se_ci = unpooled_standard_error(successes_a, total_a, successes_b, total_b)
    ci_lower, ci_upper = confidence_interval_from_se(effect, se_ci, confidence_level)

    return {
        'effect': float(effect),
        'se': float(se_test),
        'z': float(z),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),

    }