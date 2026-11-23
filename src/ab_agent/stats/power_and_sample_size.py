import math
from ab_agent.stats.normal_dist import standard_normal_ppf, standard_normal_cdf
from ab_agent.stats.welch_test import noncentral_t_cdf, student_t_critical_value

def two_proportion_required_sample_size_per_variant(baseline_rate, minimum_detectable_effect, alpha, power):
    treatment_rate = baseline_rate + minimum_detectable_effect
    p_hat = (baseline_rate + treatment_rate) / 2
    z_alpha = standard_normal_ppf(1 - alpha / 2)
    z_power = standard_normal_ppf(power)
    formula = ((z_alpha * math.sqrt(2 * p_hat * (1 - p_hat))) + (z_power * math.sqrt(baseline_rate * (1 - baseline_rate) + treatment_rate * (1- treatment_rate))))**2/minimum_detectable_effect**2
    return math.ceil(formula)

def two_proportion_statistical_power(sample_size_per_variant, baseline_rate, effect_size, alpha):
    n = sample_size_per_variant
    p_1 = baseline_rate
    p_2 = p_1 + effect_size
    p_hat = (p_1 + p_2)/2
    pooled_SE = math.sqrt((2*p_hat*(1-p_hat))/n)
    unpooled_SE = math.sqrt((p_1*(1-p_1))/n + (p_2*(1-p_2))/n)
    critical_value = standard_normal_ppf(1 - alpha/2)
    diff = abs(p_2 - p_1)
    upper_tail = standard_normal_cdf((diff - critical_value * pooled_SE) / unpooled_SE)
    lower_tail = standard_normal_cdf((-diff - critical_value * pooled_SE) / unpooled_SE)
    return upper_tail + lower_tail

def _validate_welch_inputs(std_a, std_b, delta, alpha):
    if std_a <= 0 or std_b <= 0:
        raise ValueError("std_a and std_b must be positive.")
    if delta == 0:
        raise ValueError("delta must be non-zero.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")


def welch_degrees_of_freedom_from_std(std_a, std_b, sample_size_per_variant):
    """Welch-Satterthwaite degrees of freedom for two groups of equal size n.

    std_a and std_b are standard deviations (not variances). With equal group
    sizes the general formula reduces to (n - 1) (s_a^2 + s_b^2)^2 / (s_a^4 + s_b^4).
    """
    n = sample_size_per_variant
    if n < 2:
        raise ValueError("sample_size_per_variant must be at least 2.")
    var_a = std_a**2
    var_b = std_b**2
    return (n - 1) * (var_a + var_b) ** 2 / (var_a**2 + var_b**2)


def welch_statistical_power(sample_size_per_variant, std_a, std_b, delta, alpha):
    """Power of the two-sided Welch t-test with n observations per group.

    Under the alternative the Welch statistic follows a noncentral t
    distribution with noncentrality lambda = |delta| / SE, where
    SE = sqrt(std_a^2/n + std_b^2/n), and Welch-Satterthwaite df. Both
    rejection tails are counted:
    power = P(T > t_{1-alpha/2, df}) + P(T < -t_{1-alpha/2, df}).
    """
    _validate_welch_inputs(std_a, std_b, delta, alpha)
    n = sample_size_per_variant
    df = welch_degrees_of_freedom_from_std(std_a, std_b, n)
    standard_error = math.sqrt(std_a**2 / n + std_b**2 / n)
    noncentrality = abs(delta) / standard_error
    critical_value = student_t_critical_value(1 - alpha, df)
    upper_tail = 1.0 - noncentral_t_cdf(critical_value, df, noncentrality)
    lower_tail = noncentral_t_cdf(-critical_value, df, noncentrality)
    return upper_tail + lower_tail


def welch_required_sample_size_per_variant(std_a, std_b, delta, alpha, power):
    """Smallest n per group such that welch_statistical_power(n, ...) >= power.

    The search starts from the normal-approximation formula
    n_z = ceil((z_{1-alpha/2} + z_{power})^2 (std_a^2 + std_b^2) / delta^2)
    and steps by one in the needed direction (the t correction is only a few
    observations).
    """
    _validate_welch_inputs(std_a, std_b, delta, alpha)
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1).")

    z_alpha = standard_normal_ppf(1 - alpha / 2)
    z_power = standard_normal_ppf(power)
    n = max(2, math.ceil((z_alpha + z_power) ** 2 * (std_a**2 + std_b**2) / delta**2))

    def reaches_target(size):
        return welch_statistical_power(size, std_a, std_b, delta, alpha) >= power

    if reaches_target(n):
        while n > 2 and reaches_target(n - 1):
            n -= 1
    else:
        n += 1
        while not reaches_target(n):
            n += 1
    return n
