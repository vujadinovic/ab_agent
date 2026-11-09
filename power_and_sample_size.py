import math
from normal_dist import standard_normal_ppf

def required_sample_size_per_variant(baseline_rate, minimum_detectable_effect, alpha, power):
    treatment_rate = baseline_rate + minimum_detectable_effect
    p_hat = (baseline_rate + treatment_rate) / 2
    z_alpha = standard_normal_ppf(1 - alpha / 2)
    z_power = standard_normal_ppf(power)
    formula = ((z_alpha * math.sqrt(2 * p_hat * (1 - p_hat))) + (z_power * math.sqrt(baseline_rate * (1 - baseline_rate) + treatment_rate * (1- treatment_rate))))**2/minimum_detectable_effect**2
    return math.ceil(formula)

def statistical_power(sample_size_per_variant, baseline_rate, effect_size, alpha):
    n = sample_size_per_variant
    p_1 = baseline_rate
    p_2 = p_1 + effect_size
    p_hat = (p_1 + p_2)/2
    pooled_SE = math.sqrt((2*p_hat*(1-p_hat))/n)
    unpooled_SE = math.sqrt((p_1*(1-p_1))/n + (p_2*(1-p_2))/n)
    critical_value = standard_normal_ppf(1 - alpha/2)
    return standard_normal_cdf((abs(p_2 - p_1) - critical_value * pooled_SE) / unpooled_SE)