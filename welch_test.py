# currently with normal distribution approximation

import numpy as np
from two_prop_hyp_test import confidence_interval_from_se, two_sided_p_value


def welch_standard_error(values_a, values_b):
	if len(values_a) < 2 or len(values_b) < 2:
		raise ValueError("Sample has less than 2 observations.")

	values_a_np = np.asarray(values_a, dtype=float)
	values_b_np = np.asarray(values_b, dtype=float)

	n_a = len(values_a)
	n_b = len(values_b)

	a_average = np.average(values_a_np)
	b_average = np.average(values_b_np)

	s_a = 1/(n_a-1) * np.sum(np.square(values_a_np - a_average))
	s_b = 1/(n_b-1) * np.sum(np.square(values_b_np - b_average))


	SE = np.sqrt((s_a / n_a) + (s_b / n_b))

	return SE


def welch_z_test(values_a, values_b, confidence_level = 0.95):
	values_a_np = np.asarray(values_a, dtype=float)
	values_b_np = np.asarray(values_b, dtype=float)
 
	SE = welch_standard_error(values_a_np, values_b_np)
	effect = float(np.mean(values_b_np) - np.mean(values_a_np))
 
	if SE == 0:
		z = 0.0
		p_value = 1.0
	else:
		z = effect / SE
		p_value = two_sided_p_value(z)
 
	ci_lower, ci_upper = confidence_interval_from_se(effect, SE, confidence_level)
 
	return {
		'effect': effect,
		'se': float(SE),
		'z': float(z),
		'p_value': float(p_value),
		'ci_lower': float(ci_lower),
		'ci_upper': float(ci_upper),
	}