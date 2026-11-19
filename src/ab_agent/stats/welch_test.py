# currently with normal distribution approximation
import math
import numpy as np
from ab_agent.stats.two_prop_hyp_test import confidence_interval_from_se, two_sided_p_value

def incomplete_beta_continued_fraction(a, b, x):
	# continued fraction for the incomplete beta funcion (Lentz method)
	tiny = 1e-300
	c=1.0
	d=1.0-(a+b)*x/(a+1.0)
	d=1.0 / (d if abs(d)> tiny else tiny)
	h = d

	for m in range(1, 10000):
		for aa in (
			m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m)),
			-(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1)),
		):
			d = 1.0 + aa * d
			d = 1.0 / (d if abs(d) > tiny else tiny)
			c = 1.0 + aa / c
			c = c if abs(c) > tiny else tiny
			h *= d * c

		if abs(d * c - 1.0) < 1e-15:
			break

	return h

def regularized_incomplete_beta(a, b, x):
	if x <= 0.0:
		return 0.0
	if x >= 1.0:
		return 1.0

	front = math.exp(
		math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
	)

	if x < (a + 1.0) / (a + b + 2.0):
		return front * incomplete_beta_continued_fraction(a, b, x) / a
	return 1.0 - front * incomplete_beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p_value(t, df):
	# P(|T| >= |t|), computed directly from the tail
	x = df / (df + t * t)
	return regularized_incomplete_beta(df / 2.0, 0.5, x)

def student_t_critical_value(confidence_level, df):
	# t such that P(|T| <= t) = confidence_level, found by bisection
	alpha = 1.0 - confidence_level

	low, high = 0.0, 1.0
	while student_t_two_sided_p_value(high, df) > alpha:
		low, high = high, 2.0 * high

	for _ in range(100):
		mid = (low + high) / 2.0
		if student_t_two_sided_p_value(mid, df) > alpha:
			low = mid
		else:
			high = mid

	return (low + high) / 2.0

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

def welch_degrees_of_freedom(values_a, values_b):
	values_a_np = np.asarray(values_a, dtype=float)
	values_b_np = np.asarray(values_b, dtype=float)

	n_a = len(values_a_np)
	n_b = len(values_b_np)

	v_a = np.var(values_a_np, ddof=1) / n_a
	v_b = np.var(values_b_np, ddof=1) / n_b

	return (v_a + v_b)**2 / (v_a**2 / (n_a - 1) + v_b**2 / (n_b - 1))


def welch_t_test(values_a, values_b, confidence_level = 0.95):
	values_a_np = np.asarray(values_a, dtype=float)
	values_b_np = np.asarray(values_b, dtype=float)

	SE = welch_standard_error(values_a_np, values_b_np)
	effect = float(np.mean(values_b_np) - np.mean(values_a_np))

	if SE == 0:
		df = float(len(values_a_np) + len(values_b_np) - 2)
		if effect == 0:
			t = 0.0
			p_value = 1.0
		else:
			t = np.sign(effect) * np.inf
			p_value = 0.0
	else:
		t = effect / SE
		df = welch_degrees_of_freedom(values_a_np, values_b_np)
		p_value = student_t_two_sided_p_value(t, df)

	margin = student_t_critical_value(confidence_level, df) * SE
	ci_lower, ci_upper = effect - margin, effect + margin

	return {
		'effect': effect,
		'se': float(SE),
		't': float(t),
		'df': float(df),
		'p_value': float(p_value),
		'ci_lower': float(ci_lower),
		'ci_upper': float(ci_upper),
	}
