# currently with normal distribution approximation
import math
import numpy as np
from ab_agent.stats.normal_dist import standard_normal_cdf
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

def student_t_cdf(t, df):
	# P(T <= t), from the two-sided tail by symmetry of the t distribution
	half_tail = student_t_two_sided_p_value(t, df) / 2.0
	return 1.0 - half_tail if t >= 0 else half_tail

def _incomplete_beta_step(a, b, x):
	# g(a) = Gamma(a+b) / (Gamma(a+1) Gamma(b)) x^a (1-x)^b,
	# the amount by which I_x(a, b) exceeds I_x(a+1, b)
	return math.exp(
		math.lgamma(a + b) - math.lgamma(a + 1.0) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
	)

def noncentral_t_cdf(t, df, noncentrality):
	# P(T <= t) for T = (Z + noncentrality) / sqrt(V / df), Z ~ N(0, 1), V ~ chi^2(df).
	# For t >= 0 (Lenth 1989, AS 243):
	#   F(t) = Phi(-d) + 1/2 sum_j [P_j I_x(j + 1/2, df/2) + Q_j I_x(j + 1, df/2)],
	#   x = t^2 / (t^2 + df), lam = d^2 / 2,
	#   P_j = e^-lam lam^j / j!, Q_j = d / sqrt(2) e^-lam lam^j / Gamma(j + 3/2).
	# The sum starts at the Poisson mode j = floor(lam) and runs both ways
	# (Benton and Krishnamoorthy 2003), so the weights do not underflow for large d.
	# For t < 0, F(t; d) = 1 - F(-t; -d).
	if t < 0:
		return 1.0 - noncentral_t_cdf(-t, df, -noncentrality)
	d = noncentrality
	if d == 0:
		return student_t_cdf(t, df)
	base = standard_normal_cdf(-d)
	if t == 0:
		return base

	x = t * t / (t * t + df)
	if x >= 1.0:
		return 1.0
	b = df / 2.0
	lam = d * d / 2.0
	k = int(lam)
	log_poisson = -lam + k * math.log(lam)
	p_k = math.exp(log_poisson - math.lgamma(k + 1.0))
	q_k = d / math.sqrt(2.0) * math.exp(log_poisson - math.lgamma(k + 1.5))
	ia_k = regularized_incomplete_beta(k + 0.5, b, x)
	ib_k = regularized_incomplete_beta(k + 1.0, b, x)
	width = int(10.0 * math.sqrt(lam)) + 30

	# forward: j = k, k+1, ...; I_x(a+1, b) = I_x(a, b) - g(a), g(a+1) = g(a) x (a+b) / (a+1)
	p, q, ia, ib = p_k, q_k, ia_k, ib_k
	ga = _incomplete_beta_step(k + 0.5, b, x)
	gb = _incomplete_beta_step(k + 1.0, b, x)
	total = 0.0
	for j in range(k, k + width + 1):
		total += p * ia + q * ib
		a1, a2 = j + 0.5, j + 1.0
		ia = max(ia - ga, 0.0)
		ib = max(ib - gb, 0.0)
		ga *= x * (a1 + b) / (a1 + 1.0)
		gb *= x * (a2 + b) / (a2 + 1.0)
		p *= lam / (j + 1.0)
		q *= lam / (j + 1.5)

	# backward: j = k-1, ..., 0; I_x(a-1, b) = I_x(a, b) + g(a-1), g(a-2) = g(a-1) (a-1) / (x (a-2+b))
	p, q, ia, ib = p_k, q_k, ia_k, ib_k
	if k >= 1:
		ga = _incomplete_beta_step(k - 0.5, b, x)
		gb = _incomplete_beta_step(float(k), b, x)
	for j in range(k - 1, max(k - width, 0) - 1, -1):
		a1, a2 = j + 1.5, j + 2.0  # parameters of term j + 1
		ia += ga
		ib += gb
		if j > 0:
			ga *= (a1 - 1.0) / (x * (a1 - 2.0 + b))
			gb *= (a2 - 1.0) / (x * (a2 - 2.0 + b))
		p *= (j + 1.0) / lam
		q *= (j + 1.5) / lam
		total += p * ia + q * ib

	return min(max(base + 0.5 * total, 0.0), 1.0)

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
		t = 0.0
		df = float(len(values_a_np) + len(values_b_np) - 2)
		p_value = 1.0
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