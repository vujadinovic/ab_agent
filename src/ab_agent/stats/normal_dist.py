import math
import numpy as np

def standard_normal_cdf(z):
    if np.isscalar(z):
        return 0.5*(1 + math.erf(z/np.sqrt(2.0)))

    z_arr = np.asarray(z)
    erf_vec = np.vectorize(math.erf)
    return 0.5*(1 + erf_vec(z/np.sqrt(2.0)))


# Rational approximations from Wichura, M.J. (1988), "Algorithm AS 241: The
# Percentage Points of the Normal Distribution", Applied Statistics 37, 477-484
# (routine PPND16, accurate to about 1 part in 1e16). Coefficients are listed
# from the constant term upwards; each denominator has constant term 1.

# Central region, |p - 0.5| <= 0.425, in the variable r = 0.425^2 - (p - 0.5)^2.
_SPLIT_CENTRAL = 0.425
_CENTRAL_NUM = (
    3.3871328727963666080e0,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
)
_CENTRAL_DEN = (
    1.0,
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
)

# Tail, t = sqrt(-log(min(p, 1 - p))) <= 5, in the variable t - 1.6.
_SPLIT_TAIL = 5.0
_TAIL_NUM = (
    1.42343711074968357734e0,
    4.63033784615654529590e0,
    5.76949722146069140550e0,
    3.64784832476320460504e0,
    1.27045825245236838258e0,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
)
_TAIL_DEN = (
    1.0,
    2.05319162663775882187e0,
    1.67638483018380384940e0,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
)

# Far tail, 5 < t <= 27, in the variable t - 5.
_SPLIT_FAR_TAIL = 27.0
_FAR_TAIL_NUM = (
    6.65790464350110377720e0,
    5.46378491116411436990e0,
    1.78482653991729133580e0,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
)
_FAR_TAIL_DEN = (
    1.0,
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
)


def _horner(coeffs, x):
    """Evaluate sum(coeffs[k] * x**k) for coefficients in ascending order."""
    result = np.full_like(x, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result


def _extreme_tail_quantile(log_tail):
    """Upper quantile x > 0 with log(1 - Phi(x)) = log_tail, for t > 27.

    Uses the asymptotic expansion of the Gaussian tail from Maechler, M. (2022),
    "Asymptotic tail formulas for gaussian quantiles" (CRAN package DPQ). With
    s2 = -2 * log_tail, the quantile satisfies

        x^2 = s2 - log(2 pi x^2) + 2 log(m(x^2)),

    where m is the asymptotic correction factor of Mills' ratio. Starting from
    x^2 ~ s2, the successive iterates xs_1 ... xs_5 use increasingly long
    truncations of m, the last one being
        1 - 1/(x^2 + 2) * (1 - 1/(x^2 + 4) * (1 - (5 - 9/(x^2 + 8))/(x^2 + 6))).

    For double p > 0, t = sqrt(-log(p)) never exceeds ~27.3, so only the
    subnormal range p < exp(-729) ~ 2.5e-317 reaches this function; there the
    most accurate (highest order) iterate is always the right choice.
    """
    s2 = -2.0 * log_tail
    two_pi = 2.0 * math.pi
    x2 = s2 - np.log(two_pi * s2)
    x2 = s2 - np.log(two_pi * x2) - 2.0 / (2.0 + x2)
    x2 = s2 - np.log(two_pi * x2) + 2.0 * np.log1p(-(1.0 - 1.0 / (4.0 + x2)) / (2.0 + x2))
    x2 = s2 - np.log(two_pi * x2) + 2.0 * np.log1p(
        -(1.0 - (1.0 - 5.0 / (6.0 + x2)) / (4.0 + x2)) / (2.0 + x2)
    )
    x2 = s2 - np.log(two_pi * x2) + 2.0 * np.log1p(
        -(1.0 - (1.0 - (5.0 - 9.0 / (8.0 + x2)) / (6.0 + x2)) / (4.0 + x2)) / (2.0 + x2)
    )
    return np.sqrt(x2)


def standard_normal_ppf(p):
    """Quantile function (inverse CDF) of the standard normal distribution.

    Accepts a scalar or any array-like. A scalar input returns a Python float;
    array-like input returns an ndarray of the same shape.

    Follows R's ``qnorm`` / SciPy's ``norm.ppf`` conventions at the edges:
    p = 0 -> -inf, p = 1 -> inf, p outside [0, 1] or NaN -> nan.

    Algorithm AS 241 (Wichura 1988) is used for t = sqrt(-log(min(p, 1-p))) <= 27,
    and an asymptotic tail expansion (Maechler 2022) beyond that, which is only
    reached by subnormal p.
    """
    p_arr = np.asarray(p, dtype=float)
    flat = p_arr.ravel()
    out = np.full(flat.shape, np.nan)

    out[flat == 0.0] = -np.inf
    out[flat == 1.0] = np.inf

    q = flat - 0.5
    central = np.abs(q) <= _SPLIT_CENTRAL  # False for NaN
    tail = (flat > 0.0) & (flat < 1.0) & ~central

    if central.any():
        qc = q[central]
        r = _SPLIT_CENTRAL * _SPLIT_CENTRAL - qc * qc
        out[central] = qc * _horner(_CENTRAL_NUM, r) / _horner(_CENTRAL_DEN, r)

    if tail.any():
        qt = q[tail]
        # Smaller of the two tail probabilities; 1 - p is exact when p > 0.5.
        log_tail = np.log(np.where(qt > 0.0, 1.0 - flat[tail], flat[tail]))
        t = np.sqrt(-log_tail)
        val = np.empty_like(t)

        near = t <= _SPLIT_TAIL
        far = (t > _SPLIT_TAIL) & (t <= _SPLIT_FAR_TAIL)
        extreme = t > _SPLIT_FAR_TAIL

        if near.any():
            u = t[near] - 1.6
            val[near] = _horner(_TAIL_NUM, u) / _horner(_TAIL_DEN, u)
        if far.any():
            u = t[far] - 5.0
            val[far] = _horner(_FAR_TAIL_NUM, u) / _horner(_FAR_TAIL_DEN, u)
        if extreme.any():
            val[extreme] = _extreme_tail_quantile(log_tail[extreme])

        out[tail] = np.where(qt < 0.0, -val, val)

    if p_arr.ndim == 0:
        return float(out[0])
    return out.reshape(p_arr.shape)
