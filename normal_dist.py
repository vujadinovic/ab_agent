import math
import numpy as np

def standard_normal_cdf(z):
    if np.isscalar(z):
        return 0.5*(1 + math.erf(z/np.sqrt(2.0)))

    z_arr = np.asarray(z)
    erf_vec = np.vectorize(math.erf)
    return 0.5*(1 + erf_vec(z/np.sqrt(2.0)))
    
    
def standard_normal_ppf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in interval (0, 1)")

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p_low <= p <= p_high:
        q = p - 0.5
        r = q * q
        num = (
            (
                (
                    (
                        (-39.69683028665376 * r + 220.9460984245205) * r
                        - 275.9285104469687
                    )
                    * r
                    + 138.3577518672690
                )
                * r
                - 30.66479806614716
            )
            * r
            + 2.506628277459239
        ) * q
        den = (
            (
                (
                    (
                        (-54.47609879822406 * r + 161.5858368580409) * r
                        - 155.6989798598866
                    )
                    * r
                    + 66.80131188771972
                )
                * r
                - 13.28068155288572
            )
            * r
            + 1.0
        )
        return num / den

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))

    num = (
        (
            (
                ((-0.007784894002430293 * q - 0.3223964580411365) * q - 2.400758277161838)
                * q
                - 2.549732539343734
            )
            * q
            + 4.374664141464968
        )
        * q
        + 2.938163982698783
    )
    den = (
        (
            ((0.007784695709041462 * q + 0.3224671290700398) * q + 2.445134137142996)
            * q
            + 3.754408661907416
        )
        * q
        + 1.0
    )

    res = num / den
    return res if p < p_low else -res