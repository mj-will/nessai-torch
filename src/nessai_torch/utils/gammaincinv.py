"""Implementation of the inverse incomplete gamma function.

Implementation is based directly of the implementation in tensorflow
probability: https://www.tensorflow.org/probability/api_docs/python/tfp/math/igammainv
"""
import math
import torch
from torch.distributions.utils import euler_constant as euler_gamma
from typing import List


@torch.jit.script
def _didonato_eq_twenty_three(
    log_b: torch.Tensor, v: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    return (
        -log_b + torch.xlogy(a - 1.0, v) - torch.log1p((1.0 - a) / (1.0 + v))
    )


@torch.jit.script
def _didonato_eq_twenty_five(a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Equation 25 from Didonato's paper."""
    c1 = torch.xlogy(a - 1.0, y)
    c1_sq = torch.square(c1)
    c1_cub = c1_sq * c1
    c1_fourth = torch.square(c1_sq)
    a_sq = torch.square(a)
    a_cub = a_sq * a
    c2 = (a - 1.0) * (1.0 + c1)
    c3 = (a - 1.0) * ((3.0 * a - 5.0) / 2.0 + c1 * (a - 2.0 - c1 / 2.0))
    c4 = (a - 1.0) * (
        (c1_cub / 3.0)
        - (3.0 * a - 5.0) * c1_sq / 2.0
        + (a_sq - 6.0 * a + 7.0) * c1
        + (11.0 * a_sq - 46.0 * a + 47.0) / 6.0
    )
    c5 = (a - 1.0) * (
        -c1_fourth / 4.0
        + (11.0 * a - 17.0) * c1_cub / 6
        + (-3.0 * a_sq + 13.0 * a - 13.0) * c1_sq
        + (2.0 * a_cub - 25.0 * a_sq + 72.0 * a - 61.0) * c1 / 2.0
        + (25.0 * a_cub - 195.0 * a_sq + 477 * a - 379) / 12.0
    )
    return y + c1 + (((c5 / y + c4) / y + c3 / y) + c2) / y


@torch.jit.script
def _polyval(coeffs: List[float], x: torch.Tensor) -> torch.Tensor:
    val = torch.zeros_like(x)
    for c in coeffs[:-1]:
        val = (val + c) * x
    return val + coeffs[-1]


@torch.jit.script
def _didonato_eq_thirty_two(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute Equation 32 from Didonato's paper."""
    numerator_coeffs = [
        0.213623493715853,
        4.28342155967104,
        11.6616720288968,
        3.31125922108741,
    ]
    denominator_coeffs = [
        0.36117081018842e-1,
        1.27364489782223,
        6.40691597760039,
        6.61053765625462,
        1.0,
    ]
    t = torch.where(
        p < 0.5, torch.sqrt(-2 * torch.log(p)), torch.sqrt(-2.0 * torch.log(q))
    )
    result = t - _polyval(numerator_coeffs, t) / _polyval(
        denominator_coeffs, t
    )
    return torch.where(p < 0.5, -result, result)


def _didonato_eq_thirty_four(a, x):
    """Compute Equation 34 from Didonato's paper."""
    # This function computes `S_n` in equation thirty four.

    tolerance = 1e-4

    def _taylor_series(should_stop, index, partial, series_sum):
        partial = partial * x / (a + index)
        series_sum = torch.where(should_stop, series_sum, series_sum + partial)
        should_stop = (partial < tolerance) | (index > 100)
        return should_stop, index + 1, partial, series_sum

    partial = torch.ones_like(a + x)
    series_sum = torch.ones_like(a + x)
    should_stop = torch.zeros_like(a + x, dtype=torch.bool)
    index = 1

    # Max 100 iterations
    while torch.any(~should_stop):
        should_stop, index, partial, series_sum = _taylor_series(
            should_stop, index, partial, series_sum
        )

    return series_sum


def _inverse_gammainc_initial_approx(
    a: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    use_p_for_logq: bool = True,
) -> torch.Tensor:
    """Compute an initial guess for :code:`gammaincinv(a, p)`."""
    lgamma_a = torch.lgamma(a)

    # This ensures that computing log(1 - p) avoids roundoff errors. This is
    # needed since igammacinv and igammainv both use this codepath,
    if use_p_for_logq:
        log_q = torch.log1p(-p)
    else:
        log_q = torch.log(q)

    log_b = log_q + lgamma_a

    result = _didonato_eq_twenty_five(a, -log_b)

    # The code below is for when a < 1.

    v = -log_b - (1.0 - a) * torch.log(-log_b)
    v_sq = torch.square(v)

    # This is Equation 24.
    result = torch.where(
        log_b > math.log(0.01),
        -log_b
        - (1.0 - a) * torch.log(v)
        - torch.log(
            (v_sq + 2.0 * (3.0 - a) * v + (2.0 - a) * (3 - a))
            / (v_sq + (5.0 - a) * v + 2.0)
        ),
        result,
    )

    result = torch.where(
        log_b >= math.log(0.15), _didonato_eq_twenty_three(log_b, v, a), result
    )

    t = torch.exp(-euler_gamma - torch.exp(log_b))
    u = t * torch.exp(t)
    result = torch.where(
        (a < 0.3) & (log_b >= math.log(0.35)), t * torch.exp(u), result
    )

    # These are hand tuned constants to compute (p * Gamma(a + 1)) ** (1 / a)
    # This is Equation 21.
    u = torch.where(
        (torch.exp(log_b) * q > 1e-8) & (q > 1e-5),
        torch.pow(p * torch.exp(lgamma_a) * a, torch.reciprocal(a)),
        # When (1 - p) * Gamma(a) or (1 - p) is small,
        # we can taylor expand Gamma(a + 1) ** 1 / a to get
        # exp(-euler_gamma for the zeroth order term.
        # Also p ** 1 / a = exp(log(p) / a) = exp(log(1 - q) / a)
        # ~= exp(-q / a) resulting in the following expression.
        torch.exp((-q / a) - euler_gamma),
    )

    result = torch.where(
        (log_b > math.log(0.6)) | ((log_b >= math.log(0.45)) & (a >= 0.3)),
        u / (1.0 - (u / (a + 1.0))),
        result,
    )

    # The code below is for when a < 1.

    sqrt_a = torch.sqrt(a)
    s = _didonato_eq_thirty_two(p, q)
    s_sq = torch.square(s)
    s_cub = s_sq * s
    s_fourth = torch.square(s_sq)
    s_fifth = s_fourth * s

    # This is the Cornish-Fisher 6 term expansion for x (by viewing igammainv as
    # the quantile function for the Gamma distribution). This is equation (31).
    w = a + s * sqrt_a + (s_sq - 1.0) / 3.0
    w = w + (s_cub - 7.0 * s) / (36.0 * sqrt_a)
    w = w - (3.0 * s_fourth + 7.0 * s_sq - 16.0) / (810 * a)
    w = w + (9.0 * s_fifth + 256.0 * s_cub - 433.0 * s) / (38880 * a * sqrt_a)

    # The code below is for when a > 1. and p > 0.5.
    d = torch.maximum(torch.tensor(2.0), a * (a - 1.0))
    result_a_large_p_large = torch.where(
        log_b <= -d * math.log(10.0),
        _didonato_eq_twenty_five(a, -log_b),
        _didonato_eq_twenty_three(
            log_b, _didonato_eq_twenty_three(log_b, w, a), a
        ),
    )
    result_a_large_p_large = torch.where(
        w < 3.0 * a, w, result_a_large_p_large
    )
    result_a_large_p_large = torch.where(
        (a >= 500.0) & (torch.abs(1.0 - w / a) < 1e-6),
        w,
        result_a_large_p_large,
    )

    # The code below is for when a > 1. and p <= 0.5.
    z = w
    v = torch.log(p) + torch.lgamma(a + 1.0)

    # The code below follows Equation 35 which involves multiple evaluations of
    # F_i.
    modified_z = torch.exp((v + w) / a)
    for _ in range(2):
        s = torch.log1p(
            modified_z / (a + 1.0) * (1.0 + modified_z / (a + 2.0))
        )
        modified_z = torch.exp((v + modified_z - s) / a)

    s = torch.log1p(
        modified_z
        / (a + 1.0)
        * (1.0 + modified_z / (a + 2.0) * (1.0 + modified_z / (a + 3.0)))
    )
    modified_z = torch.exp((v + modified_z - s) / a)
    z = torch.where(w <= 0.15 * (a + 1.0), modified_z, z)

    ls = torch.log(_didonato_eq_thirty_four(a, z))
    medium_z = torch.exp((v + z - ls) / a)
    result_a_large_p_small = torch.where(
        (z <= 0.01 * (a + 1.0)) | (z > 0.7 * (a + 1.0)),
        z,
        medium_z
        * (
            1.0
            - (a * torch.log(medium_z) - medium_z - v + ls) / (a - medium_z)
        ),
    )

    result_a_large = torch.where(
        p <= 0.5, result_a_large_p_small, result_a_large_p_large
    )
    result = torch.where(a < 1.0, result, result_a_large)

    # This ensures that computing log(1 - p) avoids roundoff errors. This is
    # needed since igammacinv and igammainv both use this codepath,
    # switching p and q.
    result = torch.where(torch.eq(a, torch.tensor(1.0)), -log_q, result)
    return result


def _shared_igammainv_computation(
    a: torch.Tensor, p: torch.TensorType, is_gammaincinv: bool = True
):
    """Computation for gammaincinv."""

    if is_gammaincinv:
        q = 1.0 - p
    else:
        q = p
        p = 1.0 - q

    x = _inverse_gammainc_initial_approx(
        a, p, q, use_p_for_logq=is_gammaincinv
    )

    # Run 3 steps of Newton-Halley method.
    for _ in range(3):
        factorial = torch.exp(a * torch.log(x) - x - torch.lgamma(a))

        f_over_der = torch.where(
            ((p <= 0.9) & is_gammaincinv) | ((q > 0.9) & (not is_gammaincinv)),
            (torch.igamma(a, x) - p) * x / factorial,
            -(torch.igammac(a, x) - q) * x / factorial,
        )
        second_der_over_der = -1.0 + (a - 1.0) / x
        modified_x = torch.where(
            ~torch.isfinite(second_der_over_der),
            # Use Newton's method if the second derivative is not available.
            x - f_over_der,
            # Use Halley's method otherwise. Halley's method is:
            # x_{n+1} = x_n - f(x_n) / f'(x_n) * (
            #    1 - f(x_n) / f'(x_n) * 0.5 f''(x_n) / f'(x_n))
            x - f_over_der / (1.0 - 0.5 * f_over_der * second_der_over_der),
        )
        x = torch.where(torch.eq(factorial, 0.0), x, modified_x)
    x = torch.where((a < 0.0) | (p < 0.0) | (p > 1.0), torch.nan, x)
    x = torch.where(torch.eq(p, 0.0), 0.0, x)
    x = torch.where(torch.eq(p, 1.0), torch.inf, x)

    return x


def gammaincinv(a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Inverse of the incomplete gamma function.

    Derivatives are not implemented. This function cannot be JIT compiled.
    """
    return _shared_igammainv_computation(a, p, is_gammaincinv=True)
