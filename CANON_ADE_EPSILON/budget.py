import math

# Precision profiles: label -> mantissa bits (b), so eps = 2^{-b}
PRECISION_PROFILES = {
    'float16':  10,
    'bfloat16':  7,
    'float32':  23,
    'float64':  52,
}

def delta_per_element(R: int, b: int, M: float) -> float:
    """
    Rounding error per output element of 3x3 matmul computed with a
    rank-R decomposition over inputs bounded by M, in b-bit arithmetic.

        delta = R * 2^{-b} * M^2
    """
    return R * 2.0**(-b) * M**2

def budget_frob(R: int, b: int, M: float) -> float:
    """
    Frobenius-norm budget for total allowed error across all 9 output
    elements:

        budget_frob = sqrt(9) * delta_per_element = 3 * R * 2^{-b} * M^2

    This is the maximum allowable ||eps*E||_F where eps = 2^{-b}.
    """
    return 3.0 * delta_per_element(R, b, M)

def budget_E_norm(R: int, b: int, M: float) -> float:
    """
    Budget on ||E||_F, the free variable in the decomposition equation.

    Since eps*E enters the equation and eps = 2^{-b}:

        budget_E_norm = budget_frob / eps = 3 * R * M^2

    NOTE: This is INDEPENDENT of b. The bit-depth eps appears as both
    the scale on E and the unit of rounding error, so it cancels.
    What changes with b is the absolute tolerance eps*E that is
    physically permissible; E itself is dimensionless w.r.t. precision.
    """
    return 3.0 * R * M**2

def is_loop_closed(R: int, b: int, M: float, E_norm: float) -> bool:
    """
    The feedback loop closes when the budget on ||E||_F is at least as
    large as the norm of the minimum-perturbation E* that drops rank
    below R.  Returns True if this condition is satisfied.

        budget_E_norm(R, b, M) >= E_norm
    """
    return budget_E_norm(R, b, M) >= E_norm

def summary_table(R_list, b: int, M: float) -> list:
    rows = []
    eps = 2.0**(-b)
    for R in R_list:
        rows.append({
            'R': R,
            'b': b,
            'M': M,
            'eps': eps,
            'delta_per_element': delta_per_element(R, b, M),
            'budget_frob': budget_frob(R, b, M),
            'budget_E_norm': budget_E_norm(R, b, M),
        })
    return rows
