"""Coordinate rotation: change the factor basis before/after LP.

For a CP decomposition T = sum_k a_k (x) b_k (x) g_k, any invertible
D x D matrix Q gives an equivalent decomposition via:
    a' = a @ Q,  b' = b @ Q,  g' = g @ Q_inv_T
(or any assignment of Q, Q^-T across the three modes).

This changes which entries the LP finds easy to move, breaking the
frozen active-set that causes plateaus.
"""

import numpy as np
from scipy.stats import ortho_group

from ..config import R, D, N_PARAMS


def random_rotation():
    """Sample a random D x D orthogonal matrix."""
    return ortho_group.rvs(D)


def rotate_factors(x, Q):
    """Apply rotation Q to factor matrices.

    a' = a @ Q, b' = b @ Q, g' = g @ Q^-T = g @ Q (orthogonal).
    Returns new flat parameter vector (513,).
    """
    a = x[:R * D].reshape(R, D)
    b = x[R * D:2 * R * D].reshape(R, D)
    g = x[2 * R * D:].reshape(R, D)

    a_rot = a @ Q
    b_rot = b @ Q
    g_rot = g @ Q  # Q^-T = Q for orthogonal Q

    return np.concatenate([a_rot.ravel(), b_rot.ravel(), g_rot.ravel()])


def unrotate_factors(x, Q):
    """Undo rotation: apply Q^T."""
    return rotate_factors(x, Q.T)
