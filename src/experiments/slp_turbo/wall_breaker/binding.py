"""Identify binding (near-maxabs) entries in a candidate solution."""

import numpy as np
from ..config import R, D, N_ENTRIES, T_NP


def compute_residuals(x):
    """Return signed residual vector (729,) for flat param vector x."""
    a = x[:R * D].reshape(R, D)
    b = x[R * D:2 * R * D].reshape(R, D)
    g = x[2 * R * D:].reshape(R, D)
    T_hat = np.einsum('ra,rb,rc->abc', a, b, g)
    return T_hat.ravel() - T_NP.ravel()


def binding_entries(x, top_k=10, tol=0.95):
    """Return indices of entries within `tol` fraction of maxabs.

    Parameters
    ----------
    x : (513,) flat parameter vector
    top_k : max number of binding indices to return
    tol : fraction of maxabs.  0.95 means entries >= 95% of max|r|

    Returns
    -------
    indices : 1-D int array of binding entry indices (sorted by |r| descending)
    residuals : corresponding residual values
    maxabs : the current maxabs fitness
    """
    res = compute_residuals(x)
    absres = np.abs(res)
    maxabs = absres.max()
    threshold = tol * maxabs
    mask = absres >= threshold
    idx = np.where(mask)[0]
    # Sort by descending absolute residual
    order = np.argsort(absres[idx])[::-1]
    idx = idx[order[:top_k]]
    return idx, res[idx], float(maxabs)
