"""Shared functions for use by both variational and EM mixture models."""
import numpy as np
import scipy
from scipy import linalg


def sq_maha_distance(X, loc_, scale, covariance_type='full', xp=None):
    """Computes the mahalanobis distance from each row of X
    using the k (for k clusters) rows of loc_ and the k dim3
    cholesky decompositions of the scale matrices.
    TECHNICALLY this is the squared mahalanobis distance and
    is therefore referred to as sq_maha throughout.

    Args:
        X (np.ndarray): Input data, shape N x M.
        loc_ (np.ndarray): Cluster centers, shape K x M.
        scale (np.ndarray): For 'full', M x M x K Cholesky decompositions
            of scale matrices. For 'diag', M x K diagonal elements of
            scale matrices (square roots of variances).
        covariance_type (str): One of 'full' or 'diag'. Defaults to 'full'.
        xp: Array module (numpy or cupy). Defaults to numpy.
    """
    if xp is None:
        xp = np
    if covariance_type == 'diag':
        # Vectorized: broadcast (N,1,M) - (1,K,M) -> (N,K,M), then sum
        diff = X[:, xp.newaxis, :] - loc_[xp.newaxis, :, :]
        sq_maha_dist = xp.sum(diff**2 / scale.T[xp.newaxis, :, :], axis=2)
    else:
        # Batched solve: L is (K,M,M), diff is (K,N,M)
        L = xp.transpose(scale, (2, 0, 1))
        diff = X[xp.newaxis, :, :] - loc_[:, xp.newaxis, :]
        solved = xp.linalg.solve(L, diff.transpose(0, 2, 1))
        sq_maha_dist = xp.sum(solved**2, axis=1).T
    return sq_maha_dist


def scale_update_calcs(X, ru, loc_, resp_sum, reg_covar, covariance_type='full', xp=None):
    """Updates the scale (aka covariance) matrices as part of the M-
    step for EM and as the parameter update for variational methods."""
    if xp is None:
        xp = np
    M = loc_.shape[1]
    K = loc_.shape[0]
    eps = 10 * np.finfo(np.float64).eps

    if covariance_type == 'diag':
        # Vectorized diagonal scale update
        diff = X[xp.newaxis, :, :] - loc_[:, xp.newaxis, :]  # (K, N, M)
        scale_ = xp.sum(ru.T[:, :, xp.newaxis] * diff**2, axis=1)  # (K, M)
        scale_ /= (resp_sum[:, xp.newaxis] + eps)
        scale_ += reg_covar
        scale_ = scale_.T  # (M, K) for API compat
        scale_decomp = scale_.copy()
    else:
        # Vectorized full scale update using einsum + batched Cholesky
        diff = X[xp.newaxis, :, :] - loc_[:, xp.newaxis, :]  # (K, N, M)
        weighted_diff = xp.sqrt(ru.T)[:, :, xp.newaxis] * diff  # (K, N, M)
        scale_batch = xp.einsum('kni,knj->kij', weighted_diff, weighted_diff)
        scale_batch /= (resp_sum[:, xp.newaxis, xp.newaxis] + eps)
        idx = xp.arange(M)
        scale_batch[:, idx, idx] += reg_covar
        chol_batch = xp.linalg.cholesky(scale_batch)  # (K, M, M)
        # Transpose back to (M, M, K) for API compat
        scale_ = xp.transpose(scale_batch, (1, 2, 0))
        scale_decomp = xp.transpose(chol_batch, (1, 2, 0))

    return scale_, scale_decomp
