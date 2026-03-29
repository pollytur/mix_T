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

    use_gpu = xp is not np

    if use_gpu:
        return _sq_maha_distance_gpu(X, loc_, scale, covariance_type, xp)
    return _sq_maha_distance_cpu(X, loc_, scale, covariance_type)


def _sq_maha_distance_cpu(X, loc_, scale, covariance_type):
    """CPU path: loop over K components with scipy triangular solve."""
    sq_maha_dist = np.empty((X.shape[0], loc_.shape[0]))
    for i in range(loc_.shape[0]):
        diff = X - loc_[None, i, :]
        if covariance_type == 'diag':
            sq_maha_dist[:, i] = ((diff**2) / scale[:, i][np.newaxis, :]).sum(axis=1)
        else:
            diff = scipy.linalg.solve_triangular(
                    scale[:, :, i], diff.T, lower=True)
            sq_maha_dist[:, i] = (diff**2).sum(axis=0)
    return sq_maha_dist


def _sq_maha_distance_gpu(X, loc_, scale, covariance_type, xp):
    """GPU path: vectorized operations to minimize kernel launches."""
    if covariance_type == 'diag':
        diff = X[:, xp.newaxis, :] - loc_[xp.newaxis, :, :]
        sq_maha_dist = xp.sum(diff**2 / scale.T[xp.newaxis, :, :], axis=2)
    else:
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

    use_gpu = xp is not np

    if use_gpu:
        return _scale_update_calcs_gpu(X, ru, loc_, resp_sum, reg_covar,
                                       covariance_type, xp)
    return _scale_update_calcs_cpu(X, ru, loc_, resp_sum, reg_covar,
                                   covariance_type)


def _scale_update_calcs_cpu(X, ru, loc_, resp_sum, reg_covar, covariance_type):
    """CPU path: loop over K components with direct BLAS calls."""
    M = loc_.shape[1]
    K = loc_.shape[0]

    if covariance_type == 'diag':
        scale_ = np.empty((M, K))
        scale_decomp = np.empty((M, K))
    else:
        scale_ = np.empty((M, M, K))
        scale_decomp = np.empty((M, M, K))

    for i in range(K):
        diff = X - loc_[i:i+1, :]
        with np.errstate(under='ignore'):
            if covariance_type == 'diag':
                diag_vals = np.sum(ru[:, i:i+1] * diff**2, axis=0) \
                        / (resp_sum[i] + 10 * np.finfo(scale_.dtype).eps)
                diag_vals += reg_covar
                scale_[:, i] = diag_vals
                scale_decomp[:, i] = diag_vals
            else:
                scale_[:, :, i] = np.dot(ru[:, i] * diff.T, diff) \
                        / (resp_sum[i] + 10 * np.finfo(scale_.dtype).eps)
                scale_[:, :, i].flat[::M+1] += reg_covar
                scale_decomp[:, :, i] = np.linalg.cholesky(scale_[:, :, i])
    return scale_, scale_decomp


def _scale_update_calcs_gpu(X, ru, loc_, resp_sum, reg_covar, covariance_type, xp):
    """GPU path: vectorized operations to minimize kernel launches."""
    M = loc_.shape[1]
    eps = 10 * np.finfo(np.float64).eps

    if covariance_type == 'diag':
        diff = X[xp.newaxis, :, :] - loc_[:, xp.newaxis, :]  # (K, N, M)
        scale_ = xp.sum(ru.T[:, :, xp.newaxis] * diff**2, axis=1)  # (K, M)
        scale_ /= (resp_sum[:, xp.newaxis] + eps)
        scale_ += reg_covar
        scale_ = scale_.T  # (M, K) for API compat
        scale_decomp = scale_.copy()
    else:
        diff = X[xp.newaxis, :, :] - loc_[:, xp.newaxis, :]  # (K, N, M)
        weighted_diff = xp.sqrt(ru.T)[:, :, xp.newaxis] * diff  # (K, N, M)
        scale_batch = xp.einsum('kni,knj->kij', weighted_diff, weighted_diff)
        scale_batch /= (resp_sum[:, xp.newaxis, xp.newaxis] + eps)
        idx = xp.arange(M)
        scale_batch[:, idx, idx] += reg_covar
        chol_batch = xp.linalg.cholesky(scale_batch)  # (K, M, M)
        scale_ = xp.transpose(scale_batch, (1, 2, 0))
        scale_decomp = xp.transpose(chol_batch, (1, 2, 0))

    return scale_, scale_decomp
