"""Shared functions for use by both variational and EM mixture models."""
import numpy as np
import scipy
from scipy import linalg


def sq_maha_distance(X, loc_, scale, covariance_type='full'):
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
    """
    sq_maha_dist = np.empty((X.shape[0], loc_.shape[0]))
    for i in range(loc_.shape[0]):
        diff = X - loc_[None,i,:]
        if covariance_type == 'diag':
            # scale is M x K: diagonal variances of scale matrices
            sq_maha_dist[:,i] = ((diff**2) / scale[:,i][np.newaxis, :]).sum(axis=1)
        else:
            # scale is M x M x K: Cholesky decompositions of scale matrices
            diff = scipy.linalg.solve_triangular(
                    scale[:,:,i], diff.T, lower=True)
            sq_maha_dist[:,i] = (diff**2).sum(axis=0)
    return sq_maha_dist


def scale_update_calcs(X, ru, loc_, resp_sum, reg_covar, covariance_type='full'):
    """Updates the scale (aka covariance) matrices as part of the M-
    step for EM and as the parameter update for variational methods."""
    M = loc_.shape[1]
    K = loc_.shape[0]

    if covariance_type == 'diag':
        # scale_ is M x K: diagonal variances of scale matrices
        # scale_decomp is M x K: same diagonal variances
        scale_ = np.empty((M, K))
        scale_decomp = np.empty((M, K))
    else:
        # scale_ is M x M x K: full scale matrices
        # scale_decomp is M x M x K: Cholesky decompositions
        scale_ = np.empty((M, M, K))
        scale_decomp = np.empty((M, M, K))

    for i in range(K):
        diff = X - loc_[i:i+1,:]
        with np.errstate(under='ignore'):
            if covariance_type == 'diag':
                # Compute only diagonal: weighted variance per feature
                diag_vals = np.sum(ru[:,i:i+1] * diff**2, axis=0) \
                        / (resp_sum[i] + 10 * np.finfo(scale_.dtype).eps)
                diag_vals += reg_covar
                scale_[:,i] = diag_vals
                # For 'diag', scale_decomp stores the same diagonal variances
                scale_decomp[:,i] = diag_vals
            else:
                scale_[:,:,i] = np.dot(ru[:,i] * diff.T, diff) \
                        / (resp_sum[i] + 10 * np.finfo(scale_.dtype).eps)
                scale_[:,:,i].flat[::M+1] += reg_covar
                scale_decomp[:,:,i] = np.linalg.cholesky(scale_[:,:,i])
    return scale_, scale_decomp
