"""Shared functions for use by both variational and EM mixture models."""
import numpy as np
import scipy
from scipy import linalg


def sq_maha_distance(X, loc_, scale_cholesky_):
    """Computes the mahalanobis distance from each row of X
    using the k (for k clusters) rows of loc_ and the k dim3
    cholesky decompositions of the scale matrices.
    TECHNICALLY this is the squared mahalanobis distance and
    is therefore referred to as sq_maha throughout."""
    sq_maha_dist = np.empty((X.shape[0], loc_.shape[0]))
    for i in range(loc_.shape[0]):
        diff = X - loc_[None,i,:]
        diff = scipy.linalg.solve_triangular(
                scale_cholesky_[:,:,i], diff.T, lower=True)
        sq_maha_dist[:,i] = (diff**2).sum(axis=0)
    return sq_maha_dist


def scale_update_calcs(X, ru, loc_, resp_sum, reg_covar):
    """Updates the scale (aka covariance) matrices as part of the M-
    step for EM and as the parameter update for variational methods."""
    scale_ = np.empty((loc_.shape[1], loc_.shape[1], loc_.shape[0]))
    scale_cholesky_ = np.empty((loc_.shape[1], loc_.shape[1],
        loc_.shape[0]))

    for i in range(loc_.shape[0]):
        diff = X - loc_[i:i+1,:]
        with np.errstate(under='ignore'):
            scale_[:,:,i] = np.dot(ru[:,i] * diff.T, diff) \
                    / (resp_sum[i] + 10 * np.finfo(scale_.dtype).eps)
        scale_[:,:,i].flat[::scale_.shape[0]+1] += reg_covar

        scale_cholesky_[:,:,i] = np.linalg.cholesky(scale_[:,:,i])
    return scale_, scale_cholesky_
