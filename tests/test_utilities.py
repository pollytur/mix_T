"""Unit tests for the C extension."""
import unittest
import sys
import numpy as np
import scipy
from scipy import stats, spatial
from studenttmixture.em_student_mixture import EMStudentMixture
from studenttmixture.utilities import sq_maha_distance

class TestSqMahaDistExtension(unittest.TestCase):
    
    def test_sq_maha_distance(self):
        """Test the squared mahalanobis distance calculation by generating a set
        of random datapoints and measuring squared mahalanobis distance to a
        prespecified location & scale matrix distribution, then compare the
        result with scipy's mahalanobis function."""
        print("*********************")
        np.random.seed(123)
        X = np.random.uniform(low=-10,high=10,size=(1000,3))
        #Arbitrary scale matrices...
        covmat1 = np.asarray([[0.025, 0.0075, 0.00175],
                            [0.0075, 0.0070, 0.00135],
                            [0.00175, 0.00135, 0.00043]])
        covmat2 = np.asarray([[1.2, 0.1, 0.42],
                            [0.1, 0.5, 0.0035],
                            [0.42, 0.0035, 0.35]])
        chole_covmat1 = np.linalg.cholesky(covmat1)
        chole_covmat2 = np.linalg.cholesky(covmat2)
        chole_covmat = np.stack([chole_covmat1, chole_covmat2], axis=-1)
        #Arbitrary locations...
        loc = np.asarray([[0.156,-0.324,0.456],[-2.5,3.6,1.2]])
        # In general not good to invert a matrix, but for testing
        # purposes only...
        scale_inv1 = np.linalg.inv(covmat1)
        scale_inv2 = np.linalg.inv(covmat2)

        extension_dist = np.zeros((X.shape[0], 2))
        true_dist = np.empty((X.shape[0], 2))
        extension_dist = sq_maha_distance(X, loc, chole_covmat)
        #squaredMahaDistance(X, loc, chole_inv_cov, extension_dist)

        for i in range(X.shape[0]):
            true_dist[i,0] = scipy.spatial.distance.mahalanobis(X[i,:], loc[0,:],
                            scale_inv1)**2
            true_dist[i,1] = scipy.spatial.distance.mahalanobis(X[i,:], loc[1,:],
                            scale_inv2)**2
        outcome = np.allclose(extension_dist, true_dist)
        print("Does scipy's mahalanobis match "
                f"the C extension-calculated distance? {outcome}")
        self.assertTrue(outcome)
        print('\n')



    def test_sq_maha_distance_diag(self):
        """Test the squared mahalanobis distance calculation with covariance_type='diag'
        by comparing against scipy's mahalanobis function using a diagonal scale matrix."""
        print("*********************")
        np.random.seed(123)
        X = np.random.uniform(low=-10,high=10,size=(1000,3))
        #Diagonal scale matrices
        diag1 = np.array([0.025, 0.007, 0.00043])
        diag2 = np.array([1.2, 0.5, 0.35])
        #scale for 'diag' is M x K (diagonal variances)
        scale_diag = np.stack([diag1, diag2], axis=-1)
        #Arbitrary locations
        loc = np.asarray([[0.156,-0.324,0.456],[-2.5,3.6,1.2]])

        diag_dist = sq_maha_distance(X, loc, scale_diag, covariance_type='diag')

        #Compare against scipy using inverse of diagonal matrices
        scale_inv1 = np.diag(1.0 / diag1)
        scale_inv2 = np.diag(1.0 / diag2)
        true_dist = np.empty((X.shape[0], 2))
        for i in range(X.shape[0]):
            true_dist[i,0] = scipy.spatial.distance.mahalanobis(X[i,:], loc[0,:],
                            scale_inv1)**2
            true_dist[i,1] = scipy.spatial.distance.mahalanobis(X[i,:], loc[1,:],
                            scale_inv2)**2
        outcome = np.allclose(diag_dist, true_dist)
        print("Does scipy's mahalanobis match "
                f"the diag sq_maha_distance? {outcome}")
        self.assertTrue(outcome)
        print('\n')


    def test_scale_update_calcs_diag(self):
        """Test scale_update_calcs with covariance_type='diag' produces M x K
        output with correct diagonal variance values."""
        from studenttmixture.utilities import scale_update_calcs
        print("*********************")
        np.random.seed(42)
        N, M, K = 200, 3, 2
        X = np.random.randn(N, M)
        loc_ = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        # Equal responsibilities and unit gamma weights
        ru = np.zeros((N, K))
        ru[:100, 0] = 1.0
        ru[100:, 1] = 1.0
        resp_sum = np.sum(ru, axis=0)

        scale_, scale_decomp = scale_update_calcs(X, ru, loc_, resp_sum,
                reg_covar=1e-6, covariance_type='diag')

        # Check shapes are M x K
        shape_outcome = scale_.shape == (M, K) and scale_decomp.shape == (M, K)
        print(f"Are output shapes M x K? {shape_outcome}")
        self.assertTrue(shape_outcome)

        # For 'diag', scale_decomp stores the same diagonal variances as scale_
        same_outcome = np.allclose(scale_decomp, scale_)
        print(f"Is scale_decomp equal to scale_ for diag? {same_outcome}")
        self.assertTrue(same_outcome)

        # Verify diagonal values match expected weighted variance
        for i in range(K):
            diff = X - loc_[i:i+1,:]
            expected_diag = np.sum(ru[:,i:i+1] * diff**2, axis=0) / resp_sum[i] + 1e-6
            vals_outcome = np.allclose(scale_[:,i], expected_diag)
            self.assertTrue(vals_outcome)

        print("Diagonal scale values match expected weighted variance: True")
        print('\n')


if __name__ == "__main__":
    unittest.main()
