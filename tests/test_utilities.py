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



if __name__ == "__main__":
    unittest.main()
