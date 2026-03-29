"""Tests that verify GPU and CPU code paths produce equivalent results.

All tests are skipped when CuPy is not installed.
"""
import unittest
import numpy as np
import scipy.stats
from studenttmixture._backend import gpu_available
from studenttmixture.em_student_mixture import EMStudentMixture
from studenttmixture.utilities import sq_maha_distance, scale_update_calcs

_skip_reason = "CuPy not installed — no GPU available"


def _make_toy_data(n_components=3, M=3, N=500, df=4.0, seed=42):
    """Generate synthetic Student's t mixture data with known parameters."""
    rng = np.random.RandomState(seed)
    true_loc = rng.randn(n_components, M) * 5
    true_cov = np.empty((M, M, n_components))
    for k in range(n_components):
        A = rng.randn(M, M) * 0.5
        true_cov[:, :, k] = A @ A.T + np.eye(M) * 0.5

    samples = []
    for k in range(n_components):
        s = scipy.stats.multivariate_t.rvs(
            true_loc[k], true_cov[:, :, k], df=df, size=N
        )
        samples.append(s)
    X = np.vstack(samples).astype(np.float64)
    return X, true_loc, true_cov


def _make_diag_toy_data(n_components=2, M=4, N=400, df=5.0, seed=99):
    """Generate synthetic data for diagonal covariance tests."""
    rng = np.random.RandomState(seed)
    true_loc = rng.randn(n_components, M) * 3
    true_var = np.abs(rng.randn(M, n_components)) + 0.5

    samples = []
    for k in range(n_components):
        cov_k = np.diag(true_var[:, k])
        s = scipy.stats.multivariate_t.rvs(
            true_loc[k], cov_k, df=df, size=N
        )
        samples.append(s)
    X = np.vstack(samples).astype(np.float64)
    return X, true_loc, true_var


@unittest.skipUnless(gpu_available(), _skip_reason)
class TestGPUCPUEquivalence(unittest.TestCase):
    """Verify that GPU (device='gpu') and CPU (device='cpu') produce
    numerically equivalent results within floating-point tolerance."""

    # Tolerances — GPU uses different BLAS/LAPACK kernels so results
    # will not be bit-identical, but should agree to ~1e-6 relative.
    RTOL = 1e-5
    ATOL = 1e-7

    # ------------------------------------------------------------------
    # Full covariance
    # ------------------------------------------------------------------

    def test_full_fit_equivalence(self):
        """Full-cov EM fit: GPU and CPU should converge to same parameters."""
        X, _, _ = _make_toy_data(n_components=2, M=3, N=600, seed=7)

        cpu_model = EMStudentMixture(
            n_components=2, max_iter=200, tol=1e-7,
            random_state=0, covariance_type="full", device="cpu",
        )
        gpu_model = EMStudentMixture(
            n_components=2, max_iter=200, tol=1e-7,
            random_state=0, covariance_type="full", device="gpu",
        )

        cpu_model.fit(X)
        gpu_model.fit(X)

        self.assertTrue(cpu_model.converged_)
        self.assertTrue(gpu_model.converged_)

        np.testing.assert_allclose(
            gpu_model.location_, cpu_model.location_,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="location_ mismatch (full)",
        )
        np.testing.assert_allclose(
            gpu_model.scale_, cpu_model.scale_,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="scale_ mismatch (full)",
        )
        np.testing.assert_allclose(
            gpu_model.mix_weights_, cpu_model.mix_weights_,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="mix_weights_ mismatch (full)",
        )
        np.testing.assert_allclose(
            gpu_model.df_, cpu_model.df_,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="df_ mismatch (full)",
        )

    def test_full_score_equivalence(self):
        """Full-cov: score / predict on a trained model should match."""
        X, _, _ = _make_toy_data(n_components=2, M=3, N=600, seed=11)

        cpu_model = EMStudentMixture(
            n_components=2, max_iter=300, tol=1e-8,
            random_state=0, covariance_type="full", device="cpu",
        )
        gpu_model = EMStudentMixture(
            n_components=2, max_iter=300, tol=1e-8,
            random_state=0, covariance_type="full", device="gpu",
        )
        cpu_model.fit(X)
        gpu_model.fit(X)

        cpu_score = cpu_model.score(X)
        gpu_score = gpu_model.score(X)
        np.testing.assert_allclose(
            gpu_score, cpu_score, rtol=self.RTOL, atol=self.ATOL,
            err_msg="score mismatch (full)",
        )

        np.testing.assert_array_equal(
            gpu_model.predict(X), cpu_model.predict(X),
            err_msg="predict mismatch (full)",
        )

    def test_full_fixed_df_equivalence(self):
        """Full-cov with fixed_df=False: df optimisation should match."""
        X, _, _ = _make_toy_data(n_components=2, M=3, N=800, df=6.0, seed=21)

        cpu_model = EMStudentMixture(
            n_components=2, max_iter=300, tol=1e-8,
            random_state=0, covariance_type="full",
            fixed_df=False, device="cpu",
        )
        gpu_model = EMStudentMixture(
            n_components=2, max_iter=300, tol=1e-8,
            random_state=0, covariance_type="full",
            fixed_df=False, device="gpu",
        )
        cpu_model.fit(X)
        gpu_model.fit(X)

        np.testing.assert_allclose(
            gpu_model.df_, cpu_model.df_,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="df_ mismatch (full, fixed_df=False)",
        )
        np.testing.assert_allclose(
            gpu_model.location_, cpu_model.location_,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="location_ mismatch (full, fixed_df=False)",
        )

    # ------------------------------------------------------------------
    # Diagonal covariance
    # ------------------------------------------------------------------

    def test_diag_fit_equivalence(self):
        """Diag-cov EM fit: GPU and CPU should converge to same parameters."""
        X, _, _ = _make_diag_toy_data(n_components=2, M=4, N=500, seed=33)

        cpu_model = EMStudentMixture(
            n_components=2, max_iter=200, tol=1e-7,
            random_state=0, covariance_type="diag", device="cpu",
        )
        gpu_model = EMStudentMixture(
            n_components=2, max_iter=200, tol=1e-7,
            random_state=0, covariance_type="diag", device="gpu",
        )

        cpu_model.fit(X)
        gpu_model.fit(X)

        self.assertTrue(cpu_model.converged_)
        self.assertTrue(gpu_model.converged_)

        np.testing.assert_allclose(
            gpu_model.location_, cpu_model.location_,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="location_ mismatch (diag)",
        )
        np.testing.assert_allclose(
            gpu_model.scale_, cpu_model.scale_,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="scale_ mismatch (diag)",
        )
        np.testing.assert_allclose(
            gpu_model.mix_weights_, cpu_model.mix_weights_,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="mix_weights_ mismatch (diag)",
        )

    def test_diag_score_equivalence(self):
        """Diag-cov: score / predict on a trained model should match."""
        X, _, _ = _make_diag_toy_data(n_components=2, M=4, N=500, seed=44)

        cpu_model = EMStudentMixture(
            n_components=2, max_iter=300, tol=1e-8,
            random_state=0, covariance_type="diag", device="cpu",
        )
        gpu_model = EMStudentMixture(
            n_components=2, max_iter=300, tol=1e-8,
            random_state=0, covariance_type="diag", device="gpu",
        )
        cpu_model.fit(X)
        gpu_model.fit(X)

        cpu_score = cpu_model.score(X)
        gpu_score = gpu_model.score(X)
        np.testing.assert_allclose(
            gpu_score, cpu_score, rtol=self.RTOL, atol=self.ATOL,
            err_msg="score mismatch (diag)",
        )

        np.testing.assert_array_equal(
            gpu_model.predict(X), cpu_model.predict(X),
            err_msg="predict mismatch (diag)",
        )

    # ------------------------------------------------------------------
    # Utility-level checks
    # ------------------------------------------------------------------

    def test_sq_maha_distance_full_equivalence(self):
        """sq_maha_distance (full) should match between CPU and GPU."""
        import cupy as cp

        rng = np.random.RandomState(55)
        N, M, K = 200, 5, 3
        X = rng.randn(N, M).astype(np.float64)
        loc_ = rng.randn(K, M).astype(np.float64)
        # Build valid Cholesky factors
        scale_chol = np.empty((M, M, K))
        for k in range(K):
            A = rng.randn(M, M)
            S = A @ A.T + np.eye(M)
            scale_chol[:, :, k] = np.linalg.cholesky(S)

        cpu_result = sq_maha_distance(X, loc_, scale_chol, covariance_type="full")
        gpu_result = sq_maha_distance(
            cp.asarray(X), cp.asarray(loc_), cp.asarray(scale_chol),
            covariance_type="full", xp=cp,
        )

        np.testing.assert_allclose(
            gpu_result.get(), cpu_result,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="sq_maha_distance (full) mismatch",
        )

    def test_sq_maha_distance_diag_equivalence(self):
        """sq_maha_distance (diag) should match between CPU and GPU."""
        import cupy as cp

        rng = np.random.RandomState(66)
        N, M, K = 200, 5, 3
        X = rng.randn(N, M).astype(np.float64)
        loc_ = rng.randn(K, M).astype(np.float64)
        scale_diag = np.abs(rng.randn(M, K)) + 0.1

        cpu_result = sq_maha_distance(X, loc_, scale_diag, covariance_type="diag")
        gpu_result = sq_maha_distance(
            cp.asarray(X), cp.asarray(loc_), cp.asarray(scale_diag),
            covariance_type="diag", xp=cp,
        )

        np.testing.assert_allclose(
            gpu_result.get(), cpu_result,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="sq_maha_distance (diag) mismatch",
        )

    def test_scale_update_calcs_full_equivalence(self):
        """scale_update_calcs (full) should match between CPU and GPU."""
        import cupy as cp

        rng = np.random.RandomState(77)
        N, M, K = 300, 4, 2
        X = rng.randn(N, M).astype(np.float64)
        loc_ = rng.randn(K, M).astype(np.float64)
        ru = np.abs(rng.randn(N, K)) + 0.01
        resp_sum = ru.sum(axis=0)
        reg_covar = 1e-6

        cpu_scale, cpu_chol = scale_update_calcs(
            X, ru, loc_, resp_sum, reg_covar, covariance_type="full",
        )
        gpu_scale, gpu_chol = scale_update_calcs(
            cp.asarray(X), cp.asarray(ru), cp.asarray(loc_),
            cp.asarray(resp_sum), reg_covar, covariance_type="full", xp=cp,
        )

        np.testing.assert_allclose(
            gpu_scale.get(), cpu_scale,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="scale_update_calcs scale (full) mismatch",
        )
        np.testing.assert_allclose(
            gpu_chol.get(), cpu_chol,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="scale_update_calcs cholesky (full) mismatch",
        )

    def test_scale_update_calcs_diag_equivalence(self):
        """scale_update_calcs (diag) should match between CPU and GPU."""
        import cupy as cp

        rng = np.random.RandomState(88)
        N, M, K = 300, 4, 2
        X = rng.randn(N, M).astype(np.float64)
        loc_ = rng.randn(K, M).astype(np.float64)
        ru = np.abs(rng.randn(N, K)) + 0.01
        resp_sum = ru.sum(axis=0)
        reg_covar = 1e-6

        cpu_scale, cpu_decomp = scale_update_calcs(
            X, ru, loc_, resp_sum, reg_covar, covariance_type="diag",
        )
        gpu_scale, gpu_decomp = scale_update_calcs(
            cp.asarray(X), cp.asarray(ru), cp.asarray(loc_),
            cp.asarray(resp_sum), reg_covar, covariance_type="diag", xp=cp,
        )

        np.testing.assert_allclose(
            gpu_scale.get(), cpu_scale,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="scale_update_calcs scale (diag) mismatch",
        )
        np.testing.assert_allclose(
            gpu_decomp.get(), cpu_decomp,
            rtol=self.RTOL, atol=self.ATOL,
            err_msg="scale_update_calcs decomp (diag) mismatch",
        )


if __name__ == "__main__":
    unittest.main()
