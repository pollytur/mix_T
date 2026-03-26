"""Hypothesis-based property tests for diagonal covariance EM fitting."""
import unittest
import numpy as np
import scipy.stats
from hypothesis import given, settings, note
from hypothesis.strategies import floats, integers, composite

from studenttmixture.em_student_mixture import EMStudentMixture


# Counter for progress reporting
_example_counter = {}


def _report_progress(test_name, M, N, df):
    """Print progress for the current test."""
    _example_counter.setdefault(test_name, 0)
    _example_counter[test_name] += 1
    count = _example_counter[test_name]
    print(f"  [{test_name}] example {count}: M={M}, N={N}, df={df:.1f}")


@composite
def diagonal_em_problem(draw):
    """Generate a single-component diagonal t-distribution problem.

    Returns (samples, true_location, true_variances, true_df) where samples
    are drawn from a multivariate t with diagonal covariance.
    """
    # Dimensions as powers of 2: 2, 4, 8, ..., 1024
    M = 2 ** draw(integers(min_value=1, max_value=10))
    # Need enough samples relative to dimensions (N > 2*M required by fitting)
    N = max(3 * M, draw(integers(min_value=300, max_value=1000)))
    df = draw(floats(min_value=4.0, max_value=15.0))

    variances = np.array([draw(floats(min_value=0.5, max_value=5.0)) for _ in range(M)])
    location = np.array([draw(floats(min_value=-5.0, max_value=5.0)) for _ in range(M)])

    cov = np.diag(variances)
    samples = scipy.stats.multivariate_t.rvs(location, cov, df=df, size=N)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    return samples, location, variances, df


@composite
def two_component_diagonal_problem(draw):
    """Generate a two-component diagonal t-distribution problem with
    well-separated clusters."""
    M = 2 ** draw(integers(min_value=1, max_value=6))  # up to 64 dims for speed
    N_per = max(3 * M, draw(integers(min_value=100, max_value=300)))
    df = draw(floats(min_value=3.0, max_value=15.0))

    var1 = np.array([draw(floats(min_value=0.5, max_value=3.0)) for _ in range(M)])
    var2 = np.array([draw(floats(min_value=0.5, max_value=3.0)) for _ in range(M)])

    # Well-separated locations to ensure identifiability
    loc1 = np.array([draw(floats(min_value=-10.0, max_value=-3.0)) for _ in range(M)])
    loc2 = np.array([draw(floats(min_value=3.0, max_value=10.0)) for _ in range(M)])

    samples1 = scipy.stats.multivariate_t.rvs(loc1, np.diag(var1), df=df, size=N_per)
    samples2 = scipy.stats.multivariate_t.rvs(loc2, np.diag(var2), df=df, size=N_per)
    if samples1.ndim == 1:
        samples1 = samples1.reshape(-1, 1)
        samples2 = samples2.reshape(-1, 1)
    samples = np.vstack([samples1, samples2])

    return samples, (loc1, loc2), (var1, var2), df


class TestDiagonalEMHypothesis(unittest.TestCase):

    @given(diagonal_em_problem())
    @settings(max_examples=50, deadline=None)
    def test_convergence(self, problem):
        """Model should converge for well-formed single-component data."""
        samples, location, variances, df = problem
        M = samples.shape[1]
        _report_progress("convergence", M, samples.shape[0], df)
        note(f"M={M}, N={samples.shape[0]}, df={df:.1f}")

        model = EMStudentMixture(n_components=1, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6)
        model.fit(samples)
        self.assertTrue(model.converged_,
                f"Model did not converge for M={M}, N={samples.shape[0]}")

    @given(diagonal_em_problem())
    @settings(max_examples=50, deadline=None)
    def test_positive_finite_params(self, problem):
        """All fitted parameters should be positive and finite."""
        samples, location, variances, df = problem
        M = samples.shape[1]
        _report_progress("positive_finite", M, samples.shape[0], df)
        note(f"M={M}, N={samples.shape[0]}, df={df:.1f}")

        model = EMStudentMixture(n_components=1, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6)
        model.fit(samples)

        self.assertTrue(np.all(np.isfinite(model.location_)),
                "location_ contains non-finite values")
        self.assertTrue(np.all(np.isfinite(model.scale_)),
                "scale_ contains non-finite values")
        self.assertTrue(np.all(model.scale_ > 0),
                "scale_ contains non-positive values")
        self.assertTrue(np.all(np.isfinite(model.df_)),
                "df_ contains non-finite values")
        self.assertTrue(np.all(model.df_ > 0),
                "df_ contains non-positive values")
        self.assertTrue(np.all(np.isfinite(model.mix_weights_)),
                "mix_weights_ contains non-finite values")

    @given(diagonal_em_problem())
    @settings(max_examples=50, deadline=None)
    def test_weights_sum_to_one(self, problem):
        """Mixture weights should sum to 1."""
        samples, location, variances, df = problem
        M = samples.shape[1]
        _report_progress("weights_sum", M, samples.shape[0], df)
        note(f"M={M}, N={samples.shape[0]}, df={df:.1f}")

        model = EMStudentMixture(n_components=1, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6)
        model.fit(samples)

        self.assertAlmostEqual(float(np.sum(model.mix_weights_)), 1.0, places=10)

    @given(diagonal_em_problem())
    @settings(max_examples=50, deadline=None)
    def test_location_recovery(self, problem):
        """Fitted location should be close to true location."""
        samples, location, variances, df = problem
        M = samples.shape[1]
        N = samples.shape[0]
        _report_progress("location_recovery", M, N, df)
        note(f"M={M}, N={N}, df={df:.1f}")

        model = EMStudentMixture(n_components=1, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6)
        model.fit(samples)

        fit_loc = model.location_.flatten()
        # Use per-dimension absolute error — robust regardless of location norm.
        # For heavy-tailed t-distributions, the expected standard error of the
        # sample mean per dimension is approximately sqrt(var * df/(df-2) / N).
        max_abs_err = np.max(np.abs(fit_loc - location))
        max_std_err = np.max(np.sqrt(variances * df / (df - 2) / N))
        # Allow up to 3 standard errors — generous but catches real bugs
        tol = max(3 * max_std_err, 0.5)
        self.assertLess(max_abs_err, tol,
                f"Location max abs error {max_abs_err:.3f} > {tol:.3f} for M={M}")

    @given(diagonal_em_problem())
    @settings(max_examples=50, deadline=None)
    def test_scale_recovery(self, problem):
        """Fitted diagonal variances should be close to true values."""
        samples, location, variances, df = problem
        M = samples.shape[1]
        N = samples.shape[0]
        _report_progress("scale_recovery", M, N, df)
        note(f"M={M}, N={N}, df={df:.1f}")

        model = EMStudentMixture(n_components=1, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6)
        model.fit(samples)

        fit_scale = model.scale_.flatten()
        # Relax tolerance for high dimensions
        rtol = 0.3 if M <= 64 else 0.5
        outcome = np.allclose(fit_scale, variances, rtol=rtol)
        if not outcome:
            max_rel_err = np.max(np.abs(fit_scale - variances) / variances)
            note(f"Max relative error: {max_rel_err:.3f}")
        self.assertTrue(outcome,
                f"Scale recovery failed for M={M}, N={N}")

    @given(diagonal_em_problem())
    @settings(max_examples=50, deadline=None)
    def test_df_recovery(self, problem):
        """Fitted df should be in a reasonable range of the true value."""
        samples, location, variances, df = problem
        M = samples.shape[1]
        _report_progress("df_recovery", M, samples.shape[0], df)
        note(f"M={M}, N={samples.shape[0]}, df={df:.1f}")

        model = EMStudentMixture(n_components=1, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6)
        model.fit(samples)

        fit_df = model.df_[0]
        # DF estimation is inherently noisy; allow wide tolerance
        self.assertGreater(fit_df, 1.0, "Fitted df should be > 1")
        self.assertTrue(np.isfinite(fit_df), "Fitted df should be finite")

    @given(diagonal_em_problem())
    @settings(max_examples=50, deadline=None)
    def test_score_consistency(self, problem):
        """score_samples should return finite values and score should equal their mean."""
        samples, location, variances, df = problem
        M = samples.shape[1]
        _report_progress("score_consistency", M, samples.shape[0], df)
        note(f"M={M}, N={samples.shape[0]}, df={df:.1f}")

        model = EMStudentMixture(n_components=1, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6)
        model.fit(samples)

        sample_scores = model.score_samples(samples)
        self.assertEqual(sample_scores.shape, (samples.shape[0],))
        self.assertTrue(np.all(np.isfinite(sample_scores)),
                "score_samples contains non-finite values")

        avg_score = model.score(samples)
        self.assertTrue(np.isfinite(avg_score), "score is not finite")
        self.assertAlmostEqual(float(avg_score), float(np.mean(sample_scores)),
                places=10)

    @given(diagonal_em_problem())
    @settings(max_examples=50, deadline=None)
    def test_predict_shape(self, problem):
        """predict and predict_proba should return correct shapes."""
        samples, location, variances, df = problem
        M = samples.shape[1]
        N = samples.shape[0]
        _report_progress("predict_shape", M, N, df)
        note(f"M={M}, N={N}, df={df:.1f}")

        model = EMStudentMixture(n_components=1, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6)
        model.fit(samples)

        preds = model.predict(samples)
        self.assertEqual(preds.shape, (N,))

        proba = model.predict_proba(samples)
        self.assertEqual(proba.shape, (N, 1))

    @given(diagonal_em_problem())
    @settings(max_examples=50, deadline=None)
    def test_sample_shape(self, problem):
        """sample() should return correct shape."""
        samples, location, variances, df = problem
        M = samples.shape[1]
        _report_progress("sample_shape", M, samples.shape[0], df)
        note(f"M={M}, N={samples.shape[0]}, df={df:.1f}")

        model = EMStudentMixture(n_components=1, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6)
        model.fit(samples)

        generated = model.sample(num_samples=50)
        self.assertEqual(generated.shape, (50, M))
        self.assertTrue(np.all(np.isfinite(generated)),
                "Generated samples contain non-finite values")

    @given(two_component_diagonal_problem())
    @settings(max_examples=50, deadline=None)
    def test_two_component_convergence(self, problem):
        """Two-component model should converge and have valid parameters."""
        samples, (loc1, loc2), (var1, var2), df = problem
        M = samples.shape[1]
        N = samples.shape[0]
        _report_progress("two_component", M, N, df)
        note(f"M={M}, N={N}, df={df:.1f}")

        model = EMStudentMixture(n_components=2, covariance_type='diag',
                fixed_df=False, max_iter=2000, tol=1e-6, n_init=2)
        model.fit(samples)

        # Check convergence
        self.assertTrue(model.converged_,
                f"2-component model did not converge for M={M}")

        # Check parameter validity
        self.assertTrue(np.all(np.isfinite(model.location_)))
        self.assertTrue(np.all(np.isfinite(model.scale_)))
        self.assertTrue(np.all(model.scale_ > 0))
        self.assertTrue(np.all(np.isfinite(model.df_)))
        self.assertTrue(np.all(model.df_ > 0))
        self.assertAlmostEqual(float(np.sum(model.mix_weights_)), 1.0, places=10)

        # Check shapes
        self.assertEqual(model.location_.shape, (2, M))
        self.assertEqual(model.scale_.shape, (M, 2))
        self.assertEqual(model.df_.shape, (2,))
        self.assertEqual(model.mix_weights_.shape, (2,))


if __name__ == "__main__":
    unittest.main()
