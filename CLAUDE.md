# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`studenttmixture` is a pure Python library for fitting mixtures of multivariate Student's t-distributions, analogous to scikit-learn's `GaussianMixture` and `BayesianGaussianMixture`. It provides two fitting approaches:

- **EMStudentMixture** — EM algorithm (frequentist), supports AIC/BIC for model selection, parallel restarts via `n_jobs`, optional GPU acceleration via `device='gpu'`
- **VariationalStudentMixture** — Variational mean-field approximation (Bayesian), automatically prunes unnecessary components via weight concentration prior

Dependencies: numpy, scipy, scikit-learn. Optional: cupy-cuda12x (for GPU support).

## Build & Install

Always use `uv` for dependency management. Never use `uv pip install` — use `uv sync` or `uv add` only.

```bash
uv sync              # install all dependencies (including dev)
uv add <package>     # add a new dependency
```

Build system: setuptools via `pyproject.toml`. No compiled extensions (pure Python since v1.11).

## Running Tests

Do **not** install pytest as a dependency — it is not needed. Tests use `unittest.TestCase` classes and must be run via `unittest`:

```bash
uv run python -m unittest discover -s tests -v          # all tests
uv run python -m unittest tests.test_em_mixture -v       # EM mixture tests only
uv run python -m unittest tests.test_variational_mix -v  # variational mixture tests only
uv run python -m unittest tests.test_utilities -v        # utility function tests
uv run python -m unittest tests.test_gpu_equivalence -v  # GPU/CPU equivalence (skipped without CuPy)
```

Tests generate synthetic Student's t data with known parameters, fit models, and verify recovered parameters are within tolerance of ground truth (using Herdin 2005 covariance distance for scale matrices). GPU equivalence tests are automatically skipped when CuPy is not installed.

## Architecture

### Class Hierarchy

`MixtureBaseClass` (abstract, in `mixture_base_class.py`) is the shared base for both mixture classes. It contains:
- Input validation (`check_user_params`, `check_fitting_data`, `check_inputs`)
- Prediction pipeline (`predict`, `predict_proba`, `score`, `score_samples`, `sample`)
- Log-likelihood computation (`get_loglikelihood`, `get_weighted_loglik`)
- Properties for fitted parameters (`location`, `scale`, `mix_weights`, `degrees_of_freedom`)

`EMStudentMixture` (in `em_student_mixture.py`) extends the base with EM-specific fitting (E-step/M-step), vectorized Newton-Raphson df optimization, AIC/BIC methods, parallel restarts (`n_jobs`), and optional GPU acceleration (`device`).

`VariationalStudentMixture` (in `variational_student_mixture.py`) extends the base with variational mean-field fitting, ELBO computation, and Bayesian hyperparameter updates. Uses two helper classes:
- `ParameterBundle` (`parameter_bundle.py`) — bundles all parameters updated during variational training (responsibilities, expectations, hyperparameters)
- `VariationalMixHyperparams` (`variational_hyperparams.py`) — stores fixed user-specified priors (loc_prior, scale_inv_prior, weight_concentration_prior, wishart_v0)

### Performance

- **Vectorized operations**: All loops over K components in `utilities.py` (`sq_maha_distance`, `scale_update_calcs`) and `mixture_base_class.py` (log-determinant) use batched NumPy/BLAS operations instead of Python loops.
- **Parallel restarts**: `EMStudentMixture(n_jobs=N)` runs `n_init` restarts in parallel via `joblib.Parallel`.
- **GPU backend**: `EMStudentMixture(device='gpu')` offloads E-step and M-step to GPU via CuPy. The `_backend.py` module provides `get_array_module()`, `to_device()`, and `to_numpy()` helpers. DF optimization and KMeans initialization remain on CPU. The `xp` parameter (array module) is threaded through hot-path functions.

### Key Conventions

- **Array shapes**: Data is N×M (rows=datapoints, cols=features). Scale matrices are M×M×K (stacked along axis 2). Locations are K×M. Degrees of freedom are length-K vectors.
- **Input requirements**: All data must be `np.float64` numpy arrays. 1D arrays are auto-reshaped to 2D.
- **Initialization**: Both classes support `'kmeans'` (default, better quality) and `'k++'` (faster) for cluster center initialization. Both classes duplicate this initialization logic independently.
- **Utilities** (`utilities.py`): Contains `sq_maha_distance` (squared Mahalanobis distance via batched solve) and `scale_update_calcs` (M-step scale matrix updates via einsum + batched Cholesky), shared by both fitting approaches. Both accept an optional `xp` parameter for GPU support.
- Fitted model attributes use trailing underscore convention (e.g., `df_`, `location_`, `scale_`). `df_ = None` indicates an unfitted model.
