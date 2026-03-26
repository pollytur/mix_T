# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`studenttmixture` is a pure Python library for fitting mixtures of multivariate Student's t-distributions, analogous to scikit-learn's `GaussianMixture` and `BayesianGaussianMixture`. It provides two fitting approaches:

- **EMStudentMixture** — EM algorithm (frequentist), supports AIC/BIC for model selection
- **VariationalStudentMixture** — Variational mean-field approximation (Bayesian), automatically prunes unnecessary components via weight concentration prior

Dependencies: numpy, scipy, scikit-learn.

## Build & Install

```bash
pip install -e .          # editable install from repo root
pip install studenttmixture  # from PyPI
```

Build system: setuptools via `pyproject.toml`. No compiled extensions (pure Python since v1.11).

## Running Tests

```bash
python -m pytest tests/                    # all tests
python -m pytest tests/test_em_mixture.py  # EM mixture tests only
python -m pytest tests/test_variational_mix.py  # variational mixture tests only
python -m pytest tests/test_utilities.py   # utility function tests
```

Tests use `unittest.TestCase` classes. They generate synthetic Student's t data with known parameters, fit models, and verify recovered parameters are within tolerance of ground truth (using Herdin 2005 covariance distance for scale matrices).

## Architecture

### Class Hierarchy

`MixtureBaseClass` (abstract, in `mixture_base_class.py`) is the shared base for both mixture classes. It contains:
- Input validation (`check_user_params`, `check_fitting_data`, `check_inputs`)
- Prediction pipeline (`predict`, `predict_proba`, `score`, `score_samples`, `sample`)
- Log-likelihood computation (`get_loglikelihood`, `get_weighted_loglik`)
- Properties for fitted parameters (`location`, `scale`, `mix_weights`, `degrees_of_freedom`)

`EMStudentMixture` (in `em_student_mixture.py`) extends the base with EM-specific fitting (E-step/M-step), Newton-Raphson df optimization, and AIC/BIC methods.

`VariationalStudentMixture` (in `variational_student_mixture.py`) extends the base with variational mean-field fitting, ELBO computation, and Bayesian hyperparameter updates. Uses two helper classes:
- `ParameterBundle` (`parameter_bundle.py`) — bundles all parameters updated during variational training (responsibilities, expectations, hyperparameters)
- `VariationalMixHyperparams` (`variational_hyperparams.py`) — stores fixed user-specified priors (loc_prior, scale_inv_prior, weight_concentration_prior, wishart_v0)

### Key Conventions

- **Array shapes**: Data is N×M (rows=datapoints, cols=features). Scale matrices are M×M×K (stacked along axis 2). Locations are K×M. Degrees of freedom are length-K vectors.
- **Input requirements**: All data must be `np.float64` numpy arrays. 1D arrays are auto-reshaped to 2D.
- **Initialization**: Both classes support `'kmeans'` (default, better quality) and `'k++'` (faster) for cluster center initialization. Both classes duplicate this initialization logic independently.
- **Utilities** (`utilities.py`): Contains `sq_maha_distance` (squared Mahalanobis distance via Cholesky solve) and `scale_update_calcs` (M-step scale matrix updates), shared by both fitting approaches.
- Fitted model attributes use trailing underscore convention (e.g., `df_`, `location_`, `scale_`). `df_ = None` indicates an unfitted model.
