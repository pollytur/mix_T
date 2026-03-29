"""Benchmark: loop-based vs vectorized implementations of the hot-path functions.

Run on a GPU machine with CuPy installed to compare:
  - Loop-based (original) vs vectorized for both CPU (numpy) and GPU (cupy)
  - CPU vs GPU for each approach

Usage:
    python benchmarks/bench_loop_vs_vectorized.py
"""
import numpy as np
import scipy
import time

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("CuPy not installed — GPU benchmarks will be skipped.\n")


# ── Loop-based implementations (original) ────────────────────────────

def sq_maha_loop(X, loc_, scale, covariance_type='full', xp=None):
    if xp is None:
        xp = np
    _scipy_solve = scipy.linalg.solve_triangular
    sq_maha_dist = xp.empty((X.shape[0], loc_.shape[0]))
    for i in range(loc_.shape[0]):
        diff = X - loc_[None, i, :]
        if covariance_type == 'diag':
            sq_maha_dist[:, i] = ((diff**2) / scale[:, i][xp.newaxis, :]).sum(axis=1)
        else:
            if xp is np:
                diff = _scipy_solve(scale[:, :, i], diff.T, lower=True)
            else:
                # cupy has its own solve_triangular in cupyx.scipy.linalg
                import cupyx.scipy.linalg
                diff = cupyx.scipy.linalg.solve_triangular(
                    scale[:, :, i], diff.T, lower=True)
            sq_maha_dist[:, i] = (diff**2).sum(axis=0)
    return sq_maha_dist


def scale_update_loop(X, ru, loc_, resp_sum, reg_covar, covariance_type='full', xp=None):
    if xp is None:
        xp = np
    M = loc_.shape[1]
    K = loc_.shape[0]
    eps = 10 * np.finfo(np.float64).eps
    if covariance_type == 'diag':
        scale_ = xp.empty((M, K))
        scale_decomp = xp.empty((M, K))
    else:
        scale_ = xp.empty((M, M, K))
        scale_decomp = xp.empty((M, M, K))
    for i in range(K):
        diff = X - loc_[i:i+1, :]
        if covariance_type == 'diag':
            diag_vals = xp.sum(ru[:, i:i+1] * diff**2, axis=0) \
                    / (resp_sum[i] + eps)
            diag_vals += reg_covar
            scale_[:, i] = diag_vals
            scale_decomp[:, i] = diag_vals
        else:
            scale_[:, :, i] = xp.dot(ru[:, i] * diff.T, diff) \
                    / (resp_sum[i] + eps)
            scale_[:, :, i].flat[::M+1] += reg_covar
            scale_decomp[:, :, i] = xp.linalg.cholesky(scale_[:, :, i])
    return scale_, scale_decomp


# ── Vectorized implementations (current) ─────────────────────────────

def sq_maha_vectorized(X, loc_, scale, covariance_type='full', xp=None):
    if xp is None:
        xp = np
    if covariance_type == 'diag':
        diff = X[:, xp.newaxis, :] - loc_[xp.newaxis, :, :]
        sq_maha_dist = xp.sum(diff**2 / scale.T[xp.newaxis, :, :], axis=2)
    else:
        L = xp.transpose(scale, (2, 0, 1))
        diff = X[xp.newaxis, :, :] - loc_[:, xp.newaxis, :]
        solved = xp.linalg.solve(L, diff.transpose(0, 2, 1))
        sq_maha_dist = xp.sum(solved**2, axis=1).T
    return sq_maha_dist


def scale_update_vectorized(X, ru, loc_, resp_sum, reg_covar, covariance_type='full', xp=None):
    if xp is None:
        xp = np
    M = loc_.shape[1]
    K = loc_.shape[0]
    eps = 10 * np.finfo(np.float64).eps
    if covariance_type == 'diag':
        diff = X[xp.newaxis, :, :] - loc_[:, xp.newaxis, :]
        scale_ = xp.sum(ru.T[:, :, xp.newaxis] * diff**2, axis=1)
        scale_ /= (resp_sum[:, xp.newaxis] + eps)
        scale_ += reg_covar
        scale_ = scale_.T
        scale_decomp = scale_.copy()
    else:
        diff = X[xp.newaxis, :, :] - loc_[:, xp.newaxis, :]
        weighted_diff = xp.sqrt(ru.T)[:, :, xp.newaxis] * diff
        scale_batch = xp.einsum('kni,knj->kij', weighted_diff, weighted_diff)
        scale_batch /= (resp_sum[:, xp.newaxis, xp.newaxis] + eps)
        idx = xp.arange(M)
        scale_batch[:, idx, idx] += reg_covar
        chol_batch = xp.linalg.cholesky(scale_batch)
        scale_ = xp.transpose(scale_batch, (1, 2, 0))
        scale_decomp = xp.transpose(chol_batch, (1, 2, 0))
    return scale_, scale_decomp


# ── Benchmark harness ─────────────────────────────────────────────────

def make_test_data(N, M, K, xp=np):
    rng = np.random.RandomState(42)
    X = xp.asarray(rng.randn(N, M).astype(np.float64))
    loc_ = xp.asarray(rng.randn(K, M).astype(np.float64))
    # Valid Cholesky factors
    scale_chol = np.empty((M, M, K))
    for k in range(K):
        A = rng.randn(M, M)
        S = A @ A.T + np.eye(M)
        scale_chol[:, :, k] = np.linalg.cholesky(S)
    scale_chol = xp.asarray(scale_chol)
    scale_diag = xp.asarray(np.abs(rng.randn(M, K)) + 0.5)
    ru = xp.asarray(np.abs(rng.randn(N, K)).astype(np.float64) + 0.01)
    resp_sum = ru.sum(axis=0)
    return X, loc_, scale_chol, scale_diag, ru, resp_sum


def time_fn(fn, args, warmup=3, reps=20, xp=np):
    """Time a function, with warmup and GPU sync."""
    for _ in range(warmup):
        fn(*args)
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    elapsed = (time.perf_counter() - start) / reps
    return elapsed


def run_benchmarks():
    configs = [
        ('N=500,  M=3,  K=2',   500,   3,  2),
        ('N=5k,   M=10, K=5',   5000,  10,  5),
        ('N=20k,  M=20, K=10',  20000, 20, 10),
        ('N=50k,  M=50, K=10',  50000, 50, 10),
    ]

    devices = [('CPU', np)]
    if HAS_GPU:
        devices.append(('GPU', cp))

    for label, N, M, K in configs:
        print(f'{"="*60}')
        print(f'  {label}')
        print(f'{"="*60}')

        for dev_name, xp in devices:
            X, loc_, scale_chol, scale_diag, ru, resp_sum = make_test_data(N, M, K, xp)

            for cov_type, scale in [('full', scale_chol), ('diag', scale_diag)]:
                t_loop = time_fn(
                    sq_maha_loop, (X, loc_, scale, cov_type, xp), xp=xp)
                t_vec = time_fn(
                    sq_maha_vectorized, (X, loc_, scale, cov_type, xp), xp=xp)
                ratio = t_loop / t_vec if t_vec > 0 else float('inf')
                winner = 'vectorized' if ratio > 1 else 'loop'
                print(f'  {dev_name} sq_maha {cov_type:4s}: '
                      f'loop={t_loop*1000:8.2f}ms  vec={t_vec*1000:8.2f}ms  '
                      f'ratio={ratio:.2f}x  -> {winner}')

            for cov_type, scale in [('full', scale_chol), ('diag', scale_diag)]:
                t_loop = time_fn(
                    scale_update_loop,
                    (X, ru, loc_, resp_sum, 1e-6, cov_type, xp), xp=xp)
                t_vec = time_fn(
                    scale_update_vectorized,
                    (X, ru, loc_, resp_sum, 1e-6, cov_type, xp), xp=xp)
                ratio = t_loop / t_vec if t_vec > 0 else float('inf')
                winner = 'vectorized' if ratio > 1 else 'loop'
                print(f'  {dev_name} scale   {cov_type:4s}: '
                      f'loop={t_loop*1000:8.2f}ms  vec={t_vec*1000:8.2f}ms  '
                      f'ratio={ratio:.2f}x  -> {winner}')
            print()
        print()


if __name__ == '__main__':
    run_benchmarks()
