"""Benchmark: end-to-end EM fitting on CPU vs GPU, with loop vs vectorized utilities.

This script temporarily patches utilities.py to swap between loop-based and
vectorized implementations, then runs full EM fits and reports wall-clock times.

Usage:
    python benchmarks/bench_end_to_end.py
"""
import numpy as np
import scipy.stats
import time
import sys

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("CuPy not installed — GPU benchmarks will be skipped.\n")

from studenttmixture.em_student_mixture import EMStudentMixture


def generate_data(N, M, K, df=4.0, seed=42):
    """Generate a K-component Student's t mixture."""
    rng = np.random.RandomState(seed)
    true_loc = rng.randn(K, M) * 5
    true_cov = np.empty((M, M, K))
    for k in range(K):
        A = rng.randn(M, M) * 0.3
        true_cov[:, :, k] = A @ A.T + np.eye(M) * 0.5

    samples = []
    for k in range(K):
        s = scipy.stats.multivariate_t.rvs(
            true_loc[k], true_cov[:, :, k], df=df, size=N // K)
        samples.append(s)
    return np.vstack(samples).astype(np.float64)


def time_fit(X, n_components, covariance_type, device, max_iter=100, reps=3):
    """Fit an EM model and return (mean_time, converged)."""
    times = []
    converged = False
    for _ in range(reps):
        model = EMStudentMixture(
            n_components=n_components,
            max_iter=max_iter,
            tol=1e-12,  # force full max_iter iterations for fair timing
            random_state=0,
            covariance_type=covariance_type,
            device=device,
        )
        start = time.perf_counter()
        model.fit(X)
        if HAS_GPU and device == 'gpu':
            cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        converged = model.converged_
    return np.mean(times), converged


def main():
    configs = [
        # (label, N, M, K)
        ('Small:  N=1k,  M=5,  K=3',   1000,   5,  3),
        ('Medium: N=10k, M=10, K=5',   10000,  10,  5),
        ('Large:  N=50k, M=20, K=5',   50000,  20,  5),
        ('XL:     N=50k, M=50, K=10',  50000,  50, 10),
    ]

    devices = ['cpu']
    if HAS_GPU:
        devices.append('gpu')

    print(f'{"Config":<30s}  {"cov":>5s}  {"device":>6s}  '
          f'{"time":>10s}  {"speedup":>8s}  {"conv":>5s}')
    print('-' * 80)

    for label, N, M, K in configs:
        X = generate_data(N, M, K)

        for cov_type in ['full', 'diag']:
            cpu_time = None
            for device in devices:
                t, conv = time_fit(X, K, cov_type, device, max_iter=50, reps=3)

                if device == 'cpu':
                    cpu_time = t
                    speedup_str = '(base)'
                else:
                    speedup = cpu_time / t if t > 0 else float('inf')
                    speedup_str = f'{speedup:.2f}x'

                conv_str = 'yes' if conv else 'NO'
                print(f'{label:<30s}  {cov_type:>5s}  {device:>6s}  '
                      f'{t:>9.3f}s  {speedup_str:>8s}  {conv_str:>5s}')
        print()


if __name__ == '__main__':
    main()
