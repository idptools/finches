"""
Benchmark the summed-area-table matrix_scan() against matrix_scan_legacy().

Run as a script to time both Cython implementations of the sliding-window
epsilon on a random 1000 x 1000 matrix and check that they agree::

    python finches/benchmarking/benchmark_matrix_scan.py
"""

import time

import numpy as np

from finches.utils.matrix_manipulation import matrix_scan, matrix_scan_legacy


def generate_random_matrix(rows, cols):
    """
    Generate a random matrix with entries uniform on [0, 1).

    Parameters
    ----------
    rows, cols : int
        Matrix dimensions.

    Returns
    -------
    np.ndarray
        Float matrix of shape (rows, cols).
    """
    return np.random.rand(rows, cols)


def benchmark_matrix_scan(
    w_matrix, window_size, matrix_size=None, cython=True, legacy=False
):
    """
    Time one sliding-window epsilon implementation on a matrix and print the result.

    Parameters
    ----------
    w_matrix : np.ndarray
        The matrix to scan.

    window_size : int
        Window size.

    matrix_size : int, optional
        Only used in the printed report. Default None reports the matrix's
        actual shape.

    cython : bool, optional
        Must be True; only the Cython implementations are benchmarked here.
        Default True.

    legacy : bool, optional
        If True time matrix_scan_legacy(), otherwise matrix_scan(). Default
        False.

    Returns
    -------
    np.ndarray
        The sliding-window epsilon matrix.

    Raises
    ------
    ValueError
        If cython is False.
    """
    if not cython:
        raise ValueError(
            "Only the Cython implementations are benchmarked here; use "
            "InteractionMatrixConstructor.calculate_sliding_epsilon(use_cython=False) "
            "for the pure-Python path"
        )

    # an arbitrary baseline; it shifts every window by the same constant
    null_interaction_baseline = 0.5

    scan = matrix_scan_legacy if legacy else matrix_scan

    start_time = time.perf_counter()
    everything, _, _ = scan(w_matrix, window_size, null_interaction_baseline)
    elapsed_time = time.perf_counter() - start_time

    size = (
        f"{matrix_size}x{matrix_size}"
        if matrix_size is not None
        else f"{w_matrix.shape[0]}x{w_matrix.shape[1]}"
    )
    print("Benchmark Results:")
    print(f"Implementation: {scan.__name__}")
    print(f"Matrix Size: {size}")
    print(f"Window Size: {window_size}")
    print(f"Output Shape: {everything.shape}")
    print(f"Elapsed Time: {elapsed_time:.6f} seconds")

    return everything


if __name__ == "__main__":
    window_size = 31
    w_matrix = generate_random_matrix(1000, 1000)

    ev1 = benchmark_matrix_scan(w_matrix, window_size, legacy=False)
    ev2 = benchmark_matrix_scan(w_matrix, window_size, legacy=True)

    assert np.allclose(ev1, ev2), (
        "Different results from optimized and non-optimized implementations!"
    )
