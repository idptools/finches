import time

import numpy as np

from finches.utils.matrix_manipulation import matrix_scan, matrix_scan_legacy


def generate_random_matrix(rows, cols):
    """Generate a random matrix of given dimensions."""
    return np.random.rand(rows, cols)


def benchmark_matrix_scan(
    w_matrix, window_size, matrix_size, cython=True, legacy=False
):
    """Benchmark the matrix_scan function."""
    # Generate a random matrix

    # Define a null interaction baseline
    null_interaction_baseline = 0.5

    # Measure the start time
    start_time = time.perf_counter()

    if cython and not legacy:
        everything, seq1_indices, seq2_indices = matrix_scan(
            w_matrix, window_size, null_interaction_baseline
        )
    elif cython and legacy:
        everything, seq1_indices, seq2_indices = matrix_scan_legacy(
            w_matrix, window_size, null_interaction_baseline
        )

    # Measure the end time
    end_time = time.perf_counter()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Benchmark Results:")
    print(f"Matrix Size: {matrix_size}x{matrix_size}")
    print(f"Window Size: {window_size}")
    print(f"Output Shape: {everything.shape}")
    print(f"Elapsed Time: {elapsed_time:.6f} seconds")

    return everything


if __name__ == "__main__":
    # Define parameters for benchmarking
    window_size = 31
    matrix_size = 1000  # m*n matrix (1000x1000)
    w_matrix = generate_random_matrix(matrix_size, matrix_size)
    # Run the benchmark
    ev1 = benchmark_matrix_scan(w_matrix, window_size, matrix_size, legacy=False)
    ev2 = benchmark_matrix_scan(w_matrix, window_size, matrix_size, legacy=True)

    # print(ev1,"\n")
    # print(ev2)
    assert np.allclose(ev1, ev2), (
        "Different results from optimized and non-optimized implementations!"
    )
