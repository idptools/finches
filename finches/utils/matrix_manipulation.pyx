"""
Compiled helpers for building interaction matrices and sliding-window epsilon.

matrix_scan() is the implementation behind
InteractionMatrixConstructor.calculate_sliding_epsilon(); matrix_scan_legacy() is
the original explicit-loop version, kept as a reference implementation and for
benchmarking (see finches/benchmarking/benchmark_matrix_scan.py). dict2matrix()
builds a raw interaction matrix from the nested lookup dictionary when the
faster NumPy table cannot be used.
"""

import numpy as np
cimport numpy as cnp
cimport cython

from libc.stdlib cimport rand, RAND_MAX


@cython.boundscheck(False)
def dict2matrix(str seq1, str seq2, dict lookup):
    """
    Build a matrix from two sequences and a nested lookup dictionary.

    Parameters
    ----------
    seq1 : str
        First sequence; defines the rows of the matrix.
    seq2 : str
        Second sequence; defines the columns of the matrix.
    lookup : dict
        Nested dictionary, lookup[a][b] = value for residues a and b.

    Returns
    -------
    numpy.ndarray
        Float matrix of shape (len(seq1), len(seq2)) with element [i, j] equal
        to lookup[seq1[i]][seq2[j]].

    Raises
    ------
    KeyError
        If a residue is missing from the lookup.
    """
    cdef int r1, r2, l1, l2

    l1 = len(seq1)
    l2 = len(seq2)

    # preallocate the matrix
    cdef cnp.ndarray[cnp.float64_t, ndim=2] matrix = np.empty((l1, l2), dtype=np.float64)

    for r1 in range(l1):
        for r2 in range(l2):
            matrix[r1,r2] = lookup[seq1[r1]][seq2[r2]]

    return matrix


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef matrix_scan_legacy(double[:,:] w_matrix, int window_size, double null_interaction_baseline):
    """
    Compute the sliding-window epsilon matrix with explicit nested loops.

    The original implementation, kept as a reference for matrix_scan() and for
    benchmarking. For each window of size w it:

    1. copies the w x w sub-matrix;
    2. builds the attractive deviations: val - baseline where val <= baseline,
       otherwise -baseline;
    3. builds the repulsive deviations: val - baseline where val > baseline,
       otherwise -baseline (an element exactly equal to the baseline is counted
       once, as attractive, so every element contributes val - 2 * baseline and
       the result matches matrix_scan());
    4. sums the row means of both matrices to give the window's epsilon.

    This is O((l1 - w + 1) * (l2 - w + 1) * w^2).

    Parameters
    ----------
    w_matrix : 2D memoryview of double
        Weighted interaction matrix, shape (l1, l2).
    window_size : int
        Size of the square window.
    null_interaction_baseline : double
        The forcefield's null interaction baseline.

    Returns
    -------
    tuple
        (everything, seq1_indices, seq2_indices): the epsilon matrix of shape
        (l1 - window_size + 1, l2 - window_size + 1), and the 1-based residue
        positions of the window centres along each axis.

    Raises
    ------
    ValueError
        If window_size is larger than either dimension of w_matrix.
    """
    # define the variables
    cdef int l1, l2, i, j, start, end, r1, r2;
    cdef double row_sum, total_mean_sum;

    # get dimensions of matrix
    l1 = w_matrix.shape[0]
    l2 = w_matrix.shape[1]

    # check for window size larger than matrix size
    if l1 < window_size or l2 < window_size:
        raise ValueError('Window size larger than matrix size')


    # preallocate the various matrices being used
    cdef cnp.ndarray[cnp.float64_t, ndim=2] everything = np.empty( [(l1-window_size)+1,(l2-window_size)+1], dtype=np.float64)    
    cdef cnp.ndarray[double, ndim=2] attractive_matrix = np.empty([window_size, window_size], dtype=np.double)    
    cdef cnp.ndarray[double, ndim=2] repulsive_matrix = np.empty([window_size, window_size], dtype=np.double)
    cdef cnp.ndarray[double, ndim=2] sub = np.empty([window_size, window_size], dtype=np.double)
    
    
    # calculate sliding epsilon for all possible intermolecular windows. 
    for i in range(0,(l1-window_size)+1):

        for j in range(0, (l2-window_size)+1):


            # copy the memoryview into a numpy array
            for r1 in range(i, i+window_size):
                for r2 in range(j, j+window_size):
                    sub[r1-i,r2-j] = w_matrix[r1,r2]

            # construct the attractive and repulsive matrices, and then sum them - note we do this
            # so brutally manually because it will then compile down to pure C which buys us all
            # the speed!
            for r1 in range(window_size):
                for r2 in range(window_size):

                    if sub[r1,r2] <= null_interaction_baseline:
                        attractive_matrix[r1,r2] = sub[r1,r2] - null_interaction_baseline
                    else:
                        attractive_matrix[r1,r2] = - null_interaction_baseline
                        
                    if sub[r1,r2] > null_interaction_baseline:
                        repulsive_matrix[r1,r2] = sub[r1,r2] - null_interaction_baseline
                    else:
                        repulsive_matrix[r1,r2] = -null_interaction_baseline

            # here we sum and average the attractive and repulsive means
            total_mean_sum = 0.0
            for r1 in range(window_size):
                row_sum = 0.0
                for r2 in range(window_size):
                    row_sum += attractive_matrix[r1, r2]
                total_mean_sum += row_sum / window_size  # Add the mean of the current row to the total sum
                        
            for r1 in range(window_size):
                row_sum = 0.0
                for r2 in range(window_size):
                    row_sum += repulsive_matrix[r1, r2]
                total_mean_sum += row_sum / window_size  # Add the mean of the current row to the total sum

            everything[i,j] =     total_mean_sum


    # finally, determine indices for sequence1 - note need +1 for indexing to move from Python
    # to protein space, and then these are inclusive values
    start = int((window_size-1)/2) + 1
    end   = (l1 - start) + 1
    seq1_indices = np.arange(start, end+1)

    # and sequence2
    start = int((window_size-1)/2) + 1
    end   = (l2 - start) + 1
    seq2_indices = np.arange(start, end+1)

    # finally check our matrix and indices make sense...
    assert len(seq1_indices) == everything.shape[0]
    assert len(seq2_indices) == everything.shape[1]
        
    return (everything,  seq1_indices, seq2_indices)


@cython.cdivision(True)
cdef double window_sum(cnp.ndarray[double, ndim=2] sat, int top, int left, int bottom, int right):
    """
    Retrieve the sum of elements in a rectangular region using a summed-area table.
    
    Parameters
    ----------
    sat : 2D numpy.ndarray
        Summed-area table where sat[i,j] = sum of original matrix up to (i,j).
    top, left, bottom, right : int
        Inclusive coordinates defining the rectangle to sum.
    
    Returns
    -------
    double
        Sum of the original elements in [top:bottom+1, left:right+1], computed
        as sat[bottom, right] - sat[top-1, right] (if top > 0)
        - sat[bottom, left-1] (if left > 0) + sat[top-1, left-1] (if both > 0).
    """
    cdef double total = sat[bottom, right]
    if top > 0:
        total -= sat[top-1, right]
    if left > 0:
        total -= sat[bottom, left-1]
    if top > 0 and left > 0:
        total += sat[top-1, left-1]
    return total

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef matrix_scan(double[:,:] w_matrix,
                            int window_size,
                            double null_interaction_baseline):
    """
    Compute the sliding-window epsilon matrix with a summed-area table.

    A summed-area table (SAT) is the 2D extension of a prefix sum: after an
    O(l1 * l2) pass, the sum over any rectangle takes O(1) (see
    https://en.wikipedia.org/wiki/Summed-area_table). The epsilon of a window
    only needs that sum. Every element of a window contributes val - 2 * baseline
    (see finches.epsilon_stateless.get_attractive_repulsive_matrices()), so for
    a w x w window with element sum S the sum over rows of the row means is

        epsilon = S / w - 2 * w * baseline

    which removes the need to build attractive and repulsive matrices at all and
    cuts the per-window cost from O(w^2) to O(1).

    Parameters
    ----------
    w_matrix : 2D memoryview of double
        Weighted interaction matrix, shape (l1, l2).
    window_size : int
        Size of the square window.
    null_interaction_baseline : double
        The forcefield's null interaction baseline.

    Returns
    -------
    tuple
        (everything, seq1_indices, seq2_indices): the epsilon matrix of shape
        (l1 - window_size + 1, l2 - window_size + 1), and the 1-based residue
        positions of the window centres along each axis.

    Raises
    ------
    ValueError
        If window_size is larger than either dimension of w_matrix.
    """
    cdef int l1 = w_matrix.shape[0]
    cdef int l2 = w_matrix.shape[1]
    cdef int ws = window_size
    cdef int i, j, r1, r2
    cdef int out1, out2
    cdef double raw       

    if l1 < ws or l2 < ws:
        raise ValueError('Window size larger than matrix size')

    # 1) build one SAT over raw matrix
    cdef cnp.ndarray[double, ndim=2] w_sat = np.empty((l1, l2), dtype=np.double)
    for r1 in range(l1):
        for r2 in range(l2):
            w_sat[r1, r2] = w_matrix[r1, r2]
            if r1 > 0:
                w_sat[r1, r2] += w_sat[r1-1, r2]
            if r2 > 0:
                w_sat[r1, r2] += w_sat[r1, r2-1]
            if r1 > 0 and r2 > 0:
                # need this to avoid double counting 
                w_sat[r1, r2] -= w_sat[r1-1, r2-1]

    # 2) slide a single O(1) SAT lookup per window
    out1 = l1 - ws + 1
    out2 = l2 - ws + 1
    cdef cnp.ndarray[double, ndim=2] everything = np.empty((out1, out2), dtype=np.double)

    for i in range(out1):
        for j in range(out2):
            # raw window sum
            raw = window_sum(w_sat,
                            i, j,
                            i + ws - 1,
                            j + ws - 1)
            # every element contributes val - 2*baseline to the sum of row means,
            # so the window's epsilon follows from its raw sum alone (see docstring)
            everything[i, j] = raw/ws - 2.0*ws*null_interaction_baseline

    cdef int start = int((window_size - 1) / 2) + 1
    cdef int end1 = (l1 - start) + 1
    cdef int end2 = (l2 - start) + 1
    seq1_indices = np.arange(start, end1 + 1)
    seq2_indices = np.arange(start, end2 + 1)

    # Shape checks as in original
    assert len(seq1_indices) == everything.shape[0]
    assert len(seq2_indices) == everything.shape[1]

    return everything, seq1_indices, seq2_indices


def return_random_array(int n):
    """
    Generate a random array of length n using the C library's rand().

    Nothing in FINCHES uses this function. The C generator is not seeded from
    Python, so the output is not reproducible; use numpy.random.default_rng()
    for anything that matters.

    Parameters
    ----------
    n : int
        Length of the output array.

    Returns
    -------
    numpy.ndarray
        1D float array of length n with entries in [0, 1], each rand() / RAND_MAX.
    """
    arr = np.zeros(n)
    for i in range(n):
        arr[i] = rand()/RAND_MAX
    return arr