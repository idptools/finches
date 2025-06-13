import numpy as np
cimport numpy as cnp
cimport cython 

from cpython cimport array
import array

from libc.stdlib cimport rand, srand, RAND_MAX



@cython.boundscheck(False)
@cython.cdivision(True)
def dict2matrix(str seq1, str seq2, dict lookup):
    """
    Convert two sequences into a numeric matrix using a nested lookup.
    
    Parameters
    ----------
    seq1 : str
        First sequence of characters; defines rows of output matrix.
    seq2 : str
        Second sequence of characters; defines columns of output matrix.
    lookup : dict
        Nested dictionary mapping seq1 char to dict mapping seq2 char to float value.
    
    Returns
    -------
    numpy.ndarray of shape (len(seq1), len(seq2))
        Matrix where entry [i,j] = lookup[seq1[i]][seq2[j]].
    
    Algorithm
    ---------
    1. Preallocate an empty float matrix of shape (len(seq1), len(seq2)).
    2. Iterate over all index pairs (i,j), assign matrix[i,j] = lookup[seq1[i]][seq2[j]].
    """
    cdef int r1, r2, l1, l2

    l1 = len(seq1)
    l2 = len(seq2)

    # preallocate the matrix
    cdef cnp.ndarray[cnp.float_t, ndim=2] matrix = np.empty((l1, l2), dtype=float)

    for r1 in range(l1):
        for r2 in range(l2):
            matrix[r1,r2] = lookup[seq1[r1]][seq2[r2]]

    return matrix


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef matrix_scan_legacy(double[:,:] w_matrix, int window_size, double null_interaction_baseline):
    """
    Compute sliding window epsilon using explicit nested loops.
    
    This legacy implementation performs the following for each window of size w:
      1. Copy the w*w submatrix from w_matrix.
      2. Build 'attractive' deviations: values < baseline → (val-baseline), else -baseline.
      3. Build 'repulsive' deviations: values > baseline → (val-baseline), else -baseline.
      4. Sum each row of both matrices, normalize by window_size, accumulate total mean.
      5. Store combined mean deviation as epsilon in `everything[i,j]`.
    
    Notes
    ----------
    Algorithmic Complexity
        - O((l1-w+1)*(l2-w+1)*w^2), since each window requires O(w^2) copying and computation.
    
    Returns
    -------
    everything : 2D numpy.ndarray
        Sliding window epsilon values for all positions.
    seq1_indices : 1D numpy.ndarray
        1-based indices for sequence1 in protein-space.
    seq2_indices : 1D numpy.ndarray
        1-based indices for sequence2 in protein-space.
    """
    # define the variables
    cdef int l1, l2, i, j, start, end, r1, r2;
    cdef double row_sum, total_mean_sum;

    # get dimensions of matrix
    l1 = w_matrix.shape[0]
    l2 = w_matrix.shape[1]

    # check for window size larger than matrix size
    if l1 < window_size or l2 < window_size:
        raise Exception('Window size is larger than matrix size, cannot calculate sliding epsilon')


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

                    if sub[r1,r2] < null_interaction_baseline:
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
        Sum of original elements in rectangle [top:bottom+1, left:right+1].
    
    Calculation
    -----------
    total = sat[bottom,right]
            - sat[top-1,right]   if top>0
            - sat[bottom,left-1] if left>0
            + sat[top-1,left-1]  if top>0 and left>0
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
    Compute sliding window epsilon efficiently via summed-area table (SAT).
    Summed-area tables are extensions of prefix sums to 2D matrices. 
    They are used to quickly compute sums over rectangular regions in O(1) time after an O(l1*l2) preprocessing step.
    
    See: https://en.wikipedia.org/wiki/Summed-area_table
    
    This optimized algorithm reduces per-window cost from O(w^2) to O(1).
    Since we use a large window size w=31, this is a significant speedup.
    
    Algorithm Overview
    -------------------
    The algorithm works as follows:
        1. Build SAT of w_matrix in O(l1*l2) time:
            sat[i,j] = w_matrix[i,j] + sat[i,j-1] + sat[i-1,j] - sat[i-1,j-1].
        2. For each window top-left (i,j):
            a. Retrieve raw sum via window_sum(sat, i, j, i+w-1, j+w-1) in O(1).
            b. Compute epsilon = raw/ws - 2 * ws * null_interaction_baseline.
        3. As before, assemble 1-based protein-space indices.

    Parameters
    ----------
    w_matrix : 2D memoryview of double
        Original interaction matrix.
    window_size : int
        Size of the square sliding window.
    null_interaction_baseline : double
        Baseline value to normalize interactions.
    
    Returns
    -------
    everything : 2D numpy.ndarray
        Matrix of sliding window epsilon values.
    seq1_indices : 1D numpy.ndarray
        1-based indices for sequence1.
    seq2_indices : 1D numpy.ndarray
        1-based indices for sequence2.
    """
    cdef int l1 = w_matrix.shape[0]
    cdef int l2 = w_matrix.shape[1]
    cdef int ws = window_size
    cdef int i, j, r1, r2
    cdef int out1, out2
    cdef double raw       

    if l1 < ws or l2 < ws:
        raise Exception('Window size is larger than matrix size')

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
            # I believe this is the same as the original algorithm, 
            # but avoids the need to build separate attractive and repulsive matrices. 
            # I think when garret first presented this algorithm, many of us were confused by the need to have these two matrices. 
            # as it really felt like there should've been a way to do this with a single matrix.
            # I can show the algebra why this is equivalent to explicitly constructing separate matrices and catches
            # an incredibly unlikely edge case in the original implementation where something is equal to the baseline.
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
    Generate a random array of length n using C's rand().
    
    Parameters
    ----------
    n : int
        Desired length of the output array.
    
    Returns
    -------
    numpy.ndarray
        1D array of floats in [0,1), each entry = rand()/RAND_MAX.
    
    Notes
    -----
    Uses the C standard library rand(); not reproducible without seeding via seed_C_rand.
    """
    arr = np.zeros(n)
    for i in range(n):
        arr[i] = rand()/RAND_MAX
    return arr


# ....................................................................................................
#
cdef seed_C_rand(int seedval):
    """
    Seed the C standard library RNG (rand()) for reproducibility.
    
    Parameters
    ----------
    seedval : int
        Non-negative integer seed.
    
    Notes
    -----
    Calls srand(seedval) from libc.stdlib. Affects subsequent rand() calls.
    """
    if seedval < 0:
        raise ValueError("Seed value must be non-negative")
    srand(seedval)