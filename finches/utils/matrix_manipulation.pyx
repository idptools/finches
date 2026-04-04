import numpy as np
cimport numpy as cnp
cimport cython 

from cpython cimport array
import array

from libc.stdlib cimport rand, srand, RAND_MAX


cdef inline bint _is_charged(char residue) nogil:
    return residue == 'R' or residue == 'K' or residue == 'E' or residue == 'D'


cdef inline double _charge_value(char residue) nogil:
    if residue == 'R' or residue == 'K':
        return 1.0
    if residue == 'E' or residue == 'D':
        return -1.0
    return 0.0



@cython.boundscheck(False)
@cython.cdivision(True)
def dict2matrix(str seq1, str seq2, dict lookup):
    
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
def charge_weighted_mask(str seq1, str seq2):
    cdef int i, j, start, end, idx, l1, l2
    cdef double total_charge
    cdef int total_count

    l1 = len(seq1)
    l2 = len(seq2)

    cdef cnp.ndarray[cnp.float64_t, ndim=2] attractive_matrix = np.zeros((l1, l2), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] repulsive_matrix = np.zeros((l1, l2), dtype=np.float64)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] charge_sum_1 = np.zeros(l1, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] charge_count_1 = np.zeros(l1, dtype=np.int32)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] charged_1 = np.zeros(l1, dtype=np.uint8)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] charge_sum_2 = np.zeros(l2, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] charge_count_2 = np.zeros(l2, dtype=np.int32)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] charged_2 = np.zeros(l2, dtype=np.uint8)

    for i in range(l1):
        if _is_charged(seq1[i]):
            charged_1[i] = 1

        start = i - 1
        if start < 0:
            start = 0
        end = i + 2
        if end > l1:
            end = l1

        total_charge = 0.0
        total_count = 0
        for idx in range(start, end):
            if _is_charged(seq1[idx]):
                total_charge += _charge_value(seq1[idx])
                total_count += 1
        charge_sum_1[i] = total_charge
        charge_count_1[i] = total_count

    for j in range(l2):
        if _is_charged(seq2[j]):
            charged_2[j] = 1

        start = j - 1
        if start < 0:
            start = 0
        end = j + 2
        if end > l2:
            end = l2

        total_charge = 0.0
        total_count = 0
        for idx in range(start, end):
            if _is_charged(seq2[idx]):
                total_charge += _charge_value(seq2[idx])
                total_count += 1
        charge_sum_2[j] = total_charge
        charge_count_2[j] = total_count

    for i in range(l1):
        if charged_1[i] == 0:
            continue
        for j in range(l2):
            if charged_2[j] == 0:
                continue
            total_charge = charge_sum_1[i] + charge_sum_2[j]
            total_count = charge_count_1[i] + charge_count_2[j]
            repulsive_matrix[i, j] = abs(total_charge / total_count)

    return attractive_matrix, repulsive_matrix


@cython.boundscheck(False)
@cython.cdivision(True)
def matrix_scan(double[:,:] w_matrix, int window_size, double null_interaction_baseline):
    """
    Function that calculates the sliding window epsilon from a given inter-protein matrix.
    This implementation takes about 8% of the time of our original Python implementation,
    making it a lot more feasible to use on large matrices.

    Note this returns the indices in protein space, i.e. where the first residue is 1

    Parameters
    ------------
    w_matrix : array
       Inter-protein matrix

    window_size : int
         Size of the sliding window

    null_interaction_baseline : float
        Baseline value for null interactions

    Returns
    --------
    tuple with three elements:

    everything : array
       Matrix with sliding window epsilon values

    
    seq1_indices : array
       Indices of the first sequence, starting from 1
       (i.e. in protein space)
 
    seq2_indices : array
         Indices of the second sequence, starting from 1
        (i.e. in protein space)

    """


    # define the variables
    cdef int l1, l2, start, end
    cdef cnp.ndarray[cnp.float64_t, ndim=2] matrix = np.asarray(w_matrix, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] transformed
    cdef cnp.ndarray[cnp.float64_t, ndim=2] integral
    cdef cnp.ndarray[cnp.float64_t, ndim=2] everything

    # get dimensions of matrix
    l1 = w_matrix.shape[0]
    l2 = w_matrix.shape[1]

    # check for window size larger than matrix size
    if l1 < window_size or l2 < window_size:
        raise Exception('Window size is larger than matrix size, cannot calculate sliding epsilon')


    transformed = matrix - (2.0 * null_interaction_baseline)
    transformed[matrix == null_interaction_baseline] -= null_interaction_baseline

    integral = np.pad(transformed, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    everything = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    ) / float(window_size)


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




def return_random_array(int n):
    """
    Function that returns a random array of length n

    Parameters
    ------------
    n : int
       Length of array

    Returns
    --------
    arr : array
       Array of length n with random values

    """
    arr = np.zeros(n)
    for i in range(n):
        arr[i] = rand()/RAND_MAX
    return arr


# ....................................................................................................
#
cdef seed_C_rand(int seedval):
    """
    Function that initializes C's rand() function with
    a seed value. Without this the same seed is used every
    time..

    Parameters
    ------------
    seedval : int
       Non-negative integer seed

    Returns
    --------
    None
      No return but sets the seed!
    

    """
    srand(seedval)


        
        
