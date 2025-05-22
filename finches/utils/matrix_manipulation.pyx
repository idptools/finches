import numpy as np
cimport numpy as cnp
cimport cython 

from cpython cimport array
import array

from libc.stdlib cimport rand, srand, RAND_MAX



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


        
        
