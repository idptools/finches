import numpy as np
cimport numpy as cnp
cimport cython 

from cpython cimport array
import array

from libc.stdlib cimport rand, srand, RAND_MAX



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


        
        
