"""
Constains the code for both global and local constraints of DTW algorithm.

"""

import numpy as np
from scipy.spatial.distance import cdist

# --- Step Pattern Definitions ---
# These patterns define the allowed steps from a cell (i, j) to its predecessors.
# Each entry in the dictionary is a list of tuples (i_step, j_step).
STEP_PATTERNS = {
    # Standard symmetric pattern. Allows for flexible alignment.
    'symmetric1': [(1, 1), (1, 0), (0, 1)],
    
    # A widely used symmetric pattern with a strong diagonal preference.
    # The recurrence is D[i,j] = C[i,j] + min(D[i-1,j-1], D[i-1,j]+C[i,j], D[i,j-1]+C[i,j])
    # This is handled by how we add the local cost 'C' in the main function.
    'symmetric2': [(1, 1), (1, 0), (0, 1)],

    # Asymmetric pattern. Penalizes horizontal/vertical steps, forcing progress.
    'asymmetric': [(1, 1), (1, 0)],
    
    # Rabiner-Juang Type D (symmetric). A more constrained diagonal pattern.
    'rabiner-juang-d': [(1, 1), (2, 1), (1, 2)],
}

def create_sakoe_chiba_mask(n, m, window):
    """
    Creates a boolean mask for the Sakoe-Chiba band global constraint.

    Args:
        n (int): Number of timestamps in the first sequence.
        m (int): Number of timestamps in the second sequence.
        window (int): The radius of the band around the main diagonal.

    Returns:
        np.ndarray: A boolean array of shape (n, m) where True indicates an
                    allowed cell in the pairwise cost matrix.
    """
    mask = np.full((n, m), False, dtype=bool)
    for i in range(n):
        start = max(0, i - window)
        end = min(m, i + window + 1)
        mask[i, start:end] = True
    return mask