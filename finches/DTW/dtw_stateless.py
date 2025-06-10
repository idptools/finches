"""
These are the stateless function for computing the DTW of two sequences.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def calculate_dtw_distance_flex(s1, s2, custom_dist=None, global_mask=None, step_pattern='symmetric1'):
    """
    Calculates DTW distance with generalized global and local constraints.

    Args:
        s1 (np.ndarray): The first sequence, shape (n_timestamps, n_dims).
        s2 (np.ndarray): The second sequence, shape (m_timestamps, n_dims).
        custom_dist (function, optional): A custom distance function. If None,
                                          squared Euclidean distance is used.
        global_mask (np.ndarray, optional): A boolean array of shape (n, m)
                                            where True values indicate allowed
                                            regions for the warping path. If None,
                                            the entire matrix is allowed.
        step_pattern (str, optional): The local step pattern to use. Must be a key
                                      in the STEP_PATTERNS dictionary. Defaults
                                      to 'symmetric1'.

    Returns:
        float: The DTW distance between the two sequences.
        np.ndarray: The accumulated cost matrix.
    """
    # determine the lengths of the sequences being aligned
    n, m = len(s1), len(s2)
    # reshapes the data if needed (the user might have provided the array in inverse index order)
    if s1.ndim == 1: s1 = s1.reshape(-1, 1)
    if s2.ndim == 1: s2 = s2.reshape(-1, 1)

    # 1. Pre-compute the pairwise distance matrix (local cost)
    # This is useful as it allows for custom distance metrics without the need for the user to compute them manually.
    # The preallocation also allows for the potential for faster computation.
    if custom_dist: # check that it is not None
        # cdist is an optimized function for our exact problem that scipy provides
        cost_matrix = cdist(s1, s2, metric=custom_dist)
    else: # we compute the default sqaured euclidean distance if no custom distance is provided
        cost_matrix = cdist(s1, s2, metric='sqeuclidean')

    # Apply the global constraint mask to the local costs
    # dault of the cost matrix is infinity since we are searching for a path that minimizes the cost
    if global_mask is not None:
        cost_matrix[~global_mask] = np.inf # notice we are adding this to the cost matrix (the one we compute from the precomputed distances)

    # 2. Initialize the accumulated cost matrix (this is the matrix that we will use to search for the optimal path - its terms are partially computed from the cost_matrix)
    acc_cost_matrix = np.full((n, m), np.inf) # default value is infinity
    acc_cost_matrix[0, 0] = cost_matrix[0, 0] # the starting score is 0 plus the amount of cost for the first pair
    
    # 3. Dynamic Programming with selected step pattern
    # Pre-populate the first row and column based on the pattern
    # This is a simplification; for some patterns, more complex init is needed.
    # For symmetric1/2, this works.
    for i in range(1, n):
        acc_cost_matrix[i, 0] = cost_matrix[i, 0] + acc_cost_matrix[i-1, 0]
    for j in range(1, m):
        acc_cost_matrix[0, j] = cost_matrix[0, j] + acc_cost_matrix[0, j-1]
        
    # Main loop
    for i in range(1, n):
        for j in range(1, m):
            # lets check if the cost is infinite
            # This is true when the the cell is excluded by the global mask
            if np.isinf(cost_matrix[i, j]):
                continue

            # Initialize a list to store the possible costs for the cell depending on if it is a match or skip state
            possible_costs = []
            
            # This logic handles the main symmetric step patterns
            if step_pattern in ['symmetric1', 'symmetric2']:
                prev_costs = [
                    acc_cost_matrix[i-1, j-1], # Diagonal
                    acc_cost_matrix[i-1, j],   # Up
                    acc_cost_matrix[i, j-1]    # Left
                ]
                min_prev_cost = min(prev_costs)
                
                # 'symmetric2' adds the local cost twice for non-diagonal moves
                if step_pattern == 'symmetric2':
                     # This is a common interpretation giving weight 2 to non-diagonal steps (double weighting the offdiagnols)
                     min_prev_cost = min(acc_cost_matrix[i-1, j-1], 
                                         acc_cost_matrix[i-1, j] + cost_matrix[i,j], 
                                         acc_cost_matrix[i, j-1] + cost_matrix[i,j])

                acc_cost_matrix[i, j] = cost_matrix[i, j] + min_prev_cost
            elif step_pattern == 'asymmetric':
                print(f"Not currently implemented: would also need to change the initializations for the acc_cost matrix.")

    # Return the final DTW distance and the accumulated cost matrix
    return acc_cost_matrix[-1, -1], acc_cost_matrix

def find_dtw_path_flex(acc_cost_matrix, step_pattern='symmetric1'):
    """
    Finds the optimal warping path with awareness of step patterns.

    Args:
        acc_cost_matrix (np.ndarray): The accumulated cost matrix.
        step_pattern (str): The step pattern used to generate the matrix.

    Returns:
        list of tuples: The optimal warping path.
    """
    # check that the accumulated cost matrix is a valid value
    if acc_cost_matrix is None or acc_cost_matrix.ndim != 2:
        raise ValueError(f"Invalid accumulated cost matrix was provide to the path finding function. It must not be None and must be a 2D numpy array. It is {type(acc_cost_matrix).__name__}.")
    # get the shape of the accumulated cost matrix
    # remember taht there is an extra row and column for the initialization
    i, j = acc_cost_matrix.shape
    i -= 1 # start from the last cell len(seq1) - 1
    j -= 1 # start from the last cell len(seq2) - 1

    # this is the list of tuples that holds the path
    path = [(i, j)]

    # loop over as long as we are not back at the first indexes cost.
    while i > 0 or j > 0:
        # if we match one of the middle indexes of one sequence to the first of the other we have effectivley
        # reached the end of the alignment path. So we just need to backtrack up the column or to the left in the row
        if i == 0: # check if we are already at the first index for one of the sequences
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Check predecessors based on the step pattern
            # Note: This is simplified for symmetric patterns. More complex patterns
            # would require more complex backtracking logic.
            if step_pattern in ['symmetric1', 'symmetric2']:
                options = {
                    'diag': acc_cost_matrix[i - 1, j - 1],
                    'up': acc_cost_matrix[i - 1, j],
                    'left': acc_cost_matrix[i, j - 1]
                }
                min_direction = min(options, key=options.get)

                if min_direction == 'diag':
                    i, j = i - 1, j - 1
                elif min_direction == 'up':
                    i -= 1
                else: # left
                    j -= 1
            elif step_pattern == 'asymmetric':
                print(f"Have not yet implements the asymmetric step pattern for path finding.")

        # Append the next best position to the path list
        path.append((i, j))
            
    # we are solving te index matching from the last to the first indexes so we need to flip the order to get the path
    path.reverse() # lists are mutable so this is all we need to do

    # return the correctly oriented path.
    return path







def calculate_barycenter_average(sequences, n_iterations=10, initial_center_index=0, **kwargs):
    """
    Calculates the barycenter (average) of sequences of varying lengths using DBA.

    The length of the resulting average sequence is determined by the length of the
    sequence chosen for the initial barycenter.

    Args:
        sequences (list of np.ndarray): A list of sequences to be averaged.
        n_iterations (int): The number of iterations to perform for convergence.
        initial_center_index (int): The index of the sequence in the list to use
                                    as the initial barycenter.
        **kwargs: Additional arguments to pass to the DTW functions (e.g.,
                  step_pattern, custom_dist).

    Returns:
        np.ndarray: The average sequence (barycenter).
    """
    if not sequences:
        return None

    # Use a specified sequence as the initial barycenter
    barycenter = sequences[initial_center_index].copy()

    print(f"Starting DBA with {n_iterations} iterations.")
    print(f"Initial barycenter length will be {len(barycenter)} (from sequence {initial_center_index}).")

    for iteration in range(n_iterations):
        # The 'associations' list will store, for each point in the barycenter,
        # all the points from the other sequences that are mapped to it.
        associations = [[] for _ in range(len(barycenter))]
        
        # for each sequence find the warping path to the barycenter
        for s in sequences:
            # Find the warping path between the current barycenter and the sequence
            _, acc_cost_matrix = calculate_dtw_distance_flex(barycenter, s, **kwargs)
            path = find_dtw_path_flex(acc_cost_matrix, **kwargs)
            
            # For each mapping in the path, associate the point from s with the point in barycenter
            for i, j in path:
                # take the position VALUE for each position
                associations[i].append(s[j])

        # Update the barycenter by averaging the associated points
        for i in range(len(barycenter)):
            if associations[i]:
                # Calculate the mean of all points mapped to this index
                barycenter[i] = np.mean(associations[i], axis=0)
        
        print(f"  Iteration {iteration + 1}/{n_iterations} complete.")

    return barycenter

def score_sequence_similarity(sequence, reference_sequence, **kwargs):
    """
    Scores the similarity of a sequence against a reference sequence using DTW distance.
    A lower score indicates higher similarity.

    Args:
        sequence (np.ndarray): The sequence to be scored.
        reference_sequence (np.ndarray): The reference sequence.
        **kwargs: Additional arguments to pass to calculate_dtw_distance
                  (e.g., custom_dist, window).

    Returns:
        float: The DTW distance, representing the similarity score.
    """
    distance, _ = calculate_dtw_distance_flex(sequence, reference_sequence, **kwargs)
    return distance











# -----------------------------------------
# Soft DTW
# -----------------------------------------
# --- STABILIZED Soft-DTW Implementation ---
# CURRENTLY NOT NUMERICALLY STABLE..... DO NOT USE

def soft_dtw_and_gradient(barycenter, s, gamma):
    """
    Computes the Soft-DTW value and the gradient with respect to the barycenter.
    *** This version is numerically stable using the log-sum-exp trick. ***
    """
    n, m = len(barycenter), len(s)
    barycenter = barycenter.reshape(-1, 1)
    s = s.reshape(-1, 1)
    
    cost_matrix = cdist(barycenter, s, metric='sqeuclidean')

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.
    
    # --- FIX 1: Use logsumexp for the forward pass ---
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Prepare arguments for logsumexp
            args = np.array([
                -D[i - 1, j - 1],
                -D[i - 1, j],
                -D[i, j - 1]
            ]) / gamma
            
            # The numerically stable soft-min operation
            soft_min_val = -gamma * logsumexp(args)
            D[i, j] = cost_matrix[i - 1, j - 1] + soft_min_val
            
    # Backward pass to compute the gradient alignment matrix E
    E = np.zeros((n + 2, m + 2))
    E[n, m] = 1.0
    
    # --- FIX 2: Stabilize the gradient calculation in the backward pass ---
    for i in range(n, 0, -1):
        for j in range(m, 0, -1):
            if E[i, j] > 0: # No need to propagate gradient if it's zero
                # These are the log-probabilities of taking each path
                log_p_diag = (D[i - 1, j - 1] - D[i, j] + cost_matrix[i - 1, j - 1]) / gamma
                log_p_up = (D[i - 1, j] - D[i, j] + cost_matrix[i - 1, j - 1]) / gamma
                log_p_left = (D[i, j - 1] - D[i, j] + cost_matrix[i - 1, j - 1]) / gamma
                
                # Propagate the gradient E by multiplying by the path probabilities
                # (working in normal space here is fine as exp(log_p) will be <= 1)
                E[i - 1, j - 1] += np.exp(log_p_diag) * E[i, j]
                E[i - 1, j] += np.exp(log_p_up) * E[i, j]
                E[i, j - 1] += np.exp(log_p_left) * E[i, j]

    gradient = np.zeros_like(barycenter, dtype=float)
    for i in range(n):
        for j in range(m):
            gradient[i] += E[i + 1, j + 1] * 2 * (barycenter[i] - s[j])
            
    return D[n, m], gradient.flatten()


# --- The rest of the code is unchanged ---

def softdtw_barycenter(sequences, gamma, n_iterations, learning_rate):
    """
    Computes the barycenter using the Soft-DTW method and subgradient descent.
    """
    # Initialize the barycenter (e.g., as the first sequence)
    barycenter = sequences[0].copy()
    
    print(f"Starting Soft-DTW Barycenter optimization with {n_iterations} iterations.")
    
    for i in range(n_iterations):
        total_gradient = np.zeros_like(barycenter)
        total_loss = 0.

        # Accumulate gradient and loss over all sequences
        for s in sequences:
            loss, grad = soft_dtw_and_gradient(barycenter, s, gamma)
            total_gradient += grad
            total_loss += loss
            
        # Update the barycenter by taking a step against the average gradient
        barycenter -= learning_rate * (total_gradient / len(sequences))

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{n_iterations}, Average Loss: {total_loss / len(sequences):.4f}")
            
    return barycenter