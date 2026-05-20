import numpy as np

# Stateless functions that can then be freely imported

# ============================================================================
#                    FUNCTIONS FOR MATRIX MANIPULATION
# ============================================================================


def get_attractive_repulsive_matrices(matrix, null_interaction_baseline):
    """
    Take interaction array, descritize it by above or below interaction baseline,
    Return two shaped matched matrixes for attractive and repulsive values

    The null_interaction_baseline = value to split matrix. This has been built such
        that this value recapitulates PolyGS for the specific input model being use
        to see more on how to compute a null_interaction_baseline see...

        NEED TO UPDATE HERE

    Parameters
    ---------------
    matrix : np.array
        array returned by a function in the InteractionMatrixConstructor class

    null_interaction_baseline : float
        Value to specify where to split the matrix for attractive vs repulsive interactions

    Returns
    ------------------
    attractive_matrix : np.array
        An array the same shape of the input matrix with only values below the
        null_interaction_baseline

    repulsive_matrix : np.array
        An array the same shape of the input matrix with only values above the
        null_interaction_baseline

    """
    return (matrix < null_interaction_baseline) * matrix, (
        matrix > null_interaction_baseline
    ) * matrix


def mask_matrix(matrix, column_mask):
    """
    Function to take matrix and multipy it by a mask. This
    also check to make sure the mask is the same shape.

    Parameters
    ---------------
    matrix : array
        A 2D matrix as an array with the shape of (seqence1, seqence2)

    column_mask : array
        A 2D array with the shape of the inputed matrix

    Returns
    ------------------
    out_matrix : array
        A 2D matrix with the same shape of the inputed matrix
        where the out_matrix = matrix*column_mask

    """
    if matrix.shape != column_mask.shape:
        raise ValueError(
            f"Shape mismatch: matrix {matrix.shape} vs mask {column_mask.shape}"
        )
    return matrix * column_mask


# ============================================================================
#                  BUILDING VECTORS & COMPUTING EPSILON
# ============================================================================


def get_sequence_epsilon_vectors(
    sequence1,
    sequence2,
    X,
    charge_prefactor=None,
    null_interaction_baseline=None,
    use_charge_weighting=True,
    use_aliphatic_weighting=True,
):
    """
    Function to epsilon vectors between a pair of passed sequences
    returned vectors are relative to sequence1 such that len(sequence1) equals
    the len(returned_vectors)

    NOTE this code was previously : get_weighted_sequence_epsilon_value
        It is now UPDATED to get_sequence_epsilon_vectors and get_sequence_epsilon_value
        all weighting is determined by flags.

    Parameters
    -----------
    sequence1 : str
        The first sequence to compare

    sequence2 : str
        The second sequence to compare

    X : obj
        Instance of the InteractionMatrixConstructor class with initialized pairwise interactions
        and modelspecific parameters

    Optional Parameters
    -------------------
    null_interaction_baseline : float
        threshold to differentiate between attractive and repulsive interactions

    charge_prefactor : float
        Model specific value to plug into the local charge weighting of
        the matrix

    use_charge_weighting : bool
        Flag to select whether weight the matrix by local sequence charge

    use_aliphatic_weighting : bool
        Flag to select whether weight the matrix by local patches of aliphatic
        residues

    Returns
    --------
    attractive_vector : list
        attractive epsilon vector of sequence1 relative to sequence2

    repulsive_vector : list
        repulsive epsilon vector of sequence1 relative to sequence2

    """
    # use the explicitly-passed baseline when provided (an explicit 0.0 is valid
    # and must not fall through to the instance default)
    baseline = (
        X.null_interaction_baseline
        if null_interaction_baseline is None
        else null_interaction_baseline
    )

    # get interaction matrix for said sequence
    w_matrix = X.calculate_weighted_pairwise_matrix(
        sequence1,
        sequence2,
        convert_to_custom=True,
        charge_prefactor=charge_prefactor,
        use_charge_weighting=use_charge_weighting,
        use_aliphatic_weighting=use_aliphatic_weighting,
    )

    # get attractive and repulsive matrix. The function below takes the w_matrix and separates it out
    # into two matrices, where elements that are below null_interaction_baseline are in the attractive_matrix
    # and elements that are above the null_interaction_baseline are in the repulsive_matrix
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrices(
        w_matrix, baseline
    )

    # subtract off the baselines so that 0 = non-interacting, then take row means
    return np.mean(attractive_matrix - baseline, axis=1), np.mean(
        repulsive_matrix - baseline, axis=1
    )


def get_sequence_epsilon_value(
    sequence1,
    sequence2,
    X,
    charge_prefactor=None,
    null_interaction_baseline=None,
    use_charge_weighting=True,
    use_aliphatic_weighting=True,
):
    """
    Function to epsilon value between a pair of passed sequences

    NOTE this was previously : get_weighted_sequence_epsilon_value
        It is now UPDATED to get_sequence_epsilon_value and all weighting is determined by flags.

    Parameters
    -----------
    sequence1 : str
        The first sequence to compare

    sequence2 : str
        The second sequence to compare

    X : obj
        Instance of the InteractionMatrixConstructor class with initialized pairwise interactions
        and modelspecific parameters

    Optional Parameters
    -------------------
    null_interaction_baseline : float
        threshold to differentiate between attractive and repulsive interactions

    charge_prefactor : float
        Model specific value to plug into the local charge weighting of
        the matrix

    use_charge_weighting : bool
        Flag to select whether weight the matrix by local sequence charge

    use_aliphatic_weighting : bool
        Flag to select whether weight the matrix by local patches of aliphatic
        residues

    Returns
    --------
    epsilon : float
        sequence epsilon value as computed between sequence1 and sequence2

    """
    attractive_vector, repulsive_vector = get_sequence_epsilon_vectors(
        sequence1,
        sequence2,
        X,
        charge_prefactor=charge_prefactor,
        null_interaction_baseline=null_interaction_baseline,
        use_charge_weighting=use_charge_weighting,
        use_aliphatic_weighting=use_aliphatic_weighting,
    )
    return np.sum(attractive_vector) + np.sum(repulsive_vector)
