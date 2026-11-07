"""
Stateless functions for computing FINCHES epsilon values.

Everything here takes the interaction matrix (or an InteractionMatrixConstructor
that builds it) as an argument and holds no state of its own, so these functions
can be imported anywhere without pulling in the constructor class. The
InteractionMatrixConstructor methods calculate_epsilon_vectors() and
calculate_epsilon_value() are thin wrappers around the functions below.

The epsilon convention used throughout FINCHES is

    epsilon = sum_i mean_j (M_ij - 2 * b)

where M is the (weighted) interaction matrix between two sequences and b is the
forcefield's null interaction baseline, calibrated so that poly(GS) has an
epsilon of zero against itself. The factor of two comes from splitting M into
attractive and repulsive matrices and subtracting b from both (see
get_attractive_repulsive_matrices()).

By: Garrett M. Ginell & Alex Holehouse
"""

import numpy as np

# ============================================================================
#                    FUNCTIONS FOR MATRIX MANIPULATION
# ============================================================================


def get_attractive_repulsive_matrices(matrix, null_interaction_baseline):
    """
    Split an interaction matrix into an attractive and a repulsive matrix.

    The null_interaction_baseline is the dividing line. It is the model-specific
    value calibrated so that a poly(GS) sequence has an epsilon of zero (see
    finches.data.forcefield_dependencies.get_null_interaction_baseline()). Every
    element of the input matrix ends up in exactly one of the two returned
    matrices: elements at or below the baseline go into the attractive matrix
    and elements strictly above it go into the repulsive matrix. Elements that
    belong to the other matrix are set to zero, so attractive + repulsive
    recovers the input matrix.

    An element exactly equal to the baseline is deliberately counted once (as
    attractive). This keeps the downstream epsilon calculation, which subtracts
    the baseline from every element of both matrices, continuous at the baseline
    and consistent with the summed-area-table implementation in
    finches.utils.matrix_manipulation.matrix_scan().

    Parameters
    ----------
    matrix : np.ndarray
        Interaction matrix, e.g. from
        InteractionMatrixConstructor.calculate_weighted_pairwise_matrix().

    null_interaction_baseline : float
        Value that divides attractive from repulsive interactions.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (attractive_matrix, repulsive_matrix), each the same shape as matrix.
        attractive_matrix holds the elements at or below the baseline and
        repulsive_matrix the elements above it; all other elements are zero.

    """
    return (matrix <= null_interaction_baseline) * matrix, (
        matrix > null_interaction_baseline
    ) * matrix


def mask_matrix(matrix, column_mask):
    """
    Multiply a matrix element-wise by a mask of the same shape.

    Nothing in FINCHES calls this function; the charge and aliphatic weighting
    in InteractionMatrixConstructor.calculate_weighted_pairwise_matrix() apply
    their masks directly.

    Parameters
    ----------
    matrix : np.ndarray
        2D matrix of shape (len(sequence1), len(sequence2)).

    column_mask : np.ndarray
        2D mask with the same shape as matrix.

    Returns
    -------
    np.ndarray
        matrix * column_mask, with the same shape as matrix.

    Raises
    ------
    ValueError
        If matrix and column_mask do not have the same shape.

    """
    if matrix.shape != column_mask.shape:
        raise ValueError(
            f"Shape mismatch: matrix {matrix.shape} vs mask {column_mask.shape}"
        )
    return matrix * column_mask


# ============================================================================
#                  BUILDING VECTORS & COMPUTING EPSILON
# ============================================================================


def get_matrix_epsilon_vectors(matrix, null_interaction_baseline):
    """
    Compute the attractive and repulsive epsilon vectors of an interaction matrix.

    The matrix is split with get_attractive_repulsive_matrices(), the baseline is
    subtracted from every element of both matrices so that zero means
    "non-interacting", and each row is averaged. Summing both vectors gives the
    epsilon of the matrix. This is the single place the epsilon convention is
    applied to a matrix; it is used for whole sequences by
    get_sequence_epsilon_vectors() and for every window by the pure-Python path of
    InteractionMatrixConstructor.calculate_sliding_epsilon().

    Parameters
    ----------
    matrix : np.ndarray
        Interaction matrix of shape (n, m), rows indexing the first sequence.

    null_interaction_baseline : float
        The forcefield's null interaction baseline.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (attractive_vector, repulsive_vector), each of length n: the mean over
        the columns of each row of the baseline-subtracted attractive and
        repulsive matrices.

    """
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrices(
        matrix, null_interaction_baseline
    )

    # subtract off the baselines so that 0 = non-interacting, then take row means
    return np.mean(attractive_matrix - null_interaction_baseline, axis=1), np.mean(
        repulsive_matrix - null_interaction_baseline, axis=1
    )


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
    Compute the attractive and repulsive epsilon vectors between two sequences.

    The vectors are per-residue for sequence1: element i is the mean, over every
    residue of sequence2, of residue i's baseline-subtracted attractive (or
    repulsive) interaction. Summing both vectors gives the epsilon value (see
    get_sequence_epsilon_value()).

    Parameters
    ----------
    sequence1 : str
        The sequence the vectors are reported for.

    sequence2 : str
        The sequence it is compared against.

    X : finches.epsilon_calculation.InteractionMatrixConstructor
        Constructor with the forcefield's pairwise interactions and calibrated
        constants.

    charge_prefactor : float, optional
        Charge prefactor used by the charge weighting. Default None uses
        X.charge_prefactor.

    null_interaction_baseline : float, optional
        Baseline that divides attractive from repulsive interactions. Default
        None uses X.null_interaction_baseline; an explicit 0.0 is honoured.

    use_charge_weighting : bool, optional
        Whether to apply the local charge weighting. Default True.

    use_aliphatic_weighting : bool, optional
        Whether to apply the aliphatic clustering weighting. Default True.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (attractive_vector, repulsive_vector), each of length len(sequence1).

    """
    # use the explicitly-passed baseline when provided (an explicit 0.0 is valid
    # and must not fall through to the instance default)
    baseline = (
        X.null_interaction_baseline
        if null_interaction_baseline is None
        else null_interaction_baseline
    )

    w_matrix = X.calculate_weighted_pairwise_matrix(
        sequence1,
        sequence2,
        convert_to_custom=True,
        charge_prefactor=charge_prefactor,
        use_charge_weighting=use_charge_weighting,
        use_aliphatic_weighting=use_aliphatic_weighting,
    )

    return get_matrix_epsilon_vectors(w_matrix, baseline)


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
    Compute the epsilon value between two sequences.

    Epsilon is the sum over the residues of sequence1 of the mean over the
    residues of sequence2 of M_ij - 2b, where M is the weighted interaction
    matrix and b the null interaction baseline. It is therefore extensive in
    sequence1 and intensive in sequence2. Negative values are attractive,
    positive values repulsive, and poly(GS) against itself is zero by
    construction.

    Parameters
    ----------
    sequence1 : str
        The first sequence (epsilon scales with its length).

    sequence2 : str
        The second sequence (epsilon is averaged over it).

    X : finches.epsilon_calculation.InteractionMatrixConstructor
        Constructor with the forcefield's pairwise interactions and calibrated
        constants.

    charge_prefactor : float, optional
        Charge prefactor used by the charge weighting. Default None uses
        X.charge_prefactor.

    null_interaction_baseline : float, optional
        Baseline that divides attractive from repulsive interactions. Default
        None uses X.null_interaction_baseline; an explicit 0.0 is honoured.

    use_charge_weighting : bool, optional
        Whether to apply the local charge weighting. Default True.

    use_aliphatic_weighting : bool, optional
        Whether to apply the aliphatic clustering weighting. Default True.

    Returns
    -------
    float
        The epsilon value between sequence1 and sequence2.

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
