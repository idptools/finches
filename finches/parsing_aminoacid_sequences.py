"""
Holehouse Lab - Internal Script

This script has code to parse amino acid sequences and assign groups to the
aliphatics with the proper character input needed in PIMMS
written for Ryan's development of AA params in PIMMS.

Adapted from original implementation in LAMMPS - Mpipi
see original at:
line 149 of lammpstools/lammpstools/data/configuration_v4/build_data_file.py

by: Garrett M. Ginell


"""

import numpy as np

# new characters for PIMMS aliphatic groups
aliphatic_group1 = {"A": "a", "L": "l", "M": "m", "I": "i", "V": "v"}
aliphatic_group2 = {"A": "b", "L": "o", "M": "x", "I": "y", "V": "z"}


## ---------------------------------------------------------------------------
##
def get_charge_weighted_mask(sequence1, sequence2):
    """
    Compute a charge-weighted mask for the inter-residue interaction matrix.

    For each pair of charged residues (one from each sequence), calculates a
    weight based on the local charge environment. The weight is |NCPR|/FCR of
    a 6-residue fragment formed by concatenating a ±1 window around each residue.

    This weighting allows clusters of like-charged residues to be weighted up,
    which is then used to reduce like-charge repulsion in the interaction matrix.

    Examples:
        - Fragments KKK + EEE → |NCPR|/FCR = |0|/1 = 0.0 (no weight, mixed charges)
        - Fragments EEE + EEE → |NCPR|/FCR = |-1|/1 = 1.0 (max weight, all same charge)

    Parameters
    ----------
    sequence1 : str
        First amino acid sequence (y-axis of matrix)

    sequence2 : str
        Second amino acid sequence (x-axis of matrix)

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Two matrices of shape (len(sequence1), len(sequence2)):
        - attractive_matrix: Currently all zeros (not used, kept for compatibility)
        - repulsive_matrix: Charge weights at intersections of charged residues

    """
    CHARGED_RESIDUES = {"R", "K", "E", "D"}
    POSITIVE = {"R", "K"}
    NEGATIVE = {"E", "D"}

    n1, n2 = len(sequence1), len(sequence2)

    # Pre-compute: which positions are charged in each sequence
    charged_mask1 = np.array([r in CHARGED_RESIDUES for r in sequence1])
    charged_mask2 = np.array([r in CHARGED_RESIDUES for r in sequence2])

    # Per-position positive/negative counts in the ±1 window (positions i-1, i, i+1
    # clipped to the sequence bounds). This is a width-3 moving sum over the
    # charge-indicator arrays; np.convolve(..., mode="same") reproduces the
    # boundary-truncated window exactly, replacing the old per-position Python loops.
    _kernel = np.ones(3)

    def window_counts(sequence):
        pos_ind = np.array([r in POSITIVE for r in sequence], dtype=float)
        neg_ind = np.array([r in NEGATIVE for r in sequence], dtype=float)
        n = len(sequence)
        if n == 0:
            return pos_ind, neg_ind
        # width-3 moving sum centred on each position. We use mode="full" and take
        # the centre slice [1:n+1] rather than mode="same": np.convolve("same")
        # returns max(n, kernel) elements, which is wrong when n < 3.
        return (
            np.convolve(pos_ind, _kernel, mode="full")[1 : n + 1],
            np.convolve(neg_ind, _kernel, mode="full")[1 : n + 1],
        )

    pos1, neg1 = window_counts(sequence1)
    pos2, neg2 = window_counts(sequence2)

    # Combined fragment charge counts for every (i, j) pair via broadcasting
    total_pos = pos1[:, None] + pos2[None, :]
    total_neg = neg1[:, None] + neg2[None, :]
    total_charge = total_pos + total_neg

    # Charge weight = |NCPR| / FCR. Since both NCPR and FCR share the same
    # fragment length, the length cancels and this reduces to
    # |total_pos - total_neg| / (total_pos + total_neg).
    denom = np.where(total_charge == 0, 1.0, total_charge)  # avoid divide-by-zero
    weights = np.abs(total_pos - total_neg) / denom

    # Only keep weights where BOTH residues are charged; everything else stays 0
    both_charged = charged_mask1[:, None] & charged_mask2[None, :]
    repulsive_matrix = np.zeros((n1, n2), dtype=float)
    repulsive_matrix[both_charged] = weights[both_charged]

    # attractive_matrix is all zeros (kept for API compatibility)
    attractive_matrix = np.zeros((n1, n2), dtype=float)

    return attractive_matrix, repulsive_matrix


## ---------------------------------------------------------------------------
##
def get_aliphatic_weighted_mask(sequence1, sequence2):
    """
    Compute an aliphatic clustering weight mask for the interaction matrix.

    Approximates the cooperative hydrophobic effect: isolated aliphatic residues
    may not create a large enough interface to release water molecules, but
    clusters of aliphatics can. This up-weights interactions between aliphatic
    clusters to make them effectively more "hydrophobic".

    Weight scheme based on cluster size (1=isolated, 2=pair, 3=cluster of 3+):
        - Both isolated (1) or one isolated: weight = 1.0 (no boost)
        - Both in small clusters (2-2, 2-3, 3-2): weight = 1.5
        - Both in large clusters (3-3): weight = 3.0

    Parameters
    ----------
    sequence1 : str
        First amino acid sequence (y-axis of matrix)

    sequence2 : str
        Second amino acid sequence (x-axis of matrix)

    Returns
    -------
    np.ndarray
        2D weight matrix of shape (len(sequence1), len(sequence2)).
        Values are 1.0, 1.5, or 3.0 depending on aliphatic clustering.

    """
    # Get clustering group (0, 1, 2, or 3) for each position
    groups1 = np.array(get_aliphatic_groups(sequence1))
    groups2 = np.array(get_aliphatic_groups(sequence2))

    # Create 2D matrix of minimum group values using broadcasting
    # min_groups[i,j] = min(groups1[i], groups2[j])
    min_groups = np.minimum.outer(groups1, groups2)

    # Apply weight scheme based on minimum cluster size:
    #   min >= 3 → 3.0
    #   min == 2 → 1.5
    #   min <= 1 → 1.0
    weights = np.ones_like(min_groups, dtype=float)
    weights[min_groups == 2] = 1.5
    weights[min_groups >= 3] = 3.0

    return weights


## ---------------------------------------------------------------------------
##
def get_aliphatic_groups(sequence):
    """
    Classify each residue by its local aliphatic clustering level.

    For each position in the sequence, determines how "clustered" aliphatic
    residues are in that local region. Non-aliphatic residues get 0, while
    aliphatic residues get 1, 2, or 3 based on how many nearby aliphatics
    they have.

    Clustering levels:
        0 = Not an aliphatic residue
        1 = Isolated aliphatic (no nearby aliphatics)
        2 = Small cluster (1-2 nearby aliphatics)
        3 = Large cluster (3+ nearby aliphatics)

    Aliphatic residues: A, V, I, L, M

    Parameters
    ----------
    sequence : str
        Amino acid sequence

    Returns
    -------
    list of int
        Per-residue clustering level (0, 1, 2, or 3) for each position.

    Examples
    --------
    >>> get_aliphatic_groups("GGGAGG")  # isolated A
    [0, 0, 0, 1, 0, 0]

    >>> get_aliphatic_groups("GGAAGG")  # pair of A's
    [0, 0, 2, 2, 0, 0]

    >>> get_aliphatic_groups("GAAAAG")  # cluster of A's
    [0, 3, 3, 3, 3, 0]

    """
    ALIPHATIC_RESIDUES = {"A", "V", "I", "L", "M"}
    WINDOW_SIZE = 4

    n = len(sequence)

    # Binary mask (1 = aliphatic, 0 = not aliphatic)
    mask = np.fromiter(
        (1 if r in ALIPHATIC_RESIDUES else 0 for r in sequence),
        dtype=np.int64,
        count=n,
    )
    levels = np.zeros(n, dtype=np.int64)

    one_idx = np.flatnonzero(mask)
    if one_idx.size == 0:
        return levels.tolist()

    # Split aliphatic positions into clusters: a run of >= 2 zeros between
    # consecutive aliphatics starts a new cluster, while a single-zero gap keeps
    # them together. This matches count_nearby_hits(..., max_gap=1, window_size=4)
    # / extract_fragments(mask, max_gap=1), but is computed with NumPy rather than
    # per-character Python string operations.
    split_points = np.flatnonzero(np.diff(one_idx) >= 3) + 1
    clusters = np.split(one_idx, split_points)

    for cluster in clusters:
        # character run for this cluster (first aliphatic to last, including the
        # internal single-zero gaps), and a prefix sum for O(1) window counts
        cluster_chars = mask[cluster[0] : cluster[-1] + 1]
        csum = np.concatenate(([0], np.cumsum(cluster_chars)))

        # for the m-th aliphatic in the cluster, count aliphatics in the window
        # cluster_chars[m - WINDOW_SIZE : m + WINDOW_SIZE + 1] (clipped), matching
        # the original window indexing exactly
        m = np.arange(cluster.size)
        starts = np.maximum(0, m - WINDOW_SIZE)
        ends = np.minimum(cluster_chars.size, m + WINDOW_SIZE + 1)
        levels[cluster] = np.minimum(csum[ends] - csum[starts], 3)

    return levels.tolist()


## ---------------------------------------------------------------------------
##
def get_aliphaticgroup_sequence(chain):
    """
    NOT CURRENTLY USED IN CODE...

    Function to get a sequence that is re-assigned aliphatics with
    beads for the grouped aliphatics in used in grouping of aliphatics.
    The bead assignments are those that are used in the Mpipi field.

    Takes a passed sequence and get a ready sequence of aliphatics
        aliphatic_group1 = {'A':'A', 'L':'L', 'M':'M','I':'I','V':'V'}
        aliphatic_group2 = {'A':'a', 'L':'l', 'M':'m','I':'i','V':'v'}
        aliphatic_group3 = {'A':'b', 'L':'o', 'M':'x', 'I':'y','V':'z'}

    Parameters
    --------------
    chain : str
         sequence which contains aliphatics residues
         grouped by their nearest neighbors to note local aliphatic surfaces
         in a chain

    Returns
    ---------------
    str
        sequence where aliphatic residues are grouped and re-assigned
        symbols based on the bead assignments used in the Mpipi field.

    """

    # get aliphatic groups by nearest neighbors
    aligroups = get_aliphatic_groups(chain)

    # build new proper chain string
    newsequence = []
    for i, a in enumerate(chain):
        if aligroups[i] > 1:
            if aligroups[i] == 2:
                newsequence.append(aliphatic_group1[a])
            elif aligroups[i] == 3:
                newsequence.append(aliphatic_group2[a])
        else:
            newsequence.append(a)

    return "".join(newsequence)
