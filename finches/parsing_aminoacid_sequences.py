"""
Local sequence-context weighting used by FINCHES.

FINCHES corrects the raw residue-residue interaction matrix for two local
sequence-context effects, and this module holds everything needed to compute
them:

* **Charge weighting** reduces like-charge repulsion between two charged
  residues according to the net charge of their +/-1 sequence neighbourhoods
  (see get_charge_weighted_mask()).
* **Aliphatic weighting** up-weights contacts between clusters of aliphatic
  residues, approximating the cooperative release of water from a hydrophobic
  interface (see get_aliphatic_weighted_mask()).

The residue classes and the two elementary weighting rules (a charge weight from
positive/negative counts, an aliphatic weight from two clustering levels) are
exposed as module-level constants and functions so that any other code that
applies FINCHES-style weighting - for example the structure-aware surface code
in finches.utils.structure_surface, which takes its local context from a 3D
surface patch rather than a linear window - uses exactly the same definitions.

Adapted from the original Mpipi implementation in LAMMPS (line 149 of
lammpstools/lammpstools/data/configuration_v4/build_data_file.py).

By: Garrett M. Ginell & Alex Holehouse
"""

import numpy as np

# residue classes used by the context weighting
POSITIVE_RESIDUES = frozenset("RK")
NEGATIVE_RESIDUES = frozenset("ED")
CHARGED_RESIDUES = POSITIVE_RESIDUES | NEGATIVE_RESIDUES
ALIPHATIC_RESIDUES = frozenset("AVILM")

# alternative one-letter codes that mark an aliphatic residue's clustering level:
# aliphatic_group1 for a residue in a pair, aliphatic_group2 for a residue in a
# cluster of three or more (see get_aliphaticgroup_sequence())
aliphatic_group1 = {"A": "a", "L": "l", "M": "m", "I": "i", "V": "v"}
aliphatic_group2 = {"A": "b", "L": "o", "M": "x", "I": "y", "V": "z"}


## ---------------------------------------------------------------------------
##
def get_charge_window_counts(sequence):
    """
    Count positive and negative residues in the +/-1 window around each position.

    For position i the window is residues i-1, i and i+1, truncated at the ends of
    the sequence. This is the local charge context used by the FINCHES charge
    weighting.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (pos_counts, neg_counts), each a float array of length len(sequence)
        giving the number of positive (R, K) and negative (E, D) residues in the
        window around each position.
    """
    pos_ind = np.array([r in POSITIVE_RESIDUES for r in sequence], dtype=float)
    neg_ind = np.array([r in NEGATIVE_RESIDUES for r in sequence], dtype=float)
    n = len(sequence)
    if n == 0:
        return pos_ind, neg_ind

    # width-3 moving sum centred on each position. We use mode="full" and take
    # the centre slice [1:n+1] rather than mode="same": np.convolve("same")
    # returns max(n, kernel) elements, which is wrong when n < 3.
    kernel = np.ones(3)
    return (
        np.convolve(pos_ind, kernel, mode="full")[1 : n + 1],
        np.convolve(neg_ind, kernel, mode="full")[1 : n + 1],
    )


## ---------------------------------------------------------------------------
##
def charge_weight_from_counts(total_pos, total_neg):
    """
    Charge weight |NCPR| / FCR of a fragment from its charged-residue counts.

    Both NCPR and FCR share the fragment length, so the length cancels and the
    weight reduces to |n_pos - n_neg| / (n_pos + n_neg). It is 1 for a fragment
    whose charges are all of one sign, 0 for a net-neutral fragment, and 0 (by
    convention) for a fragment with no charges.

    Parameters
    ----------
    total_pos : float or np.ndarray
        Number of positive residues in the fragment(s).

    total_neg : float or np.ndarray
        Number of negative residues in the fragment(s).

    Returns
    -------
    float or np.ndarray
        The charge weight, between 0 and 1, with the same shape as the inputs.
    """
    total_pos = np.asarray(total_pos, dtype=float)
    total_neg = np.asarray(total_neg, dtype=float)
    total_charge = total_pos + total_neg
    denom = np.where(total_charge == 0, 1.0, total_charge)  # avoid divide-by-zero
    weight = np.abs(total_pos - total_neg) / denom
    return float(weight) if weight.ndim == 0 else weight


## ---------------------------------------------------------------------------
##
def aliphatic_weight_from_levels(level_a, level_b):
    """
    Aliphatic weight for a pair of residues from their clustering levels.

    The weight depends on the smaller of the two levels (see
    get_aliphatic_groups() for what a level means): 3.0 if both residues sit in
    clusters of three or more aliphatics, 1.5 if the smaller is a pair, and 1.0
    otherwise (either residue isolated or not aliphatic).

    Parameters
    ----------
    level_a, level_b : int or np.ndarray
        Clustering levels (0, 1, 2 or 3). Arrays broadcast against each other.

    Returns
    -------
    float or np.ndarray
        The weight, 1.0, 1.5 or 3.0, with the broadcast shape of the inputs.
    """
    min_level = np.minimum(level_a, level_b)
    weight = np.ones_like(min_level, dtype=float)
    weight[min_level == 2] = 1.5
    weight[min_level >= 3] = 3.0
    return float(weight) if weight.ndim == 0 else weight


## ---------------------------------------------------------------------------
##
def get_charge_weighted_mask(sequence1, sequence2):
    """
    Compute the charge-weighting mask for an inter-residue interaction matrix.

    For every pair of charged residues (one from each sequence) we build a
    fragment from the +/-1 window around each residue (up to six residues, fewer
    at the ends of a sequence) and compute its |NCPR| / FCR (see
    charge_weight_from_counts()). The weight is 1 when every charge in the
    fragment has the same sign and 0 when the fragment is net neutral. The
    caller scales the weight by the forcefield's charge prefactor and uses it to
    reduce like-charge repulsion, so that clusters of like charges are penalised
    less than the raw pair potential would suggest. Pairs in which either residue
    is uncharged get a weight of 0.

    For example, the fragments KKK + EEE give |0| / 1 = 0 (no correction), while
    EEE + EEE give |-1| / 1 = 1 (the full correction).

    Parameters
    ----------
    sequence1 : str
        First amino acid sequence (the rows of the matrix).

    sequence2 : str
        Second amino acid sequence (the columns of the matrix).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (attractive_matrix, repulsive_matrix), both of shape
        (len(sequence1), len(sequence2)). attractive_matrix is always all zeros
        and is kept for backwards compatibility; repulsive_matrix holds the
        charge weights, which are non-zero only where both residues are
        charged.

    """
    n1, n2 = len(sequence1), len(sequence2)

    # Pre-compute: which positions are charged in each sequence
    # explicit bool dtype: an empty sequence would otherwise give a float array
    # and the & below would raise
    charged_mask1 = np.array([r in CHARGED_RESIDUES for r in sequence1], dtype=bool)
    charged_mask2 = np.array([r in CHARGED_RESIDUES for r in sequence2], dtype=bool)

    # Per-position positive/negative counts in the +/-1 window of each sequence
    pos1, neg1 = get_charge_window_counts(sequence1)
    pos2, neg2 = get_charge_window_counts(sequence2)

    # Charge weight |NCPR| / FCR of the combined fragment for every (i, j) pair
    weights = charge_weight_from_counts(
        pos1[:, None] + pos2[None, :], neg1[:, None] + neg2[None, :]
    )

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
    Compute the aliphatic-weighting mask for an inter-residue interaction matrix.

    Approximates the cooperative hydrophobic effect: an isolated aliphatic
    residue may not bury enough surface to release water, but a cluster of them
    can. Each residue is assigned a clustering level by get_aliphatic_groups(),
    and each pair is weighted on the smaller of its two levels (see
    aliphatic_weight_from_levels()):

    * both residues in clusters of three or more (3-3): 3.0
    * the smaller level is a pair (2-2, 2-3, 3-2): 1.5
    * otherwise (either residue isolated or not aliphatic): 1.0

    Parameters
    ----------
    sequence1 : str
        First amino acid sequence (the rows of the matrix).

    sequence2 : str
        Second amino acid sequence (the columns of the matrix).

    Returns
    -------
    np.ndarray
        Weight matrix of shape (len(sequence1), len(sequence2)) whose elements
        are 1.0, 1.5 or 3.0. The caller multiplies the interaction matrix by it
        element-wise.

    """
    # Get clustering group (0, 1, 2, or 3) for each position
    groups1 = np.array(get_aliphatic_groups(sequence1))
    groups2 = np.array(get_aliphatic_groups(sequence2))

    # weight scheme on the smaller of the two levels for every (i, j) pair:
    # min >= 3 -> 3.0, min == 2 -> 1.5, otherwise 1.0
    return aliphatic_weight_from_levels(groups1[:, None], groups2[None, :])


## ---------------------------------------------------------------------------
##
def get_aliphatic_groups(sequence):
    """
    Assign each residue its aliphatic clustering level.

    Aliphatic residues (A, V, I, L, M) are grouped into clusters: runs of
    aliphatic residues in which a single non-aliphatic residue is tolerated (so
    "AGA" is one cluster of two), while two or more consecutive non-aliphatic
    residues end a cluster. Each aliphatic residue's level is the size of its
    cluster, capped at 3; non-aliphatic residues are 0.

    Clustering levels:

    * 0 - not an aliphatic residue
    * 1 - isolated aliphatic (the only aliphatic in its cluster)
    * 2 - in a pair (exactly one other aliphatic in its cluster)
    * 3 - in a cluster of three or more aliphatics

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    list of int
        Per-residue clustering level (0, 1, 2 or 3), one per position.

    Examples
    --------
    >>> get_aliphatic_groups("GGGAGG")  # isolated A
    [0, 0, 0, 1, 0, 0]

    >>> get_aliphatic_groups("GGAAGG")  # pair of A's
    [0, 0, 2, 2, 0, 0]

    >>> get_aliphatic_groups("GAAAAG")  # cluster of A's
    [0, 3, 3, 3, 3, 0]

    >>> get_aliphatic_groups("AGAGGA")  # one gap joins, two gaps split
    [2, 0, 2, 0, 0, 1]

    """
    aliphatic = np.fromiter(
        (r in ALIPHATIC_RESIDUES for r in sequence), dtype=bool, count=len(sequence)
    )
    levels = np.zeros(len(sequence), dtype=np.int64)

    positions = np.flatnonzero(aliphatic)
    if positions.size == 0:
        return levels.tolist()

    # consecutive aliphatics 3 or more positions apart have at least two
    # non-aliphatic residues between them, which starts a new cluster
    clusters = np.split(positions, np.flatnonzero(np.diff(positions) >= 3) + 1)
    for cluster in clusters:
        levels[cluster] = min(cluster.size, 3)

    return levels.tolist()


## ---------------------------------------------------------------------------
##
def get_aliphaticgroup_sequence(chain):
    """
    Rewrite a sequence so that clustered aliphatic residues carry their level.

    Aliphatic residues are recoded according to their clustering level from
    get_aliphatic_groups(): residues in a pair are replaced with the codes in
    aliphatic_group1 (A->a, L->l, M->m, I->i, V->v), residues in a cluster of
    three or more with the codes in aliphatic_group2 (A->b, L->o, M->x, I->y,
    V->z), and isolated aliphatics and every other residue are left unchanged.
    These codes match the context-specific aliphatic beads of the Mpipi-GG
    parameter set.

    Nothing in FINCHES calls this function: the aliphatic context is applied as
    a weight on the interaction matrix (get_aliphatic_weighted_mask()) rather
    than by recoding the sequence.

    Parameters
    ----------
    chain : str
        Amino acid sequence.

    Returns
    -------
    str
        The recoded sequence, the same length as chain.

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
