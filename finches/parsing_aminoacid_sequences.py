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
from finches import sequence_tools

# new characters for PIMMS aliphatic groups 
aliphatic_group1 = {'A':'a', 'L':'l', 'M':'m', 'I':'i', 'V':'v'}
aliphatic_group2 = {'A':'b', 'L':'o', 'M':'x', 'I':'y', 'V':'z'}

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
    CHARGED_RESIDUES = {'R', 'K', 'E', 'D'}
    POSITIVE = {'R', 'K'}
    NEGATIVE = {'E', 'D'}

    n1, n2 = len(sequence1), len(sequence2)

    # Pre-compute: which positions are charged in each sequence
    charged_mask1 = np.array([r in CHARGED_RESIDUES for r in sequence1])
    charged_mask2 = np.array([r in CHARGED_RESIDUES for r in sequence2])

    # Pre-compute: the ±1 window fragment for each position in both sequences
    # This avoids calling get_neighbors_window_of3() inside the loop
    def get_window(i, seq):
        """Get residues at positions i-1, i, i+1 (clipped to sequence bounds)."""
        start = max(0, i - 1)
        end = min(len(seq), i + 2)
        return seq[start:end]

    windows1 = [get_window(i, sequence1) for i in range(n1)]
    windows2 = [get_window(j, sequence2) for j in range(n2)]

    # Pre-compute: FCR and NCPR components for each window
    # For a fragment, FCR = (n_pos + n_neg) / len, NCPR = (n_pos - n_neg) / len
    def count_charges(fragment):
        """Count positive and negative residues in a fragment."""
        n_pos = sum(1 for r in fragment if r in POSITIVE)
        n_neg = sum(1 for r in fragment if r in NEGATIVE)
        return n_pos, n_neg

    # Pre-compute charge counts for all windows
    charges1 = [count_charges(w) for w in windows1]  # list of (n_pos, n_neg)
    charges2 = [count_charges(w) for w in windows2]

    # Initialize output matrices
    repulsive_matrix = np.zeros((n1, n2), dtype=float)
    # attractive_matrix is always zeros in current implementation

    # Get indices where residues are charged
    charged_indices1 = np.where(charged_mask1)[0]
    charged_indices2 = np.where(charged_mask2)[0]

    # Only compute weights where BOTH residues are charged
    for i in charged_indices1:
        pos1, neg1 = charges1[i]
        len1 = len(windows1[i])

        for j in charged_indices2:
            pos2, neg2 = charges2[j]
            len2 = len(windows2[j])

            # Combined fragment stats (6 residues max)
            total_pos = pos1 + pos2
            total_neg = neg1 + neg2
            total_len = len1 + len2

            # Calculate FCR and NCPR of combined fragment
            fcr = (total_pos + total_neg) / total_len
            ncpr = (total_pos - total_neg) / total_len

            # Charge weight = |NCPR| / FCR
            # fcr > 0 is guaranteed since both central residues are charged
            repulsive_matrix[i, j] = abs(ncpr) / fcr

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
    ALIPHATIC_RESIDUES = ['A', 'V', 'I', 'L', 'M']

    # Step 1: Create binary mask (1 = aliphatic, 0 = not aliphatic)
    aliphatic_mask = sequence_tools.mask_sequence(sequence, ALIPHATIC_RESIDUES)

    # Step 2: For each aliphatic, count how many aliphatics are nearby
    # This returns: 0 for non-aliphatics, N for aliphatics (where N = count of
    # aliphatics in local window including self)
    neighbor_counts = sequence_tools.count_nearby_hits(aliphatic_mask, max_gap=1, window_size=4)

    # Step 3: Bin into clustering levels (cap at 3 for "large cluster")
    # neighbor_count=1 means isolated (just itself), 2 means one neighbor, etc.
    clustering_levels = [min(count, 3) for count in neighbor_counts]

    return clustering_levels

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
    
    return ''.join(newsequence)
