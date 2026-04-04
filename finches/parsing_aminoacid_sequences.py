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
from finches.utils import matrix_manipulation

# new characters for PIMMS aliphatic groups 
aliphatic_group1 = {'A':'a', 'L':'l', 'M':'m', 'I':'i', 'V':'v'}
aliphatic_group2 = {'A':'b', 'L':'o', 'M':'x', 'I':'y', 'V':'z'}

## ---------------------------------------------------------------------------
##
def get_charge_weighted_mask(sequence1, sequence2):
    """
    Function to get the charge-weighted mask of the inter-residue interaction
    matrix.

    Specifically, this function loops over all cross-interacting residues from
    the two sequences (i.e. every pair of r1:r2 (where r1 is from seq1 and r2 
    is from seq1), and if BOTH residues are charged, then a 'charge weight' is
    calculated whereby the +/- 1 residues around those two residues are extracted
    and the |NCPR|/FCR of the resulting concatenated sequence is a weighting factor.

    What does this mean, practically?

    If I have two fragments that are KKK and EEE then my charge weighting will be

    |NCPR/FCR| = 0/1 = 0.0 - NO WEIGHT

    If I have two fragments that are EEE and EEE, then my charge weighting will be
    
    |NCPR/FCR| = |-1/1| = 1.0 - MAX POSSIBLE WEIGHT

    In this way, clusters of like-charged residues are weighted up, and then
    subtracted off the repulsive cross terms to weaken like-charge repulsion.

    This means we ONLY generate a so-called repulsive matrix.

    BONUS CONTENT:

    We ALSO tested a version where charge weight was done by determining if 
    the local context of a charged residue is expected to enhance attractive 
    interactions or suppress repulsive interactions compared to an unweighted 
    value. Specifically, for each unique pair of residues in sequence 
    1 and sequence 2 we ask:
    
    1. Are both residues charged? If yes, continue.

    2. In the +1/-1 window around the two residues, are any charged residues 
       found the same sign as the central residue? If yes, for both residues,
       continue.

    3. Charge weight is calculated as the product of the NCPR from the two
       fragments. If these are the same sign, this is a repulsive weight, whereas
       if the opposite sign, this is an attractive weight.

    Note that attractive weights make oppositely-attractive residues MORE 
    attractive, whereas repulsive weights make like-charged residues LESS 
    repulsive. The result from this is two matrices that can be used to add 
    or subtract values from the overall interaction matrix. HOWEVER, we found
    this implementation just worked less well across the board, so the final
    implementation is the simpler |NCPR/FCR| weighting.
    
    Parameters
    --------------
    sequence1 : str 
        Input sequence 1 on y axis of matrix 

    sequence2 : str 
        Input sequence 2 on x axis of matrix

    Returns
    ---------------
    Tuple 

        This returns a tuple of two np.arrays (matrices) that are weighted 
        masks of the same  shape of (len(sequence1), len(sequence2)) where 
        at intersections of charged residues between the two sequences we 
        get a charge weighting factor. 
    
        Matrix 1 is the attractive matrix and matrix 2 is the repulsive 
        matrix. NOTE that we currently do not actually use the attractive 
        matrix here, but this function does return
    """
    
    attractive_matrix, repulsive_matrix = matrix_manipulation.charge_weighted_mask(sequence1, sequence2)
    return np.asarray(attractive_matrix), np.asarray(repulsive_matrix)



## ---------------------------------------------------------------------------
##
def get_charge_weighted_FD_mask(sequence1, sequence2):
    """
    Function to get the charge-weighted mask of a Matrix EXCEPT 
    that residues in sequence 1 are treated in isolation. 

    Here sequence1 is the SAFD sequence, so weighting is computed 
    between the 1FD residue and the normal three residues in the IDR

    Parameters
    --------------
    sequence1 : str 
        Input sequence 1 on y axis of the matrix (this is the SAFD sequence)
        the sequence that represents the solvent-accessible residues on 
        the surface of the folded domain of interest.

    sequence2 : str 
        Input sequence 2 on x axis of the matrix

    Returns
    ---------------
    np.array 
        returns a 2D mask the same shape of (len(sequence1), len(sequence2))

    """
    charges = ['R','K','E','D']

    matrix = []
    n2 = len(sequence2)
    for i,r1 in enumerate(sequence1):
        tmp = []
        if r1 in charges:
            for j,r2 in enumerate(sequence2):  
                if r2 in charges: 
                    l_resis = r1 + sequence_tools.get_neighbors_window_of3(j,sequence2) #this line is the difference HERE 
                    
                    [local_fcr, local_ncpr] = sequence_tools.calculate_FCR_and_NCPR(l_resis)
                    chrg_weight = np.abs(local_ncpr / local_fcr)
                    
                    tmp.append(chrg_weight)
                else:
                    tmp.append(0)
        else:
            tmp = [0]*n2
            
        matrix.append(tmp)

    return np.array(matrix)



## ---------------------------------------------------------------------------
##  
def get_aliphatic_weighted_mask(sequence1, sequence2):
    """
    Function to get the aliphatic weighted mask of a Matrix. This is am emprical
    approximation to the fact that in an implicit solvent approximation aliphatic/
    hydrophobic residues aren't sticky for one another because they like each other,
    but because water release is entropically favourable. Because water is quantized 
    two leucines next to each other may not create a sufficiently large interface
    to release many water molecules, whereas two clusters of leucines can and do. To
    approximate this phenomenon, we up-weight the attractive interactions between 
    clusters of aliphatic residues for themselves. In this way, clusters of aliphatic
    residues are actually more "hydrophobic" than the same number of individually
    spaced aliphatic residues.
    
    Tentatively - this is probably true for aromatic residues as well, but we plan
    to investigate this more systematically going forward...

    Parameters
    --------------
    sequence1 : str 
        Input sequence 1 on y axis of the matrix (this is the SAFD sequence)
        the sequence that represents the solvent accessable resisues on 
        surface of the folded domain of itrest.

    sequence2 : str 
        Input sequence 2 on x axis of matrix

    Returns
    ---------------
    np.array 
        returns a 2D MATRIX mask the same shape of (len(sequence1), len(sequence2))
   
    """
    multiplier_weighting = np.ones((4, 4), dtype=float)
    multiplier_weighting[2, 2] = 1.5
    multiplier_weighting[2, 3] = 1.5
    multiplier_weighting[3, 2] = 1.5
    multiplier_weighting[3, 3] = 3.0

    ali_mask1 = np.asarray(get_aliphatic_groups(sequence1), dtype=np.int8)
    ali_mask2 = np.asarray(get_aliphatic_groups(sequence2), dtype=np.int8)
    return multiplier_weighting[ali_mask1[:, None], ali_mask2[None, :]]


## ---------------------------------------------------------------------------
##
def get_aliphatic_groups(chain):
    """
    Function to get groups of aliphatic residues based on their 
    local clustering relative to each other. Returns a 1D SEQUENCE 
    mask of the passed sequence.

    Parameters
    --------------
    chain : str 
        sequence that contains aliphatics residues grouped by their 
        nearest neighbors to note local aliphatic surfaces in a chain
    
    Returns
    ---------------
    list 
        1D mask of sequence where aliphatic residues are grouped 
        groups 1, 2, or 3 based on their local clustering. Non-aliphatics 
        in the mask are returned as a 0. 

    """
    
    # get binary mask of aliphatics, so ali_mask is 1 for aliphatics and 0 for non-aliphatics 
    ali_mask = sequence_tools.mask_sequence(chain, ['A','V','I','L','M'])
    
    # count the number of nearest neighbors per aliphatic
    aliphaticgrouping = sequence_tools.MASK_n_closest_nearest_neighbors(ali_mask)
    
    # filter aliphatic grouping in 3 groups 
    aliphaticgrouping = [sub_a if sub_a < 4 else 3 for sub_a in aliphaticgrouping]
    
    return aliphaticgrouping

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
