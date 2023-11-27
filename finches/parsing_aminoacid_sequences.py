"""
Holehouse Lab - Internal Script

This script has code to parse amino acid sequences and assign groups to the 
aliphatics with the proper charecter input needed in PIMMS 
written for Ryan's development of AA params in PIMMS.

Attapted from original implimentation in LAMMPS - Mpipi
see origial at:
line 149 of lammpstools/lammpstools/data/configuration_v4/build_data_file.py 

by: Garrett M. Ginell 

updated: 2022-07-22
"""
import numpy as np

# NB from Alex; underscored functions should not be exported from a model so should rename _get_neighbors_3
# _get_neighbors_3 replaced with get_neighbors_window_of3
from .sequence_tools import MASK_n_closest_nearest_neighbors, mask_sequence, get_neighbors_window_of3, calculate_FCR_and_NCPR

# new charecters for PIMMS aliphatic groups 
aliphatic_group1 = {'A':'a', 'L':'l', 'M':'m','I':'i','V':'v'}
aliphatic_group2 = {'A':'b', 'L':'o', 'M':'x', 'I':'y','V':'z'}

## ---------------------------------------------------------------------------
##
def get_charge_weighted_mask(sequence1, sequence2):
    """
    Function to get the charge weighted mask of the inter-residue interaction
    matrix.

    Specifically, this function loops over all cross-interacting residues from
    the two sequences (i.e. every pair of r1-r2 (where r1 is from seq1 and r2 
    is from seq1) and if BOTH residues are charged then a 'charge weight' is
    calculated whereby the +/- 1 residues around that two residues are extracted
    and the |NCPR|/FCR of the resulting concatenated sequence is a weighting factor.

    What does this mean, practically?

    If I have 2 fragements that are KKK and EEE then my charge weighting will be

    |NCPR/FCR| = 0/1 = 0.0

    If I have 2 fragments that are EEE and EEE then my charge weighting will be
    
    |NCPR/FCR| = |-1/1| = 1.0

    In this way, clusters of like-charged residues are weighted up

    NOTE ADD MORE DETAIL HERE 

    depends on: sequence_tools.get_neighbors_window_of3
    
    Parameters
    --------------
    sequence1 : str 
        Input sequence 1 on y axis of matrix 

    sequence2 : str 
        Input sequence 2 on x axis of matrix

    Returns
    ---------------
    np.array 
        returns a 2D mask the same shape of (len(sequence1), len(sequence2)) 
        where at intersections of charged residues between the two sequences
        we get a charge weighting factor calculated as |NCPR|/FCR 

        
    """
    charges = ['R','K','E','D']

    matrix = []
    n2 = len(sequence2)
    for i,r1 in enumerate(sequence1):
        tmp = []
        if r1 in charges:
            for j,r2 in enumerate(sequence2):  
                if r2 in charges: 

                    # this generates a string of max 6 residues (for terminal residues 5 or 4 residues)
                    # which is basically a concatenated fragment 
                    l_resis = get_neighbors_window_of3(i,sequence1) + get_neighbors_window_of3(j,sequence2)

                    # for that fragment, calculate the local fcr and ncpr
                    [local_fcr, local_ncpr] = calculate_FCR_and_NCPR(l_resis)

                    # calculate the charge weight as |NCPR/FRC|
                    chrg_weight = np.abs(local_ncpr / local_fcr)

                    tmp.append(chrg_weight)
                else:
                    tmp.append(0)
        else:

            # if r1 was not charged create an empty vector
            tmp = [0]*n2
            
        matrix.append(tmp)

    return np.array(matrix)

## ---------------------------------------------------------------------------
##
def get_charge_weighted_FD_mask(sequence1, sequence2):
    """
    Function to get the charge weighted mask of a Matrix EXCEPT 
    that residues in sequence1 are treated in issolation. 

    depends on: sequence_tools.get_neighbors_window_of3
                 
    Here sequence1 is the the SAFD sequence so weighting is computed 
    between the 1FD residue and the normal 3 residues in the IDR

    Parameters
    --------------
    sequence1 : str 
        Input sequence 1 on y axis of matrix (this is the SAFD sequence)
        the sequence that represents the solvent accessable resisues on 
        surface of the folded domain of itrest.

    sequence2 : str 
        Input sequence 2 on x axis of matrix

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
                    l_resis = r1 + get_neighbors_window_of3(j,sequence2) #this line is the difference HERE 
                    
                    [local_fcr, local_ncpr] = calculate_FCR_and_NCPR(l_resis)
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
    Function to get the aliphatic weighted mask of a Matrix. This 
    is done to account for the interactions between local aliphaic surfaces 
    likly to have stronger interactions. Functionaly this builds a mask to 
    which strenthens interactions between clusters of aliphatic residues.

    Parameters
    --------------
    sequence1 : str 
        Input sequence 1 on y axis of matrix (this is the SAFD sequence)
        the sequence that represents the solvent accessable resisues on 
        surface of the folded domain of itrest.

    sequence2 : str 
        Input sequence 2 on x axis of matrix

    Returns
    ---------------
    np.array 
        returns a 2D MATRIX mask the same shape of (len(sequence1), len(sequence2))
   
    """
    multiplier_weighting = {'1_1':1, '1_2':1, '1_3':1, '2_1':1, '3_1':1,
                            '2_2':1.5, '2_3':1.5, '3_2':1.5, 
                            '3_3':3}

    ali_mask1 = get_aliphatic_groups(sequence1)
    ali_mask2 = get_aliphatic_groups(sequence2)
    n2 = len(sequence2)
    matrix = []
    for i,v1 in enumerate(ali_mask1):
        tmp = [] 
        if v1 > 0:
            for j,v2 in enumerate(ali_mask2):  
                if v2 > 0: 
                    tmp.append(multiplier_weighting[f'{v1}_{v2}'])
                else:
                    tmp.append(1)
        else:
            tmp = [1]*n2
            
        matrix.append(tmp)

    return np.array(matrix) 


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
         sequence which contains aliphatics residues 
         grouped by there nearest neighbors to note local aliphaic surfaces 
         in a chain
    
    Returns
    ---------------
    list 
        1D mask of sequence where aliphatic residues are grouped 
        groups 1, 2, or 3 base on their local clustering. Non-alphatics 
        in the mask are returned as a 0. 

    """
    
    # get bionary mask of aliphatics 
    ali_mask = mask_sequence(chain, ['A','V','I','L','M'])
    
    # count number of nearest neighbors per aliphatic
    aliphaticgrouping = MASK_n_closest_nearest_neighbors(ali_mask)
    
    # filter aliphaticgrouping in 3 groups 
    aliphaticgrouping = [sub_a if sub_a < 4 else 3 for sub_a in aliphaticgrouping]
    
    return aliphaticgrouping

## ---------------------------------------------------------------------------
##
def get_aliphaticgroup_sequence(chain):
    """
    Function to get sequence that is re-assigned aliphatics with 
    beads for the grouped aliphatics in used in grouping of aliphatics. 
    The bead assignments are those that are used in the mPiPi forcfeild.
    
    Takes a passed sequence and get PIMMS ready sequence of aliphatics 
        aliphatic_group1 = {'A':'A', 'L':'L', 'M':'M','I':'I','V':'V'}
        aliphatic_group2 = {'A':'a', 'L':'l', 'M':'m','I':'i','V':'v'}
        aliphatic_group3 = {'A':'b', 'L':'o', 'M':'x', 'I':'y','V':'z'}

    Parameters
    --------------
    chain : str 
         sequence which contains aliphatics residues 
         grouped by there nearest neighbors to note local aliphaic surfaces 
         in a chain
    
    Returns
    ---------------
    str 
        sequence where aliphatic residues are grouped and re-assigned 
        symbols based on the bead assignments used in the mPiPi forcfeild.

    """
    
    # get aliphatic groups by neirest neighbors 
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
