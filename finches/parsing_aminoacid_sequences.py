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

# NB from Alex; underscored functions should not be exported from a model so should rename _get_neighboors_3
from .sequence_tools import MASK_n_closest_nearest_neighbors, mask_sequence, _get_neighboors_3, calculate_FCR_and_NCPR


# new charecters for PIMMS aliphatic groups 
aliphatic_group1 = {'A':'a', 'L':'l', 'M':'m','I':'i','V':'v'}
aliphatic_group2 = {'A':'b', 'L':'o', 'M':'x', 'I':'y','V':'z'}

## ---------------------------------------------------------------------------
##
def get_charge_weighed_mask(sequence1, sequence2):
    """
    depends on: sequence_tools._get_neighboors_3
                sparrow.Protein 
    """
    charges = ['R','K','E','D']

    matrix = []
    n2 = len(sequence2)
    for i,r1 in enumerate(sequence1):
        tmp = []
        if r1 in charges:
            for j,r2 in enumerate(sequence2):  
                if r2 in charges:
                    
                    l_resis = _get_neighboors_3(i,sequence1) + _get_neighboors_3(j,sequence2)
                    
                    # old way using Sparrow
                    #chrg_weight = np.abs(Protein(l_resis).NCPR / Protein(l_resis).FCR)

                    # new way that avoids Sparrow; note this has not been tested because finches
                    # doesn't currently work...
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
def get_aliphatic_groups(chain):
    """
    pass sequence and get mask of aliphatics 
    grouped by there nearest neighbors 
    
    """
    
    # get bionary mask of aliphatics 
    ali_mask = mask_sequence(chain, ['A','V','I','L','M'])
    
    # count number of nearest neighboors per aliphatic
    aliphaticgrouping = MASK_n_closest_nearest_neighbors(ali_mask)
    
    # filter aliphaticgrouping in 3 groups 
    aliphaticgrouping = [sub_a if sub_a < 4 else 3 for sub_a in aliphaticgrouping]
    
    return aliphaticgrouping

## ---------------------------------------------------------------------------
##
def get_aliphaticgroup_sequence(chain):
    """
    pass sequence and get PIMMS ready sequence of aliphatics 
    grouped by there nearest neighbors with properly assigned 
    grouped resis.
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
