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
from .sequence_tools import MASK_n_closest_nearest_neighbors, mask_sequence


# new charecters for PIMMS aliphatic groups 
aliphatic_group1 = {'A':'a', 'L':'l', 'M':'m','I':'i','V':'v'}
aliphatic_group2 = {'A':'b', 'L':'o', 'M':'x', 'I':'y','V':'z'}


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