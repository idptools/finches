"""
Module to support forcefields passed to the Interaction_Matrix_Constructor

By : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06
"""

from scipy.stats import linregress
import finches
import numpy as np

## ------------------------------------------------------------------------------
##
def _GS_length_generator(N, start_AA="G"):
    """
    Takes in a number and generates "GS" string of the certain length
    
    Parameters
    ---------------
    N : int
        int that defines the length of the returned polyGS sequence

    start_AA : str ['G', 'S']
        flag to note wheather to build the sequence starting with Gly or Ser 

    Returns
    ---------------
    GS_string : str
        sequence of polyGS 

    """
    AA_choices = {"G":["G","S"] , "S":["S","G"]}[start_AA]
    seq=[]
    c=0
    while c < N:
        for i in AA_choices:
            seq.append(i)
            c+=1
    #if N is odd 
    if N % 2 != 0:
        seq= seq[:-1]
    
    if len(seq)!= N:
        return 0
        
    return ''.join(seq)

## ------------------------------------------------------------------------------
##
def get_null_interaction_baseline(X_model, min_len=10, max_len=500):
    """
    Function to arrive at null interaction baseline for specific passed model
    This works by testing a bunch of different null_interaction_baselines and 
    computing the sequence_epsilon_value for polyGS of lengths 10-500.

    The null_interaction_baseline is then fit of fits. IE null_interaction_baseline
    which results in a slope of 0 when epsilon vs polyGS(n) is plotted. 
    
    Parameters
    ---------------
    X_model : obj
        An instantiation of the Interaction_Matrix_Constructor class 

    min_len : int 
        The minimum length of a polyGS sequence used 

    max_len : int 
        The minimum length of a polyGS sequence used

    Returns
    ---------------

    null_interaction_baseline : float 
        The model specific baseline used to split the matrix into attrative 
        and repulsive interactions.

    """

    null_baseline_values_dict = {}
    base_lines = np.arange(-0.2,0,.01) # range may need to be updated

    GS_refseq = {i: _GS_length_generator(i, start_AA="G") for i in range(min_len,max_len)}
    SG_refseq = {i: _GS_length_generator(i, start_AA="S") for i in range(min_len,max_len)}

    # iterate possible baselines
    for ibl in base_lines:
       
        # not charge prefactor can be set to 0 here because the null_interaction_baseline
        # generation does not depend on any sequences containing charge residues 

        GS_ref_dict = {i:finches.epsilon_calculation.get_sequence_epsilon_value(a, a, null_interaction_baseline=ibl, prefactor=0, X=X_model, CHARGE=False, ALIPHATICS=False) for i,a in GS_refseq.items()}
        SG_ref_dict = {i:finches.epsilon_calculation.get_sequence_epsilon_value(a, a, null_interaction_baseline=ibl, prefactor=0, X=X_model, CHARGE=False, ALIPHATICS=False) for i,a in SG_refseq.items()}

        GS_avg_reference_dic = {i: (GS_ref_dict[i] + SG_ref_dict[i])/2 for i in SG_refseq}

        # get slope for this specific baseline
        slope_of_fit = linregress(x=list(GS_avg_reference_dic.keys()), y=list(GS_avg_reference_dic.values())).slope

        null_baseline_values_dict[ibl] = slope_of_fit

    # return the theretical baseline (ibl) where the slope of epsilon vs polyGS(n) == 0 
    null_interaction_baseline = linregress(y=list(null_baseline_values_dict.keys()), x=list(null_baseline_values_dict.values())).intercept

    return null_interaction_baseline




######################################################################
##                                                                  ##
##                                                                  ##
##              PRECOMPUTED FORCEFEILD DEPENDNED PARAMS             ##
##                                                                  ##
##                                                                  ##
######################################################################


precomputed_forcefield_dependent_values =  {'charge_prefactor':{'mPiPi_GGv1':0.216145,
                                                                'CALVADOS2':1.442590
                                                                },

                                            'null_interaction_baseline':{'mPiPi_GGv1':-0.128539,
                                                                         'CALVADOS2':-0.047859,
                                                                        },
                                            }