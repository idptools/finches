"""
Module to support forcefields passed to the Interaction_Matrix_Constructor

By : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06
"""

from scipy.stats import linregress
import finches
from .reference_sequence_info import DAS_KAPPA_RG_MPIPI
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

## ------------------------------------------------------------------------------
##
def get_charge_prefactor(X_model, refference_data='DAS_KAPPA_RG_MPIPI', prefactor_range=None):
    """
    Function to arrive at the charge prefactor for weighting the specific passed
    model based on local charged residues. This works by computing the epsilon 
    values for many prefactors and then matching the slope of the epsilon value 
    or radius of gyration (Rg) vs Kappa for a pre computed set charge sequences.

    The charge sequences used are the Das Kappa sequences from the below paper: 

        Das, R.K. & Pappu, R.V. Proc. Natl. Acad. Sci. U. S. A. 110, 13392–13397 (2013).

    The valitity of the this prefactor depends on the trusted corilary 
    relationship between Rg and homotypic epsilon.The prefactor that returned is 
    that which has a slope which a matches the kappa to Rg slope of refference data. 
    
    Parameters
    ---------------
    X_model : obj
        An instantiation of the Interaction_Matrix_Constructor class 

    refference_data : list 
        dataset to be used for computing the charge prefactor, this dataset 
        should be organized as a list of tuples where the tuples contain 
        three values in the order of (sequence, Rg(y-value), Kappa(x-value)) 
        where the slope of the computed epsion vs x-value will be matched to 
        that of Rg(y-value) vs Kappa(x-value). 

        The defult here is DAS_KAPPA_RG_MPIPI stored: 
            finches.data.reference_sequence_info.DAS_KAPPA_RG_MPIPI
    
    prefactor_range : tuple 
        A two position tuple defining the minimum and maximum prefactors 
        to itterate over (min, max). If None, defult is hard coded in as (0,2)

    Returns
    ---------------

    charge_prefactor : float 
        The model specific prefactor used to weight the matrix for local  
        charge patterning this is specifically implimented in the function:
            finches.epsilon_calculation.calculate_weighted_pairwise_matrix()
    """

    charge_prefactor_dict = {}
    if not prefactor_range:
        prefactors = np.arange(0,2,.01) # defult range may need to be updated
    else:
        prefactors = np.arange(prefactor_range[0],prefactor_range[1],.01)

    refseqs = list(a[0] for a in DAS_KAPPA_RG_MPIPI)
    ref_Y = list(a[1] for a in DAS_KAPPA_RG_MPIPI)
    ref_X = list(a[2] for a in DAS_KAPPA_RG_MPIPI)
    ref_slope = linregress(x=ref_X, y=ref_Y).slope

    # check that null_interaction_baseline is set
    if X_model.null_interaction_baseline == None:
        raise Exception(f'NOTE - a null_interaction_baseline must be set or computed prior to computing a charge_prefactor')
    else:
        ibl = X_model.null_interaction_baseline

    # iterate possible prefactors
    for prf in prefactors:

        seq_ref_list = [finches.epsilon_calculation.get_sequence_epsilon_value(s, s, null_interaction_baseline=ibl, prefactor=prf, X=X_model, CHARGE=True, ALIPHATICS=False) for s in refseqs]
        
        # get slope for this specific prefactor
        slope_of_fit = linregress(x=ref_X, y=seq_ref_list).slope

        # because we are looking for the slope that matches that of the 
        # refference slope, here we subtract off the refference slope 
        # from the computed slope 
        charge_prefactor_dict[prf] = slope_of_fit - ref_slope

    # return the theretical prefactor (prf) where the (slope of epsilon vs ref_X) - ref_slope == 0 
    charge_prefactor = linregress(y=list(charge_prefactor_dict.keys()), x=list(charge_prefactor_dict.values())).intercept

    return charge_prefactor



######################################################################
##                                                                  ##
##                                                                  ##
##              PRECOMPUTED FORCEFEILD DEPENDNED PARAMS             ##
##                                                                  ##
##                                                                  ##
######################################################################


precomputed_forcefield_dependent_values =  {'charge_prefactor':{'mPiPi_default': 0.184890,
                                                                'mPiPi_GGv1': 0.216145,
                                                                'CALVADOS2': 1.442590,
                                                                },

                                            'null_interaction_baseline':{'mPiPi_default':-0.066265,
                                                                         'mPiPi_GGv1':-0.128539,
                                                                         'CALVADOS2':-0.047859,
                                                                        },
                                            }