"""
Module to support forcefields passed to the Interaction_Matrix_Constructor

By : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06
"""

from scipy.optimize import root_scalar

from scipy.stats import linregress
import finches
from .reference_sequence_info import DAS_KAPPA_RG_MPIPI
import numpy as np

## ------------------------------------------------------------------------------
##
def get_null_interaction_baseline(X_model, lower_end=-10.0, upper_end=10.0, alternative_sequence=None):
    """
    Function that calibrates a null interaction baseline against a GS repeat,
    which we assume to be correctly described as having an epsilon of 0.

    NOTE this may not good be assumption to make, and for CALVADOS (for example)
    we do actually have a small empyrical correction to make from the GS value.

    Parameters
    ---------------
    X_model : obj
        An instantiation of the Interaction_Matrix_Constructor class

    lower_end : float
        The lower end of the bracket to search for the null_interaction_baseline

    upper_end : float
        The upper end of the bracket to search for the null_interaction_baseline

    alternative_sequence : str
        An alternative sequence to use for the calibration. If None, the default
        400-residue GS sequence is used.

    Returns
    ---------------

    null_interaction_baseline : float 
        The model specific baseline used to split the matrix into attrative 
        and repulsive interactions.

    """

    if alternative_sequence is not None:
        seq = alternative_sequence
    else:
        seq='GS'*200

    def f(null_interaction_baseline):        
        return finches.epsilon_calculation.get_sequence_epsilon_value(seq,
                                                                      seq,
                                                                      null_interaction_baseline=null_interaction_baseline,
                                                                      charge_prefactor=None,
                                                                      X=X_model,
                                                                      use_charge_weighting=False,                                                                      
                                                                      use_aliphatic_weighting=False)
    
    # expect null to be between -10 and +10 but this could be adjusted if needed; solve for null_interaction_baseline
    result = root_scalar(f, bracket=[lower_end, upper_end])

    return result.root

## ------------------------------------------------------------------------------
##
def get_charge_prefactor(X_model, reference_data='DAS_KAPPA_RG_MPIPI', prefactor_range=None):
    """
    Function to arrive at the charge prefactor for weighting the specific passed
    model based on local charged residues. This works by computing the epsilon 
    values for many prefactors and then matching the slope of the epsilon value 
    or radius of gyration (Rg) vs Kappa for a pre computed set charge sequences.

    The charge sequences used are the Das Kappa sequences from the below paper: 

        Das, R.K. & Pappu, R.V. Proc. Natl. Acad. Sci. U. S. A. 110, 13392–13397 (2013).

    The valitity of the this prefactor depends on the trusted corilary 
    relationship between Rg and homotypic epsilon.The prefactor that returned is 
    that which has a slope which a matches the kappa to Rg slope of reference data. 
    
    Parameters
    ---------------
    X_model : obj
        An instantiation of the Interaction_Matrix_Constructor class 

    reference_data : list 
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

        seq_ref_list = [finches.epsilon_calculation.get_sequence_epsilon_value(s, s, null_interaction_baseline=ibl, prefactor=prf, X=X_model, use_charge_weighting=True, use_aliphatic_weighting=False) for s in refseqs]
        
        # get slope for this specific prefactor
        slope_of_fit = linregress(x=ref_X, y=seq_ref_list).slope

        # because we are looking for the slope that matches that of the 
        # reference slope, here we subtract off the reference slope 
        # from the computed slope 
        charge_prefactor_dict[prf] = slope_of_fit - ref_slope

    # return the theretical prefactor (prf) where the (slope of epsilon vs ref_X) - ref_slope == 0 
    charge_prefactor = linregress(y=list(charge_prefactor_dict.keys()), x=list(charge_prefactor_dict.values())).intercept

    return charge_prefactor
