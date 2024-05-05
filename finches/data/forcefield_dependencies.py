"""
Module to support forcefields passed to the Interaction_Matrix_Constructor

By : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06
"""

from scipy.optimize import root_scalar

from scipy.stats import linregress
from finches import epsilon_stateless
from .reference_sequence_info import DAS_KAPPA_RG_MPIPI
import numpy as np

## ------------------------------------------------------------------------------
##
def get_null_interaction_baseline(X_model, lower_end=-10.0, upper_end=10.0, alternative_sequence=None):
    """
    A function that calibrates a null interaction baseline against a GS repeat,
    which we assume to be correctly described as having an epsilon of 0.

    NOTE: this may not good be a good assumption to make, and for CALVADOS (for example)
    we do have a minor empyrical correction to make from the GS value.

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
        The model-specific baseline used to split the matrix into attractive 
        and repulsive interactions.

    """

    if alternative_sequence is not None:
        seq = alternative_sequence
    else:
        seq='GS'*200

    def f(null_interaction_baseline):        
        return epsilon_stateless.get_sequence_epsilon_value(seq,
                                                            seq,
                                                            null_interaction_baseline=null_interaction_baseline,
                                                            charge_prefactor=None,
                                                            X=X_model,
                                                            use_charge_weighting=False,
                                                            use_aliphatic_weighting=False)

                                                            
    
    # expect null to be between -10 and +10 but this could be adjusted if needed; solve for null_interaction_baseline
    result = root_scalar(f, bracket=[lower_end, upper_end])

    return result.root


# to compute charge prefactor see the prefactor_calibration.ipynb notebook associated with finches
