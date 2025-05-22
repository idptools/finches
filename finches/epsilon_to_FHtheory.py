"""
Module to build FH-based phase dygrams from computed epsilon

Flory-Huggins phase diagrams

This Module provides direct examples of using the binodal calculating code from:

Qian, D., Michaels, T. C. T., & Knowles, T. P. J. (2022). Analytical Solution to the Flory-Huggins Model. 
    Journal of Physical Chemistry Letters, 13(33), 7853–7860.

In the context of generating phase diagrams. We have integrated the code for computing FH phase diagrams into finches.
Specifically, there's an easy-to=use wrapper around the code released with the paper. 

The actual code can be found in finches/analytical_fh. In here backend.py is code taken from the original GitHub repo:

 https://github.com/KnowlesLab-Cambridge/FloryHugginsfloryhuggins.py 

This module is a wrapper that takes the analytical_fh method and builds a phase diagram using the Analytical approximation 
    of FH theory to build a phase diagram and our computed mean-field interaction parameter (epsilon).

By : Garrett M. Ginell & Alex S. Holehouse 
2023-08-18
"""

import numpy as np

from finches.epsilon_stateless import get_sequence_epsilon_value
from finches.analytical_fh import floryhuggins


## ---------------------------------------------------------------------------
##
def build_DIELECTRIC_dependent_phase_diagrams(seq,
                                              X_class,
                                              condition_list,
                                              prefactor=None,
                                              null_interaction_baseline=None):
    """
    Fuction that iterativly calls return_phase_diagram base on a list 
    of passed conditions for dielectric that re-intialize the input parameter_model

    Parameters
    ---------------
    seq : str
        Input sequence

    X_model : object
        instantiation of the InteractionMatrixConstructor object for which gets
         re-initialized for each condition in the condition dict

    prefactor : object
        instantiation

    null_interaction_baseline : object
        instantiation

    condition_list : list
        list of conditions values where:
        conditions_list = list of values (v) passed one at 
                            a time to X_model.parameters.salt = v

    Returns
    ---------------
    out_list : list 
    
        The function returns a list with the following elements
        
        [0] - List of concentrations
        [1] - Dictionary of outputs from return_phase_diagram with keys as concentrations
        [2] - Dictionary of outputs from epsilons for each concentrations keys as concentrations
    """

    out_diagrams = {}
    out_epsilons = {}

    # check if condition_parameter exists and if so set base condition_parameter 
    try:
        base_value = X_class.parameters.dielectric 
    except Exception as E:
        print(E)
        raise Exception(f'''Condition "dielectric" NOT FOUND - Input parameter model {X_class.parameters.version} of passed X_model does 
                            not contain parameters.dielectric as viable condition. Properties of the passed parameter model are below:
                            {vars(X_class.parameters).keys()}''')

    base_params = X_class.parameters

    # itterate conditions to get epsilon values at different conditions
    for i in condition_list:

        # get local parameters 
        l_param = X_class.parameters
        l_param.dielectric = i

        # update constructor with for new parameters
        X_class._update_lookup_dict()

        # get phase diagram for condition
        out_diagrams[i] = return_phase_diagram(seq, X_class)
        out_epsilons[i] = get_sequence_epsilon_value(seq, seq, X_class, prefactor=prefactor, 
                                                        null_interaction_baseline=null_interaction_baseline, 
                                                        use_charge_weighting=True, use_aliphatic_weighting=True)

    # reset parameters to base_value 
    X_class.parameters.salt = base_value

    # update base parameters 
    X_class._update_parameters(base_params)

    return [condition_list, out_diagrams, out_epsilons]

## ---------------------------------------------------------------------------
##
def build_PH_dependent_phase_diagrams(seq,
                                      X_class,
                                      condition_list,
                                      prefactor=None,
                                      null_interaction_baseline=None):
    """
    Fuction that iterativly calls return_phase_diagram base on a list 
    of passed conditions for ph that re-intialize the input parameter_model

    Parameters
    ---------------
    seq : str
        Input sequence

    X_model : object
        instantiation of the InteractionMatrixConstructor object for which gets
         re-initialized for each condition in the condition dict

    prefactor : object
        instantiation

    null_interaction_baseline : object
        instantiation

    condition_list : list
        list of conditions values where:
        conditions_list = list of values (v) passed one at 
                            a time to X_model.parameters.salt = v

    Returns
    ---------------
    out_list : list 
    
        The function returns a list with the following elements
        
        [0] - List of concentrations
        [1] - Dictionary of outputs from return_phase_diagram with keys as concentrations
        [2] - Dictionary of outputs from epsilons for each concentrations keys as concentrations
    """

    out_diagrams = {}
    out_epsilons = {}

    # check if condition_parameter exists and if so set base condition_parameter 
    try:
        base_value = X_class.parameters.pH 
    except Exception as E:
        print(E)
        raise Exception(f'''Condition "pH" NOT FOUND - Input parameter model {X_class.parameters.version} of passed X_model does 
                            not contain parameters.pH as viable condition. Properties of the passed parameter model are below:
                            {vars(X_class.parameters).keys()}''')

    base_params = X_class.parameters

    # itterate conditions to get epsilon values at different conditions
    for i in condition_list:

        # get local parameters 
        l_param = X_class.parameters
        l_param.pH = i

        # update constructor with for new parameters
        X_class._update_lookup_dict()

        # get phase diagram for condition
        out_diagrams[i] = return_phase_diagram(seq, X_class)
        out_epsilons[i] = get_sequence_epsilon_value(seq, seq, X_class, prefactor=prefactor, 
                                                        null_interaction_baseline=null_interaction_baseline, 
                                                        use_charge_weighting=True, use_aliphatic_weighting=True)

    # reset parameters to base_value 
    X_class.parameters.salt = base_value

    # update base parameters 
    X_class._update_parameters(base_params)

    return [condition_list, out_diagrams, out_epsilons]


## ---------------------------------------------------------------------------
##
def build_SALT_dependent_phase_diagrams(seq,
                                        X_class,
                                        condition_list):
    """
    Fuction that iterativly calls return_phase_diagram base on a list 
    of passed conditions for salt that re-intialize the input parameter_model

    Parameters
    ---------------
    seq : str
        Input sequence

    X_model : object
        instantiation of the InteractionMatrixConstructor object for which gets
         re-initialized for each condition in the condition dict

    prefactor : object
        instantiation

    null_interaction_baseline : object
        instantiation

    condition_list : list
        list of conditions values where:
        conditions_list = list of values (v) passed one at 
                            a time to X_model.parameters.salt = v

    Returns
    ---------------
    out_list : list 
    
        The function returns a list with the following elements
        
        [0] - List of concentrations
        [1] - Dictionary of outputs from return_phase_diagram with keys as concentrations
        [2] - Dictionary of outputs from epsilons for each concentrations keys as concentrations
    """

    out_diagrams = {}
    out_epsilons = {}

    # check if condition_parameter exists and if so set base condition_parameter 
    try:
        base_value = X_class.parameters.salt 
    except Exception as E:
        print(E)
        raise Exception(f'''Condition "salt" NOT FOUND - Input parameter model {X_class.parameters.version} of passed X_model does 
                            not contain parameters.salt as viable condition. Properties of the passed parameter model are below:
                            {vars(X_class.parameters).keys()}''')
        

    null_interaction_baseline = X_class.null_interaction_baseline
    charge_preactor = X_class.charge_prefactor
    

    # itterate conditions to get epsilon values at different conditions
    for i in condition_list:

        # set local parameters salt
        X_class.parameters.salt = i

        # update constructor with for new parameters
        X_class._update_lookup_dict()

        # get phase diagram for condition
        out_diagrams[i] = return_phase_diagram(seq, X_class)
        out_epsilons[i] = get_sequence_epsilon_value(seq, seq, X_class, charge_prefactor=X_class.charge_prefactor, 
                                                        null_interaction_baseline=null_interaction_baseline, 
                                                        use_charge_weighting=True, use_aliphatic_weighting=True)

    # reset parameters to base_value 
    X_class.parameters.salt = base_value

    # update base parameters
    X_class._update_lookup_dict()

    return [condition_list, out_diagrams, out_epsilons]


## ---------------------------------------------------------------------------
##
def return_phase_diagram(seq, X_class):
                         
    """
    Wrapper function that takes in a sequence and builds a phase diagram using the 
    Analytical approximation of FH theory to build a phase diagram. This works by 
    computing a phase diagram in terms of chi vs. T, and then converting this to 
    chi vs. phi.

    We can recast chi in terms of T given that.
    
        chi = delta_eps / (kB * T)
    
    where delta_eps here is the site-to-site contact energy, i.e. a larger (positive) 
    value means more attractive. 
       
    If kB=1 the delta_eps is in terms of kB, therefore
    
        chi = delta_eps / (T)
    
    and therefore
    
        T = delta_eps/chi
    
    The code below does a few things:
    
    1. Uses get_sequence_epsilon_value() to calculate the mean-field 
       interaction parameter (epsilon)
    
    2. Using sequence length calculates the phi vs. chi phase diagram 
    
    3. Converts epsilon so a positive (repulsive) epsilon becomes mildly 
       attractive but not enough to cause phase separation. We do this 
       as a hack otherwise the math breaks down...
              
    4. Converts the chi array into T by converting in terms of 
       delta_eps * 1/chi
        
    This approach is repeated to also calculate the spinodal, which is also 
    then returned. 
    
    Parameters
    ---------------
    seq : str
        Input sequence
        
    Returns
    ---------------
    list    
        The function returns a list with the following elements
        
        [0] - Dilute phase concentrations (array of len=N) in Phi
        [1] - Dense phase concentrations (array of len=N) in Phi
        [2] - List with [0]: critical T and [1]: Critical phi
        [3] - List of temperatures that match with the dense and dilute phase concentrations
        [4] - Dilute phase concentrations (array of len=N) in Phi for spinodal
        [5] - Dense phase concentrations (array of len=N) in Phi for spinodal
        [6] - List with [0]: critical T and [1]: Critical phi  for spinodal
        [7] - List of temperatures that match with the dense and dilute phase concentrations for spinodal

    """

    charge_prefactor = X_class.charge_prefactor
    null_interaction_baseline = X_class.null_interaction_baseline
    
    
    # calculate the epilson value for the sequence
    eps_real = get_sequence_epsilon_value(seq,
                                          seq,
                                          X_class,
                                          charge_prefactor=charge_prefactor,
                                          null_interaction_baseline=null_interaction_baseline, 
                                          use_charge_weighting=True,
                                          use_aliphatic_weighting=True)

    # get phase diagram
    return epsilon_to_phase_diagram(seq, eps_real)
    
    

## ---------------------------------------------------------------------------
##
def epsilon_to_phase_diagram(seq, epsilon):
    """
    Function that uses input sequence (only for length) and epsilon value to
    calculate the phase diagram.

    Specifically this does the following:

    1. Using sequence length calculates the phi vs. chi phase diagram 
    
    2. Converts epsilon so a positive (repulsive) epsilon becomes mildly 
       attractive but not enough to cause phase separation. We do this 
       as a hack otherwise the math breaks down...
              
    3. Converts the chi array into T by converting in terms of 
       delta_eps * 1/chi
        
    This approach is repeated to also calculate the spinodal, which is also 
    then returned. 
    
    Parameters
    ---------------
    seq : str
        Input sequence
        
    Returns
    ---------------
    list    
        The function returns a list with the following elements
        
        [0] - Dilute phase concentrations (array of len=N) in Phi
        [1] - Dense phase concentrations (array of len=N) in Phi
        [2] - List with [0]: critical T and [1]: Critical phi
        [3] - List of temperatures that match with the dense and dilute phase concentrations
        [4] - Dilute phase concentrations (array of len=N) in Phi for spinodal
        [5] - Dense phase concentrations (array of len=N) in Phi for spinodal
        [6] - List with [0]: critical T and [1]: Critical phi  for spinodal
        [7] - List of temperatures that match with the dense and dilute phase concentrations for spinodal

    """

    
    # if we have a positive (i.e. repulsive) epsilon set this to -0.01, which means super 
    # weakly attractive. We do this because otherwise a positive epsilon value leads to a
    # negative kB_eps which means we get 'inverse' phase diagrams, because we're converting chi
    # to T in a universe where delta-epsilon is a measure of site-specific interaction that 
    # CANNOT be repulsive (worset case 0). We have to set it to a small attractive value because if 
    # its 0 you get a divide by 0 error
    if epsilon > 0:
        epsilon = -0.01
    
    # note we have to 
    # 1. reverse the sign (i.e. negative eps = positive flory chi)
    # 2. Divide epsilon by seq(len) because delta_eps is a site-specific energy whereas 
    #    epsilon is in terms of the overall chain, but the impact of length is already taken
    #    into account in the calculate_binodal, so we need to divide epsilon by length to get the
    #    average per-residue contribution.
    #
    delta_eps = -epsilon/len(seq) 
    
    # use standard binodal from the floryhuggins module 
    out = floryhuggins.calculate_binodal(len(seq),'analytic_binodal',n_points=50000, chi_max=0.8)
    # out here has the following elements:
    # [0] - chi values
    # [1] - dilute concentration
    # [2] - dense concentration
    # [3] - critical volume fraction
    # [4] - critical chi value

    # convert from chi to T in kelvin by 
    # 1. taking 1/chi and multiplying that by delta_eps - i.e. T = delta_eps/chi
    Ts_in_Kelvin = delta_eps*(1/np.array(out[0]))

    # get the critical temperature using same conversion factor
    crit_T = (1/out[4])*delta_eps 
    
    # get dense and dilute concentrations and critical concentration
    dilute = out[1]
    dense = out[2]
    crit_phi = out[3]

    ## Do it all again for the spinodal
    out_s = floryhuggins.calculate_spinodal(len(seq),n_points=10000, chi_max=0.8)

    S_crit_T = (1/out_s[4])*delta_eps
    S_Ts_in_Kelvin = delta_eps*(1/np.array(out_s[0]))

    S_dilute = out_s[1]
    S_dense = out_s[2]
    S_crit_phi = out_s[3]

            
    return [dilute, dense, [crit_phi, crit_T], Ts_in_Kelvin, S_dilute, S_dense, [S_crit_phi, S_crit_T], S_Ts_in_Kelvin]
