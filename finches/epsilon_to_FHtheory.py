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

from .epsilon_calculation import get_weighted_sequence_epsilon_value
from .analytical_fh import floryhuggins

## ---------------------------------------------------------------------------
##
def build_condition_dependent_phase_diagrams(seq, X_class, prefactor, null_interaction_baseline, condition_list):
    """
    Fuction that iterativly calls return_phase_diagram base on a list 
    of passed conditions that re-intialize the input parameter_model

    Parameters
    ---------------
    seq : str
        Input sequence

    X_model : object
        instantiation of the Interaction_Matrix_Constructor object for which gets re-initialized for
        each condition in the condition dict

    prefactor : object
        instantiation

    null_interaction_baseline : object
        instantiation

    condition_list : list
        list of [condition_parameter , conditions_list] where:
            condition_parameter = input parameter of the X_model
            conditions_list = list of values (v) passed one at 
                                a time to X_model.parameters.{condition_parameter} = v

    Returns
    ---------------
    list
    
        The function returns a list with the following elements
        
        [0] - List of concentrations
        [1] - List of outputs from return_phase_diagram
        [2] - List epsilons for each concentrations
    """
    """
    SUDO testing functionalitly here 


    mPiPi_model = mPiPi.mPiPi_model('mPiPi_GGv1', salt=0.1)
    XG_mPiPi = epsilon_calculation.Interaction_Matrix_Constructor(mPiPi_model)
    def test(X_mPiPi, condition_list):
        
        l_param = X_mPiPi.parameters 
        
        print(X_mPiPi.lookup["R"]["E"], X_mPiPi.parameters.salt)
        l_con = condition_list[1]
        base_value = X_mPiPi.parameters.salt
        for i in condition_list[1]:
            # print(l_param.salt)
            l_param.salt = i
            X_mPiPi._update_lookup_dict()
            # print(l_param.salt)
            
            ### you would call return_phase_diagram here
            return_phase_diagram(seq, X_mPiPi)

        print(X_mPiPi.lookup["R"]["E"], X_mPiPi.parameters.salt)
        
        #reset_parameters to base value 
        l_param.salt = base_value
        X_mPiPi._update_parameters(l_param)
        print(X_mPiPi.lookup["R"]["E"], X_mPiPi.parameters.salt)
        
    condition_list = ['salt',[0.1,0.2,0.3,0.4]]
    print(XG_mPiPi.lookup["R"]["E"], XG_mPiPi.parameters.salt)

    test(XG_mPiPi, condition_list)

    print(XG_mPiPi.lookup["R"]["E"], XG_mPiPi.parameters.salt)

    """


    pass  


## ---------------------------------------------------------------------------
##
def return_phase_diagram(seq, X_class, prefactor, null_interaction_baseline):
    """
    Wrapper function that takes in a sequence and builds a phase diagram using the Analytical approximation 
    of FH theory to build a phase diagram. This works by computing a phase diagram in terms of chi vs. T, and then
    recast chi in terms of T given that.
    
    chi = delta_eps / (kB * T)
    
    where delta_eps here is the site-to-site contact energy, i.e. a larger (positive) value means more attractive. This 
    
    if kB=1 the delta_eps is in terms of kB, therefore
    
    chi = delta_eps / (T)
    
    and therefore
    
    T = delta_eps/chi
    
    The code below does a few things:
    
    1. Using the get_weighted_sequence_epsilon_value() to calculate the mean-field interaction parameter (epsilon)
    
    2. Using sequence length calculates the phi vs. chi phase diagram 
    
    3. Converts epsilon so a positive (repulsive) epsilon becomes mildly attractive but not enough to 
       cause phase separation. We do this as a hack otherwise the math breaks down...
       
    4. Converts the chi array into T by converting in terms of delta_eps * 1/chi
    
    
    This approach is repeated to also calculate the spinodal, although we don't plot that here
    
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
    
    
    # calculate the 
    eps_real = get_weighted_sequence_epsilon_value(seq, seq, X_class, prefactor=prefactor,null_interaction_baseline=null_interaction_baseline)
    
    # if we have a positive (i.e. repulsive) epsilon set this to -0.01, which means super 
    # weakly attractive. We do this because otherwise a positive epsilon value leads to a
    # negative kB_eps which means we get 'inverse' phase diagrams, because we're converting chi
    # to T in a universe where delta-epsilon is a measure of site-specific interaction that 
    # CANNOT be repulsive (worset case 0). We have to set it to a small attractive value because if 
    # its 0 you get a divide by 
    if eps_real > 0:
        eps_real = -0.01
    
    # note we have to 
    # 1. reverse the sign (i.e. negative eps = positive flory chi)
    # 2. Divide eps_real by seq(len) because delta_eps is a site-specific energy whereas 
    #    eps_real is in terms of the overall chain, but the impact of length is already taken
    #    into account in the calculate_binodal, so we need to divide epsilon by length to get the
    #    average per-residue contribution.
    #
    delta_eps = -eps_real/len(seq) 
    
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
    