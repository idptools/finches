"""
Holehouse Lab - Internal Script

This script has code to update pairwise interaction strengths for 
mPiPi force field tooning written for development of AA params in mPiPi.




updated: 2024-01-13
"""


from . import defined_RNA_params
### defined functions to modify parameters ###

## -----------------------------------------------------------
#
#
def STRENGTHEN_small_polar_EPSILON(EPSILON_ALL):
    """
    Function to strengtehn G:S and G:G interaction strengths by 119%.
    i.e. +119% from the original value.

    Parameters
    ----------
    EPSILON_ALL : dict
        Dictionary of all pairwise epsilon values where keys are AA and values are 
        dictionaries of AA:epsilon, e.g. EPSILON_ALL['A']['A'] = 0.5
    

    Returns
    -------
    EPSILON_ALL : dict
        Updated dictionary of all pairwise epsilon values
    
    """

    # 119%
    local_scalar = 1.19

    # Increasing G:G and G:S epsilon to +119%
    for i in ['G']:
        for j in ['G','S']:

            # i.e. G-G
            if i == j:
                EPSILON_ALL[i][j] = EPSILON_ALL[i][j] + EPSILON_ALL[i][j]*local_scalar
                
            # i.e. G-S
            else:
                EPSILON_ALL[i][j] = EPSILON_ALL[i][j] + EPSILON_ALL[i][j]*local_scalar
                EPSILON_ALL[j][i] = EPSILON_ALL[j][i] + EPSILON_ALL[j][i]*local_scalar

                
    return EPSILON_ALL 


## -----------------------------------------------------------
#
#
def WEAKEN_Aromatic_Charge_EPSILON(EPSILON_ALL):

    """
    Function to weaken Aromatic:Charge interactions - i.e. YFW:RED by 60%
    i.e. -60% of the original value for all except lysine, which was already
    sufficiently weak.

    Parameters
    ----------
    EPSILON_ALL : dict
        Dictionary of all pairwise epsilon values where keys are AA and values are
        dictionaries of AA:epsilon, e.g. EPSILON_ALL['A']['A'] = 0.5

    Returns
    -------
    EPSILON_ALL : dict
        Updated dictionary of all pairwise epsilon values.

    """

    # 60%
    local_scalar = 0.60

    
    for i in ['Y','F','W']:
        for j in ['R','E','D']:
            EPSILON_ALL[i][j] = EPSILON_ALL[i][j] - EPSILON_ALL[i][j]*local_scalar
            EPSILON_ALL[j][i] = EPSILON_ALL[j][i] - EPSILON_ALL[j][i]*local_scalar

                                                     
    return EPSILON_ALL 


## -----------------------------------------------------------
#
#
def ENLARGE_Proline_SIGMA(SIGMA_ALL):
    """
    Function to enlarge Proline:X interaction by increasing sigma by 33%.

    Parameters
    ----------
    SIGMA_ALL : dict
        Dictionary of all pairwise sigma values where keys are AA and values are
        dictionaries of AA:sigma, e.g. SIGMA_ALL['A']['A'] = 0.5

    Returns
    -------
    SIGMA_ALL : dict
        Updated dictionary of all pairwise sigma values.

    """

    # 33%
    local_scalar = 0.33
                                                     
    # Tuning P:X sigma +33% for all amino acids except proline. Note previously
    # we hardcoded in the 20 AAs, but now we default to everything so this fix
    # is applied universally.
    for i in SIGMA_ALL.keys():

        if i != 'P':            
            SIGMA_ALL['P'][i] = SIGMA_ALL['P'][i] + SIGMA_ALL['P'][i] * local_scalar
            SIGMA_ALL[i]['P'] = SIGMA_ALL[i]['P'] + SIGMA_ALL[i]['P'] * local_scalar
                                                     
    # tuning P:P sigma +33%
    SIGMA_ALL['P']['P'] = SIGMA_ALL['P']['P'] + SIGMA_ALL['P']['P'] * local_scalar
                                                     
    return SIGMA_ALL


## -----------------------------------------------------------
#
#
def CORRECT_aliphatic_aliphatic_ALL(CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL):
    """
    Function to correct aliphatic:aliphatic interactions in mPiPi force field.

    This function ONLY updates the EPISLON values for aliphatic:aliphatic 
    interactions.


    # CORRECT aliphatic:aliphatic interation 
    # ILVAM:ILVAM epsilon rebuild 
    # epsilons are now relative Kyte-doolittle Hydrophobisity scale
    # NEW_epsilon = 0.008X + 0.018 ; where X = sum(KDhydro_A1, KDhydro_A2) ;
    #         where A1 and A2 are the aliphatic residues in the pairwise interaction
    """

    # Fit from original aliphatic epsilons vs. sum of hydrophobicity scale values for
    # each pair of aliphatic residues
    linear_fit = [-0.00838733 , 0.07555277]

    # original Mpipi shows a reduction in eps as pariwise hydrophobicity increases, but
    # this is not what we want. We want to increase eps as hydrophobicity increases, so
    # we take the absolute value of the slope.
    corrected_dependency = abs(linear_fit[0])

    # Kyte-Doolitle hydrophobicty scale per AA
    intercept = linear_fit[1]

    # this is a scaling factor that converts the KD hydrophobicity scale to the
    # epsilon scale used in the mPiPi force field. 
    alpha = 4.3

    # Kyte-Doolitle hydrophobicty scale per AA
    AA_KD_scale =  {'A':1.80,'C':2.50,'D':-3.50,'E':-3.50,'F':2.80,'G':-0.40,'H':-3.20,'I':4.50,
                    'K':-3.90,'L':3.80,'M':1.90,'N':-3.50,'P':-1.60,'Q':-3.50,'R':-4.50,'S':-0.80,
                    'T':-0.70,'V':4.20, 'W':-0.90, 'Y':-1.30}

    # scaling epsilon based on Kyte-doolittle Hydrophobisity
    for a1 in ['A','L','M','I','V']:
        for a2 in ['A','L','M','I','V']:

            # sum of hydrophobicity scale values
            KDSSUM = AA_KD_scale[a1] + AA_KD_scale[a2]

            # new epsilon value
            NEWEPS = (KDSSUM * corrected_dependency) + (intercept/alpha)

            # update epsilon values (NOTE this loops over self pairs
            # multiple times, but this is not a problem because we're not basing
            # NEWPS on the original value, but rather the sum of the hydrophobicity
            # scale values
            EPSILON_ALL[a1][a2] = NEWEPS
            EPSILON_ALL[a2][a1] = NEWEPS

    # FIX MU for I:I and I:V such that it is 2.0
    for i in ['I']:
        for j in ['V','I']:
            MU_ALL[i][j] = 2.0
            MU_ALL[j][i] = 2.0

            
    return CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL


## -----------------------------------------------------------
#
#
def CREATE_new_aliphatic_residues_ALL(CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL):
    """
    Function to generate new aliphatic residues that account for the contribution from the 
    local aliphatic context by introducing groups for aliphatic residues based on 
    local number of aliphatic nearest neighbors, then scaling each group proportionately.

    Parameters
    ----------
    CHARGE_ALL : dict
        Dictionary of AA to charge assignments.

    MU_ALL : dict
        Dictionary of all pairwise mu values where keys are AA and values are
        dictionaries of AA:sigma, e.g. MU_ALL['A']['A'] = 0.5

    NU_ALL : dict
        Dictionary of all pairwise nu values where keys are AA and values are
        dictionaries of AA:sigma, e.g. NU_ALL['A']['A'] = 0.5

    EPSILON_ALL : dict
        Dictionary of all pairwise epsilon values where keys are AA and values are
        dictionaries of AA:sigma, e.g. EPSILON_ALL['A']['A'] = 0.5

    SIGMA_ALL : dict
        Dictionary of all pairwise sigma values where keys are AA and values are
        dictionaries of AA:sigma, e.g. SIGMA_ALL['A']['A'] = 0.5

    Returns
    -------
    CHARGE_ALL : dict
        Updated dictionary of AA to charge assignments.

    MU_ALL : dict
        Updated dictionary of all pairwise mu values.

    NU_ALL : dict
        Updated dictionary of all pairwise nu values.
    
    EPSILON_ALL : dict
        Updated dictionary of all pairwise epsilon values.

    SIGMA_ALL : dict
        Updated dictionary of all pairwise sigma values.

    """

    aliphatic_group1 = {'a':'A', 'l':'L', 'm':'M', 'i':'I', 'v':'V'}
    aliphatic_group2 = {'b':'A', 'o':'L', 'x':'M', 'y':'I', 'z':'V'}

    
    all_aligroup_conversions = {'a':'A', 'l':'L', 'm':'M', 'i':'I', 'v':'V', 'b':'A', 'o':'L', 
                                'x':'M', 'y':'I', 'z':'V'}

    # define the new residues we're going to create
    new_residues = list(all_aligroup_conversions.keys())

    # for each of the new residues (a,l,...z)
    for i in new_residues:

        # get the real residue name associated with this rescode
        a1 = all_aligroup_conversions[i]

        # set default values for new residue to match the base residue
        EPSILON_ALL[i] = EPSILON_ALL[a1].copy()
        SIGMA_ALL[i] = SIGMA_ALL[a1].copy()
        MU_ALL[i] = MU_ALL[a1].copy()
        NU_ALL[i] = NU_ALL[a1].copy()

        # set default charge to zero
        CHARGE_ALL[i] = 0.0


    # create a list that encompasses all input residues
    # and the new residues we're creeaing here
    all_residues = list(EPSILON_ALL.keys())
    all_residues.extend(new_residues)
    
    # for each of the new residues (a,l,...z)
    # add aliphatic group residues to other residue dicts 
    for i in new_residues:

        # get the real residue name associated with this rescode
        a1 = all_aligroup_conversions[i]

        # for ALL residue codes (both base codes and new codes)
        for a2 in all_residues:
            # for pairs where BOTH residues are new residues (i.e. at end of the a2 list)
            if a2 in new_residues:
                EPSILON_ALL[a2][i] = EPSILON_ALL[a2][a1]
                SIGMA_ALL[a2][i] = SIGMA_ALL[a2][a1]
                MU_ALL[a2][i] = MU_ALL[a2][a1]
                NU_ALL[a2][i] = NU_ALL[a2][a1]

            # for pairs where only 1 of the 2 residues (i) is a new residue
            else:
                EPSILON_ALL[a2][i] = EPSILON_ALL[i][a2]
                SIGMA_ALL[a2][i] = SIGMA_ALL[i][a2]
                MU_ALL[a2][i] = MU_ALL[i][a2]
                NU_ALL[a2][i] = NU_ALL[i][a2]
                
    return CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL


# ---------------------------------------------------------
def SCALE_aliphatic_group_EPSILON(EPSILON_ALL):
    """
    Function to scale the epsilon values of the aliphatic groups in a way that
    allows us to account for the context that an aliphatic group finds itself
    in. Specifically, we consider three types of aliphatic groups:

    1) Aliphatic groups that are not surrounded by other aliphatic groups (grp0)

    2) Aliphatic groups that are surrounded by one other aliphatic group (grp1)

    3) Aliphatic groups that are surrounded by two other aliphatic groups (grp2)

    We then scale the epsilon values of each group proportionately to account for
    the local aliphatic context.

    Parameters
    ----------
    EPSILON_ALL : dict
        Dictionary of all pairwise epsilon values where keys are AA and values are
        dictionaries of AA:sigma, e.g. EPSILON_ALL['A']['A'] = 0.5

    Returns
    -------
    EPSILON_ALL : dict
        Updated dictionary of all pairwise epsilon values.

    """

    # scaling associated with one residue in grp0 
    Ali0_scaler = 0    

    # scaling when one residue is in grp1 and the other is in grp1 or grp2
    Ali1_scaler = 0.5  

    # scaling when both residues are in grp2
    Ali2_scaler = 2    

    # grp0 scaling
    # Aligroup0 : Aligroup0, Aligroup1, and Aligroup2
    for i in ['A','L','M','I','V']:
        for j in ['A','L','M','I','V','a', 'l', 'm', 'i', 'v', 'b', 'o', 'x', 'y', 'z']:

            # calculate epsilon after scaling
            NEWEPS = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j] * Ali0_scaler)
            
            EPSILON_ALL[i][j] = NEWEPS
            EPSILON_ALL[j][i] = NEWEPS

    # grp1 scaling
    # Aligroup1 : Aligroup1 and Aligroup2
    for i in ['a', 'l', 'm', 'i', 'v']:
        for j in ['a', 'l', 'm', 'i', 'v', 'b', 'o', 'x', 'y', 'z']:

            # calculate epsilon after scaling
            NEWEPS = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j] * Ali1_scaler)
            
            EPSILON_ALL[i][j] = NEWEPS
            EPSILON_ALL[j][i] = NEWEPS

    # grp2 scaling
    # Aligroup2 : Aligroup2
    for i in ['b', 'o', 'x', 'y', 'z']:
        for j in ['b', 'o', 'x', 'y', 'z']:

            # calculate epsilon after scaling
            NEWEPS = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j] * Ali2_scaler)
            
            EPSILON_ALL[i][j] = NEWEPS
            EPSILON_ALL[j][i] = NEWEPS

    return EPSILON_ALL

# ----------------------------------------------------------------------------------------
def ADD_RNA_U_ALL(CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL):
    """
    Function to add RNA U residue to all other residues. This is done by copying the
    parameters from U to all other residues. 

    Parameters
    ----------
    CHARGE_ALL : dict
        Dictionary of all pairwise charge values where keys are AA and values are
        dictionaries of AA:charge, e.g. CHARGE_ALL['A']['A'] = 0.5

    MU_ALL : dict
        Dictionary of all pairwise mu values where keys are AA and values are
        dictionaries of AA:mu, e.g. MU_ALL['A']['A'] = 0.5

    NU_ALL : dict
        Dictionary of all pairwise nu values where keys are AA and values are
        dictionaries of AA:nu, e.g. NU_ALL['A']['A'] = 0.5

    EPSILON_ALL : dict
        Dictionary of all pairwise epsilon values where keys are AA and values are
        dictionaries of AA:sigma, e.g. EPSILON_ALL['A']['A'] = 0.5

    SIGMA_ALL : dict
        Dictionary of all pairwise sigma values where keys are AA and values are
        dictionaries of AA:sigma, e.g. SIGMA_ALL['A']['A'] = 0.5

    Returns
    -------
    CHARGE_ALL : dict
        Updated dictionary of all pairwise charge values.

    MU_ALL : dict
        Updated dictionary of all pairwise mu values.

    NU_ALL : dict
        Updated dictionary of all pairwise nu values.

    EPSILON_ALL : dict
        Updated dictionary of all pairwise epsilon values.

    SIGMA_ALL : dict
        Updated dictionary of all pairwise sigma values.

    """

    
    AAs = ['P','E','D','R','K','I','V','L','T','H','M','G','A','C','S','N','Q','F','Y','W']

    for i in SIGMA_ALL:
        if i not in AAs:
            raise Exception('Error: input directory contains something beyond the standard 20 amino acids. This function should be run first when updating the parameters.')

    # update charge as defined in the RNA params
    for r in defined_RNA_params.CHARGE:
        CHARGE_ALL[r] = defined_RNA_params.CHARGE[r]
        
    # update existing parameters for amino acids with AA:RNA interaction
    for a1 in defined_RNA_params.CHARGE:
        
        for a2 in AAs :

            # get the parameters for the residue; note l_params
            # has format [residue, epsilon, sigma, nu, mu]
            l_params = defined_RNA_params.PARAMS[a2]
            
            EPSILON_ALL[a2][a1] = l_params[1]
            SIGMA_ALL[a2][a1] = l_params[2]

            # ASH - flipped to be correct
            NU_ALL[a2][a1] = l_params[3]
            MU_ALL[a2][a1] = l_params[4]
            

    # create new parameters for U:everything
    for a1 in defined_RNA_params.CHARGE:

        EPSILON_ALL[a1] = {}
        SIGMA_ALL[a1] = {}
        MU_ALL[a1] = {}
        NU_ALL[a1] = {}

        for a2, l_params in defined_RNA_params.PARAMS.items():    
            EPSILON_ALL[a1][a2] = l_params[1]
            SIGMA_ALL[a1][a2] = l_params[2]

            # ASH - flipped to be correct
            NU_ALL[a1][a2] = l_params[3]
            MU_ALL[a1][a2] = l_params[4]


    return CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL

# list of tuples for updating parameters ### 
# where tuple[0] = class function to call
#       tuple[1] = list of functions to pass to tuple[0]
# 
# this is parsed by iterativly passing each item in tuple[1] to tuple[0]. Note the ORDER here REALLY
# matters.
update_to_mPiPi_GGv1 = [('update_ALL', [CORRECT_aliphatic_aliphatic_ALL, CREATE_new_aliphatic_residues_ALL]),
                        ('update_SIGMA_ALL', [ENLARGE_Proline_SIGMA]),
                        ('add_ALL_RNA_U', [ADD_RNA_U_ALL]),
                        ('update_EPSILON_ALL', [STRENGTHEN_small_polar_EPSILON,
                                                WEAKEN_Aromatic_Charge_EPSILON, 
                                                SCALE_aliphatic_group_EPSILON])
                        ]
