"""
Holehouse Lab - Internal Script

This script has code to update pairwise interaction strengths for 
mPiPi force field tooning written for development of AA params in mPiPi.

Attapted from original implimentation in Mpipi
see origial at: holehouse-lab/lammpstools/data/configuration_v4/modify_params.py

by: Garrett M. Ginell 

updated: 2022-09-26
"""
from . import defined_RNA_params
### defined functions to modify parameters ###

# -------------------------------------------------
def STRENGTHEN_small_polar_EPSILON(EPSILON_ALL):
    """# STRENGTHEN Small Polar interaction
    # Tune G:GS epsilon +119%""" 
    for i in ['G']:
        for j in ['G','S']:
            if i == j:
                EPSILON_ALL[i][j] = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j]*1.19)
            else:
                EPSILON_ALL[i][j] = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j]*1.19)
                EPSILON_ALL[j][i] = EPSILON_ALL[j][i] + (EPSILON_ALL[j][i]*1.19)
    return EPSILON_ALL 
            
# ----------------------------------------------------------
def WEAKEN_Aromatic_Charge_EPSILON(EPSILON_ALL):               
    """# WEAKEN Aromatic:Charge interaction 
    # Tuning YFW:RED -60%""" 
    for i in ['Y','F','W']:
        for j in ['R','E','D']:
            EPSILON_ALL[i][j] = EPSILON_ALL[i][j] - (EPSILON_ALL[i][j]*0.60)
            EPSILON_ALL[j][i] = EPSILON_ALL[j][i] - (EPSILON_ALL[j][i]*0.60)
    return EPSILON_ALL 

# ---------------------------------------------------
def ENLARGE_Proline_SIGMA(SIGMA_ALL):
    """ENLARGE Proline:X interaction
    # make Proline:X interaction occur at LONGER distances""" 
    # Tuning P:X sigma +33%
    for i in ['E','D','R','K','I','V','L','T','H','M','G','A','C','S','N','Q','F','Y','W']:
        SIGMA_ALL['P'][i] = SIGMA_ALL['P'][i] + (SIGMA_ALL['P'][i] * 0.33)
        SIGMA_ALL[i]['P'] = SIGMA_ALL[i]['P'] + (SIGMA_ALL[i]['P'] * 0.33)
    SIGMA_ALL['P']['P'] = SIGMA_ALL['P']['P'] + (SIGMA_ALL['P']['P'] * 0.33)
    return SIGMA_ALL

# ----------------------------------------------------------------------------------------
def CORRECT_aliphatic_aliphatic_ALL(CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL):
    """
    # CORRECT aliphatic:aliphatic interation 
    # ILVAM:ILVAM epsilon rebuild 
    # epsilons are now relative Kyte-doolittle Hydrophobisity scale
    # NEW_epsilon = 0.008X + 0.018 ; where X = sum(KDhydro_A1, KDhydro_A2) ;
    #         where A1 and A2 are the aliphatic residues in the pairwise interaction
    """

    # Fit from original aliphatic epsilons vs sum of Hydro-Scale Values
    popt = [-0.00838733 , 0.07555277]

    # hydrophobicty scale per AA
    AA_KD_scale =  {'A':1.80,'C':2.50,'D':-3.50,'E':-3.50,'F':2.80,'G':-0.40,'H':-3.20,'I':4.50,
                    'K':-3.90,'L':3.80,'M':1.90,'N':-3.50,'P':-1.60,'Q':-3.50,'R':-4.50,'S':-0.80,
                    'T':-0.70,'V':4.20, 'W':-0.90, 'Y':-1.30}

    # scaling epsilon based on Kyte-doolittle Hydrophobisity
    for a1 in ['A','L','M','I','V']:
        for a2 in ['A','L','M','I','V']:
            KDSSUM = AA_KD_scale[a1] + AA_KD_scale[a2]
            NEWEPS = ((KDSSUM * 0.008 ) + popt[1]/4.3)
            EPSILON_ALL[a1][a2] = NEWEPS
            EPSILON_ALL[a2][a1] = NEWEPS

    # FIX MU for I:I and I:V such that it is 2.0
    for i in ['I']:
        for j in ['V','I']:
            MU_ALL[i][j] = 2
            MU_ALL[j][i] = 2
    return CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL

# ----------------------------------------------------------------------------------------
def ACCOUNT_local_aliphatic_surfaces_ALL(CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL):
    """
    # ACCOUNT for contributions from local aliphatic surfaces by
    # INDRODUCING groups for aliphatic residues based on local number 
    # of aliphatic nearest neighbors. Then SCALE each group proportionately
    # Aligroup1 and Aligroup2  
    """

    aliphatic_group1 = {'a':'A', 'l':'L', 'm':'M', 'i':'I', 'v':'V'}
    aliphatic_group2 = {'b':'A', 'o':'L', 'x':'M', 'y':'I', 'z':'V'}
    all_aligroup_conversions = {'a':'A', 'l':'L', 'm':'M', 'i':'I', 'v':'V', 'b':'A', 'o':'L', 
                                'x':'M', 'y':'I', 'z':'V'}
    new_residues = list(all_aligroup_conversions.keys())

    # add aliphatic aliphatic_group1 & aliphatic_group2
    for i in new_residues:
        a1 = all_aligroup_conversions[i]
        EPSILON_ALL[i] = EPSILON_ALL[a1].copy()
        SIGMA_ALL[i] = SIGMA_ALL[a1].copy()
        MU_ALL[i] = MU_ALL[a1].copy()
        NU_ALL[i] = NU_ALL[a1].copy()
        CHARGE_ALL[i] = 0.0

    # add aliphatic group residues to other residue dicts 
    for i in new_residues:
        a1 = all_aligroup_conversions[i]
        for a2 in ['P','E','D','R','K','I','V','L','T','H','M','G','A','C','S','N',
                   'Q','F','Y','W','a','l','m','i','v','b','o','x','y','z']:
            if a2 in new_residues:
                EPSILON_ALL[a2][i] = EPSILON_ALL[a2][a1]
                SIGMA_ALL[a2][i] = SIGMA_ALL[a2][a1]
                MU_ALL[a2][i] = MU_ALL[a2][a1]
                NU_ALL[a2][i] = NU_ALL[a2][a1]
            else:
                EPSILON_ALL[a2][i] = EPSILON_ALL[i][a2]
                SIGMA_ALL[a2][i] = SIGMA_ALL[i][a2]
                MU_ALL[a2][i] = MU_ALL[i][a2]
                NU_ALL[a2][i] = NU_ALL[i][a2]
                
    return CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL


# ---------------------------------------------------------
def SCALE_aliphatic_group_EPSILON(EPSILON_ALL):
    """
    # SCALE epsilons of each aliphatic group proportional to each other to account 
    # to for local aliphatic surfaces. SCALER for % increase in value of epsilon for 
    # each group of aliphatic residues is followed: 
    """

    Ali0_scaler = 0    # Aligroup0 : Aligroup0, Aligroup1, and Aligroup2
    Ali1_scaler = 0.5  # Aligroup1 : Aligroup1 and Aligroup2
    Ali2_scaler = 2    # Aligroup2 : Aligroup2

    # Aligroup0 : Aligroup0, Aligroup1, and Aligroup2
    for i in ['A','L','M','I','V']:
        for j in ['A','L','M','I','V','a', 'l', 'm', 'i', 'v', 'b', 'o', 'x', 'y', 'z']:
            NEWEPS = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j] * Ali0_scaler)
            EPSILON_ALL[i][j] = NEWEPS
            EPSILON_ALL[j][i] = NEWEPS

    # Aligroup1 : Aligroup1 and Aligroup2
    for i in ['a', 'l', 'm', 'i', 'v']:
        for j in ['a', 'l', 'm', 'i', 'v', 'b', 'o', 'x', 'y', 'z']:
            NEWEPS = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j] * Ali1_scaler)
            EPSILON_ALL[i][j] = NEWEPS
            EPSILON_ALL[j][i] = NEWEPS

    # Aligroup2 : Aligroup2
    for i in ['b', 'o', 'x', 'y', 'z']:
        for j in ['b', 'o', 'x', 'y', 'z']:
            NEWEPS = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j] * Ali2_scaler)
            EPSILON_ALL[i][j] = NEWEPS
            EPSILON_ALL[j][i] = NEWEPS

    return EPSILON_ALL

# ----------------------------------------------------------------------------------------
def ADD_RNA_U_ALL(CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL):
    """
    # ADD RNA for polyU interaction with U and all amino acids #
    
    CHARGE:
    U = -0.75
    """

    AAs = ['P','E','D','R','K','I','V','L','T','H','M','G','A','C','S','N','Q','F','Y','W']

    # update charge
    for r,v in defined_RNA_params.CHARGE.items():
        CHARGE_ALL[r] = v
        
    #update ALL other parameters 
    a1 = 'U'
    # U to all subdictionary
    for a2 in AAs :
        l_params = defined_RNA_params.PARAMS[a2]
        
        EPSILON_ALL[a2][a1] = l_params[1]
        SIGMA_ALL[a2][a1] = l_params[2]
        MU_ALL[a2][a1] = l_params[3]
        NU_ALL[a2][a1] = l_params[4]

    # U to all dictionarys 
    EPSILON_ALL[a1] = {}
    SIGMA_ALL[a1] = {}
    MU_ALL[a1] = {}
    NU_ALL[a1] = {}

    for a2, l_params in defined_RNA_params.PARAMS.items():    
        EPSILON_ALL[a1][a2] = l_params[1]
        SIGMA_ALL[a1][a2] = l_params[2]
        MU_ALL[a1][a2] = l_params[3]
        NU_ALL[a1][a2] = l_params[4]


    return CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL

### list of tuples for updating parameters ### 
### where tuple[0] = class function to call
###       tuple[1] = list of functions to pass to tuple[0]
### 
### this is parsed by iterativly passing each item in tuple[1] to tuple[0]
update_to_mPiPi_GGv1 = [('update_ALL', [CORRECT_aliphatic_aliphatic_ALL, ACCOUNT_local_aliphatic_surfaces_ALL]),
                        ('update_SIGMA_ALL', [ENLARGE_Proline_SIGMA]),
                        ('add_ALL_RNA_U', [ADD_RNA_U_ALL]),
                        ('update_EPSILON_ALL', [STRENGTHEN_small_polar_EPSILON, WEAKEN_Aromatic_Charge_EPSILON, 
                                                    SCALE_aliphatic_group_EPSILON])
                        ]
