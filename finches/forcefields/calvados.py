"""
This Script pulls and builds the pair-wise potentials for the calvados model 
see here:

https://github.com/KULL-Centre/CALVADOS

The Code below is then paired with a jupyter-notebook, and contains code 
directly pulled and adapted from the calvados package. 

By: Garrett M. Ginell
2023.03.07

######################################################################
BUILDING THE MAIN ENERGY POTENTIAL
#
# NOTES:
#  direct_coexistance uses openMM 
#  single_chain uses hoomd
# 
#  Lennard-Jones potential is a Ashbaugh-Hatch potential
#  Debye-Hückel potential is a Yukawa potentials
#
#
# FOR:
#
#  General SI and explanation:
#    https://www.pnas.org/doi/suppl/10.1073/pnas.2111696118/suppl_file/pnas.2111696118.sapp.pdf
#
#  Prefactors calculation:
#    (openmm) 
#    https://github.com/KULL-Centre/CALVADOS/blob/main/direct_coexistence/simulate.py
#     - SEE lines 19-20
#    https://github.com/KULL-Centre/CALVADOS/blob/main/direct_coexistence/analyse.py
#     - SEE fxns genParamsLJ & genParamsDH
#
#    (hoomd) 
#    https://github.com/KULL-Centre/CALVADOS/blob/main/single_chain/simulate.py
#     - SEE fxn genParams (line 18) & lines 65-73, 99-108
#
#  Potential calculation:
#    (openmm)
#    https://github.com/KULL-Centre/CALVADOS/blob/main/direct_coexistence/simulate.py
#     - SEE lines 92-105
#  
#    (hoomd)
#    DH - https://github.com/joaander/hoomd-blue/blob/master/hoomd/md/EvaluatorPairYukawa.h
#    LJ - https://github.com/mphowardlab/azplugins/blob/af9bc8d63c4b45c3e79c756dc1b47b3a1ac44795/
                    azplugins/PairEvaluatorAshbaugh.h
#
#######################################################################
"""

import itertools
import numpy as np
import finches
from os.path import exists
import pandas as pd

types = ['M', 'G', 'K', 'T', 'Y', 'A', 'D', 'E', 'V', 'L', 'Q', 'W', 'R', 'F', 'S', 'H', 'N', 'P', 'C', 'I']
pairs = np.array(list(itertools.combinations_with_replacement(types,2)))

CALVADOS_CONFIGS = {}
CALVADOS_CONFIGS['CALVADOS1'] = {}
CALVADOS_CONFIGS['CALVADOS2'] = {}

CALVADOS_CONFIGS['CALVADOS1']['charge_prefactor'] = np.nan # not computed yet
CALVADOS_CONFIGS['CALVADOS2']['charge_prefactor'] = 1.442590

CALVADOS_CONFIGS['CALVADOS1']['null_interaction_baseline'] = np.nan # not computed yet
CALVADOS_CONFIGS['CALVADOS2']['null_interaction_baseline'] = -0.047859



class calvados_model:

    def __init__(self, version, salt=0.2, pH=7.4, temp=278, input_directory='default'):
        """
        The calvados_model class defines an calvados_model Object which lets you calculate 
        and return both individual components of the calvados forcefield potential 
        alongside an 'interaction parameter' that reflects 
        Parameters
        ---------------
        input_directory : str
            Defines the directory where the input parameter files are found. 

            If defult, the data is pulled from the finches.data.calvados

            Each of these dictionaries contains a 20x20 dictionary keyed by the 20 
            amino acids, (or NxN dictionary keyed by the 20 residues in the model)
            that defines the pairwise parameter for each of the five 
            parameters named here. In this way, you could in principle define
            and save arbitrary parameters and then feed these into the mPiPi_object
            via the input_directory keyword. Alternatively a series of update  
            commands exist so you can update parameters on the fly.
        
        version : str
            Defines the version of the CALVADOS parameters to use for the model
            the options here are based off of those defined in the CALVADOS data files. 

            select CALVADOS1 or CALVADOS2 to choose which stickiness parameters to use

            Current options are: [CALVADOS1, CALVADOS2]

        salt : float
            Defines the general salt concentration build the reference model.
            this salt values tune the electrostatic interactions 

        pH : float 
            Defines the general pH to build the reference model.

        temp : int
            Defines the tempurature at which the the focefield model is computed 
            functionally this really just modulates the streghts of the interactions. 

        Returns
        -------------
        
        model_object : obj 
            Returned is a finches.forcefields.calvados object that can then be passed to 
            the Interaction_Matrix_Constructor class. 


        """

        # if 'default' is passed, use the default parameters. Note that if in the future
        # we want to add additional precomputed parameters this is easy and we just need
        # to include an additional directory
        if input_directory == 'default':
            # initialize the prarameters from a pickle file
            data_prefix = finches.get_data('calvados')
            
        else:
            data_prefix = input_directory

        # check files are present
        for n in ['calvados_residues.pickle']:                
            if not exists(f'{data_prefix}/{n}'):
                raise Exception(f'Using [{data_prefix}] as our data directory but no {n} file found')

        # pull residue information
        self.version = version

        r = pd.read_pickle(f'{data_prefix}/calvados_residues.pickle').set_index('three')
        
        # Added Check to ensure passed version is acceptable 
        try:
            r.lambdas = r[f'{self.version}'] 
        except Exception as e:
            print(e)
            raise Exception(f'''Passed vesion of model unknown. \n Check {data_prefix}/calvados_residues.pickle 
                                for available versions. Current stable versions are [CALVADOS1, CALVADOS2]''')

        self.residue_df = r.set_index('one',drop=False)
        
        # initiate parameters 
        self.ionic = salt 
        self.salt = salt
        self.pH = pH
        self.temp = temp
        self.CONFIGS = CALVADOS_CONFIGS[self.version]

        # calculate other parameters based off of defults in the CALVADOS model
        # line 65 single_chain/simulate.py & (line 138 of direct_coexistence/analyse.py and also toonable in prot obj)
        self.eps_factor = 0.2
        # line 65 single_chain/simulate.py & line 
        self.lj_eps = 4.184 * self.eps_factor 
        # line 29 of direct_coexistence/submit.py
        self.cutoff = 2.0 
        # line 105 - single_chain/simulate.py & line 104 direct_coexistence/simulate.py
        self.yukawa_r_cut = 4.0 


        # generate ionic, pH, temp specific to initialize calvados object 
        yukawa_kappa, yukawa_eps, residues = self._genParams(self.residue_df)
        self.yukawa_kappa = yukawa_kappa
        self.yukawa_eps = yukawa_eps
        self.residue_df = residues 


        # build averages map of sigma and lambda values
        # from lines 70-73 of single_chain/simulate.py
        sigmamap = pd.DataFrame((residues.sigmas.values+residues.sigmas.values.reshape(-1,1))/2,
                                    index=residues.sigmas.index,columns=residues.sigmas.index)

        lambdamap = pd.DataFrame((residues.lambdas.values+residues.lambdas.values.reshape(-1,1))/2,
                            index=residues.lambdas.index,columns=residues.lambdas.index)

        self.sigmamap = sigmamap
        self.lambdamap = lambdamap

        # name of parameters below must match naming in other forcefield module
        # NOTE THIS IS A NESTED LIST FOR WHICH:
        #   every residue in each sublist can occur in the same sequeuce, for sequences with residues 
        #   found in multible sublist and error will be thrown
        self.ALL_RESIDUES_TYPES = [['M', 'G', 'K', 'T', 'Y', 'A', 'D', 'E', 'V', 'L', 'Q', 'W', 'R', 'F', 'S', 'H', 'N', 'P', 'C', 'I'],
                                    ]

        self.conditions = ['salt', 'pH', 'temp']

    # ----------------------------------------------------------
    #
    def _genParams(self, r):
        """
        Function directly pulled from line 18 of calvados/single_chain/simulate.py
        """
        RT = 8.3145*self.temp*1e-3
        
        # Set the charge on HIS based on the pH of the protein solution? Not needed if pH=7.4
        r.loc['H','q'] = 1. / ( 1 + 10**(self.pH-6) )
        
        fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
        epsw = fepsw(self.temp)
        lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
        
        # Calculate the inverse of the Debye length
        yukawa_kappa = np.sqrt(8*np.pi*lB*self.ionic*6.022/10)
        
        # Calculate the prefactor for the Yukawa potential
        qq = pd.DataFrame(r.q.values*r.q.values.reshape(-1,1),index=r.q.index,columns=r.q.index)
        yukawa_eps = qq*lB*RT
        
        return yukawa_kappa, yukawa_eps, r



    # ----------------------------------------------------------
    #
    def compute_full_calvados(self, p1, p2, r_range):
        
        s = self.sigmamap[p1][p2] # sigma 
        l = self.lambdamap[p1][p2] # lambda 
        q = self.yukawa_eps[p1][p2] # q or yukawa_eps
        
        e_out = []
        for r in r_range:
            e_out.append(compute_calvados_energy(r,s,l,q,self.yukawa_kappa, cutoff=self.cutoff, lj_eps=self.lj_eps, yukawa_r_cut=self.yukawa_r_cut))
            
        return e_out

    # ----------------------------------------------------------
    #
    def compute_interaction_parameter(self, residue_1, residue_2, r=None):
        """
        NOTE - the name of this function must match name in other forcefield 
        modules.

        Standalone function that computes pariwise interaction parameter
        between two residue types based on the finite integral between 1
        and 3 sigma.
        
        Parameters
        --------------
        residue_1 : str
            Must be one of the 20 valid amino acid one-letter codes
        residue_2 : str
            Must be one of the 20 valid amino acid one-letter codes
        r : array-like
            Actual distance (in Angstroms) between the beads. Can be a single 
            number or a numpy array. If not provided uses 0.1 to 30 in 
            increments of 0.01; suggest this is  safe range to basically always
            use unless your sigmas start getting really big...

        Returns
        -----------
        tuple
            Returns a tuple with several values:
            [0] - float -  interaction parameter (sum of integral)
            [1] - np.array - full pairwise potential energy vs. distance profile
            [2] - int - index for one sigma
            [3] - int - index for 3 sigma
            [4] - np.array - distance array
        
        """

        if r is None:
            r = np.arange(0.01,3,0.001)
        else:
            # convert Angstroms to nM
            r = r * 10

        # determine 
        sig1 = self.sigmamap[residue_1][residue_2]
        sig3 = self.sigmamap[residue_1][residue_2]*3
            
        # get index in distance-dependent energy that matches 1 sigma
        s1 = np.argmin(abs(sig1 - np.array(r)))

        # get index of 3*sigma
        s3 = np.argmin(abs(sig3 - np.array(r)))

        # calculate the combined energy vector of the rage 
        combo = self.compute_full_calvados(residue_1, residue_2, r)

        # take the numerical finite integral between 1 and 3 sigma to calculate
        # an interacion parameter
        interaction_param = np.trapz(combo[s1:s3], x=r[s1:s3])

        return (interaction_param, combo, s1, s3, r)



# ----------------------------------------------------------
#
def compute_calvados_energy(r,s,l,q,yukawa_kappa, cutoff=2.0, lj_eps=4.184*0.2, yukawa_r_cut=4.0):
     

    unit_nanometer = 1 
    unit_kilojoules_per_mole = 1
    # 
    # calculate Ashbaugh-Hatch potential
    #
    rc = cutoff * unit_nanometer
    eps = lj_eps * unit_kilojoules_per_mole
    
    shift = (s/rc)**12 - (s/rc)**6
    
    # to parse select and step in line 92 of https://github.com/KULL-Centre/CALVADOS/blob/main/direct_coexistence/simulate.py
    #   see http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.CustomNonbondedForce.html
    # step(x) operation
    x = r-2**(1/6)*s
    if x < 0: 
        x_out = 0
    else:
        x_out = 1 
    
    # select(x,y,z) operation
    y = 4*eps*l*((s/r)**12-(s/r)**6-shift)
    z = 4*eps*((s/r)**12-(s/r)**6-l*shift)+eps*(1-l)
    if x_out == 0:
        ah = z
    else:
        ah = y 
 
    #
    # calculate Yukawa potential parameters
    #
    kappa = yukawa_kappa
    shift = np.exp(-yukawa_kappa*yukawa_r_cut)/yukawa_r_cut/unit_nanometer
    yu = q*(np.exp(-kappa*r)/r-shift)
    
    return yu + ah 

