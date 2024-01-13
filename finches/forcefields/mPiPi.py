import pickle
import numpy as np
import finches
from os.path import exists

## Code that implements the key forcefield functions in the mPiPi model. For more details on this see
#     Joseph, J. A., Reinhardt, A., Aguirre, A., Chew, P. Y., Russell, K. O., Espinosa, J. R., Garaizar, A., 
#     & Collepardo-Guevara, R. (2021). Physics-driven coarse-grained model for biomolecular phase separation 
#     with near-quantitative accuracy. Nature Computational Science, 1(11), 732–743.
#
# Note - as of right now the model loads the parameters for amino acids in IDRs and does not support nucleic acids, 
# although this of course could easily be extended in the future!
#
# For a demo of these functions

VALID_AA= ['A','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','C']
VALID_RNA=['U']

MPIPI_CONFIGS = {}

# different versions can have different parameters
MPIPI_CONFIGS['mPiPi_default'] = {}
MPIPI_CONFIGS['mPiPi_GGv1'] = {}
MPIPI_CONFIGS['OLD_mPiPi_GGv1'] = {}

MPIPI_CONFIGS['mPiPi_default']['charge_prefactor'] = 0.184890
MPIPI_CONFIGS['mPiPi_default']['null_interaction_baseline'] = -0.066265

MPIPI_CONFIGS['mPiPi_GGv1']['charge_prefactor'] = 0.216145
MPIPI_CONFIGS['mPiPi_GGv1']['null_interaction_baseline'] = -0.128539


MPIPI_CONFIGS['OLD_mPiPi_GGv1']['charge_prefactor'] = 0.216145
MPIPI_CONFIGS['OLD_mPiPi_GGv1']['null_interaction_baseline'] = -0.128539

class mPiPi_model:

    def __init__(self, version='mPiPi_default', input_directory='default', dielectric=80.0, salt=0.150):
        """
        The mPiPi_model class defines an mPiPi_model Object which lets you calculate 
        and return both individual components of the mPiPi forcefield potential 
        alongside an 'interaction parameter' that reflects 


        Parameters
        ---------------
        version : str
            Defines which base parameter set to read-in and use. Default reads in the base 
            pickle files un-modified. If not defult the parameter calls a ansociated  
            list of tuples where: 
                x = (class_function update_XY, [list of input functions to pass x[0]])     
        
            CURRENT OPTIONS : ['mPiPi_default', 'mPiPi_GGv1']

            mPiPi_GGv1 calls: 
                finches.data.mPiPi.mPiPi_GGv1_modification_fxns import update_to_mPiPi_GGv1 
        
        input_directory : str
            Defines the directory where the input parameter files are found. The mPiPi 
            forcefield depends on a set of per-residue (and per-pairwise residue) 
            parameters. These parameters should be stored in a Python dictionary and 
            will be loaded into the mPiPi_model object for manipulation.

            By default, mPiPi_model uses the original parameters defined by Joseph et 
            al. 2021, and you can use the default directory ('default') to load these. 

            However, if you wanted to load your own pre-computed parameters, you can 
            define a directory that must have five pickled dictionaries in them with 
            the following names:

            sigma.pickle   : 2D dictionary that enables indexing in as [R1][R2] to get
                             the R1,R2 sigma parameter.

            epsilon.pickle : 2D dictionary that enables indexing in as [R1][R2] to get
                             the R1,R2 epsilon parameter.

            nu.pickle      : 2D dictionary that enables indexing in as [R1][R2] to get
                             the R1,R2 nu parameter.

            mu.pickle      : 2D dictionary that enables indexing in as [R1][R2] to get
                             the R1,R2 mu parameter.

            charge.pickle : 1D dictionary that enables indexing in as [R1] to get the
                            charge associated with the residue

            Each of these dictionaries contains a 20x20 dictionary keyed by the 20 
            amino acids that defines the pairwise parameter for each of the five 
            parameters named here. In this way, you could in principle define
            and save arbitrary parameters and then feed these into the mPiPi_object
            via the input_directory keyword. Alternatively a series of update  
            commands exist so you can update parameters on the fly.
        
        dielectric : float 
            Defines the dielectric constant of the solvent when computing the interactions 
            in the reference model.

        salt : float
            Defines the general salt concentration build the reference model.
            this salt values tune the electrostatic interactions 

        Returns
        -------------
        finches.methods.forcefields.mPiPi object

        """

        # if 'default' is passed, use the default parameters. Note that if in the future
        # we want to add additional precomputed parameters this is easy and we just need
        # to include an additional directory
        if input_directory == 'default':
            # initialize the prarameters from a pickle file
            data_prefix = finches.get_data('mPiPi')

        else:
            data_prefix = input_directory


        if version == 'mPiPi_default':

            # check files are present    
            for n in ['sigma.pickle', 'epsilon.pickle', 'nu.pickle', 'mu.pickle', 'charge.pickle']:                
                if not exists(f'{data_prefix}/{n}'):
                    raise Exception(f'Using [{data_prefix}] as our data directory but no {n} file found')
            
                with open(f'{data_prefix}/sigma.pickle', 'rb') as fh:
                    self.SIGMA_ALL = pickle.load(fh)
            
                with open(f'{data_prefix}/epsilon.pickle', 'rb') as fh:
                    self.EPSILON_ALL = pickle.load(fh)
            
                with open(f'{data_prefix}/nu.pickle', 'rb') as fh:
                    self.NU_ALL = pickle.load(fh)    
            
                with open(f'{data_prefix}/mu.pickle', 'rb') as fh:
                    self.MU_ALL = pickle.load(fh)        
            
                with open(f'{data_prefix}/charge.pickle', 'rb') as fh:
                    self.CHARGE_ALL = pickle.load(fh)
                    
        elif version == 'mPiPi_GGv1':
            
            # check files are present    
            for n in ['sigma_ggv1.pickle', 'epsilon_ggv1.pickle', 'nu_ggv1.pickle', 'mu_ggv1.pickle', 'charge_ggv1.pickle']:                
                if not exists(f'{data_prefix}/{n}'):
                    raise Exception(f'Using [{data_prefix}] as our data directory but no {n} file found')
            
                with open(f'{data_prefix}/sigma_ggv1.pickle', 'rb') as fh:
                    self.SIGMA_ALL = pickle.load(fh)
            
                with open(f'{data_prefix}/epsilon_ggv1.pickle', 'rb') as fh:
                    self.EPSILON_ALL = pickle.load(fh)
            
                with open(f'{data_prefix}/nu_ggv1.pickle', 'rb') as fh:
                    self.NU_ALL = pickle.load(fh)    
            
                with open(f'{data_prefix}/mu_ggv1.pickle', 'rb') as fh:
                    self.MU_ALL = pickle.load(fh)        
            
                with open(f'{data_prefix}/charge_ggv1.pickle', 'rb') as fh:
                    self.CHARGE_ALL = pickle.load(fh)


        elif version == 'OLD_mPiPi_GGv1':
            
            # check files are present    
            for n in ['old_sigma_ggv1.pickle', 'old_epsilon_ggv1.pickle', 'old_nu_ggv1.pickle', 'old_mu_ggv1.pickle', 'old_charge_ggv1.pickle']:                
                if not exists(f'{data_prefix}/{n}'):
                    raise Exception(f'Using [{data_prefix}] as our data directory but no {n} file found')
            
                with open(f'{data_prefix}/old_sigma_ggv1.pickle', 'rb') as fh:
                    self.SIGMA_ALL = pickle.load(fh)
            
                with open(f'{data_prefix}/old_epsilon_ggv1.pickle', 'rb') as fh:
                    self.EPSILON_ALL = pickle.load(fh)
            
                with open(f'{data_prefix}/old_nu_ggv1.pickle', 'rb') as fh:
                    self.NU_ALL = pickle.load(fh)    
            
                with open(f'{data_prefix}/old_mu_ggv1.pickle', 'rb') as fh:
                    self.MU_ALL = pickle.load(fh)        
            
                with open(f'{data_prefix}/old_charge_ggv1.pickle', 'rb') as fh:
                    self.CHARGE_ALL = pickle.load(fh)
                    
        else:
            raise Exception(f"Unrecognized version [{version}] passed to mPiPi_model. Must be one of 'mPiPi_default' or 'mPiPi_GGv1'")

        # set precomputed forcfield parameters
        self.CONFIGS = MPIPI_CONFIGS[version]

        # initialize object variables
        self.version =  version

        # intu
        self.dielectric = dielectric 
        self.salt = salt 

        # name the identity of the conditions we've assigned
        self.conditions = ['salt', 'dielectric']



    # .....................................................................................
    #
    def compute_wang_frenkel(self, residue_1, residue_2, r):
        """
        Function that returns the values in kJ/mol for the Wang-Frenkel 
        potential associated with the pairwise interaction of the two
        passed residues. 

        Takes two residues and an input distance, which can be a single 
        value or a np.array.

        Parameters
        -------------
        residue_1 : str
            Must be one of the 20 valid amino acid one-letter codes

        residue_2 : str
            Must be one of the 20 valid amino acid one-letter codes

        r : int, float, array-like
            Actual distance (in Angstroms) between the beads. Can be a single 
            number or a numpy array.

        Returns
        ------------
        float or np.array
            Returns energy in kJ/mol that corresponds to the distance provided 
            from r.
        
        """
        
        return wang_frenkel(r, self.SIGMA_ALL[residue_1][residue_2], self.EPSILON_ALL[residue_1][residue_2], self.MU_ALL[residue_1][residue_2], self.NU_ALL[residue_1][residue_2])


    # .....................................................................................
    #
    def compute_colulomb(self, residue_1, residue_2, r, dielectric=None, salt=None):
        """
        Function that returns the values in kJ/mol for the Coulombic 
        potential associated with the pairwise interaction of the two
        passed residues. 

        Takes two residues and an input distance, which can be a single 
        value or a np.array.

        Parameters
        -------------
        residue_1 : str
            Must be one of the 20 valid amino acid one-letter codes

        residue_2 : str
            Must be one of the 20 valid amino acid one-letter codes

        r : int, float, array-like
            Actual distance (in Angstroms) between the beads. Can be a single 
            number or a numpy array.

        dielectric : float
            Dielectric constant of the solvent. Generally safe to assume 80.0

        salt : float
            Salt concentration in M, used for Debye screening. Default=0.15.

        Returns
        ------------
        float or np.array
            Returns energy in kJ/mol that corresponds to the distance provided 
            from r.
        
        """
        if salt is None:
            salt = self.salt

        if dielectric is None:
            dielectric = self.dielectric
            
        return coulomb(self.CHARGE_ALL[residue_1], self.CHARGE_ALL[residue_2], r, dielectric=dielectric, salt=salt)



    # .....................................................................................
    #        
    def compute_full_mPiPi(self, residue_1,  residue_2, r, dielectric=None, salt=None):

        """
        Function that returns the values in kJ/mol for the full Wang-Frenkel 
        potential associated with the pairwise interaction of the two
        passed residues. 

        Takes two residues and an input distance, which can be a single 
        value or a np.array.

        Parameters
        -------------

        residue_1 : str
            Must be one of the 20 valid amino acid one-letter codes

        residue_2 : str
            Must be one of the 20 valid amino acid one-letter codes

        r : int, float, array-like
            Actual distance (in Angstroms) between the beads. Can be a single 
            number or a numpy array.

        dielectric : float
            Dielectric constant of the solvent. Generally safe to assume 80.0

        salt : float
            Salt concentration in M, used for Debye screening. Default=0.15.


        Returns
        ------------
        float or np.array
            Returns energy in J/mol that corresponds to the distance provided 
            from r.
        
        """
        if salt is None:
            salt = self.salt

        if dielectric is None:
            dielectric = self.dielectric

        wf = self.compute_wang_frenkel(residue_1, residue_2, r)
        c = self.compute_colulomb(residue_1, residue_2, r, dielectric=dielectric, salt=salt)
        return wf+c



    # ................................................................................
    #
    #
    def compute_interaction_parameter(self, residue_1, residue_2, r=None, dielectric=None, salt=None):
        """
        NOTE - the name of this function must match name in other forcefield modules

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

        dielectric : float
            Dielectric constant of the solvent. Generally safe to assume 80.0

        salt : float
            Salt concentration in M, used for Debye screening. Default=0.15.

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
            r = np.arange(0.1,30,0.01)

        if salt is None:
            salt = self.salt

        if dielectric is None:
            dielectric = self.dielectric

        # determine 
        sig1 = self.SIGMA_ALL[residue_1][residue_2]
        sig3 = self.SIGMA_ALL[residue_1][residue_2]*3
            
        # get index in distance-dependent energy that matches 1 sigma
        s1 = np.argmin(abs(sig1 - np.array(r)))

        # get index of 3*sigma
        s3 = np.argmin(abs(sig3 - np.array(r)))

        # calculate the combined energy vector of the rage 
        combo = self.compute_full_mPiPi(residue_1, residue_2, r, dielectric=dielectric, salt=salt)

        # take the numerical finite integral between 1 and 3 sigma to calculate
        # an interacion parameter
        interaction_param = np.trapz(combo[s1:s3], x=r[s1:s3])

        return (interaction_param, combo, s1, s3, r)



######################################
# non - parameter specific functions #
######################################

# ................................................................................
#
#
def harmonic(r, mode='protein', k=8.03, ideal_length=None):
    """
    Standard harmonic potential. Equation (2) from mPiPi paper.
    
    Parameters
    -------------
    r : float 
        Actual distance (in Angstroms)
        
    mode : str
        mode to be used. Must be 'protein' or 'nucleic'. 
        Default='protein'
        
    k : float
        If provided allows you to change the spring constant. Default
        in the mPiPi paper is 8.03 J mol-1 pm-2.
        
    ideal_length : float or None
        IF provided over-rides the standard ideal_length distances which are
        defined based on mode. Default = None.
        
    Returns
    ------------
    float
        Returns an energy in kJ/mol 
    
    """
    
    # define the ideal bond length; if we passed an ideal_length
    # in this gets used and we ignore other things.
    if ideal_length is not None:
        r_ref = ideal_length
        
    # if not, use one of the two default values defined in the 
    # paper
    else:            
        if mode == 'protein':
            r_ref = 3.81
        
        elif mode == 'nucleic':
            r_ref = 5.00
        else:
            raise Exception('Invalid mode passed to harmonic()')

    # calculate (r - r_ref)^2 and convert from Angstroms
    # to picometers        
    distance_deviation_in_pm = np.power(100*(r-r_ref),2)
        
    
    # the /1000 is so we go from J to kJ
    return 0.5*k*distance_deviation_in_pm/1000

# ................................................................................
#
#
def coulomb(qi, qj, rij, dielectric=80.0, salt=0.150):
    """
    Coulombic potential with a Debye-Huckel screening term. Takes in two point
    charge values and either a single distance or a numpy array of distances and
    returns the distance dependent energy.

    Note that this is a stand-alone Coloumbic potential with Debye screening and
    actually does not use any of the mPiPi parameters.
    
    Parameters
    -------------
    
    qi : float
        Charge (in 'elemental' units - i.e. 1, 0 -1 0.5 etc) of the first
        bead.
        
    qj : float
        Charge (in 'elemental' units - i.e. 1, 0 -1 0.5 etc) of the second
        bead.
        
    r : int, float, array-like
        Actual distance (in Angstroms) between the beads. Can be a single 
        number or a numpy array.
        
    dielectric : float
        Defines the overal medium dielectric. Default = 80 for water.
    
    salt : float
        Defines the concentration of monovalent salt in molar (i.e.)
        0.150 = 150 mM. Default=0.150.
        
    Returns
    ------------
    float
        Returns an energy in kJ/mol 
    
    """

    # based on Dan's code - thanks Dan!
    sqrt_salt = np.sqrt(salt)
    
    # 3.06 comes the Debye equation - notably see    
    # https://minds.wisconsin.edu/bitstream/handle/1793/79225/1923-debye-huckel-theory-2020-braus-translation-with-preface.pdf?sequence=3&isAllowed=y
    # which has a nice translation of the paper where 3.06 is explicitly deconvolved
    kappa = sqrt_salt/3.06
    
    # define some constants
    C = 1.602176634e-19  # elementary charge constant (in Coulombs)
    Na = 6.023e23        # Avogadro's number (unitless - just a big ol number)
    
    # converts charge from elemental units into columbs and then into C mol, and then
    # divides by pi *4 * e0 - this is in units of j m mol-1
    conversion_constant = (C*C*Na)/(np.pi*4*8.854187812799999e-12)
    
    # convert A to meters
    rij_m = rij*1e-10
    
    # debye dampening factor (this is unitless). Note if salt=0 this goes
    # to 1 and screening is turned off, otherwise this will be a value between
    # 1 and 0. Note that rij is in Angstroms here because of the units that
    # emerge for kappa above
    DH = np.exp(-kappa*rij)
    
    energy_in_J = (conversion_constant*qi*qj)/(dielectric*rij_m)*DH
        
    return energy_in_J/1000



# ................................................................................
#
#
def wang_frenkel(r, sigma_ij, epsilon_ij, mu_ij=2, nu_ij=1):
    """
    Wang-Frenkel potential, takes four parameters and a distance 
    (or numpy array of distances) and returns the distance-dependent
    potential energy in kj/mol.

    Note that this is a stand-alone Wang-Frenkel potential and
    actually does not use any of the mPiPi parameters.
    
    Parameters
    -------------
    r : int, float, array-like
        Actual distance (in Angstroms) between the beads. Can be a single 
        number or a numpy array.
    
    sigma_ij : float 
        Size parameter in Angstroms. Reports on the size of the bead
        in terms of cutoff distance; specifically defines the distance
        at which the WF potential crosses the 0 line on the Y axis. 
        Should be positive.

    epsilon_ij : float 
        Energy in kj/mol for bead-bead energy (well depth). Specifically
        defines the lowest value in the minimum. Should be positive.

    mu_ij : float
        Value that controls the width and steepness of the well

    nu_ij : float
        Second that controls the width and steepness of the well
        
    Returns
    ------------
    float
        Returns an energy in kJ/mol 
    
    """
    
    # define R_ij
    R_ij = 3*sigma_ij
    
    # DO MATH!
        
    alpha_ij_term1 = 2*nu_ij*np.power(R_ij/sigma_ij,2*mu_ij)
    
    alpha_ij_term2 = (2*nu_ij + 1) / (2*nu_ij*(np.power(R_ij/sigma_ij,2*mu_ij) - 1))
    
    alpha_ij = alpha_ij_term1*np.power(alpha_ij_term2,2*nu_ij+1)
    
    main_term1 = epsilon_ij*alpha_ij
    
    main_term2 = np.power(sigma_ij/r, 2*mu_ij) - 1
    
    main_term3 = np.power(np.power(R_ij/r, 2*mu_ij) - 1, 2*nu_ij)
    
    return main_term1*main_term2*main_term3
            
      



