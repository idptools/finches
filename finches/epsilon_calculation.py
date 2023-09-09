"""
Class to build Interation Matrix from Mpipi forcefield values


By : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06
"""
import numpy as np
import math

from .data.forcefeild_dependencies import precomputed_forcefield_dependent_values, get_null_interaction_baseline

from .parsing_aminoacid_sequences import get_charge_weighed_mask, get_charge_weighed_FD_mask
from .parsing_aminoacid_sequences import get_aliphatic_weighted_mask

from .PDB_structure_tools import build_column_mask_based_on_xyz

# -------------------------------------------------------------------------------------------------
class Interaction_Matrix_Constructor:
    
    def __init__(self, parameters, sequence_converter=False, prefactor=None, null_interaction_baseline=None, 
                 compute_forcefield_dependencies=False):
        
        """
        This initializes the constructor for building a interaction matrix 
        between two inmputed sequences, based on the associated constructor functions
        defines all pairwise interactions. 
        
        Parameters
        -----------
        parameters : obj 
            Instance of one of the forcefield objects found in finches.data.forcefields 
            an example of this is below: 

            from finches.data.forcefields import mPiPi
            mPiPi_GGv2_params = mPiPi.mPiPi_model(version = 'mPiPi_GGv1')

            where mPiPi_GGv2_params is then passed as the parameters. 

            This parameters object has several key associated functions that are required:

                parameters.ALL_RESIDUES_TYPES 

                    NOTE - THIS IS A NESTED LIST FOR WHICH:
                        every residue in each sublist can occur in the same sequeuce,
                        for sequences with residues found in multible sublist and error will be thrown
                    Example: 
                        AAs [['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'],
                        RNA  ['U']]
                
                parameters.compute_interaction_parameter()

        sequence_converter : function 
            A function that takes in a sequence and converts the sequence to analternative 
            sequnence that matches the acceptable residue types of the parameters object 

        prefactor : float 
            Model specific value to plug into the local charge wheighting of 
            the matrix. This value is specific to the version of the paremters being used.

            NOTE -  charge_prefactor is specific to the parameter version. Charge weighing of the 
                    the matrix will not work. Precomputed charge_prefactor values can be added to
                    the following location:

                        data.forcefeild_dependencies.precomputed_forcefield_dependent_values 
                
                    To compute a new charge_prefactor see data.forcefeild_dependencies.get_charge_prefactor

        null_interaction_baseline : float  
            Model specific threshold to differentiate between attractive and repulsive interactions
            This is parameterized based off of a polyGS sequence from that fuction.

            NOTE -  null_interaction_baseline is specific to the parameter version. The null_interaction_baseline is
                    the value used to split matrix. This has been built such that this value recapitulates PolyGS for
                    the specific input model being used. Precomputed charge_prefactor values can be added to
                    the following location: 

                        data.forcefeild_dependencies.precomputed_forcefield_dependent_values 
                
                    To compute a new charge_prefactor see data.forcefeild_dependencies.get_null_interaction_baseline

        compute_forcefield_dependencies : bool 
            Flag to specify whether to recompute the model specific 
            charge_prefactor and null_interaction_baseline used when computing epsilon. 

        """

        self.valid_residue_groups = parameters.ALL_RESIDUES_TYPES 
        # this parameter should be in all finches.methods.forcefields
        
        self.parameters = None 
        if not sequence_converter:
            self.sequence_converter = lambda a: a 
        else:
            self.sequence_converter = sequence_converter

        # NOTE ADD CHECKS/ AUTO PULL of values here 
        self.charge_prefactor = prefactor
        self.null_interaction_baseline = null_interaction_baseline

        self.lookup = {}
        self._update_parameters(parameters)

        # charge charge_prefactor 
        if self.charge_prefactor == None: 
            try:
                self.charge_prefactor = precomputed_forcefield_dependent_values['charge_prefactor'][parameters.version]
            except Exception as e: 
                if compute_forcefield_dependencies:

                    # NOTE NOT computing not working yet 
                    # raise Exception(f'compute_forcefield_dependencies not working yet')
                    pass
                    # self.charge_prefactor = 
                else:
                    print(f'''NOTE - charge_prefactor NOT found or defined for the parameter version "{parameters.version}". 
                    Update or set compute_forcefield_dependencies = True. Without charge weighing of the the matrix will not work. 
                    Precomputed charge_prefactor values can be added to the following location:
                    data.reference_sequence_info.precomputed_forcefield_dependent_values
                    To compute a new charge_prefactor see data.forcefeild_dependencies.get_charge_prefactor''')
        # check null_interaction_baseline
        if self.null_interaction_baseline == None: 
            try:
                self.null_interaction_baseline = precomputed_forcefield_dependent_values['null_interaction_baseline'][parameters.version]
            except Exception as e: 
                if compute_forcefield_dependencies: 
                    self._update_null_interaction_baseline(min_len=10, max_len=500, verbose=True)
                else:
                    print(f'''NOTE - null_interaction_baseline NOT found or defined for the parameter version "{parameters.version}". 
                    Update or set compute_forcefield_dependencies = True. Without epsilon calcualtion of matrix will not work. 
                    Precomputed null_interaction_baseline values can be added to the following location: 
                    data.reference_sequence_info.precomputed_forcefield_dependent_values
                    To compute a new null_interaction_baseline see data.forcefeild_dependencies.null_interaction_baseline''')

        

    def _update_lookup_dict(self):
        # make sure all resigroup pairs can be calculated based on passed parameter 
        #  and build reference dictionary
        ## NOTE NEED TO ADD CHECKERS FOR UNFOUND PAIRS ##
        lookup = {}
        valid_aa = sum(self.valid_residue_groups,[])
        for r1 in valid_aa:
            self.lookup[r1] = {}
            for r2 in valid_aa:
                # this parameter function should be in all finches.methods.forcefields
                self.lookup[r1][r2] = self.parameters.compute_interaction_parameter(r1,r2)[0]

    def _update_parameters(self, new_parameters):
        self.parameters = new_parameters
        self._update_lookup_dict()

    def _update_null_interaction_baseline(self, min_len=10, max_len=500, verbose=True):
        """
        Function to compute and update the null interaction baseline 
        for specific passed self.parameters model. This works by calling: 
            data.forcefeild_dependencies.get_null_interaction_baseline

        The self.null_interaction_baseline parameter is then updated.

        *thanks Alex K. for the recomendation on how to organize this feature 
        
        Parameters
        ---------------
        model : obj
            An instantiation of the one of the forcefield class object 
            IE self.parameters 

        min_len : int 
            The minimum length of a polyGS sequence used 

        max_len : int 
            The minimum length of a polyGS sequence used
        """
        if verbose:
            # remind user that the null_interaction_baseline is being updated
            print(f'Recomputing the null_interaction_baseline for {self.parameters.version}...')

        # return the theretical baseline (ibl) where the slope of epsilon vs polyGS(n) == 0 
        null_interaction_baseline = get_null_interaction_baseline(self, min_len=min_len, max_len=max_len)

        if verbose:
            # remind user that the null_interaction_baseline is being updated
            print(f' the new baseline = {null_interaction_baseline}')

        self.null_interaction_baseline = null_interaction_baseline

    def _check_sequence(self, sequence):
        """
        Function that checks that passed sequence contains valid residues to the model
        """
        list_count = 0
        found_lists = set()
        unique_resis = set(sequence)

        for lst in self.valid_residue_groups:
            common_values = unique_resis.intersection(lst)
            if common_values:
                list_count += 1
                found_lists.update(common_values)
                
        if len(unique_resis) > 0:
            if list_count > 1:
                raise Exception(f'INVALID SEQUENCE - input sequence below contains a mix of valid residues: \n {sequence}')
            elif len(found_lists - unique_resis) > 0:
                raise Exception(f'INVALID SEQUENCE - unknow residue found in input sequence below: \n {sequence}')
        else:
            raise Exception(f'INVALID SEQUENCE - no input sequence passed')

    ## ------------------------------------------------------------------------------
    ##
    def get_converted_sequence(self, sequence):
        """
        Function to return sequence of pasted through the sequence converter
        *use to be called get_custom_sequence

        Parameters
        ----------
        sequence : str
            Amino acid sequence of interest
        
        Returns
        ----------
        str
            converted sequence from self.sequence_converter(sequence)
        
        """
        if self.sequence_converter:
            outseq = self.sequence_converter(sequence)
        else:
            outseq = sequence
        _check_sequence(outseq) 
        return outseq


    ## ------------------------------------------------------------------------------
    ##
    def calculate_pairwise_homotypic_matrix(self, sequence, convert_to_custom=True):
        
        """
        Interaction_Matrix_Constructor.calculate_pairwise_homotypic_matrix

        Note in reality you should just do all pairwise-residues ONCE 
        at the start and then look 'em up, but this code below does the
        dynamic non-redudant on-the-fly calculation of unique pairwise
        residues.

        Parameters
        ---------------
        sequence : str
            Amino acid sequence of interest

        Returns
        ------------------
        np.array
            Returns an (n x n) matrix with pairwise interactions; recall
            that negative values are attractive and positive are repulsive!

        """
        
        if convert_to_custom:
            sequence = self.sequence_converter(sequence)
        
        self._check_sequence(sequence)

        matrix = []
        for r1 in sequence:
            tmp = []
            
            for r2 in sequence:                
                tmp.append(self.lookup[r1][r2])
            matrix.append(tmp)
            
        return np.array(matrix)
        
    ## ------------------------------------------------------------------------------
    ## 
    def calculate_pairwise_heterotypic_matrix(self, sequence1, sequence2, convert_to_custom=True):
        
        """
        Interaction_Matrix_Constructor.calculate_pairwise_heterotypic_matrix

        Note in reality you should just do all pairwise-residues ONCE 
        at the start and then look 'em up, but this code below does the
        dynamic non-redudant on-the-fly calculation of unique pairwise
        residues.

        Parameters
        ---------------
        sequence : str
            Amino acid sequence of interest

        Returns
        ------------------
        np.array
            Returns an (n x n) matrix with pairwise interactions; recall
            that negative values are attractive and positive are repulsive!

        """
        
        if convert_to_custom:
            sequence1 = self.sequence_converter(sequence1)
            sequence2 = self.sequence_converter(sequence2)
        else:
            self._check_sequence(sequence1)
            self._check_sequence(sequence2)

        matrix = []
        for r1 in sequence1:
            tmp = []
            
            for r2 in sequence2:                
                tmp.append(self.lookup[r1][r2])
            matrix.append(tmp)
            
        return np.array(matrix)

    ## ------------------------------------------------------------------------------
    ## 
    def calculate_weighted_pairwise_matrix(self, sequence1, sequence2, convert_to_custom=True, 
                                           prefactor=None, CHARGE=True, ALIPHATICS=True):
        
        """
        Interaction_Matrix_Constructor.calculate_pairwise_heterotypic_matrix


        Calculate heterotypic matrix and weight the matrix.

        Parameters
        ---------------
        sequence1 : str
            Amino acid sequence of interest

        sequence2 : str
            Amino acid sequence of interest
        
        prefactor : float 
            Model specific value to plug into the local charge wheighting of 
            the matrix 

        CHARGE : bool
            Flag to select whether weight the matrix by local sequence charge 

        ALIPHATICS : bool
            Flag to select whether weight the matrix by local patches of aliphatic 
            residues

        Returns
        ------------------
        np.array
            Returns an (n x n) matrix with pairwise interactions; recall
            that negative values are attractive and positive are repulsive, 
            this matrix is weighted by local sequence context of the two 
            inputed sequences.

        """
        # compute matrix
        if sequence1 == sequence2:
            matrix = self.calculate_pairwise_homotypic_matrix(sequence1, convert_to_custom=True)
        else:
            matrix = self.calculate_pairwise_heterotypic_matrix(sequence1, sequence2, convert_to_custom=True)

        # weight the matrix by local sequence charge
        if CHARGE:

            if prefactor == None:
                prefactor = self.charge_prefactor 

            w_mask = get_charge_weighed_mask(sequence1, sequence2)
            try:
                w_matrix = matrix - (matrix*w_mask*prefactor)
            except Exception as e:
                raise Exception('INVALID charge_prefactor, check to ensure self.charge_prefactor is defined.')
                print(e)
        else:
            w_matrix = matrix

        # weigt the matrix by local patches of aliphatic residues
        if ALIPHATICS:
            w_ali_mask = get_aliphatic_weighted_mask(sequence1, sequence2)
            w_matrix = w_matrix*w_ali_mask

        return w_matrix


    

######################################################################
##                                                                  ##
##                                                                  ##
##              FUNCTIONS FOR MATRIX MANIPULATION                   ##
##                                                                  ##
##                                                                  ##
######################################################################

## ---------------------------------------------------------------------------
##
def get_attractive_repulsive_matrixes(matrix, null_interaction_baseline):
    """
    Take interaction array, descritize it by above or below interaction baseline,
    Return two shaped matched matrixes for attractive and repulsive values
    
    The null_interaction_baseline = value to split matrix. This has been built such 
        that this value recapitulates PolyGS for the specific input model being use 
        to see more on how to compute a null_interaction_baseline see... 

        NEED TO UPDATE HERE 
    
    Parameters
    ---------------
    matrix : np.array
        array returned by a function in the Interaction_Matrix_Constructor class

    null_interaction_baseline : float
        Value to specify where to split the matrix for attractive vs repulsive interactions
    
    Returns
    ------------------
    attractive_matrix : np.array
        An array the same shape of the input matrix with only values below the 
        null_interaction_baseline

    repulsive_matrix : np.array
        An array the same shape of the input matrix with only values below the 
        null_interaction_baseline

    """
    attractive_matrix = (matrix < null_interaction_baseline)*matrix
    repulsive_matrix =  (matrix > null_interaction_baseline)*matrix

    return attractive_matrix, repulsive_matrix


## ---------------------------------------------------------------------------
##
def mask_matrix(matrix, column_mask):
    """
    Function to take matrix and multipy it by a mask. This 
    also check to make sure the mask is the same shape. 

    Parameters
    ---------------
    matrix : array 
        A 2D matrix as an array with the shape of (seqence1, seqence2)

    column_mask : array 
        A 2D array with the shape of the inputed matrix

    Returns
    ------------------
    out_matrix : array 
        A 2D matrix with the same shape of the inputed matrix
        where the out_matrix = matrix*column_mask

    """
    # check to ensure matrix and mask are same shape 
    if matrix.shape == column_mask.shape:
        return matrix*column_mask
    else:
        raise Exception('column_mask and matrix are not the same shape')

## ---------------------------------------------------------------------------
##
def flatten_matrix_to_vector(matrix, orientation=[0,1]):
    """
    Function to convert matrix into interaction vectors.

    Parameters
    ---------------
    matrix : array 
        A 2D matrix as an array with the shape of (seqence1, seqence2)
    
    orientation : int
        Flag to specify whether to flatten to matrix along the X or Y axis. 
        1 refers to X axis (mean of rows) (vector relitive to sequence1)
        0 refers to Y axis (mean of columns) (vector relitive to sequence2)


    Returns
    ------------------
    vector : array 
        A 1D array the length of the columns in input matrix. This vector of 
        ist the mean along the vertical axis for each column in the matrix and 
        is normalized by the model specific null_interaction_baseline.
    """
    return np.mean(matrix, axis=orientation)


######################################################################
##                                                                  ##
##                                                                  ##
##               BUILDING VECTORS & COMPUTING EPSILON               ## 
##                                                                  ##
##                                                                  ##
######################################################################

## ---------------------------------------------------------------------------
##
def get_sequence_epsilon_vectors(sequence1, sequence2, X, prefactor=None, null_interaction_baseline=None,
                                CHARGE=True, ALIPHATICS=True):
    """
    Function to epsilon vectors between a pair of passed sequences
    returned vectors are relitive to sequence1 such that len(sequence1) equals 
    the len(returned_vectors)

    NOTE this code was previously : get_weighted_sequence_epsilon_value 
        It is now UPDATED to get_sequence_epsilon_vectors and  get_sequence_epsilon_value
        all weighting is determined by flags. 
    
    Parameters
    -----------
    sequence1 : str
        The first sequence to compare 

    sequence2 : str
        The second sequence to compare 

    X : obj 
        Instance of the Interaction_Matrix_Constructor class with intialized pairpwise interactions 
        and modelspecific parameters 

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    prefactor : float 
        Model specific value to plug into the local charge wheighting of 
        the matrix

    CHARGE : bool
        Flag to select whether weight the matrix by local sequence charge 

    ALIPHATICS : bool
        Flag to select whether weight the matrix by local patches of aliphatic 
        residues
    
    Returns
    --------
    attractive_vector : list 
        attractive epsilon vector of sequence1 relitive to sequence2

    repulsive_vector : list 
        repulsive epsilon vector of sequence1 relitive to sequence2 
    
    """

    # check for baseline 
    if not null_interaction_baseline:
        null_interaction_baseline = X.null_interaction_baseline

    # get interaction matrix for said sequence
    w_matrix = X.calculate_weighted_pairwise_matrix(sequence1, sequence2, convert_to_custom=True, 
                                           prefactor=prefactor, CHARGE=CHARGE, ALIPHATICS=ALIPHATICS)
    
    # get attractive and repulsive matrix
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrixes(w_matrix, null_interaction_baseline)
    
    attractive_matrix = attractive_matrix - null_interaction_baseline
    repulsive_matrix = repulsive_matrix - null_interaction_baseline

    # itegerate under vectors to get attractive and repulsive values
    attractive_vector = flatten_matrix_to_vector(attractive_matrix, orientation=1) 
    repulsive_vector = flatten_matrix_to_vector(repulsive_matrix, orientation=1)

    # return attractive and repulsive vectors
    return attractive_vector, repulsive_vector


## ---------------------------------------------------------------------------
##
def get_sequence_epsilon_value(sequence1, sequence2, X, prefactor=None, null_interaction_baseline=None,
                                CHARGE=True, ALIPHATICS=True):
    """
    Function to epsilon value between a pair of passed sequences

    NOTE this was previously : get_weighted_sequence_epsilon_value 
        It is now UPDATED to get_sequence_epsilon_value and all weighting is determined by flags. 
    
    Parameters
    -----------
    sequence1 : str
        The first sequence to compare 

    sequence2 : str
        The second sequence to compare 

    X : obj 
        Instance of the Interaction_Matrix_Constructor class with intialized pairpwise interactions 
        and modelspecific parameters 

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    prefactor : float 
        Model specific value to plug into the local charge wheighting of 
        the matrix

    CHARGE : bool
        Flag to select whether weight the matrix by local sequence charge 

    ALIPHATICS : bool
        Flag to select whether weight the matrix by local patches of aliphatic 
        residues
    
    Returns
    --------
    epsilon : float 
        sequence epsilon value as computed between sequence1 and sequence2 
    
    """

    # get attractive and repulsive vectors 
    attractive_vector, repulsive_vector = get_sequence_epsilon_vectors(sequence1, sequence2, X,
                                                    prefactor=prefactor,
                                                    null_interaction_baseline=null_interaction_baseline,
                                                    CHARGE=CHARGE, ALIPHATICS=ALIPHATICS)

    # sum vectors to get attractive and repulsive values
    attractive_value = np.sum(attractive_vector)
    repulsive_value = np.sum(repulsive_vector)

    # sum attractive and repulsive values
    return attractive_value + repulsive_value


## ---------------------------------------------------------------------------
##
def get_interdomain_epsilon_vectors(sequence1, sequence2, X, SAFD_cords, prefactor=None, null_interaction_baseline=None,
                                    CHARGE=True, IDR_positon=['Cterm','Nterm'], 
                                    sequence_of_reff=['sequence1','sequence2']):
    """
    Function to epsilon vectors between the surface of a folded domain 
    and a directly ajoining attached IDR sequence. This epsilon value is weighted by
    the local sequence context and the likly reachable Solvent Accessable esidues on 
    the surface of the folded domain.

    NOTE this was previously part of : get_XYZ_weighted_sequence_epsilon_value 
        It is now UPDATED to get_interdomain_epsilon_value and all weighting is determined by flags. 
    
    Parameters
    -----------
    sequence1 : str
        the first sequence (FOLDED DOMAIN) and only SAFD residues.
        everyresidue in this sequence should be SA and in FD. 
        
        To generate this from a PDB see:
            PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs 
        
        sequence1 is the first output of the above function.

    sequence2 : str
        The second sequence to compare (the IDR)

    X : obj 
        Instance of the Interaction_Matrix_Constructor class with intialized pairpwise
        interactions and modelspecific parameters 

    SAFD_cords : list 
        Sequence mask of sequence1 containing the solvent accessable folded domain (SAFD)
        residue cordinates where len(SAFD_cords) == len(sequence1) 

        This list should be organized such that: 
          values that are NOT solvent accessable and NOT in a folded domain = 0 
          values that are solvent accessable and NOT in a folded domain = [x, y, z]
        
        This SAFD_cords can be returned by PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs

        SAFD_cords is the third output of the above function in PDB_structure_tools.

    IDR_positon : str 
        Flag to denote whether the IDR sequence (sequence2) is directly 'C-terminal' or 'N-terminal'
        of the inputed Folded Domain (sequence1)

    sequence_of_reff : str 
        Flag to denote whether to build the interaction vectors relitive to 'sequence1' or 'sequence2'

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    prefactor : float 
        Model specific value to plug into the local charge wheighting of 
        the matrix

    CHARGE : bool
        Flag to select whether weight the matrix by local sequence charge 

        NOTE - NO weighting of aliphatics is conducted here because aliphatic weighting is 
        only performed between groups of local aliphatic residues, and no groups are caluculated 
        on the surface of folded domains, therefor all aliphatics who still be treated as if they 
        they are in isolation. 
    
    Returns
    --------
    epsilon : float 
        sequence epsilon value as computed between sequence1 and sequence2 
    
    """
    # check for prefactor  
    if not prefactor:
        prefactor = X.charge_prefactor

    # check for baseline 
    if not null_interaction_baseline:
        null_interaction_baseline = X.null_interaction_baseline

    # parse sequence_of_reff flag 
    orientation = {'sequence1':1,'sequence2':0}[sequence_of_reff] 

    # check to make sure sequence1 is the Folded Domain
    if len(sequence1) != len(SAFD_cords):
        raise Exception('Length of sequence1 does not match length of SAFD cordinates \n sequence1 should be the fold domain sequence')

    # get interaction matrix for said sequence
    matrix = X.calculate_pairwise_heterotypic_matrix(sequence1, sequence2, convert_to_custom=True)
    
    # get mask for IDR residues relitive resisdues on FD
    # just returns bionary mask
    w_xyz_mask = build_column_mask_based_on_xyz(matrix, SAFD_cords, IDR_positon=IDR_positon)

    # NOTE - NO weighting of aliphatics is conducted here because aliphatic weighting is 
    #   only performed between groups of local aliphatic residues, and no groups are caluculated 
    #   on the surface of folded domains, therefor all aliphatics who still be treated as if they 
    #   they are in isolation.  

    #  only occurs between on surface residue and IDR window.
    w_mask = get_charge_weighed_FD_mask(sequence1, sequence2) 
    w_matrix = matrix - (matrix*w_mask*prefactor)

    # get attractive and repulsive matrix
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrixes(w_matrix, null_interaction_baseline)

    # NOTE - filtering the matrix is done after the matrix is split give filtering just multiplys 
    #        by 0 and null_interaction_baseline may fluctuate and is likly not zero 

    # multiply matrix by 1&0s to screen out IDRs residues that cant reach
    wXYZ_attractive_matrix = mask_matrix(attractive_matrix - null_interaction_baseline, w_xyz_mask) 
    wXYZ_repulsive_matrix = mask_matrix(repulsive_matrix - null_interaction_baseline, w_xyz_mask) 

    attractive_vector = flatten_matrix_to_vector(wXYZ_attractive_matrix, orientation=orientation)
    repulsive_vector = flatten_matrix_to_vector(wXYZ_repulsive_matrix, orientation=orientation)

    # return attractive and repulsive vectors
    return attractive_vector, repulsive_vector


## ---------------------------------------------------------------------------
##
def get_interdomain_epsilon_value(sequence1, sequence2, X, SAFD_cords, prefactor=None, null_interaction_baseline=None,
                                  CHARGE=True, IDR_positon=['Cterm','Nterm'],
                                  sequence_of_reff=['sequence1','sequence2']):
    """
    Function to compute epsilon value between the surface of a folded domain 
    and a directly ajoining attached IDR sequence. This epsilon value is weighted by
    the local sequence context and the likly reachable Solvent Accessable esidues on 
    the surface of the folded domain.

    NOTE this was previously : get_XYZ_weighted_sequence_epsilon_value 
        It is now UPDATED to get_interdomain_epsilon_value and all weighting is determined by flags. 
    
    Parameters
    -----------
    sequence1 : str
        the first sequence (FOLDED DOMAIN) and only SAFD residues.
        everyresidue in this sequence should be SA and in FD. 
        
        To generate this from a PDB see:
            PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs 
        
        sequence1 is the first output of the above function.

    sequence2 : str
        The second sequence to compare (the IDR)

    X : obj 
        Instance of the Interaction_Matrix_Constructor class with intialized pairpwise
        interactions and modelspecific parameters 

    SAFD_cords : list 
        Sequence mask of sequence1 containing the solvent accessable folded domain (SAFD)
        residue cordinates where len(SAFD_cords) == len(sequence1) 

        This list should be organized such that: 
          values that are NOT solvent accessable and NOT in a folded domain = 0 
          values that are solvent accessable and NOT in a folded domain = [x, y, z]
        
        This SAFD_cords can be returned by PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs

        SAFD_cords is the third output of the above function in PDB_structure_tools.

    IDR_positon : str 
        Flag to denote whether the IDR sequence (sequence2) is directly 'C-terminal' or 'N-terminal'
        of the inputed Folded Domain (sequence1)

    sequence_of_reff : str 
        Flag to denote whether to build the interaction vectors relitive to 'sequence1' or 'sequence2'

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    prefactor : float 
        Model specific value to plug into the local charge wheighting of 
        the matrix

    CHARGE : bool
        Flag to select whether weight the matrix by local sequence charge 

        NOTE - NO weighting of aliphatics is conducted here because aliphatic weighting is 
        only performed between groups of local aliphatic residues, and no groups are caluculated 
        on the surface of folded domains, therefor all aliphatics who still be treated as if they 
        they are in isolation. 
    
    Returns
    --------
    epsilon : float 
        sequence epsilon value as computed between sequence1 and sequence2 
    
    """
    # get attractive and repulsive matrixes 
    attractive_vector, repulsive_vector = get_interdomain_epsilon_vectors(sequence1, sequence2, X, SAFD_cords, 
                                                prefactor=prefactor, 
                                                null_interaction_baseline=null_interaction_baseline,
                                                CHARGE=CHARGE, IDR_positon=IDR_positon, 
                                                sequence_of_reff=sequence_of_reff)

    # itegerate under vectors to get attractive and repulsive values
    attractive_value = np.sum(attractive_vector)
    repulsive_value = np.sum(repulsive_vector)

    # sum attractive and repulsive vectors to get sequence1 centric vector
    return attractive_value + repulsive_value

