"""
Class to build Interation Matrix from Mpipi forcefield values


By : Garrett M. Ginell & Alex S. Holehouse 
2022-10-18
"""
import numpy as np
import math

from .data.reference_sequence_info import GS_seqs_dic

from .parsing_aminoacid_sequences import get_charge_weighed_mask

# -------------------------------------------------------------------------------------------------
class Interaction_Matrix_Constructor:
    
    def __init__(self, parameters, sequence_converter=False):
        
        ## code below initializes a self.matrix which defines all pairwise interactions ##

        self.valid_residue_groups = parameters.ALL_RESIDUES_TYPES 
        # this parameter should be in all finches.methods.forcefields
        # NOTE parameters.ALL_RESIDUES_TYPES IS A NESTED LIST FOR WHICH:
        #   every residue in each sublist can occur in the same sequeuce, for sequences with residues 
        #   found in multible sublist and error will be thrown
        #   Example: 
        #     AAs [['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'],
        #     RNA  ['U']]

        
        self.parameters = None 
        self.sequence_converter = sequence_converter
        self.lookup = {}


        self._update_parameters(parameters)

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

    def _check_sequence(self, sequence):
        """
        Check passed sequence for valid residues 
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
            elif found_lists != unique_resis:
                raise Exception(f'INVALID SEQUENCE - unknow residue found in input sequence below: \n {sequence}')
        else:
            raise Exception(f'INVALID SEQUENCE - no input sequence passed')

    ## ------------------------------------------------------------------------------
    ##
    def get_convert_sequence(self, sequence):
        """
        public facing function to get parameter specific sequence 
        *use to be called get_custom_sequence
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
        Calculate homotypic matrix.

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
        Calculate heterotypic matrix.

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




# ---------------------------------------------------------------------------
def get_attractive_repulsive_matrixes(matrix, null_interaction_baseline):
    """
    Take interaction array, descritize it by above or below interaction baseline,
    
    Return two shape matched matrixes for attractive and repulsive values
    
    A null_interaction_baseline = value to split matrix that recapitulate PolyGS
    
    """
    attractive_matrix = (matrix < null_interaction_baseline)*matrix
    repulsive_matrix =  (matrix > null_interaction_baseline)*matrix

    return attractive_matrix, repulsive_matrix

# ---------------------------------------------------------------------------
def flatten_matrix_to_vector(matrix, null_interaction_baseline):
    """
    Args:
        matrix 

        null_interaction_baseline

    Returns:

        Vector of the summed matrix along the vertical axis
    """
    return np.mean(matrix - null_interaction_baseline, axis=1)


# ---------------------------------------------------------------------------
def get_weighted_matrix(matrix, null_interaction_baseline):
    """
    Args:
        matrix 

        null_interaction_baseline

    Returns:

        matrix-null_interaction_baseline
    """
    return matrix - null_interaction_baseline



## ---------------------------------------------------------------------------
##
def get_sequence_epsilon_value(sequence1, sequence2, X, null_interaction_baseline = -0.15):
    """get per sequence pair chi value for passed sequences
    
    Args:
        sequence1 (string): the first sequence 
        sequence2 (string): the second sequence
        X (object): class object with intialized pairpwise interactions adn  
        null_interaction_baseline (float): threshold to differentiate between attractive 
            and repulsive interactions
    Returns:
        sequence epsilon value as (float)
    
    """
    # get interaction matrix for said sequence
    matrix = X.calculate_pairwise_heterotypic_matrix(sequence1, sequence2, convert_to_custom=True)

    # w_matrix = get_weighted_matrix(matrix, null_interaction_baseline)

    # get attractive and repulsive matrix
    # attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrixes(matrix, null_interaction_baseline)
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrixes(matrix, null_interaction_baseline)

    
    # itegerate under vectors to get attractive and repulsive values
    attractive_value = np.sum(flatten_matrix_to_vector(attractive_matrix, null_interaction_baseline))
    repulsive_value = np.sum(flatten_matrix_to_vector(repulsive_matrix, null_interaction_baseline)) 

    # sum attractive and repulsive values
    return attractive_value + repulsive_value

## ---------------------------------------------------------------------------
##
def get_weighted_sequence_epsilon_value(sequence1, sequence2, X, prefactor, null_interaction_baseline):
    """get per sequence pair epsilon value for passed sequences
    
    Args:
        sequence1 (string): the first sequence 
        sequence2 (string): the second sequence
        X (object): class object with intialized pairpwise interactions adn  
        null_interaction_baseline (float): threshold to differentiate between attractive 
            and repulsive interactions
    Returns:
        sequence epsilon value as (float)
    
    """
    # get interaction matrix for said sequence
    matrix = X.calculate_pairwise_heterotypic_matrix(sequence1, sequence2, convert_to_custom=True)
    
    w_mask = _get_charge_weighed_mask(sequence1, sequence2)
    w_matrix = matrix - (matrix*w_mask*prefactor)
    
    # get attractive and repulsive matrix
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrixes(w_matrix, null_interaction_baseline)
    
    # itegerate under vectors to get attractive and repulsive values
    attractive_value = np.sum(flatten_matrix_to_vector(attractive_matrix, null_interaction_baseline))
    repulsive_value = np.sum(flatten_matrix_to_vector(repulsive_matrix, null_interaction_baseline)) 

    # sum attractive and repulsive values
    return attractive_value + repulsive_value

## ---------------------------------------------------------------------------
##
def mask_matrix_columns(matrix, column_mask):
    """
    """
    # check to ensure matrix and mask are same shape 
    if matrix.shape() == column_mask.shape():
        return matrix*column_mask
    else:
        raise Exception('column_mask and matrix are not the same shape')

## ---------------------------------------------------------------------------
##
def calculate_distance(coord1, coord2):
    # Calculate the squared differences between corresponding coordinates
    squared_diffs = [(c1 - c2)**2 for c1, c2 in zip(coord1, coord2)]
    
    # Calculate the square root of the sum of squared differences
    return math.sqrt(sum(squared_diffs))

## ---------------------------------------------------------------------------
##
def build_column_mask_based_on_xyz(matrix, SAFD_cords, IDR_positon=['Cterm','Nterm']):
    """
    SAFD_cords are read in as IDR1 and is on the X axis

    return binary mask where 1 is accessable idr resdiues
          shape == matrix.shape()
    """
    # get the xyz position of where the IDR attaches to the FD
    if IDR_positon == 'Cterm':
        IDR0_xyz == SAFD_cords[-1]

    elif IDR_positon == 'Nterm':
        IDR0_xyz == SAFD_cords[0]

    out_matrix = []
    # for each SAFD residue col
    for i_fdres, col in enumerate(matrix.T):
        SAFD_distance = calculate_distance(IDR0_xyz, SAFD_cords[i_fdres]) 
        l_out_row = []
        # for each IDR residue row
        for i, v in enumerate(col):
            if SAFD_distance < GS_seqs_dic[i]:
                l_out_row.append(0)
            else:
                l_out_row.append(1)
        out_matrix.append(l_out_row)

    out_mask = np.array(out_matrix).T
    if out_matrix.shape() != matrix.shape():
        raise Exception('New built mask does not match shape of passed matrix')

    return out_mask 

## ---------------------------------------------------------------------------
##
def get_XYZ_weighted_sequence_epsilon_value(sequence1, sequence2, X, prefactor, null_interaction_baseline,
                                            SAFD_cords, IDR_positon=['Cterm','Nterm']):
    """get per sequence pair epsilon value for passed sequences
    
    Args:
        sequence1 (string): the first sequence (FOLDED DOMAIN) and only SAFD residues.
                            everyresidue in this sequence should be SA and in FD.
        sequence2 (string): the second sequence
        X (object): class object with intialized pairpwise interactions adn  
        null_interaction_baseline (float): threshold to differentiate between attractive 
            and repulsive interactions

        SAFD_cords : full sequence mask of SAFD residue cordinates where
             len(SAFD_cords) == len(sequence1)
    Returns:
        sequence epsilon value as (float)
    
    """
    # get interaction matrix for said sequence
    matrix = X.calculate_pairwise_heterotypic_matrix(sequence1, sequence2, convert_to_custom=True)
    
    # get mask for IDR residues relitive resisdues on FD
    # just returns bionary mask
    w_xyz_mask = build_column_mask_based_on_xyz(matrix, SAFD_cords, IDR_positon=['Cterm','Nterm'])


    ###
    ### UPDATE CODE to include aliphatics mask instead of aliphatic beads 
    ###
    w_mask = _get_charge_weighed_mask(sequence1, sequence2) ### NEED FIX SO WEIGHTING..
    ##  only occurs between on surface residue and IDR window.

    w_matrix = matrix - (matrix*w_mask*prefactor)
    
    # multiply matrix by 1&0s to screen out IDRs residues that can reach
    w_XYZ_matrix = mask_matrix_columns(w_matrix, column_mask)

    # get attractive and repulsive matrix
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrixes(w_matrix, null_interaction_baseline)
    
    # itegerate under vectors to get attractive and repulsive values
    attractive_value = flatten_matrix_to_vector(attractive_matrix, null_interaction_baseline)
    repulsive_value = flatten_matrix_to_vector(repulsive_matrix, null_interaction_baseline)

    # sum attractive and repulsive vectors to get sequence1 centric vector
    return attractive_value + repulsive_value

