"""
Class to build Interation Matrix from Mpipi forcefield values


By : Garrett M. Ginell & Alex S. Holehouse 
2022-10-18
"""
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------
class Calculate_Interation_Matrix:
    
    def __init__(self, parameters, sequence_converter=False):
        
        ## code below initializes a self.matrix which defines all pairwise interactions
        valid_aa = parameters.ALL_RESIDUES_TYPES # this parameter should be in all housetools.methods.forcefields
#         ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',
#                    'a', 'l', 'm', 'i', 'v', 'b', 'o', 'x', 'y', 'z']

        self.lookup = {}
        self.parameters = parameters 
        self.sequence_converter = sequence_converter
        
        for r1 in valid_aa:
            self.lookup[r1] = {}
            for r2 in valid_aa:
                # this parameter function should be in all housetools.methods.forcefields
                self.lookup[r1][r2] = self.parameters.compute_interaction_parameter(r1,r2)[0]
                
    def get_custom_seq(self, sequence):
         if self.sequence_converter:
            return self.sequence_converter(sequence)
         else:
            return sequence
                
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
        
        matrix = []
        
        for r1 in sequence:
            tmp = []
            
            for r2 in sequence:                
                tmp.append(self.lookup[r1][r2])
            matrix.append(tmp)
            
        return np.array(matrix)
        
     
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
    
    A null_interaction_baseline = -0.15 seems to recapitulate PolyGS
    
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



# ---------------------------------------------------------------------------
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
