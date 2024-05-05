import numpy as np

# Stateless functions that can then be freely imported 
    

######################################################################
##                                                                  ##
##                                                                  ##
##              FUNCTIONS FOR MATRIX MANIPULATION                   ##
##                                                                  ##
##                                                                  ##
######################################################################

## ---------------------------------------------------------------------------
##
def get_attractive_repulsive_matrices(matrix, null_interaction_baseline):
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

# commented out and to remove
"""
def flatten_matrix_to_vector(matrix, orientation=[0,1]):

    Function to convert matrix into interaction vectors.

    Parameters
    ---------------
    matrix : array 
        A 2D matrix as an array with the shape of (seqence1, seqence2)
    
    orientation : int
        Flag to specify whether to flatten to matrix along the X or Y axis. 
        1 refers to X axis (mean of rows) (vector relative to sequence1)
        0 refers to Y axis (mean of columns) (vector relative to sequence2)


    Returns
    ------------------
    vector : array 
        A 1D array the length of the columns in input matrix. This vector of 
        ist the mean along the vertical axis for each column in the matrix and 
        is normalized by the model specific null_interaction_baseline.

    return np.mean(matrix, axis=orientation)
    """

######################################################################
##                                                                  ##
##                                                                  ##
##               BUILDING VECTORS & COMPUTING EPSILON               ## 
##                                                                  ##
##                                                                  ##
######################################################################

## ---------------------------------------------------------------------------
##
def get_sequence_epsilon_vectors(sequence1,
                                 sequence2,
                                 X,
                                 charge_prefactor=None,
                                 null_interaction_baseline=None,
                                 use_charge_weighting=True,
                                 use_aliphatic_weighting=True):
    """
    Function to epsilon vectors between a pair of passed sequences
    returned vectors are relative to sequence1 such that len(sequence1) equals 
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

    charge_prefactor : float 
        Model specific value to plug into the local charge weighting of 
        the matrix

    use_charge_weighting : bool
        Flag to select whether weight the matrix by local sequence charge 

    use_aliphatic_weighting : bool
        Flag to select whether weight the matrix by local patches of aliphatic 
        residues
    
    Returns
    --------
    attractive_vector : list 
        attractive epsilon vector of sequence1 relative to sequence2

    repulsive_vector : list 
        repulsive epsilon vector of sequence1 relative to sequence2 
    
    """

    # check for baseline 
    if not null_interaction_baseline:
        null_interaction_baseline = X.null_interaction_baseline

    # get interaction matrix for said sequence
    w_matrix = X.calculate_weighted_pairwise_matrix(sequence1,
                                                    sequence2,
                                                    convert_to_custom=True, 
                                                    charge_prefactor=charge_prefactor,
                                                    use_charge_weighting=use_charge_weighting,
                                                    use_aliphatic_weighting=use_aliphatic_weighting)
    
    # get attractive and repulsive matrix
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrices(w_matrix, null_interaction_baseline)

    attractive_matrix = attractive_matrix - null_interaction_baseline
    repulsive_matrix = repulsive_matrix - null_interaction_baseline

    # take average over the matrix rows to get attractive and repulsive values
    attractive_vector = np.mean(attractive_matrix, axis=1) 
    repulsive_vector = np.mean(repulsive_matrix, axis=1)   

    # return attractive and repulsive vectors
    return attractive_vector, repulsive_vector


## ---------------------------------------------------------------------------
##
def get_sequence_epsilon_value(sequence1,
                               sequence2,
                               X,
                               charge_prefactor=None,
                               null_interaction_baseline=None,
                               use_charge_weighting=True,
                               use_aliphatic_weighting=True):
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

    charge_prefactor : float 
        Model specific value to plug into the local charge weighting of 
        the matrix

    use_charge_weighting : bool
        Flag to select whether weight the matrix by local sequence charge 

    use_aliphatic_weighting : bool
        Flag to select whether weight the matrix by local patches of aliphatic 
        residues
    
    Returns
    --------
    epsilon : float 
        sequence epsilon value as computed between sequence1 and sequence2 
    
    """

    # get attractive and repulsive vectors 
    attractive_vector, repulsive_vector = get_sequence_epsilon_vectors(sequence1,
                                                                       sequence2,
                                                                       X,
                                                                       charge_prefactor=charge_prefactor,
                                                                       null_interaction_baseline=null_interaction_baseline,
                                                                       use_charge_weighting=use_charge_weighting,
                                                                       use_aliphatic_weighting=use_aliphatic_weighting)

    # sum vectors to get attractive and repulsive values
    attractive_value = np.sum(attractive_vector)
    repulsive_value = np.sum(repulsive_vector)

    # sum attractive and repulsive values
    return attractive_value + repulsive_value


## ---------------------------------------------------------------------------
##
def get_interdomain_epsilon_vectors(sequence1,
                                    sequence2,
                                    X,
                                    SAFD_cords,
                                    charge_prefactor=None,
                                    null_interaction_baseline=None,
                                    use_charge_weighting=True,
                                    IDR_positon=['Cterm','Nterm','CUSTOM'],
                                    origin_index=None, 
                                    sequence_of_ref='sequence1'):
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
w        the first sequence (FOLDED DOMAIN) and only SAFD residues.
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
        of the inputed Folded Domain (sequence1). If 'CUSTOM' the origin_index flag must be set to 
        a specific index in SAFD_cords.

    origin_index : int 
        Optional value formated like on of indexes in the SAFD_cords list that will be used as the 
        point of origin for where the IDR is attached to the fold domain. Defult here is None.  

        NOTE - IF THIS IS PASSED IDR_positon must be set to CUSTOM)

    sequence_of_ref : str 
        Flag to denote whether to build the interaction vectors relative to 'sequence1' or 'sequence2'

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    charge_prefactor : float 
        Model specific value to plug into the local charge weighting of 
        the matrix

    use_charge_weighting : bool
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
    if IDR_positon not in ['Cterm','Nterm','CUSTOM']:
        raise Exception(f'INVALID IDR_positon passed')
    
    if sequence_of_ref not in ['sequence1','sequence2']:
        raise Exception(f'INVALID sequence_of_ref passed')

    # check for charge_prefactor  
    if not charge_prefactor:
        charge_prefactor = X.charge_prefactor

    # check for baseline 
    if not null_interaction_baseline:
        null_interaction_baseline = X.null_interaction_baseline

    # check origin location 
    if IDR_positon == 'CUSTOM':
        if not origin_index:
            raise Exception('When IDR_positon is set to CUSTOM the origin_index must be set an index in (X,Y,Z) the cordinate')
        try:
            list(map(lambda a: float(a)), origin_index)
        except:
            raise Exception('origin_index must be set to (X,Y,Z) cordinate where XYZ can be floats')

    # parse sequence_of_ref flag 
    orientation = {'sequence1':1,'sequence2':0}[sequence_of_ref] 

    # check to make sure sequence1 is the Folded Domain
    if len(sequence1) != len(SAFD_cords):
        raise Exception('Length of sequence1 does not match length of SAFD cordinates \n sequence1 should be the fold domain sequence')

    # get interaction matrix for said sequence
    matrix = X.calculate_pairwise_heterotypic_matrix(sequence1, sequence2, convert_to_custom=True)
    
    # get mask for IDR residues relative resisdues on FD
    # just returns bionary mask
    w_xyz_mask = build_column_mask_based_on_xyz(matrix, SAFD_cords, IDR_positon=IDR_positon, origin_index=origin_index)

    # NOTE - NO weighting of aliphatics is conducted here because aliphatic weighting is 
    #   only performed between groups of local aliphatic residues, and no groups are calculated 
    #   on the surface of folded domains, therefor all aliphatics who still be treated as if they 
    #   they are in isolation.  

    #  only occurs between on surface residue and IDR window.
    w_mask = parsing_aminoacid_sequences.get_charge_weighted_FD_mask(sequence1, sequence2) 
    w_matrix = matrix - (matrix*w_mask*charge_prefactor)

    # get attractive and repulsive matrix
    attractive_matrix, repulsive_matrix = get_attractive_repulsive_matrices(w_matrix, null_interaction_baseline)

    # NOTE - filtering the matrix is done after the matrix is split give filtering just multiplys 
    #        by 0 and null_interaction_baseline may fluctuate and is likly not zero 

    # multiply matrix by 1&0s to screen out IDRs residues that cant reach
    wXYZ_attractive_matrix = mask_matrix(attractive_matrix - null_interaction_baseline, w_xyz_mask) 
    wXYZ_repulsive_matrix = mask_matrix(repulsive_matrix - null_interaction_baseline, w_xyz_mask) 

    # original code left here...
    #attractive_vector = flatten_matrix_to_vector(wXYZ_attractive_matrix, orientation=orientation)
    #repulsive_vector = flatten_matrix_to_vector(wXYZ_repulsive_matrix, orientation=orientation)
    attractive_vector = np.mean(wXYZ_attractive_matrix, axis=orientation)
    repulsive_vector = np.mean(wXYZ_repulsive_matrix, axis=orientation)

    # return attractive and repulsive vectors
    return attractive_vector, repulsive_vector


## ---------------------------------------------------------------------------
##
def get_interdomain_epsilon_value(sequence1,
                                  sequence2,
                                  X,
                                  SAFD_cords,
                                  charge_prefactor=None,
                                  null_interaction_baseline=None,
                                  use_charge_weighting=True,
                                  IDR_positon=['Cterm','Nterm', 'CUSTOM'],
                                  origin_index=None,
                                  sequence_of_ref='sequence1'):
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
        of the inputed Folded Domain (sequence1). If 'CUSTOM' the origin_index flag must be set to 
        a specific index in SAFD_cords.

    origin_index : int 
        Optional value formated like thes indexes in the SAFD_cords list that will be used as the 
        point of origin for where the IDR is attached to the fold domain. Defult here is None. 

        NOTE - IF THIS IS PASSED IDR_positon must be set to CUSTOM

    sequence_of_ref : str 
        Flag to denote whether to build the interaction vectors relative to 'sequence1' or 'sequence2'

    Optional Parameters
    -------------------
    null_interaction_baseline : float  
        threshold to differentiate between attractive and repulsive interactions

    charge_prefactor : float 
        Model specific value to plug into the local charge weighting of 
        the matrix

    use_charge_weighting : bool
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
    attractive_vector, repulsive_vector = get_interdomain_epsilon_vectors(sequence1,
                                                                          sequence2,
                                                                          X,
                                                                          SAFD_cords, 
                                                                          charge_prefactor=charge_prefactor,
                                                                          origin_index=origin_index,
                                                                          null_interaction_baseline=null_interaction_baseline,
                                                                          use_charge_weighting=use_charge_weighting,
                                                                          IDR_positon=IDR_positon, 
                                                                          sequence_of_ref=sequence_of_ref)

    # itegerate under vectors to get attractive and repulsive values
    attractive_value = np.sum(attractive_vector)
    repulsive_value = np.sum(repulsive_vector)

    # sum attractive and repulsive vectors to get sequence1 centric vector
    return attractive_value + repulsive_value

