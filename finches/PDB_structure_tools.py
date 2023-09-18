"""
Holehouse Lab - Internal Script

This script has code to parse amino acid sequences PDBs and 
extract the surface residues for a given folded domain in the 
in the PDB.

by: Garrett M. Ginell 

updated: 2022-07-22
"""
from soursop.sstrajectory import SSTrajectory
import metapredict
import math
import numpy as np

from .data.reference_sequence_info import GS_seqs_dic


## ---------------------------------------------------------------------------
##
def _accessibility_parse(PO):
    """
    Function that takes in proteinTrajectory object and returns the 
    SASA as list using a of stride=1 and probe_radius=7. For more 
    information see soursop documentation.

    Parameters
    --------------
    PO : obj 
        Instantiation of a proteinTrajectory object in soursop.

    Returns
    ---------------
    list 
        The per residues SASA of the passed protein object specific 
        output is the first parameter returned from PO.get_all_SASA

    """
    return list(PO.get_all_SASA(stride=1, probe_radius=7)[0])

## ---------------------------------------------------------------------------
##
def _check_bounds(SAFD_idxs, FD_start, FD_end):
    """
    Function to check that the passed indicies are 
    with bounds of FD_start and FD_end
    
    Parameters
    --------------
    SAFD_idxs : list 
        List of indicies (for example the output of _get_accessible_indicies)
    
    FD_start : int 
        Lower bound for mimimum index included 
    
    FD_end : int 
        Upper bound for maximim index included 

    """
    for i in SAFD_idxs:
        if FD_start<=i<=FD_end:
            pass
        else:
            raise Exception(f'INDEX ERROR - index {i} outside of bounds {FD_start}-{FD_end}')


## ---------------------------------------------------------------------------
##
def _get_accessible_indicies(l_acc, threshold=10):
    """
    Function that takes in a list of solvent accessablitiy scores 
    and returns a list of indicies that are considered accessable residues.
     
    An accessable residue is defined as one with a score > 10 

    Parameters
    --------------
    l_acc : list 
        List of values containing accessibilty scores in the units of 
        NOTE UPDATE DOCS HERE 

    threshold : float 
        Value that denotes accessibilty (if score > threshold == accessibile)

    Returns
    ---------------
    list 
        The indicies for the residues in the list that are considered
        accessable (IE above the threshold)

    """
    out_SA_track=[]
    for r in l_acc:
        if r > threshold:
            out_SA_track.append(1)
        else:
            out_SA_track.append(0)
    return [index for index, v in enumerate(out_SA_track) if v == 1]

## ---------------------------------------------------------------------------
##
def _filter_indicies(l_acc_idxs, FD_start, FD_end):
    """
    Function that filters a list of indicies by an inputed 
    minimum and maximum index value. 

    Parameters
    --------------
    l_acc_idxs : list 
        List of indicies (for example the output of _get_accessible_indicies)
    
    FD_start : int 
        Lower bound for mimimum index included 
    
    FD_end : int 
        Upper bound for maximim index included 

    Returns
    ---------------
    list 
        A filtered list of indicies that are within the bounds of the 
        inputed values  FD_start and FD_end

    """
    return [i for i in l_acc_idxs if FD_start<=i<=FD_end]

## ---------------------------------------------------------------------------
##
def calculate_distance(coord1, coord2):
    """
    Function to calculate the distance between two sets of XYZ coridnates 
    
    Parameters
    --------------
    coord1 : tuple 
        tuple containing cordinates of (X, Y, Z)
    
    coord2 : tuple 
        tuple containing cordinates of (X, Y, Z)

    Returns
    ---------------
    float 
        Value that represents the distance in 3D between coord1 and coord2

    """
    # Calculate the squared differences between corresponding coordinates
    squared_diffs = [(c1 - c2)**2 for c1, c2 in zip(coord1, coord2)]
    
    # Calculate the square root of the sum of squared differences
    return math.sqrt(sum(squared_diffs))


## ---------------------------------------------------------------------------
##
def build_column_mask_based_on_xyz(matrix, SAFD_cords, IDR_positon=['Cterm','Nterm','CUSTOM'], origin_index=None):
    """
    Function to read in SAFD_cords and build a 2D mask for the matrix based on which
    residues are generally reachable in 3D space on the IDR. The matrix is read in such
    that sequence1 is the Folded Domain and matrix.shape()[0] == len(SAFD_cords)

    Parameters
    ---------------
    matrix : array 
        A 2D matrix as an array with the shape of (seqence1, seqence2)

    SAFD_cords : list 
        Sequence mask of sequence1 containing the solvent accessable folded domain (SAFD)
        residue cordinates where len(SAFD_cords) == len(sequence1) 
        where sequence1 == SAFD_seq

        This list should be organized such that: 
          values that are NOT solvent accessable and NOT in a folded domain are NOT included 
          values that are solvent accessable and in a folded domain contain cordinate in
            xyz form
        
        This SAFD_cords can be returned by PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs

    IDR_positon : str 
        Flag to denote whether the IDR sequence (sequence2) is directly 'C-terminal' or 'N-terminal'
        of the inputed Folded Domain (sequence1). If 'CUSTOM' the origin_index flag must be set to 
        a specific index in SAFD_cords.

    Optional value formated like on of indexes in the SAFD_cords list that will be used as the 
        point of origin for where the IDR is attached to the fold domain. Defult here is None.  

        NOTE - IF THIS IS PASSED IDR_positon must be set to CUSTOM)
    
    Returns
    --------
    out_mask : np.array
        binary 2D mask where 1 is an accessable inderaction between and IDR and a
        SAFD residue. 0 are indicies in the matrix where the residue pair is not within a 
        physical distance that could result in an interaction. IE the distance between the 
        residue in the on the surface of the FD and that in the IDR is beyond the expanded 
        polymer limits of equiviant length polymer relitive to the position of the residue 
        in the IDR and where the IDR is attached to the folded domain.

    NOTE THIS IS BROKEN

    """

    # get the xyz position of where the IDR attaches to the FD
    if IDR_positon == 'Cterm':
        IDR0_xyz = SAFD_cords[-1]

    elif IDR_positon == 'Nterm':
        IDR0_xyz = SAFD_cords[0]

    elif IDR_positon == 'CUSTOM':
        if not origin_index:
            raise Exception('origin_index can not be NONE if IDR_positon = CUSTOM')
        else:
            IDR0_xyz = SAFD_cords[origin_index]

    out_matrix = []
    # for each SAFD residue row
    for i_fdres, col in enumerate(matrix):
        # compute the distance between the SA residue on the FD and terminal 
        #    end of FD where the IDR is attached either (it is assumed this is 
        #    either the first of last solvent accessable residue) 
        SAFD_distance = calculate_distance(IDR0_xyz, SAFD_cords[i_fdres]) 
        l_out_row = []
        # for each IDR residue col
        for i, v in enumerate(col):
            if SAFD_distance < GS_seqs_dic[i]:
                l_out_row.append(1)
            else:
                l_out_row.append(0)

        out_matrix.append(l_out_row)

    out_mask = np.array(out_matrix)
    if out_mask.shape != np.array(matrix).shape:
        raise Exception('New built mask does not match shape of passed matrix')

    return out_mask 


## ---------------------------------------------------------------------------
##
def pdb_to_SDFDresidues_and_xyzs(pdb, FD_start, FD_end, issolate_domain=False):
    """
    Function that takes in pdb and returns the solvent accessable surface residues
    based on the in the imputed range on folded domain and the calculated solvent 
    accessable residues on the surface of the folded domain. 

    Parameters
    ---------------
    pdb : str 
        Path to the .pdb file that is then passed to soursop

    FD_start : int 
        Lower bound for mimimum residue index that includes the folded domain
    
    FD_end : int 
        Upper bound for maximim residue index that includes the folded domain

    issolate_domain : bool 
        Flag to seperate the domain out from the rest of the pdb and calculate
        the SASA on the domain in issolation of the rest of the surrounded 
        residues and chains found in the pdb.

    Returns
    --------
    SAFD_seq : str
        A string of the solvent accessible residues found in the folded domain 
        concatinated in index order. len(SAFD_seq) == the number of solvent 
        accessible residues found in the folded domain.

    SAFD_idxs : list
        The sequence indicies for the residues in that are considered solvent 
        accessable and in the folded domain. The positions of these indicies 
        corispond to the those in the returned SAFD_seq, such that 
        SAFD_seq[i] ~= SAFD_idxs[i]. The index value of SAFD_idxs[i] is in reference
        to the full len sequence found in the pdb.

    SAFD_cords : list 
        Sequence mask containing the solvent accessable folded domain (SAFD)
        residue cordinates where The positions of these cordinates corispond 
        to the those in the returned in SAFD_seq and SAFD_idxs

        This list is organized such that: 
          Values that are NOT solvent accessable and are NOT in
          a folded domain are NOT returned.

          Values that are solvent accessable and in a folded domain 
            are included with there (x, y, z)

    """
    try:
        PO = SSTrajectory(pdb, pdb).proteinTrajectoryList[0]
    except Exception as e:
        print(f"Exception thrown on [{pdb}] while parsing file with soursop\n{e}\n")

    # this for building the SASA for the domain out of comlex of everything else
    if issolate_domain:
        # here is where we would redefine PO
        #l_PO = PO.

        # get solvent accessibitly score of PO in pdb
        # here len(accessibility_track) = len(full_seq)
        # accessibility_track = 
        ### FOR ALEX OR RYAN TO FILL IN ###
        raise Exception(f'''This functionaliy is not yet built set issolate_domain=False. For more information
                            contact Ryan and Alex who have code.''')
        pass 
    else:
        # full protein object casted as protein object
        l_PO = PO 
        
        # get solvent accessibitly score of PO in pdb
        # here len(accessibility_track) = len(full_seq)
        accessibility_track = _accessibility_parse(PO)

    # extract SAFD Residues 
    l_seq = PO.get_amino_acid_sequence(oneletter=True)

    # get accessible indicies
    l_acc_idxs = _get_accessible_indicies(accessibility_track)

    # filter accessible indicies for only those in domain
    SAFD_idxs = _filter_indicies(l_acc_idxs, FD_start, FD_end)

    # get SAFD Sequence
    SAFD_seq = ''.join([l_seq[i] for i in SAFD_idxs])

    # get xyz cordinate track for SAFD_idxs
    ca_idxs = PO.get_multiple_CA_index(resID_list=SAFD_idxs)
    SAFD_cords = PO.traj.xyz[0][ca_idxs]

    return SAFD_seq, SAFD_idxs, SAFD_cords


## ---------------------------------------------------------------------------
##
def pdb_to_xyz_cordinate_track(pdb):
    """
    Function that takes in a pdb and returns a track of XYZs cordinates for
    that residue in the pdb. The XYZs cordinates are determined based off the 
    cordinates of the CA atom for each residue in the pdb. 

    Parameters
    ---------------
    pdb : str 
        Path to the .pdb file that is then passed to soursop

    Returns
    --------
    full_cordinate_track : list 
        list the length of chain in the PDB where each position is a residue 
        and contains the XYZs cordinates for that residue.

    """
    try:
        PO = SSTrajectory(pdb, pdb).proteinTrajectoryList[0]
    except Exception as e:
        print(f"Exception thrown on [{pdb}] while parsing file with soursop\n{e}\n")

    all_indx = [i for i in range(len(PO.get_amino_acid_sequence(oneletter=True)))]
    ca_idxs = PO.get_multiple_CA_index(resID_list=all_indx)
    full_cordinate_track = PO.traj.xyz[0][ca_idxs]

    return full_cordinate_track


## ---------------------------------------------------------------------------
##
def accesssvector_to_SAFD_residues(sequence, FD_start, FD_end, accessibility_track):
    """
    Function that takes in a solvent accessibility track and gets the solvent 
    accessable surface residues based on the in the imputed range of folded domain
    bounds and computed SASA. 

    Returned is the concatinated SAFD amino acid sequence and indicies of the
    accessable residues in the sequence. 

    Parameters
    ---------------
    sequence : str 
        full length amino acid sequence equal in length to the accessibility_track

    FD_start : int 
        Lower bound for mimimum residue index that includes the folded domain
    
    FD_end : int 
        Upper bound for maximim residue index that includes the folded domain
    
    accessibility_track : list 
        List of values containing accessibilty scores for each residue in the 
        sequence. This can be returned from a PDB by calling _accessibility_parse()

        NOTE - PDB parsing is often timely, hence it is best to parse the pbd once 
               extracting the SAFD_idxs and SAFD_cords in one go. This can be done 
               with pdb_to_SDFDresidues_and_xyzs().
    
    Returns
    --------
    SAFD_seq : str
        A string of the solvent accessible residues found in the folded domain 
        concatinated in index order. len(SAFD_seq) == the number of solvent 
        accessible residues found in the folded domain.

    SAFD_idxs : list
        The sequence indicies for the residues in that are considered solvent 
        accessable and in the folded domain. The positions of these indicies 
        corispond to the those in the returned SAFD_seq, such that 
        SAFD_seq[i] ~= SAFD_idxs[i]. The index value of SAFD_idxs[i] is in reference
        to the full len sequence found in the pdb.

    """
    l_acc_idxs = _get_accessible_indicies(accessibility_track)

    # filter accessible indicies for only those in domain
    SAFD_idxs = _filter_indicies(l_acc_idxs, FD_start, FD_end)

    # get SAFD Sequence
    SAFD_seq = ''.join([l_seq[i] for i in SAFD_idxs])

    return SAFD_seq, SAFD_idxs


## ---------------------------------------------------------------------------
##
def tracks_to_SAFD_residues_and_xyzs(sequence, FD_start, FD_end, accessibility_track, full_cordinate_track):
    """
    Function to get the SAFD_seq, SAFD_idxs, and SAFD_cords from a amino acid sequence
    a precomputed solvent accesibility track, pre-extracted xyz_cordinate_track. 

    Parameters
    ---------------
    sequence : str 
        full length amino acid sequence equal in length to the accessibility_track

    FD_start : int 
        Lower bound for mimimum residue index that includes the folded domain
    
    FD_end : int 
        Upper bound for maximim residue index that includes the folded domain
    
    accessibility_track : list 
        List of values containing accessibilty scores for each residue in the 
        sequence. This can be returned from a PDB by calling _accessibility_parse()

        NOTE - PDB parsing is often timely, hence it is best to parse the pbd once 
               extracting the SAFD_idxs and SAFD_cords in one go. This can be done 
               with pdb_to_SDFDresidues_and_xyzs().
    
    full_cordinate_track : list 
        list the length of chain in the PDB where each position is a residue 
        and contains the XYZs cordinates for that residue.

    Returns
    ---------------
    SAFD_seq : str
        A string of the solvent accessible residues found in the folded domain 
        concatinated in index order. len(SAFD_seq) == the number of solvent 
        accessible residues found in the folded domain.

    SAFD_idxs : list
        The sequence indicies for the residues in that are considered solvent 
        accessable and in the folded domain. The positions of these indicies 
        corispond to the those in the returned SAFD_seq, such that 
        SAFD_seq[i] ~= SAFD_idxs[i]. The index value of SAFD_idxs[i] is in reference
        to the full len sequence found in the pdb.

    SAFD_cords : list 
        Sequence mask containing the solvent accessable folded domain (SAFD)
        residue cordinates where The positions of these cordinates corispond 
        to the those in the returned in SAFD_seq and SAFD_idxs

        This list is organized such that: 
          Values that are NOT solvent accessable and are NOT in
          a folded domain are NOT returned.

          Values that are solvent accessable and in a folded domain 
            are included with there (x, y, z)

    """
    SAFD_seq, SAFD_idxs = accesssvector_to_SAFD_residues(sequence, FD_start, FD_end, accessibility_track)

    SAFD_cords = [full_cordinate_track[i] for i in SAFD_idxs]

    return SAFD_seq, SAFD_idxs, SAFD_cords

## ---------------------------------------------------------------------------
##
def map_SAFD_vector_to_full_folded_domain(partial_vector, SAFD_idxs, FD_start, FD_end, null_value=0):
    """
    Function that takes a partial vector and the orderd indcies of the values in the 
    parial vector (an SAFD interaction_vector) and builds a full vector length of 
    in domain based on domain index bounds. Functionally this is usefull to build 
    a full mask of the fold domain from an interaction vector computed using 
    just the SAFD_sequence. 

    SAFD_idxs, and SAFD_cords from a amino acid sequence
    a precomputed solvent accesibility track, pre-extracted xyz_cordinate_track. 

    Parameters
    ---------------
    partial_vector : list 
        vector of equivalent length to SAFD_idxs where each position has a value 
        For example this can be a SAFD attractive_vector outputed from: 

            epsilon_calculation.get_interdomain_epsilon_vectors 

        where in the above function sequence_of_reff = 'sequence1' IE. the 
        returned attractive_vector is equal in length to the SAFD_seq 

    SAFD_idxs : list
        The sequence indicies for the residues in that are considered solvent 
        accessable and in the folded domain. The positions of these indicies 
        corispond to the those in the partial_vector, such that 
        partial_vector[i] ~= SAFD_idxs[i]. The index value of SAFD_idxs[i] is 
        in reference to the sequence space found in FD_start and FD_end. 

    FD_start : int 
        Lower bound for mimimum residue index that includes the folded domain
    
    FD_end : int 
        Upper bound for maximim residue index that includes the folded domain


    null_value : float 
        Value to insert for missing indicies not defined in SAFD_idxs but within 
        the bounds of the FD_start and FD_end

    Returns
    ---------------
    
    FULL_FD_vector : list 
        vector of length (FD_end - FD_start + 1) with values for of the partial_vector 
        for the indicies in  SAFD_idxs and the other values as the null_value
        
        funtionally this represents the a vector that can then be used as a color 
        map for visulalization of folded domain.

    FULL_FD_idxs : list 
        list of all indicies ranging FD_start to FD_end
    """

    # check to make sure all indicies are in the bounds of FD_start of FD_end
    _check_bounds(SAFD_idxs, FD_start, FD_end)

    
    FULL_FD_idxs = [i for i in range(FD_start, FD_end+1)]
    FULL_length = FD_end - FD_start + 1
    FULL_FD_vector = [null_value] * FULL_length
    
    # make full vector 
    for ei, i in enumerate(SAFD_idxs):
        if i in SAFD_idxs: 
            l_i = i - FD_start
            FULL_FD_vector[l_i] = partial_vector[ei]

    # check lengths 
    if len(FULL_FD_vector) != FULL_length and len(FULL_FD_vector) != FULL_FD_idxs:
        raise Exception(f'ERROR building new vectors lengths are not expected.') 

    return FULL_FD_vector, FULL_FD_idxs


## ---------------------------------------------------------------------------
##
def extract_flanking_domain_combinations(pdb, return_domain_lists=False):
    """
    Function takes a pdb and using metapredict extracts directly ajoining pairs
    of IDR and Folded domains these pairs can then be used to compute epsilon values 
    between the IDR and Solvent accessable residues on the folded domain. 

    Parameters
    ---------------
    pdb : str 
        Path to the .pdb file that is then passed to soursop
    
    return_domain_lists : bool 
        Flag to return list of IDRs and Folded Domains in PDB


    Returns
    ---------------
    flanking_combinations : list 
        list of tuples where each tuple contains the bounds of a pair of 
        directly adjoining domains listed as (idr, fd)
    
    IDRS : list [OPTIONAL]
        list of IDRs found in the PDB

    FDs : list [OPTIONAL]
        list of FDs found in the PDB
    """

    # attempt to read pdb
    try:
        PO = SSTrajectory(pdb, pdb).proteinTrajectoryList[0]
    except Exception as e:
        print(f"Exception thrown on [{pdb}] while parsing file with soursop\n{e}\n")

    ### FOR ALEX OR RYAN TO INSERT PDB PARSER HERE IF WE WANT ###

    # get sequence from pdb
    l_seq = PO.get_amino_acid_sequence(oneletter=True)

    # get list of domains in found in PDB
    disorder = metapredict.predict_disorder_domains(l_seq)
    IDRs = disorder.disordered_domain_boundaries
    FDs = disorder.folded_domain_boundaries
    
    # find_flanking_combinations of IDRs and FDs 
    flanking_combinations = []
    for idr in IDRs:
        for fd in FDs:
            if abs(idr[1] - fd[0]) == 0 or abs(fd[1] - idr[0]) == 0:
                flanking_combinations.append((idr, fd))

    if return_domain_lists:
        return flanking_combinations, IDRs, FDs
    else:
        return flanking_combinations


