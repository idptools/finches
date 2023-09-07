"""
Module to build inaction vector plots and indentify interacting residues 
between IDRs and other sequences and attached folded domains.


By : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06
"""
import matplotlib.pyplot as plt

from .epsilon_calculation import get_sequence_epsilon_vectors, get_interdomain_epsilon_vectors
from .PDB_structure_tools import pdb_to_SDFDresidues_and_xyzs, map_SAFD_vector_to_full_folded_domain


## ---------------------------------------------------------------------------
##
def show_folded_domain_interaction_on_sequence(pdb, FD_start, FD_end,  
                                                sequence2, X, IDR_positon=['Cterm','Nterm'], issolate_domain=False,
                                                prefactor=None, null_interaction_baseline=None, CHARGE=True, 
                                                sequence_of_reff='sequence2',
                                                sequence_names=None, title=None): 
    """
    Function to plot the interaction vector bewteen a IDR and an ajoining 
    folded domain. This is just a wrapper over the functions: 
        
        finches.PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs
        finches.epsilon_calculation.get_interdomain_epsilon_vectors

    For full documentation of input parameters see docs of the two 
    above functions.

    NOTE  - by defult this returns a plot computing the interaction 
            vector relitive to the IDR (sequence2). 

    Returns
    ----------
    
    out_figure
        figure object of plot of interaction vectors of SAFD residues 
        relitive to sequence1

    """

    # extract solvent accessable residues for sequence1 
    SAFD_seq, SAFD_idxs, SAFD_cords = pdb_to_SDFDresidues_and_xyzs(pdb, FD_start, FD_end, 
                                                                   issolate_domain=issolate_domain)


    # compute interaction vectors (NOTE defult of sequence_of_reff is sequence2 the IDR)
    attractive_vector, repulsive_vector = get_interdomain_epsilon_vectors(SAFD_seq, sequence2, X, SAFD_cords,
                                                            prefactor=prefactor,
                                                            null_interaction_baseline=null_interaction_baseline,
                                                            CHARGE=CHARGE, IDR_positon=IDR_positon, 
                                                            sequence_of_reff=sequence_of_reff)

    if not sequence_names:
        sequence_names = ['sequence1', 'sequence2']

    if not title:
        if sequence_of_reff == 'sequence2':
            title = f'''Interaction Vector of a {IDR_positon} {sequence_names[1]} relitve to residues\n
                         on the surface of the folded domain ({FD_start}-{FD_end}) in {sequence_names[0]}\n 
                        computed with the {pdb} pdb file and using the {X.parameters.version} model.'''
        else: 
            title = f'''Interaction Vector of the surface residue on the folded domain ({FD_start}-{FD_end})\n
                        in {sequence_names[0]} relitve to the {IDR_positon} {sequence_names[1]} \n
                        computed with the {pdb} pdb file and using the {X.parameters.version} model.'''

    f = make_interaction_vector_plot(attractive_vector, repulsive_vector, sequence1, sequence_names=sequence_names, 
                                     title=title)
    return f


## ---------------------------------------------------------------------------
##
def show_sequence_interaction_vector(sequence1, sequence2, X, prefactor=None, 
                                     null_interaction_baseline=None, CHARGE=True,
                                     ALIPHATICS=True, sequence_names=None, title=None):
    """
    Function to plot the interaction vector bewteen two sequences. 
    This is just a wrapper over: 

        finches.epsilon_calculation.get_sequence_epsilon_vectors
    
    Parameters
    ---------- 
    
    SEE finches.epsilon_calculation.get_sequence_epsilon_vectors

    Returns
    ----------
    
    out_figure
        figure object of plot of interaction vectors relitive to sequence1

    """
    attractive_vector, repulsive_vector = get_sequence_epsilon_vectors(sequence1, sequence2, X, 
                                                            prefactor=prefactor, 
                                                            null_interaction_baseline=null_interaction_baseline,
                                                            CHARGE=CHARGE, ALIPHATICS=ALIPHATICS)

    if not sequence_names:
        sequence_names = ['sequence1', 'sequence2']

    if not title:
        title = f'''Interaction Vector of {sequence_names[1]} Relitve to {sequence_names[0]}\n 
                    computed using the {X.parameters.version} model.'''

    f = make_interaction_vector_plot(attractive_vector, repulsive_vector, sequence1, 
                                        sequence_names=sequence_names, title=title)

    return f


## ---------------------------------------------------------------------------
##
def make_interaction_vector_plot(attractive_vector, repulsive_vector, sequence1, sequence_names=None, 
                                    title=None):
    """
    Function that takes in attractive_vector and repulsive_vector
    and returns a plt figure object
    
    Parameters
    ---------- 
    attractive_vector : list 
        attractive epsilon vector of sequence1 relitive to sequence2
        as returned by epsilon_calculation.get_sequence_epsilon_value

    repulsive_vector : list 
        repulsive epsilon vector of sequence1 relitive to sequence2 
        as returned by epsilon_calculation.get_sequence_epsilon_value
    
    sequence1 : str
        Amino acid sequence of sequence1. NOTE: 
            len(sequence1) == len(input vectors)

    sequence_names : list
        List of names of the sequences used to compute the interaction
        vectors. NOTE the sequence_names[0] refers to sequence1 and 
        sequence_names[1] refers to sequence2

    title : str
        Option input title for the ploted figure 
    
    Returns
    ----------
    out_figure
        figure object of plot of interaction vectors

    """
    if not sequence_names:
        sequence_names = ['sequence1', 'sequence2']

    # figure axis 
    f, ax = plt.subplots(1,1, dpi=300, facecolor='w', edgecolor='k')

    # plot repulsive vector
    ax.plot(repulsive_vector, linewidth=0.5,color='Blue', ls='-', alpha=.9)

    # plot attractive vector
    ax.plot(attractive_vector, linewidth=0.5, color='Red',  ls='-',alpha=.9)

    ax.set_ylabel('Interaction',fontsize=6)
    ax.set_xlabel(sequence_names[0],fontsize=6)
    ax.hlines(0, 0,len(repulsive_vector),ls='--',lw=.5,color='grey')

    if not title:
        title = f'Interaction vector of {sequence_names[1]} and {sequence_names[0]}'

    ax.set_title(title)
    plt.xticks(np.arange(0,len(repulsive_vector)), sequence1, fontsize=6, ha="center")
    ptl.yticks(fontsize=7)

    return f


## ---------------------------------------------------------------------------
##
def make_interaction_vector_for_folded_domain(pdb, FD_start, FD_end, sequence2, X,
                                                issolate_domain=False, prefactor=None,
                                                null_interaction_baseline=None,
                                                CHARGE=True, IDR_positon=['Cterm','Nterm'],
                                                sequence_names=None, title=None): 
    """
    Function to build a per residue interaction vector for every residue
    in the folded domain of intrest relitive to an ajoining IDR.
    This is just a wrapper over the functions: 
        
        finches.PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs
        finches.epsilon_calculation.get_interdomain_epsilon_vectors

    For full documentation of input parameters see docs of the two 
    above functions.

    In this function the returned vectors of are the length of defined 
    folded domain. 

    Returns
    ----------
    
    FULL_FD_attractive_vector : list 
        attractive epsilon vector of SAFD relitive to the ajoining IDR
        length is equal to that of the entire folded domain INCLUDING 
        non_solvent accessble residues 

    FULL_FD_repulsive_vector : list 
        repulsive epsilon vector of SAFD relitive to the ajoining IDR
        length is equal to that of the entire folded domain INCLUDING 
        non_solvent accessble residues 

    """
    # extract solvent accessable residues for sequence1 
    SAFD_seq, SAFD_idxs, SAFD_cords = pdb_to_SDFDresidues_and_xyzs(pdb, FD_start, FD_end, 
                                                                    issolate_domain=issolate_domain)


    # compute interaction vectors (NOTE sequence_of_reff is sequence1, the SAFD)
    attractive_vector, repulsive_vector = get_interdomain_epsilon_vectors(SAFD_seq, sequence2, X, SAFD_cords,
                                                            prefactor=prefactor,
                                                            null_interaction_baseline=null_interaction_baseline,
                                                            CHARGE=CHARGE, IDR_positon=IDR_positon, 
                                                            sequence_of_reff='sequence1')

    # convert SAFD_vector to FULL_FD_vector 
    FULL_FD_attractive_vector, FULL_FD_idxs = map_SAFD_vector_to_full_folded_domain(attractive_vector, SAFD_idxs, 
                                                                                    FD_start, FD_end, null_value=0)
    FULL_FD_repulsive_vector, FULL_FD_idxs = map_SAFD_vector_to_full_folded_domain(repulsive_vector, SAFD_idxs, 
                                                                                    FD_start, FD_end, null_value=0)

    return FULL_FD_attractive_vector, FULL_FD_repulsive_vector



