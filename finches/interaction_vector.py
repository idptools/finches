"""
Module to build inaction vector plots and indentify interacting residues 
between IDRs and other sequences and attached folded domains.


By : Garrett M. Ginell & Alex S. Holehouse 
2023-08-06
"""
import numpy as np
import re
import matplotlib.pyplot as plt

from .epsilon_calculation import get_sequence_epsilon_vectors, get_interdomain_epsilon_vectors
from .PDB_structure_tools import pdb_to_SDFDresidues_and_xyzs, map_SAFD_vector_to_full_folded_domain
from .sequence_tools import show_sequence_HTML

## ---------------------------------------------------------------------------
##
def show_folded_domain_interaction_on_sequence(pdb, FD_start, FD_end,  
                                                sequence2, X, IDR_positon=['Cterm','Nterm'], issolate_domain=False,
                                                prefactor=None, null_interaction_baseline=None, CHARGE=True, 
                                                sequence_of_reff='sequence2', return_vectors=False,
                                                sequence_names=None, title=None): 
    """
    Function to plot the interaction vector bewteen a IDR and an ajoining 
    folded domain. This is just a wrapper over the functions: 
        
        finches.PDB_structure_tools.pdb_to_SDFDresidues_and_xyzs
        finches.epsilon_calculation.get_interdomain_epsilon_vectors

    For full documentation of input parameters see docs of the two 
    above functions.

    NOTE  - by defult this returns a plot computing the interaction 
            vector onto the IDR (sequence2). 

    Returns
    ----------
    
    out_figure
        figure object of plot of interaction vectors of SAFD residues 
        relitive to sequence2 (the IDR)

    attractive_vector
        first output of get_interdomain_epsilon_vectors

    repulsive_vector 
        second output of get_interdomain_epsilon_vectors
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
            title = f'''Interaction Vector of a {IDR_positon} {sequence_names[1]} relitve to residues on the surface of the folded domain ({FD_start}-{FD_end}) in {sequence_names[0]} 
            computed with the pdb file and using the {X.parameters.version} model.'''
        else: 
            title = f'''Interaction Vector of the surface residue on the folded domain ({FD_start}-{FD_end}) in {sequence_names[0]} relitve to the {IDR_positon} {sequence_names[1]}
            computed with the pdb file and using the {X.parameters.version} model.'''

    if sequence_of_reff == 'sequence2' :
        f = make_interaction_vector_plot(attractive_vector, repulsive_vector, sequence2, sequence_names=[sequence_names[1],'surface of '+sequence_names[0]], 
                                     title=title)
    else:
        f = make_interaction_vector_plot(attractive_vector, repulsive_vector, SAFD_seq, sequence_names=['surface of '+sequence_names[0],sequence_names[1]], 
                                     title=title)
    if return_vectors:
        return attractive_vector, repulsive_vector
    else:
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
        title = f'Interaction Vector of {sequence_names[1]} Relitve to {sequence_names[0]} computed using the {X.parameters.version} model.'

    f = make_interaction_vector_plot(attractive_vector, repulsive_vector, sequence1, 
                                        sequence_names=sequence_names, title=title)

    return f


## ---------------------------------------------------------------------------
##
def make_interaction_vector_plot(attractive_vector, repulsive_vector, sequence1, sequence_names=None, fig=None,
                                    title=None, all_resi_labels=True):
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
    
    fig : obj
        matplotlib figure object to plot ontop of, defult = None

    sequence_names : list
        List of names of the sequences used to compute the interaction
        vectors. NOTE the sequence_names[0] refers to sequence1 and 
        sequence_names[1] refers to sequence2

    title : str
        Option input title for the ploted figure 

    all_resi_labels : bool 
        Flag to determine whether to show and color the residues in 
        the sequence plot on the X-axis labels or to show tradional 
        tickmarks by residue index. Residues are colored by thier 
        sequence chemistry.
    
    Returns
    ----------
    out_figure
        figure object of plot of interaction vectors

    """
    if not sequence_names:
        sequence_names = ['sequence1', 'sequence2']

    if not fig:
        # figure axis 
        if all_resi_labels:
            f, ax = plt.subplots(1,1, figsize=(len(sequence1)/5.5, 4), dpi=300, facecolor='w', edgecolor='k')
        else:
            f, ax = plt.subplots(1,1, figsize=(6, 4), dpi=300, facecolor='w', edgecolor='k')
    else:
        # check if X axis is same size as sequence 
        if len(fig.axes[0].get_children()[2].get_xdata(orig=True)) == len(sequence1):
            f = fig
            ax = fig.axes[0]
        else:
            raise Exception('Lenth of X-axis in passed figure "fig" does NOT match length of attractive_vector to be plotted')

    # plot repulsive vector
    ax.plot(repulsive_vector, linewidth=0.5,color='Blue', ls='-', alpha=.9)

    # plot attractive vector
    ax.plot(attractive_vector, linewidth=0.5, color='Red',  ls='-',alpha=.9)

    ax.set_ylabel('Interaction',fontsize=10, fontfamily='avenir')
    ax.set_xlabel(sequence_names[0],fontsize=10, fontfamily='avenir')
    ax.hlines(0, 0,len(repulsive_vector),ls='--',lw=.5,color='grey')

    if not title:
        title = f'Interaction vector of {sequence_names[1]} and {sequence_names[0]}'

    ax.set_title(title, fontfamily='avenir')
    plt.yticks(fontsize=10, fontfamily='avenir')

    if all_resi_labels: 

        # get 
        out_html = show_sequence_HTML(sequence1, return_raw_string=True, font_family='avenir', fontsize=12)

        # parse HTML out 
        fontsize = int(out_html.split('px')[0].split(' ')[-1])
        fontfamily = out_html.split('font-family:')[1].split(';')[0]
        html_seq = out_html.split('<span ')[1:]
        seq = [re.findall(r'>(.*?)</span',i)[0] for i in html_seq]
        colors = re.findall(r'style="color:(.*?)">', out_html)

        # update ticklabel text 
        ax.set_xticks([i for i in range(len(sequence1))])
        ax.set_xticklabels(sequence1, font=fontfamily)
        ax.tick_params(axis='x',colors='black', labelsize=fontsize)  

        # update ticklable color 
        for i in range(len(attractive_vector)):
            ax.get_xticklabels()[i].set_color(colors[i])
    else:
        plt.xticks(np.arange(0,len(repulsive_vector)), sequence1, fontsize=10, ha="center")

    return f


## ---------------------------------------------------------------------------
##
def make_interaction_vector_for_folded_domain(pdb, FD_start, FD_end, sequence2, X,
                                                issolate_domain=False, prefactor=None,
                                                null_interaction_baseline=None,
                                                CHARGE=True, IDR_positon=['Cterm','Nterm']): 
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



