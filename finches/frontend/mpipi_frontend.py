import metapredict as meta
from finches.forcefields.mPiPi import mPiPi_model
from finches import epsilon_calculation

import matplotlib.pyplot as plt
import numpy as np


def build_intermolecular_idr_matrix(seq1,
                                    seq2,
                                    window_size=31,
                                    use_cython=True,
                                    use_aliphatic_weighting=True,
                                    use_charge_weighting=True):

    """
    Function to build the interaction matrix between two sequences, and also
    calculate the disorder profiles for each sequence.

    Parameters
    --------------
    seq1 : str
        Input sequence 1 

    seq2 : str
        Input sequence 2

    window_size : int
        Size of the window to use for the interaction matrix calculation. Note
        this must be an odd number and will be converted to an odd number if it
        is not. Default is 31.

    use_cython : bool
        Whether to use the cython implementation of the interaction matrix
        calculation. This is faster but requires the cython package to be
        installed. Default is True.

    use_aliphatic_weighting : bool
        Whether to use the aliphatic weighting scheme for the interaction matrix
        calculation. This weights local aliphatic residues based on the number of
        aliphatic residues adjacent to them. Default is True.

    use_charge_weighting : bool
        Whether to use the charge weighting scheme for the interaction matrix


    """
    
    # initialize an Mpipi forcefield opbject
    mPiPi_GGv1_model = mPiPi_model('mPiPi_GGv1')

    # build an interaction matrix constructor object
    IMC_object = epsilon_calculation.Interaction_Matrix_Constructor(mPiPi_GGv1_model)

    B = IMC_object.calculate_sliding_epsilon(seq1, seq2, window_size=window_size, use_cython=use_cython, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)
                                             

    disorder_1 = meta.predict_disorder(seq1)[B[1][0]:B[1][-1]+1]
    disorder_2 = meta.predict_disorder(seq2)[B[2][0]:B[2][-1]+1]

    # this assertion clause just checks our x values match for matrix dimensions vs. disorder profile
    assert B[0].shape[0] == len(disorder_1)

    # and our y protein
    assert B[0].shape[1] == len(disorder_2)

    return (B, disorder_1, disorder_2)


def generate_interaction_figure(seq1,
                                seq2,
                                window_size=31,
                                use_cython=True,
                                use_aliphatic_weighting=True,
                                use_charge_weighting=True,
                                tic_frequency=100,
                                seq1_domains=[],
                                seq2_domains=[],
                                vmin=-3,
                                vmax=3,
                                cmap='PRGn',
                                fname=None):
                                
                                

    B, disorder_1, disorder_2 = build_intermolecular_idr_matrix(seq1,
                                                                seq2,
                                                                window_size=window_size,
                                                                use_cython=use_cython,
                                                                use_aliphatic_weighting=use_aliphatic_weighting,
                                                                use_charge_weighting=use_charge_weighting)

    # extract out the interaction matrix
    matrix = B[0]
        
    # Create a figure and a grid of subplots; note we can tweak the figure size params to 
    # make the figure look good depending on how long the two sequences are, here 5.5 and 5
    # works but this is adjustable ofc.
    fig = plt.figure(figsize=(8.5, 8.5), dpi=350)
    
    # Main matrix plot; create axis and then plot using seismic so 0=white
    ax_main = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
    im = ax_main.imshow(matrix.T, extent=[B[1][0], B[1][-1], B[2][0], B[2][-1]], origin='lower', aspect='auto', vmax=vmax, vmin=vmin, cmap=cmap)
    
    # edt here to change tickmarks;  note again the tic_frequency, this again
    # probably can be edited manually depending on the system
    ax_main.set_xticks(np.arange(B[1][0],B[1][-1], tic_frequency))
    ax_main.set_yticks(np.arange(B[2][0],B[2][-1], tic_frequency))
    ax_main.tick_params(axis='x', rotation=45)  # Rotates the x-tick labels by 45 degrees
    
    
    #cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    
    
    ## .....................................................................
    # Bar plot for X protein 
    ax_top = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=ax_main)
    
    # plot disorder profile 
    ax_top.bar(B[1], disorder_1, width=1, color='k', alpha=0.3)
    
    # if we wanted to plot mean per-residue inteaction value, uncomment
    ax_top.set_xlim(B[1][0], B[1][-1])
    
    # highlight some specific regions manually
    for r in seq1_domains:
        region_start = r[0]
        region_end   = r[1]
        ax_top.axvspan(region_start,region_end, color='k', linewidth=0, alpha=0.7)

    
    ## .....................................................................
    # Bar plot for Y protein
    ax_right = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
    ax_right.barh(B[2], disorder_2, align='center', height=1, color='k', alpha=0.3)
    
    ax_right.set_yticks(ax_main.get_yticks())
    ax_right.set_ylim(ax_main.get_ylim())
    
    # highlight some specific regions manually
    
    
    for r in seq2_domains:
        region_start = r[0]
        region_end   = r[1]
        ax_right.axhspan(region_start,region_end, color='k', linewidth=0, alpha=0.7)
    
    ax_colorbar = plt.subplot2grid((4, 4), (0, 3))
    cbar = fig.colorbar(im, cax=ax_colorbar, orientation='vertical')
    ax_colorbar.yaxis.set_ticks_position('left')
    ax_colorbar.yaxis.set_label_position('left')

    plt.tight_layout()
    # finally save the figure
    if fname is not None:

        plt.savefig(fname, dpi=350)

    return fig, im, ax_main, ax_top, ax_right, ax_colorbar


def build_protein_nucleic_vector():
    pass

