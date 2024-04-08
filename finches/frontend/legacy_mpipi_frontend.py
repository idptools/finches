import metapredict as meta
from finches.forcefields.mPiPi import mPiPi_model
from finches import epsilon_calculation

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter

# initialize an Mpipi forcefield opbject
mPiPi_GGv1_model = mPiPi_model('mPiPi_GGv1')

# build an interaction matrix constructor object
IMC_object = epsilon_calculation.Interaction_Matrix_Constructor(mPiPi_GGv1_model)

import matplotlib

# ensure text is editable in illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# set to define axes linewidths
matplotlib.rcParams['axes.linewidth'] = 0.5


# ....................................................................................
#
#
def intermolecular_idr_matrix(seq1,
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
    
    B = IMC_object.calculate_sliding_epsilon(seq1, seq2, window_size=window_size, use_cython=use_cython, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)
                                             
    # accomodate nucleic acids 
    if seq1.find('U') == -1:
        disorder_1 = meta.predict_disorder(seq1)[B[1][0]:B[1][-1]+1]
    else:
        disorder_1 = np.array([1]*B[0].shape[0])

    # accomodate nucleic acids 
    if seq2.find('U') == -1:
        disorder_2 = meta.predict_disorder(seq2)[B[2][0]:B[2][-1]+1]
    else:
        disorder_2 = np.array([1]*B[0].shape[1])

    # this assertion clause just checks our x values match for matrix dimensions vs. disorder profile
    assert B[0].shape[0] == len(disorder_1)

    # and our y protein
    assert B[0].shape[1] == len(disorder_2)

    return (B, disorder_1, disorder_2)


# ....................................................................................
#
#
def epsilon(seq1, seq2,  use_aliphatic_weighting=True, use_charge_weighting=True):
    """
    Function to calculate the interaction epsilon value between two sequences.

    Parameters
    --------------
    seq1 : str
        Input sequence 1 

    seq2 : str
        Input sequence 2

    use_aliphatic_weighting : bool
        Whether to use the aliphatic weighting scheme for the interaction matrix
        calculation. This weights local aliphatic residues based on the number of
        aliphatic residues adjacent to them. Default is True.

    use_charge_weighting : bool
        Whether to use the charge weighting scheme for the interaction matrix

    """
    
    eps = IMC_object.calculate_epsilon_value(seq1, seq2, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)
    
    return eps
    

# ....................................................................................
#
#
def interaction_figure(seq1,
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
    
    """
    Function to generate an interaction matrix figure between two sequences. This does
    all the calculation on the backend and formats a figure with parallel disorder tracks 
    alongside the interaction matrix.

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
        Whether to use the cython implementation of the interaction matrix (always
        use this if you can). Default is True.

    use_aliphatic_weighting : bool
        Whether to use the aliphatic weighting scheme for the interaction matrix
        calculation. This weights local aliphatic residues based on the number of
        aliphatic residues adjacent to them. Default is True.

    use_charge_weighting : bool
        Whether to use the charge weighting scheme for the interaction matrix. This
        weights local charged residues based on the number of charged residues adjacent
        to them. Default is True.

    tic_frequency : int
        Frequency of the TICs on the plot. Default is 100.

    seq1_domains : list
        List of tuples/lists containing the start and end positions of domains in 
        sequence 1. Means these can be easily higlighted in the plot.

    seq2_domains : list
        List of tuples/lists containing the start and end positions of domains in
        sequence 2. Means these can be easily higlighted in the plot.

    vmin : float
        Minimum value for the interaction matrix color scale. Default is -3.

    vmax : float
        Maximum value for the interaction matrix color scale. Default is 3.

    cmap : str
        Colormap to use for the interaction matrix. Default is 'PRGn'.

    fname : str
        Filename to save the figure to. If None, the figure will be displayed

    Returns
    --------------
    A tuple containing the figure and the axes objects for the main plot, the top
    disorder plot, the right disorder plot and the colorbar.

        fig : matplotlib.figure.Figure (from plt.figure()

        im : matplotlib.image.AxesImage (from plt.imshow())

        ax_main : matplotlib.axes.Axes (from plt.subplot2grid()

        ax_top : matplotlib.axes.Axes  (from plt.subplot2grid()

        ax_right : matplotlib.axes.Axes  (from plt.subplot2grid()

        ax_colorbar : matplotlib.axes.Axes  (from plt.subplot2grid()
    

    """
                                
                                

    B, disorder_1, disorder_2 = intermolecular_idr_matrix(seq1,
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


# ....................................................................................
#
#
def per_residue_attractive_vector(s1,
                                  s2,
                                  window_size=31,
                                  return_total=False,
                                  attractive_threshold=0,
                                  smoothing_window=20,
                                  poly_order=3):
    
    """
    Function to calculate the per-residue attractive vector for a given pair 
    of sequences. This is calculated as the sum of the attractive interactions 
    for each residue in the first sequence with all residues in the second 
    sequence. Specifically, this is an average over all attractive values (i.e.
    where value < 0) using the inter-sequence matrix.

    If return_total is True, the function will return the total sum of attractive
    interactions between the two sequences instead of the average.

    This is (potentially) interesting inasmuch as if we just tak the AVERAGE of
    a region it may be very attractive in some place but repulsive elsewhere, 
    however, repulsive regions in an IDR can avoid each other while attractive
    things attract, so this allows you to identify the putative 'sticker' regions
    without confounding by repulsive regions. 


    Parameters
    ----------
    s1 : str
        The first sequence

    s2 : str
        The second sequence

    window_size : int
        The window size for the intermolecular matrix. Default is 31.
   
    return_total : bool
        If True, return the total sum of attractive interactions between 
        the two sequences. Sometimes you may want this.

    attractive_threshold : float
        The threshold for what is considered attractive. Default is 0 (i.e. 
        only negative values are considered attractive). If changed anything
        less than this value will be considered attractive.

    smoothing_window : int
        The window size for the Savgol filter. This is used to
        smooth the per-residue attractive vector. If set to False no
        smoothing is applied. Default is 20.

    poly_order : int
        The polynomial order for the Savgol filter. This is used to
        smooth the per-residue attractive vector. If set to False no
        smoothing is applied. Default is 3.


    Returns
    -------
    tuple of np.arrays

        [0] : np.array
            Indices of the residues in the first sequence

        [1] : np.array
            The per-residue attractive vector

    """

    # do the thing
    B = intermolecular_idr_matrix(s1,s2, window_size=window_size)[0]

    # extract the raw matrix
    raw_matrix = B[0]

    # get dem indices
    idx = np.arange(B[1][0], B[1][-1]+1)

    # create a mask for attractive values
    attractive_mask = raw_matrix < attractive_threshold

    # sum attractive values in each column
    attractive_sums = np.sum(raw_matrix * attractive_mask, axis=1)

    if len(attractive_sums) != len(idx):
        print(len(attractive_sums), len(idx))
        raise ValueError('Length of attractive sums does not match length of indices; this is a bug.')

    # count attractive values in each column
    attractive_counts = np.sum(attractive_mask, axis=1)

    # Avoid division by zero for columns with no attractive values
    attractive_counts[attractive_counts == 0] = 1

    # alculate the average of attractive values in each column
    if return_total:
        vals = attractive_sums
    else:
        vals = attractive_sums / attractive_counts

    # if no smoothing is requested
    if smoothing_window is False or poly_order is False:
        pass
    else:

        
        try:
            vals = savgol_filter(vals, smoothing_window, poly_order)
        except Exception as e:
            print('')
            print('Error when trying to apply savgol filter; error message below')
            raise(e)


    return idx, vals
            


# ....................................................................................
#
#
def protein_nucleic_vector(seq, fragsize=31, smoothing_window=30, poly_order=3):
    """
    Function to calculate the per-residue attractive vector for a given protein
    sequence. This is calculated as the sliding-window average of a fragsize
    region of the protein with a fragsize region of poly-U RNA. 

    The two-position return vector returns the indices of the residues in the
    protein sequence and the per-residue attractive vector. Note that indices
    START at fragsize-1/2 and END at len(seq)-fragsize+1/2. This is because the
    sliding window is centered on each residue.

    Parameters
    ----------
    seq : str
        The protein sequence

    fragsize : int
        The size of the sliding window. Must be an odd number. Default is 31.

    smoothing_window : int
        The window size for the Savgol filter. This is used to smooth the
        per-residue attractive vector. If set to False no smoothing is applied.

    poly_order : int
        The polynomial order for the Savgol filter. This is used to smooth the
        per-residue attractive vector. If set to False no smoothing is applied.


    Returns
    -------
    tuple of np.arrays

        [0] : np.array
            Indices of the residues in the protein sequence

        [1] : np.array
            The per-residue attractive vector

    """

    if fragsize % 2 == 0:
        raise Exception('fragsize must be odd')


    # define the function to calculate the epsilon value for a given sequence; the closure
    # used with the fragsize and IMC_object variables is to avoid having to pass these
    # as arguments to the function
    def RNA_bind(seq):
        return epsilon_calculation.get_sequence_epsilon_value(seq, len(seq)*'U', IMC_object)/fragsize

    # initialize the return vector
    return_vector = []
    idx = []

    # if the sequence is longer than the fragment size, calculate the per-residue
    # attractive vector for each fragment
    if len(seq) > fragsize:
        
        for i in range(0,1+(len(seq)-fragsize)):
            return_vector.append(RNA_bind(seq[i:i+fragsize]))
            idx.append(i+(fragsize-1)/2)
        


    # if the sequence is shorter than the fragment size, calculate the per-residue
    # attractive vector for the whole sequence, which is all we can really do here
    else:
        return_vector.append(RNA_bind(seq))
        idx = [int(len(seq)/2)]

    return_vector = np.array(return_vector)
    idx = np.array(idx)

    if smoothing_window is False or poly_order is False:
        return [idx, return_vector]
    else:
        return [idx, savgol_filter(return_vector, smoothing_window, poly_order)]




