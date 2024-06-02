from .frontend_base import FinchesFrontend

# for model construction
from finches.forcefields.calvados import calvados_model
from finches import epsilon_calculation

# other stuff
import numpy as np

# needed so we preserve docstrings after decorator is applied...
from functools import wraps



##
## This is a decorator that checks for RNA in the input sequences and throws an
## exception if it is found. NOTE the @wraps(func) decorator is needed to preserve
## the docstring of the function being decorated.
##
def RNA_check(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        # args[1] = seq1
        # args[2] = seq2
        if 'U' in args[1] or 'U' in args[2]:
            raise ValueError("CALVADOS cannot handled RNA ('U')")
        return func(*args, **kwargs)
    return wrapper





class CALVADOS_frontend(FinchesFrontend):


    # ....................................................................................
    #
    #            
    def __init__(self, salt=0.150, pH=7.4, temp=288):

        # call superclass constructor 
        super().__init__()

        # initialize the CALVADOS forcefield opbject
        self.model = calvados_model('CALVADOS2', salt=salt, pH=pH, temp=temp)

        # build an interaction matrix constructor object
        self.IMC_object = epsilon_calculation.InteractionMatrixConstructor(self.model)


    # decorator checks for RNA in CALVADOS input

    # ....................................................................................
    #
    #        
    @RNA_check
    def intermolecular_idr_matrix(self,
                                  seq1,
                                  seq2,
                                  window_size=31,
                                  use_cython=True,
                                  use_aliphatic_weighting=True,
                                  use_charge_weighting=True):
        """
        Returns the interaction matrix for the two sequences. Specifically this involves
        decomposing the two sequences into window_size fragments and calculating the inter-fragment
        epsilon values using a sliding window approach.

        Note that we don't pad the sequence here, so the edges of the matrix start
        and end at indices that depend on the window size. To avoid confusion, the
        function also returns the indices for sequence1 and sequence2.

        Parameters
        --------------
        seq1 : str
            Input sequence 1

        seq2 : str
            Input sequence 2

        window_size : int
            The window size to use for the interaction matrix calculation. 
            Default is 31. 

        use_cython : bool
            Whether to use the cython implementation of the interaction matrix calculation.
            Default is True. 

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme for the interaction matrix
            calculation. This weights local aliphatic residues based on the number of
            aliphatic residues adjacent to them. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme for the interaction matrix. This
            weights local charged residues based on the number of charged residues adjacent
            to them. Default is True.

        disorder_1 : bool
            Whether to generate the disorder profile for sequence 1. Default is True. If False,
            a uniform disorder profile is used (all values=1).

        disorder_2 : bool
            Whether to generate the disorder profile for sequence 2. Default is True. If False,
            a uniform disorder profile is used (all values=1).

        Returns
        --------------
        tuple
            A tuple containing the interaction matrix, disorder profile for sequence 1, and disorder
            profile for sequence 2. 

            [0] : This is interaction matrix, and is itself a tuple of 3 elements. The first 
            is the matrix of sliding epsilon values, and the second and 3rd are the indices that map            
            sequence position from sequence1 and sequence2 to the matrix

            [1] disorder profile for sequence 1. Will be all 1s if disorder_1 is False

            [2] disorder profile for sequence 2. Will be all 1s if disorder_2 is False

        """
        

        
        # call the superclass function
        return super().intermolecular_idr_matrix(seq1,
                                                 seq2,
                                                 window_size=window_size,
                                                 use_cython=use_cython,
                                                 use_aliphatic_weighting=use_aliphatic_weighting,
                                                 use_charge_weighting=use_charge_weighting,
                                                 disorder_1=True,
                                                 disorder_2=True)

    
    # ....................................................................................
    #
    #        
    @RNA_check
    def epsilon(self,
                seq1,
                seq2,
                use_aliphatic_weighting=True,
                use_charge_weighting=True):

        """
        Returns the epilson value associated with the two sequences. Note that CALVADOS
        does not currently support RNA.

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
            Whether to use the charge weighting scheme for the interaction matrix. This
            weights local charged residues based on the number of charged residues adjacent
            to them. Default is True.

        Returns
        --------------
        float
            The epsilon value for the two sequences.

        """
        
        return self.IMC_object.calculate_epsilon_value(seq1,
                                                       seq2,
                                                       use_aliphatic_weighting=use_aliphatic_weighting,
                                                       use_charge_weighting=use_charge_weighting)


    
    # ....................................................................................
    #
    #        
    @RNA_check
    def interaction_figure(self,
                           seq1,
                           seq2,
                           window_size=31,
                           use_cython=True,
                           use_aliphatic_weighting=True,
                           use_charge_weighting=True,
                           tic_frequency=100,
                           seq1_domains=[],
                           seq2_domains=[],
                           seq1_lines=[],
                           seq2_lines=[],                           
                           vmin=-7.5,
                           vmax=7.5,
                           cmap='PRGn',
                           fname=None,
                           zero_folded=True):

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
            sequence 1. This means these can be easily highlighted in the plot.

        seq2_domains : list
            List of tuples/lists containing the start and end positions of domains in
            sequence 2. This means these can be easily highlighted in the plot.

        seq1_lines : list
            List of values that will draw lines onto the plot along sequence 1.

        seq2_lines : list
            List of values that will draw lines onto the plot along sequence 1.
                                                      
        vmin : float
            Minimum value for the interaction matrix color scale. Default is -0.75.

        vmax : float
            Maximum value for the interaction matrix color scale. Default is 0.75.
        
        cmap : str
            Colormap to use for the interaction matrix. Default is 'PRGn'.

        fname : str
            Filename to save the figure to. If None, the figure will be displayed

        disorder_1 : bool
            Whether to include the disorder profile for sequence 1. Default is True.

        disorder_2 : bool
            Whether to include the disorder profile for sequence 2. Default is True.


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
        
        # call the superclass function
        return super().interaction_figure(seq1,
                                          seq2,
                                          window_size=window_size,
                                          use_cython=use_cython,
                                          use_aliphatic_weighting=use_aliphatic_weighting,
                                          use_charge_weighting=use_charge_weighting,
                                          tic_frequency=tic_frequency,
                                          seq1_domains=seq1_domains,
                                          seq2_domains=seq2_domains,
                                          seq1_lines=seq1_lines,
                                          seq2_lines=seq2_lines,                                          
                                          vmin=vmin,
                                          vmax=vmax,
                                          cmap=cmap,
                                          fname=fname,
                                          zero_folded=zero_folded,
                                          disorder_1=True,
                                          disorder_2=True)
    

    
    # ....................................................................................
    #
    #        
    def protein_nucleic_vector(seq, fragsize=31, smoothing_window=30, poly_order=3):
        """
        Stub function to calculate the protein-nucleic acid interaction vector. CALVADOS
        does not currently support RNA.
        """
        raise Exception('CALVADOS cannot currently handle RNA')
