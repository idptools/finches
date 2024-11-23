# other stuff
import numpy as np

from finches import epsilon_calculation

# for model construction
from finches.forcefields.mpipi import Mpipi_model

from .frontend_base import FinchesFrontend


class Mpipi_frontend(FinchesFrontend):
    def __init__(self, salt=0.150, dielectric=80.0):
        # call superclass constructor
        super().__init__()

        # initialize an Mpipi forcefield opbject
        self.model = Mpipi_model("Mpipi_GGv1", salt=salt, dielectric=dielectric)

        # build an interaction matrix constructor object
        self.IMC_object = epsilon_calculation.InteractionMatrixConstructor(self.model)

    # functions defined in superclass listed below for clarity
    # epslion() define in super exclusively
    # per_residue_attractive_vector() defined in super exclusively

    def intermolecular_idr_matrix(
        self,
        seq1,
        seq2,
        window_size=31,
        use_cython=True,
        use_aliphatic_weighting=True,
        use_charge_weighting=True,
        disorder_1=None,
        disorder_2=None,
        null_shuffle=False):    
        """
        Returns the interaction matrix for the two sequences. Specifically this involves
        decomposing the two sequences into window_size fragments and calculating the inter-fragment
        epsilon values using a sliding window approach.

        Note that we don't pad the sequence here, so the edges of the matrix start
        and end at indices that depend on the window size. To avoid confusion, the
        function also returns the indices for sequence1 and sequence2.

        If sequence 1 or sequence 2 contain 'U', then the disorder profile is not generated for
        that sequence.

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

        null_shuffle : bool
            Whether to shuffle the sequence before calculating the interaction matrix. Default
            is False. If set to a number defines the number of shuffles used for each sequence;
            recommended to use 100 shuffles.

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

        if "U" in seq1:
            disorder_1 = False
        elif disorder_1 is None:
            disorder_1 = True

        if "U" in seq2:
            disorder_2 = False
        elif disorder_2 is None:
            disorder_2 = True

        # call the superclass function
        return super().intermolecular_idr_matrix(
            seq1,
            seq2,
            window_size=window_size,
            use_cython=use_cython,
            use_aliphatic_weighting=use_aliphatic_weighting,
            use_charge_weighting=use_charge_weighting,
            disorder_1=disorder_1,
            disorder_2=disorder_2,
            null_shuffle=null_shuffle
        )

    def interaction_figure(
        self,
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
        linewidth=1,
        vmin=-3,
        vmax=3,
        cmap="PRGn",
        fname=None,
        zero_folded=True,
        no_disorder=False,
        null_shuffle=False,
        plot_rectangles=None):
        
        """
        Function to generate an interaction matrix figure between two sequences. This does
        all the calculation on the backend and formats a figure with parallel disorder tracks
        alongside the interaction matrix.

        If sequence 1 or sequence 2 contain 'U', then the disorder profile is not generated for
        that sequence.

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
            Minimum value for the interaction matrix color scale. Default is -3.

        vmax : float
            Maximum value for the interaction matrix color scale. Default is 3.

        cmap : str
            Colormap to use for the interaction matrix. Default is 'PRGn'.

        fname : str
            Filename to save the figure to. If None, the figure will be displayed

        disorder_1 : bool
            Whether to include the disorder profile for sequence 1. Default is True.

        disorder_2 : bool
            Whether to include the disorder profile for sequence 2. Default is True.

        no_disorder : bool
            Whether to include the disorder profiles. Default is False. If True, the disorder
            profiles will not be included.

        null_shuffle : bool
            Whether to shuffle the sequence before calculating the interaction matrix. Default
            is False. If set to a number defines the number of shuffles used for each sequence;
            recommended to use 100 shuffles.

        plot_rectangles : list
            If a list is provided it should be a list of lists, where each sublist has the
            folowing information [seq1_start, seq1_end, seq2_start, seq2_end, color, alpha, kwargs].
            Based on this information, rectangles will be drawn on the plot to highlight
            specific regions. Default is None.

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

        # Mpipi can accomdate RNA as polyU only
        if seq1.find("U") == -1:
            disorder_1 = True
        else:
            disorder_1 = False

        if seq2.find("U") == -1:
            disorder_2 = True
        else:
            disorder_2 = False

        # call the superclass function
        return super().interaction_figure(
            seq1,
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
            linewidth=linewidth,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            fname=fname,
            zero_folded=zero_folded,
            disorder_1=disorder_1,
            disorder_2=disorder_2,
            no_disorder=no_disorder,
            null_shuffle=null_shuffle,
            plot_rectangles=plot_rectangles)
        
