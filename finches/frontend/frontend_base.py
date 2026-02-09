import metapredict as meta

from finches import epsilon_to_FHtheory
from finches import epsilon_stateless
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
import functools


# ensure text is editable in illustrator
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42

# set to define axes linewidths
#matplotlib.rcParams['axes.linewidth'] = 0.5

# Decorator to apply consistent plot styles
import matplotlib
import functools

def apply_publication_styles(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Font settings
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
        
        # PDF/PS settings
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        # Line settings
        matplotlib.rcParams['axes.linewidth'] = 0.5

        # X-Ticks
        matplotlib.rcParams['xtick.major.width'] = 0.5 # Match axis width
        matplotlib.rcParams['xtick.minor.width'] = 0.25 # Match axis width
        matplotlib.rcParams['xtick.minor.size'] = 0.5  # Length of the tick
        matplotlib.rcParams['xtick.major.size'] = 2.5  # Length of the tick
        matplotlib.rcParams['xtick.direction'] = 'out' # Ticks point outside
        
        # Y-Ticks
        matplotlib.rcParams['ytick.major.width'] = 0.5
        matplotlib.rcParams['ytick.minor.width'] = 0.25 # Match axis width
        matplotlib.rcParams['ytick.minor.size'] = 0.5
        matplotlib.rcParams['ytick.major.size'] = 2.5
        matplotlib.rcParams['ytick.direction'] = 'out'        
        
        return func(*args, **kwargs)
    return wrapper

class FinchesFrontend:
    """
    Base class for FINCHES frontend interfaces.
    
    This class provides a unified interface for calculating protein-protein
    and protein-RNA interactions using coarse-grained forcefields. It should
    not be instantiated directly; instead, use one of the derived classes:
    
    - **Mpipi_frontend**: Uses the Mpipi forcefield (supports protein-RNA)
    - **CALVADOS_frontend**: Uses the CALVADOS forcefield (protein only)
    
    The frontend classes provide high-level methods for:
    
    - Computing interaction parameters (epsilon) between sequences
    - Generating interaction matrices (spatially resolved)
    - Building phase diagrams
    - Identifying sticker/spacer regions
    - Analyzing RNA-binding propensity
    - Deep mutational scanning (DMS)
    
    All methods handle the underlying forcefield calculations and provide
    publication-ready visualization options.

    Attributes
    ----------
    IMC_object : InteractionMatrixConstructor
        The underlying interaction matrix constructor that performs
        the actual calculations. Set by derived classes.

    Examples
    --------
    Use via a derived class::\n
        from finches.frontend.mpipi_frontend import Mpipi_frontend
        from finches.frontend.calvados_frontend import CALVADOS_frontend
        
        # For protein-protein and protein-RNA analysis
        mf = Mpipi_frontend()
        
        # For protein-protein analysis only
        cf = CALVADOS_frontend()
        
        # Calculate homotypic epsilon
        seq = \"MSKGEELFTGVVPILVELDGDVNGHKFSVS\"
        eps = mf.epsilon(seq, seq)
        
        # Generate interaction matrix figure
        fig, im, ax, *_ = mf.interaction_figure(seq, seq)
        
        # Build phase diagram
        phase_data = mf.build_phase_diagram(seq)

    Note
    ----
    Attempting to instantiate FinchesFrontend directly will raise a TypeError.
    Always use the appropriate derived class for your forcefield of choice.

    See Also
    --------
    finches.frontend.mpipi_frontend.Mpipi_frontend : Mpipi forcefield frontend.
    finches.frontend.calvados_frontend.CALVADOS_frontend : CALVADOS forcefield frontend.

    """

    def __init__(self):

        # ensure we don't accidentally instantiate this class!
        if type(self) == FinchesFrontend:
            raise TypeError("FinchesFrontend class should not be instantiated directly, but instead derived classes should be used.")
        # this must be defined in the subclass
        self.IMC_object = None


    # ....................................................................................
    #
    #        
    def intermolecular_idr_matrix(self,
                                  seq1,
                                  seq2,
                                  window_size=31,
                                  use_cython=True,
                                  use_aliphatic_weighting=True,
                                  use_charge_weighting=True,
                                  disorder_1=True,
                                  disorder_2=True,
                                  null_shuffle=False):

        """
        Compute the interaction matrix between two sequences using a sliding window approach.
        
        This function calculates pairwise epsilon (interaction) values between all
        window-sized fragments of two sequences. The resulting matrix shows how different
        regions of the two proteins interact with each other, which is useful for
        identifying interaction hotspots or analyzing domain-domain interactions.

        The matrix dimensions depend on the sequence lengths and window size:
        - Matrix rows correspond to positions in seq1
        - Matrix columns correspond to positions in seq2
        - Edge positions are trimmed based on window_size (half-window on each side)

        Parameters
        ----------
        seq1 : str
            First amino acid sequence.

        seq2 : str
            Second amino acid sequence. Can be the same as seq1 for homotypic analysis.

        window_size : int, optional
            Size of the sliding window for fragment extraction. Must be odd.
            Larger windows provide more context but reduce resolution.
            Default is 31.

        use_cython : bool, optional
            If True, use the faster Cython implementation. Highly recommended.
            Default is True.

        use_aliphatic_weighting : bool, optional
            If True, apply weighting to aliphatic residues based on local
            aliphatic context (adjacent aliphatic residues enhance contribution).
            Default is True.

        use_charge_weighting : bool, optional
            If True, apply weighting to charged residues based on local
            charge context (adjacent same-sign charges enhance contribution).
            Default is True.

        disorder_1 : bool, optional
            If True, compute disorder profile for seq1 using metapredict.
            If False, use uniform values (all 1s). Default is True.

        disorder_2 : bool, optional
            If True, compute disorder profile for seq2 using metapredict.
            If False, use uniform values (all 1s). Default is True.

        null_shuffle : bool or int, optional
            If False, return raw interaction matrix.
            If an integer, perform that many shuffled controls and subtract
            the mean shuffled matrix from the raw matrix to get a 
            sequence-specific signal. Recommended value: 100.
            Default is False.

        Returns
        -------
        tuple
            A 3-element tuple containing:
            
            [0] : tuple (matrix, idx1, idx2)
                - matrix : np.ndarray
                    2D array of epsilon values. Shape is (len1, len2) where
                    len1 and len2 are the trimmed sequence lengths.
                    Negative values = attractive, Positive values = repulsive.
                - idx1 : np.ndarray
                    1-indexed positions in seq1 corresponding to matrix rows.
                - idx2 : np.ndarray
                    1-indexed positions in seq2 corresponding to matrix columns.
            
            [1] : np.ndarray
                Disorder profile for seq1 (values 0-1, where 1 = disordered).
                Array of 1s if disorder_1=False.
            
            [2] : np.ndarray
                Disorder profile for seq2 (values 0-1, where 1 = disordered).
                Array of 1s if disorder_2=False.

        Examples
        --------
        Basic usage for two different sequences::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            
            mf = Mpipi_frontend()
            seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
            seq2 = "EKEKEKEKEKEKEKEKEKEK"
            
            result, disorder1, disorder2 = mf.intermolecular_idr_matrix(seq1, seq2)
            
            # Unpack the matrix tuple
            matrix, idx1, idx2 = result
            
            # matrix[i, j] gives the epsilon between position idx1[i] of seq1
            # and position idx2[j] of seq2
            print(f"Matrix shape: {matrix.shape}")
            print(f"Seq1 positions: {idx1[0]} to {idx1[-1]}")
            print(f"Seq2 positions: {idx2[0]} to {idx2[-1]}")

        Homotypic (self-interaction) analysis::

            seq = "FYWFYWFYWFYWFYWFYWFY"
            result, dis1, dis2 = mf.intermolecular_idr_matrix(seq, seq)
            matrix = result[0]
            
            # For homotypic, matrix is symmetric
            # Diagonal represents self-interaction of each region

        With null shuffling to get sequence-specific signal::

            # Subtract shuffled background (slower but more specific)
            result, dis1, dis2 = mf.intermolecular_idr_matrix(
                seq1, seq2, 
                null_shuffle=100  # 100 shuffled controls
            )
            # Resulting matrix shows sequence-specific interactions
            # relative to composition-matched random sequences

        See Also
        --------
        interaction_figure : Visualize the interaction matrix as a heatmap.
        epsilon : Get a single epsilon value for two full sequences.

        """                 
        
        # compute the matrix
        B = self.IMC_object.calculate_sliding_epsilon(seq1,
                                                      seq2,
                                                      window_size=window_size,
                                                      use_cython=use_cython,
                                                      use_aliphatic_weighting=use_aliphatic_weighting,
                                                      use_charge_weighting=use_charge_weighting)
        
        # if we're shuffling the sequence
        if null_shuffle is not False:

            # sanity check if input is provided for null_shuffle; note we explicitly include
            # bool because bool is a subclass of int and we don't want to allow that to avoid
            # any potential confusion.
            if not isinstance(null_shuffle, (int, float)) or isinstance(null_shuffle, bool):
                raise ValueError('null_shuffle must be an integer or float')

            # initialize for storing 
            shuffle_matrices = []

            # define an iterator so we print progress if this is gonna take a while 
            if null_shuffle > 50:
                local_iterator = tqdm(range(null_shuffle))
            else:
                local_iterator = range(null_shuffle)

            
            # cycle through the number of shuffles
            for i in local_iterator:

                # nb for homotypic we ensure the shuffles remain homotypic
                seq_1_shuffled = ''.join(np.random.permutation(list(seq1)))
                seq_2_shuffled = ''.join(np.random.permutation(list(seq2)))

                B_tmp = self.IMC_object.calculate_sliding_epsilon(seq_1_shuffled,
                                                                  seq_2_shuffled,
                                                                  window_size=window_size,
                                                                  use_cython=use_cython,
                                                                  use_aliphatic_weighting=use_aliphatic_weighting,
                                                                  use_charge_weighting=use_charge_weighting)
                shuffle_matrices.append(B_tmp[0])

            if seq1 == seq2:
                # note - we do this to symmetrize the null matrix without introducing artifacts by 
                # enforcing shuffled sequences to always be the same
                new_B_0 = B[0] - (np.mean(shuffle_matrices, axis=0) + np.mean(np.array(shuffle_matrices).swapaxes(1,2), axis=0))/2

            else:
                # overwrite the original matrix with the shuffle-normalized matrix
                new_B_0 = B[0] - np.mean(shuffle_matrices, axis=0)
            #new_B_0 = np.mean(shuffle_matrices, axis=0)
            new_B_1 = B[1]
            new_B_2 = B[2]
            B = (new_B_0, new_B_1, new_B_2)

        # compute disorder profile for sequence 1 assuming disorder_1 is set to True
        if disorder_1:
            start_python_slice = B[1][0] - 1
            end_python_slice   = B[1][-1] # no -1 because python slices are not inclusive
            disorder_1 = meta.predict_disorder(seq1)[start_python_slice:end_python_slice]
        else:
            disorder_1 = np.array([1]*B[0].shape[0])

        # compute disorder profile for sequence 1 assuming disorder_2 is set to True
        if disorder_2:
            start_python_slice = B[2][0] - 1 # -1 to move into python indexing
            end_python_slice   = B[2][-1] # no -1 because python slices are not inclusive            
            disorder_2 = meta.predict_disorder(seq2)[start_python_slice:end_python_slice]
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
    def epsilon(self,
                seq1,
                seq2,
                use_aliphatic_weighting=True,
                use_charge_weighting=True):
        """
        Calculate the mean-field interaction parameter (epsilon) between two sequences.
        
        Epsilon quantifies the overall interaction strength between two protein sequences.
        It is computed as the sum of all pairwise residue-residue interactions, weighted
        by local sequence context (charge and aliphatic patterns).

        Interpretation of epsilon values:
        
        - **Negative epsilon**: Net attractive interaction (favorable for interaction)
        - **Positive epsilon**: Net repulsive interaction (unfavorable for interaction)
        - **More negative**: Stronger attraction (more likely to phase separate)
        - **Near zero**: Weak or balanced interactions

        Parameters
        ----------
        seq1 : str
            First amino acid sequence.

        seq2 : str
            Second amino acid sequence. Use seq1=seq2 for homotypic epsilon.

        use_aliphatic_weighting : bool, optional
            If True, weight aliphatic residues by their local aliphatic context.
            Adjacent aliphatic residues enhance the contribution. Default is True.

        use_charge_weighting : bool, optional
            If True, weight charged residues by their local charge context.
            Adjacent like-charges enhance the contribution. Default is True.

        Returns
        -------
        float
            The epsilon value. Negative = attractive, Positive = repulsive.
            Typical range is approximately -10 to +5 for most sequences.

        Examples
        --------
        Calculate homotypic (self-interaction) epsilon::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            
            mf = Mpipi_frontend()
            seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
            
            eps = mf.epsilon(seq, seq)
            print(f"Homotypic epsilon: {eps:.3f}")
            
            if eps < 0:
                print("Sequence is self-attractive (may phase separate)")
            else:
                print("Sequence is self-repulsive (unlikely to phase separate)")

        Compare interactions between different sequences::

            seq_a = "FYWFYWFYWFYWFYWFYWFY"  # Aromatic-rich
            seq_b = "EKEKEKEKEKEKEKEKEKEK"  # Charged
            
            # Homotypic interactions
            eps_aa = mf.epsilon(seq_a, seq_a)
            eps_bb = mf.epsilon(seq_b, seq_b)
            
            # Heterotypic interaction
            eps_ab = mf.epsilon(seq_a, seq_b)
            
            print(f"A-A: {eps_aa:.3f}, B-B: {eps_bb:.3f}, A-B: {eps_ab:.3f}")

        Effect of weighting schemes::

            # Compare with and without weighting
            eps_weighted = mf.epsilon(seq, seq)
            eps_unweighted = mf.epsilon(seq, seq, 
                                        use_aliphatic_weighting=False,
                                        use_charge_weighting=False)
            print(f"Weighted: {eps_weighted:.3f}")
            print(f"Unweighted: {eps_unweighted:.3f}")

        See Also
        --------
        epsilon_vectors : Get the attractive and repulsive components separately.
        build_phase_diagram : Use epsilon to construct a phase diagram.

        """        
        return self.IMC_object.calculate_epsilon_value(seq1,
                                                       seq2,
                                                       use_aliphatic_weighting=use_aliphatic_weighting,
                                                       use_charge_weighting=use_charge_weighting)

    # ....................................................................................
    #
    #            
    def epsilon_vectors(self,
                seq1,
                seq2,
                use_aliphatic_weighting=True,
                use_charge_weighting=True):
        """
        Calculate the per-residue attractive and repulsive interaction vectors.
        
        This function decomposes the epsilon calculation into two components:
        
        1. **Attractive vector**: Per-residue sum of attractive (negative) interactions
        2. **Repulsive vector**: Per-residue sum of repulsive (positive) interactions
        
        These vectors show how each residue in seq1 contributes to attraction or
        repulsion when interacting with seq2. This is useful for identifying which
        residues are "stickers" (drive attraction) vs. "spacers" (neutral or repulsive).

        Parameters
        ----------
        seq1 : str
            First amino acid sequence. The returned vectors correspond to
            positions in this sequence.

        seq2 : str
            Second amino acid sequence.

        use_aliphatic_weighting : bool, optional
            If True, weight aliphatic residues by local aliphatic context.
            Default is True.

        use_charge_weighting : bool, optional
            If True, weight charged residues by local charge context.
            Default is True.

        Returns
        -------
        tuple of np.ndarray
            Two arrays, each of length len(seq1):
            
            [0] : np.ndarray
                Attractive vector. Each element is the sum of attractive (negative)
                interactions for that residue position. More negative = stronger
                attraction contributed by that residue.
            
            [1] : np.ndarray
                Repulsive vector. Each element is the sum of repulsive (positive)
                interactions for that residue position. More positive = stronger
                repulsion contributed by that residue.

        Examples
        --------
        Identify sticker and spacer residues::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            import numpy as np
            
            mf = Mpipi_frontend()
            seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
            
            attractive, repulsive = mf.epsilon_vectors(seq, seq)
            
            # Find the most attractive positions (stickers)
            sticker_positions = np.argsort(attractive)[:5]  # Top 5 most negative
            print("Top sticker positions (1-indexed):")
            for pos in sticker_positions:
                print(f"  Position {pos+1} ({seq[pos]}): {attractive[pos]:.3f}")
            
            # Find the most repulsive positions (spacers)
            spacer_positions = np.argsort(repulsive)[-5:]  # Top 5 most positive
            print("Top spacer positions (1-indexed):")
            for pos in spacer_positions:
                print(f"  Position {pos+1} ({seq[pos]}): {repulsive[pos]:.3f}")

        Plot the interaction profile::

            import matplotlib.pyplot as plt
            
            attractive, repulsive = mf.epsilon_vectors(seq, seq)
            positions = np.arange(1, len(seq) + 1)
            
            plt.figure(figsize=(10, 4))
            plt.bar(positions, attractive, alpha=0.7, label='Attractive', color='green')
            plt.bar(positions, repulsive, alpha=0.7, label='Repulsive', color='purple')
            plt.xlabel('Residue Position')
            plt.ylabel('Interaction Strength')
            plt.legend()
            plt.axhline(0, color='black', linewidth=0.5)
            plt.show()

        Note
        ----
        The total epsilon equals sum(attractive) + sum(repulsive).

        See Also
        --------
        epsilon : Get the single combined epsilon value.
        per_residue_attractive_vector : Get smoothed per-residue attractive profile.

        """        
        return self.IMC_object.calculate_epsilon_vectors(seq1,
                                                         seq2,
                                                         use_aliphatic_weighting=use_aliphatic_weighting,
                                                         use_charge_weighting=use_charge_weighting)

    
    # ....................................................................................
    #
    #          
    @apply_publication_styles  
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
                           linewidth=1,
                           vmin=-3,
                           vmax=3,
                           cmap='PRGn',
                           fname=None,
                           zero_folded=True,
                           disorder_1=True,
                           disorder_2=True,
                           no_disorder=False,
                           null_shuffle=False,
                           plot_rectangles=None):
    
        """
        Generate a publication-ready interaction matrix heatmap between two sequences.
        
        This function creates a comprehensive figure showing:
        
        1. **Main panel**: Heatmap of pairwise epsilon values between sequence fragments
        2. **Top panel**: Disorder profile for sequence 1
        3. **Right panel**: Disorder profile for sequence 2
        4. **Colorbar**: Scale for interpreting epsilon values
        
        Green/negative values indicate attractive interactions, while purple/positive
        values indicate repulsive interactions (with default PRGn colormap).

        Parameters
        ----------
        seq1 : str
            First amino acid sequence (displayed on x-axis).

        seq2 : str
            Second amino acid sequence (displayed on y-axis).

        window_size : int, optional
            Size of sliding window for fragment comparison. Must be odd.
            Default is 31.

        use_cython : bool, optional
            Use faster Cython implementation. Default is True.

        use_aliphatic_weighting : bool, optional
            Apply aliphatic context weighting. Default is True.

        use_charge_weighting : bool, optional
            Apply charge context weighting. Default is True.

        tic_frequency : int, optional
            Spacing between axis tick labels. Default is 100.

        seq1_domains : list, optional
            List of [start, end] pairs marking domains in seq1 to highlight
            on the disorder track. Default is [].

        seq2_domains : list, optional
            List of [start, end] pairs marking domains in seq2 to highlight
            on the disorder track. Default is [].

        seq1_lines : list, optional
            List of positions to draw vertical lines on the matrix.
            Useful for marking domain boundaries. Default is [].

        seq2_lines : list, optional
            List of positions to draw horizontal lines on the matrix.
            Default is [].

        linewidth : float, optional
            Width of domain boundary lines. Default is 1.

        vmin : float, optional
            Minimum value for color scale. Default is -3.

        vmax : float, optional
            Maximum value for color scale. Default is 3.

        cmap : str, optional
            Matplotlib colormap name. Default is 'PRGn' (purple-green diverging).

        fname : str, optional
            If provided, save figure to this filepath. Default is None (display only).

        zero_folded : bool, optional
            If True, set epsilon=0 for regions predicted to be folded.
            Helps focus on IDR-IDR interactions. Default is True.

        disorder_1 : bool, optional
            Show disorder profile for seq1. Default is True.

        disorder_2 : bool, optional
            Show disorder profile for seq2. Default is True.

        no_disorder : bool, optional
            If True, omit disorder panels entirely. Default is False.

        null_shuffle : bool or int, optional
            If an integer, subtract shuffled background (that many shuffles).
            Default is False.

        plot_rectangles : list, optional
            List of rectangle specifications to highlight regions:
            [[x1_start, x1_end, x2_start, x2_end, kwargs_dict], ...]
            where kwargs_dict contains matplotlib Rectangle parameters.
            Default is None.

        Returns
        -------
        tuple
            If no_disorder=False (default):
                (fig, im, ax_main, ax_top, ax_right, ax_colorbar)
            If no_disorder=True:
                (fig, im, ax_main)
            
            - fig : matplotlib.figure.Figure
            - im : matplotlib.image.AxesImage (the heatmap)
            - ax_main : matplotlib.axes.Axes (main heatmap axis)
            - ax_top : matplotlib.axes.Axes (seq1 disorder track)
            - ax_right : matplotlib.axes.Axes (seq2 disorder track)
            - ax_colorbar : matplotlib.axes.Axes (colorbar)

        Examples
        --------
        Basic usage::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            
            mf = Mpipi_frontend()
            seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS" * 3  # ~90 residues
            seq2 = "EKEKEKEKEKEKEKEKEKEK" * 3
            
            mf.interaction_figure(seq1, seq2
            

        Homotypic analysis with domain annotations::

            seq = "MSKGEELFT" * 10  # 90 residues
            
            # Mark a domain boundary at position 45
            x = mf.interaction_figure(
                seq, seq,
                seq1_lines=[45],
                seq2_lines=[45],
                seq1_domains=[[1, 30]],  # Highlight first 30 residues
                vmin=-2,
                vmax=2,
                fname='homotypic_matrix.png'
            )

        With shuffled background subtraction::

            x = mf.interaction_figure(
                seq1, seq2,
                null_shuffle=100,  # 100 shuffled controls
                vmin=-1,
                vmax=1
            )

        Minimal figure without disorder tracks::

            x = mf.interaction_figure(
                seq1, seq2,
                no_disorder=True
            )

        See Also
        --------
        intermolecular_idr_matrix : Get the raw matrix data without plotting.
        per_residue_attractive_vector : Summarize attractive interactions per residue.

        """

        # note - when this is called from a derived class it's the DERIVED CLASS
        # version of the intermolecular_idr_matrix function that's called, so RNA
        # is handled in this correctly
        B, disorder_1, disorder_2 = self.intermolecular_idr_matrix(seq1,
                                                                   seq2,
                                                                   window_size=window_size,
                                                                   use_cython=use_cython,
                                                                   use_aliphatic_weighting=use_aliphatic_weighting,
                                                                   use_charge_weighting=use_charge_weighting,
                                                                   null_shuffle=null_shuffle)


        if zero_folded:        
            try:
                folded_1 = meta.predict_disorder_domains(seq1).folded_domain_boundaries
                folded_2 = meta.predict_disorder_domains(seq2).folded_domain_boundaries

            except Exception as e:
                folded_1 = []
                folded_2 = []

        else:
            folded_1 = []
            folded_2 = []

        B1_start = B[1][0]
        B2_start = B[2][0]

            
        for i in range(B[1][0]-1, B[1][-1]):
            for j in range(B[2][0]-1, B[2][-1]):
                
                # if we are zeroing out folded regions, do that here
                for fd in folded_1:
                    if i >= fd[0] and i <= fd[1]:
                        
                        B[0][i - B1_start, j - B2_start ] = 0

                for fd in folded_2:
                    if j >= fd[0] and j <= fd[1]:
                        B[0][i - B1_start, j - B2_start ] = 0


        # extract out the interaction matrix
        matrix = B[0]
        
        # Create a figure and a grid of subplots; note we can tweak the figure size params to 
        # make the figure look good depending on how long the two sequences are, here 5.5 and 5
        # works but this is adjustable ofc.
        fig = plt.figure(figsize=(8.5, 8.5), dpi=350)
    
        # Main matrix plot; create axis and then plot using seismic so 0=white
        ax_main = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)

        # note we need that +1
        im = ax_main.imshow(matrix.T, extent=[B[1][0], B[1][-1]+1, B[2][0], B[2][-1]+1], origin='lower', aspect='auto', vmax=vmax, vmin=vmin, cmap=cmap)
            
        # edit here to change tickmarks;  note again the tic_frequency, this again
        # probably can be edited manually depending on the system       
        ax_main.set_xticks(np.arange(B[1][0],B[1][-1]+1, tic_frequency))
        ax_main.set_yticks(np.arange(B[2][0],B[2][-1]+1, tic_frequency))
        ax_main.tick_params(axis='x', rotation=45)  # Rotates the x-tick labels by 45 degrees


        # add any lines onto the main axis as vertical (seq1) or horizontal (seq2) lines
        for line in seq1_lines:
            ax_main.axvline(line, color='k', linewidth=linewidth)
    
        for line in seq2_lines:
            ax_main.axhline(line, color='k', linewidth=linewidth)


        if no_disorder:
            pass
        else:
    
            ## .....................................................................
            # Bar plot for X protein 
            ax_top = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=ax_main)
    
            # plot disorder profile 
            ax_top.bar(B[1], disorder_1, width=1, color='k', alpha=0.3)
    
            # if we wanted to plot mean per-residue inteaction value, uncomment
            ax_top.set_xlim(B[1][0], B[1][-1])

            # disorder goes 0 to 1
            ax_top.set_ylim(0,1.05)
    
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
            
            # disorder goes 0 to 1
            ax_right.set_xlim(0,1.05)
    
            # highlight some specific regions manually
    
    
            for r in seq2_domains:
                region_start = r[0]
                region_end   = r[1]
                ax_right.axhspan(region_start,region_end, color='k', linewidth=0, alpha=0.7)
    
            ax_colorbar = plt.subplot2grid((4, 4), (0, 3))
            cbar = fig.colorbar(im, cax=ax_colorbar, orientation='vertical')
            ax_colorbar.yaxis.set_ticks_position('left')
            ax_colorbar.yaxis.set_label_position('left')

        if plot_rectangles is not None:
            for r in plot_rectangles:
                region_start_1 = r[0]
                region_end_1   = r[1]
                region_start_2 = r[2]
                region_end_2   = r[3]                
                kwargs = r[4]
                ax_main.add_patch(matplotlib.patches.Rectangle((region_start_1, region_start_2), 
                                                               region_end_1-region_start_1, 
                                                               region_end_2-region_start_2,                                                                
                                                               linewidth=2,                                                               
                                                               facecolor='none',
                                                               **kwargs))
                                                               
        # add this to ensure the main axes do not extend beyond the data limits
        ax_main.set_xlim(B[1][0], B[1][-1]+1)
        ax_main.set_ylim(B[2][0], B[2][-1]+1)
        plt.tight_layout()
        
        # finally save the figure
        if fname is not None:
            plt.savefig(fname, dpi=350)

        if no_disorder:
            return fig, im, ax_main
        else:
            return fig, im, ax_main, ax_top, ax_right, ax_colorbar


    # ....................................................................................
    #
    #
    def per_residue_attractive_vector(self,
                                      seq1,
                                      seq2,
                                      window_size=31,
                                      use_cython=True,
                                      use_aliphatic_weighting=True,
                                      use_charge_weighting=True,                                      
                                      return_total=False,
                                      attractive_threshold=0,
                                      smoothing_window=20,
                                      poly_order=3):
        
        """
        Calculate the per-residue average of attractive interactions from the interaction matrix.
        
        For each position in seq1, this function computes the mean of all attractive
        (negative) epsilon values from the interaction matrix with seq2. This identifies
        which regions of seq1 serve as "stickers" - positions that drive favorable
        interactions regardless of local repulsive contributions.

        The key insight is that when analyzing IDR interactions:
        
        - **Repulsive regions can be avoided** through conformational flexibility
        - **Attractive regions will find each other** and drive association
        - This function isolates the attractive contribution per residue

        Parameters
        ----------
        seq1 : str
            First amino acid sequence. The output vector corresponds to positions
            in this sequence.

        seq2 : str
            Second amino acid sequence.

        window_size : int, optional
            Window size for the interaction matrix. Default is 31.

        use_cython : bool, optional
            Use Cython implementation. Default is True.

        use_aliphatic_weighting : bool, optional
            Apply aliphatic weighting. Default is True.

        use_charge_weighting : bool, optional
            Apply charge weighting. Default is True.

        return_total : bool, optional
            If True, return the sum instead of the average of attractive values.
            Default is False.

        attractive_threshold : float, optional
            Values below this threshold are considered attractive.
            Default is 0 (only negative values).

        smoothing_window : int or False, optional
            Window size for Savitzky-Golay smoothing filter.
            Set to False to disable smoothing. Default is 20.

        poly_order : int or False, optional
            Polynomial order for Savitzky-Golay filter.
            Set to False to disable smoothing. Default is 3.

        Returns
        -------
        tuple of np.ndarray
            [0] : np.ndarray
                1-indexed position indices for seq1.
            
            [1] : np.ndarray
                Per-residue attractive values (smoothed by default).
                More negative = stronger average attraction at that position.

        Examples
        --------
        Identify sticker regions in a sequence::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            import matplotlib.pyplot as plt
            
            mf = Mpipi_frontend()
            seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS" * 3
            
            # Get per-residue attractive profile
            positions, attractive = mf.per_residue_attractive_vector(seq, seq)
            
            # Plot the sticker profile
            plt.figure(figsize=(10, 3))
            plt.plot(positions, attractive, 'g-', lw=1)
            plt.fill_between(positions, attractive, 0, 
                            where=(attractive < 0), alpha=0.3, color='green')
            plt.xlabel('Residue Position')
            plt.ylabel('Average Attractive Interaction')
            plt.axhline(0, color='black', lw=0.5)
            plt.title('Sticker Profile')
            plt.show()

        Compare homotypic vs heterotypic attraction::

            seq_a = "FYWFYWFYWFYWFYWFYWFY" * 2
            seq_b = "EKEKEKEKEKEKEKEKEKEK" * 2
            
            # A-A homotypic stickers
            pos_aa, attr_aa = mf.per_residue_attractive_vector(seq_a, seq_a)
            
            # A-B heterotypic stickers (which residues in A attract B)
            pos_ab, attr_ab = mf.per_residue_attractive_vector(seq_a, seq_b)
            
            plt.plot(pos_aa, attr_aa, label='A-A (homotypic)')
            plt.plot(pos_ab, attr_ab, label='A-B (heterotypic)')
            plt.legend()

        Without smoothing::

            positions, attractive = mf.per_residue_attractive_vector(
                seq, seq,
                smoothing_window=False,  # Disable smoothing
                poly_order=False
            )

        See Also
        --------
        per_residue_repulsive_vector : Get the repulsive component.
        intermolecular_idr_matrix : Get the full interaction matrix.

        """

        # do the thing; note this class from the derived class so model-specific
        # sanity checking is handled implicitly here
        B = self.intermolecular_idr_matrix(seq1, seq2, window_size=window_size, use_cython=use_cython, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)[0]

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

        # calculate the average of attractive values in each column
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
    def per_residue_repulsive_vector(self,
                                      seq1,
                                      seq2,
                                      window_size=31,
                                      use_cython=True,
                                      use_aliphatic_weighting=True,
                                      use_charge_weighting=True,                                      
                                      return_total=False,
                                      repulsive_threshold=0,
                                      smoothing_window=20,
                                      poly_order=3):
        
        """
        Calculate the per-residue average of repulsive interactions from the interaction matrix.
        
        For each position in seq1, this function computes the mean of all repulsive
        (positive) epsilon values from the interaction matrix with seq2. This identifies
        which regions of seq1 serve as "spacers" - positions that contribute unfavorable
        interactions and may help maintain solubility or prevent aggregation.

        This is the counterpart to `per_residue_attractive_vector`, focusing on
        repulsive rather than attractive contributions.

        Parameters
        ----------
        seq1 : str
            First amino acid sequence. The output vector corresponds to positions
            in this sequence.

        seq2 : str
            Second amino acid sequence.

        window_size : int, optional
            Window size for the interaction matrix. Default is 31.

        use_cython : bool, optional
            Use Cython implementation. Default is True.

        use_aliphatic_weighting : bool, optional
            Apply aliphatic weighting. Default is True.

        use_charge_weighting : bool, optional
            Apply charge weighting. Default is True.

        return_total : bool, optional
            If True, return the sum instead of the average of repulsive values.
            Default is False.

        repulsive_threshold : float, optional
            Values above this threshold are considered repulsive.
            Default is 0 (only positive values).

        smoothing_window : int or False, optional
            Window size for Savitzky-Golay smoothing filter.
            Set to False to disable smoothing. Default is 20.

        poly_order : int or False, optional
            Polynomial order for Savitzky-Golay filter.
            Set to False to disable smoothing. Default is 3.

        Returns
        -------
        tuple of np.ndarray
            [0] : np.ndarray
                1-indexed position indices for seq1.
            
            [1] : np.ndarray
                Per-residue repulsive values (smoothed by default).
                More positive = stronger average repulsion at that position.

        Examples
        --------
        Identify spacer regions in a sequence::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            import matplotlib.pyplot as plt
            
            mf = Mpipi_frontend()
            seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS" * 3
            
            # Get per-residue repulsive profile
            positions, repulsive = mf.per_residue_repulsive_vector(seq, seq)
            
            # Plot the spacer profile
            plt.figure(figsize=(10, 3))
            plt.plot(positions, repulsive, 'purple', lw=1)
            plt.fill_between(positions, 0, repulsive, 
                            where=(repulsive > 0), alpha=0.3, color='purple')
            plt.xlabel('Residue Position')
            plt.ylabel('Average Repulsive Interaction')
            plt.axhline(0, color='black', lw=0.5)
            plt.title('Spacer Profile')
            plt.show()

        Compare stickers and spacers in the same plot::

            pos, attractive = mf.per_residue_attractive_vector(seq, seq)
            pos, repulsive = mf.per_residue_repulsive_vector(seq, seq)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            
            ax1.plot(pos, attractive, 'g-')
            ax1.fill_between(pos, attractive, 0, alpha=0.3, color='green')
            ax1.set_ylabel('Attractive')
            ax1.set_title('Sticker Profile')
            
            ax2.plot(pos, repulsive, 'purple')
            ax2.fill_between(pos, 0, repulsive, alpha=0.3, color='purple')
            ax2.set_ylabel('Repulsive')
            ax2.set_xlabel('Residue Position')
            ax2.set_title('Spacer Profile')
            
            plt.tight_layout()

        See Also
        --------
        per_residue_attractive_vector : Get the attractive component.
        intermolecular_idr_matrix : Get the full interaction matrix.

        """

        # do the thing; note this class from the derived class so model-specific
        # sanity checking is handled implicitly here
        B = self.intermolecular_idr_matrix(seq1, seq2, window_size=window_size, use_cython=use_cython, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)[0]

        # extract the raw matrix
        raw_matrix = B[0]

        # get dem indices
        idx = np.arange(B[1][0], B[1][-1]+1)

        # create a mask for repulsive values
        repulsive_mask = raw_matrix > repulsive_threshold

        # sum repulsive values in each column
        repulsive_sums = np.sum(raw_matrix * repulsive_mask, axis=1)

        if len(repulsive_sums) != len(idx):
            print(len(repulsive_sums), len(idx))
            raise ValueError('Length of repulsive sums does not match length of indices; this is a bug.')

        # count repulsive values in each column
        repulsive_counts = np.sum(repulsive_mask, axis=1)

        # Avoid division by zero for columns with no repulsive values
        repulsive_counts[repulsive_counts == 0] = 1

        # calculate the average of repulsive values in each column
        if return_total:
            vals = repulsive_sums
        else:
            vals = repulsive_sums / repulsive_counts

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
    def protein_nucleic_vector(self, seq, fragsize=21, smoothing_window=30, poly_order=3):
        """
        Calculate the per-residue RNA-binding propensity for a protein sequence.
        
        This function computes the interaction strength between sliding windows
        of a protein sequence and poly-U RNA (a simple RNA model). It produces
        a profile showing which regions of the protein are predicted to have
        favorable interactions with nucleic acids.

        The calculation uses a sliding window approach:
        
        1. Extract a fragment of `fragsize` residues from the protein
        2. Calculate epsilon between that fragment and poly-U of the same length
        3. Normalize by fragment length
        4. Slide the window and repeat
        5. Optionally smooth the resulting profile

        Parameters
        ----------
        seq : str
            The protein amino acid sequence.

        fragsize : int, optional
            Size of the sliding window. Must be odd. Default is 21.

        smoothing_window : int or False, optional
            Window size for Savitzky-Golay smoothing filter.
            Set to False to disable smoothing. Default is 30.

        poly_order : int or False, optional
            Polynomial order for Savitzky-Golay filter.
            Set to False to disable smoothing. Default is 3.

        Returns
        -------
        list of np.ndarray
            [0] : np.ndarray
                Residue positions (centered on each window).
                Positions start at (fragsize-1)/2 and end at len(seq)-(fragsize+1)/2.
            
            [1] : np.ndarray
                Per-residue RNA interaction values.
                Negative = favorable RNA binding.
                Positive = unfavorable RNA binding.

        Examples
        --------
        Calculate RNA-binding profile::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            import matplotlib.pyplot as plt
            
            mf = Mpipi_frontend()
            
            # Example: FUS protein (known RNA-binding protein)
            fus_seq = "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQ"
            
            positions, rna_binding = mf.protein_nucleic_vector(fus_seq)
            
            plt.figure(figsize=(10, 3))
            plt.plot(positions, rna_binding, 'b-')
            plt.fill_between(positions, rna_binding, 0,
                            where=(rna_binding < 0), alpha=0.3, color='blue')
            plt.xlabel('Residue Position')
            plt.ylabel('RNA Interaction (ε/residue)')
            plt.axhline(0, color='black', lw=0.5)
            plt.title('RNA-Binding Profile')
            plt.show()

        Compare two proteins::

            seq_rbd  = "ARGARGARGARGARGARG"  # Arg-rich (RNA-binding)
            seq_ctrl = "AAAAAAAAAAAAAAAAAA"  # Control
            
            pos1, bind1 = mf.protein_nucleic_vector(seq_rbd, fragsize=11)
            pos2, bind2 = mf.protein_nucleic_vector(seq_ctrl, fragsize=11)
            
            print(f"RBD mean binding: {np.mean(bind1):.3f}")
            print(f"Control mean binding: {np.mean(bind2):.3f}")

        Note
        ----
        This method is designed for the Mpipi forcefield which has explicit
        RNA parameters. For CALVADOS, RNA interactions may not be available.

        See Also
        --------
        plot_protein_nucleic_vector : Visualize the RNA-binding profile.

        """

        if fragsize % 2 == 0:
            raise Exception('fragsize must be odd')

        
        # define the function to calculate the epsilon value for a given sequence; the closure
        # used with the fragsize and IMC_object variables is to avoid having to pass these
        # as arguments to the function
        def RNA_bind(seq):
            return epsilon_stateless.get_sequence_epsilon_value(seq, len(seq)*'U', self.IMC_object)/len(seq)

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


    # ....................................................................................
    #
    #
    def protein_peptide_vector(self, seq, peptide, fragsize=21, smoothing_window=30, poly_order=3):
        """
        Calculate the per-residue peptide-binding propensity for a protein sequence.
        
        This function computes the interaction strength between sliding windows
        of a protein sequence and a specified peptide. It produces a profile 
        showing which regions of the protein are predicted to have favorable 
        interactions with the peptide.

        The calculation uses a sliding window approach:
        
        1. Extract a fragment of `fragsize` residues from the protein
        2. Calculate epsilon between that fragment and the peptide
        3. Normalize by peptide length
        4. Slide the window and repeat
        5. Optionally smooth the resulting profile

        Parameters
        ----------
        seq : str
            The protein amino acid sequence.

        peptide : str
            The peptide sequence to calculate binding propensity against.

        fragsize : int, optional
            Size of the sliding window. Must be odd. Default value is
            21,.

        smoothing_window : int or False, optional
            Window size for Savitzky-Golay smoothing filter.
            Set to False to disable smoothing. Default is 30.

        poly_order : int or False, optional
            Polynomial order for Savitzky-Golay filter.
            Set to False to disable smoothing. Default is 3.

        Returns
        -------
        list of np.ndarray
            [0] : np.ndarray
                Residue positions (centered on each window).
                Positions start at (fragsize-1)/2 and end at len(seq)-(fragsize+1)/2.
            
            [1] : np.ndarray
                Per-residue peptide interaction values.
                Negative = favorable peptide binding.
                Positive = unfavorable peptide binding.

        Examples
        --------
        Calculate peptide-binding profile::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            import matplotlib.pyplot as plt
            
            mf = Mpipi_frontend()
            
            protein_seq = "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQ"
            peptide = "FYWFYW"
            
            positions, binding = mf.protein_peptide_vector(protein_seq, peptide)
            
            plt.figure(figsize=(10, 3))
            plt.plot(positions, binding, 'b-')
            plt.fill_between(positions, binding, 0,
                            where=(binding < 0), alpha=0.3, color='blue')
            plt.xlabel('Residue Position')
            plt.ylabel('Peptide Interaction (ε/residue)')
            plt.axhline(0, color='black', lw=0.5)
            plt.title(f'Peptide ({peptide}) Binding Profile')
            plt.show()

        Compare binding of different peptides::

            protein = "EKEKEKEKEKEKEKEKEKEK" * 3
            peptide_aromatic = "FYWFYW"
            peptide_charged = "RKRKRK"
            
            pos1, bind1 = mf.protein_peptide_vector(protein, peptide_aromatic)
            pos2, bind2 = mf.protein_peptide_vector(protein, peptide_charged)
            
            print(f"Aromatic peptide mean binding: {np.mean(bind1):.3f}")
            print(f"Charged peptide mean binding: {np.mean(bind2):.3f}")

        See Also
        --------
        protein_nucleic_vector : Similar function for RNA binding.
        epsilon : Get epsilon for two full sequences.

        """

        if fragsize % 2 == 0:
            raise Exception('fragsize must be odd')
        
        # define the function to calculate the epsilon value for a given sequence; the closure
        # uses the peptide and IMC_object variables to avoid passing them as arguments
        def peptide_bind(seq):
            return epsilon_stateless.get_sequence_epsilon_value(seq, peptide, self.IMC_object)/len(peptide)

        # initialize the return vector
        return_vector = []
        idx = []

        # if the sequence is longer than the fragment size, calculate the per-residue
        # interaction for each fragment
        if len(seq) > fragsize:
        
            for i in range(0, 1+(len(seq)-fragsize)):
                return_vector.append(peptide_bind(seq[i:i+fragsize]))
                idx.append(i+(fragsize-1)/2)
                
        # if the sequence is shorter than the fragment size, calculate the per-residue
        # interaction for the whole sequence, which is all we can really do here
        else:
            return_vector.append(peptide_bind(seq))
            idx = [int(len(seq)/2)]
        
        return_vector = np.array(return_vector)
        idx = np.array(idx)

        if smoothing_window is False or poly_order is False:
            return [idx, return_vector]
        else:
            return [idx, savgol_filter(return_vector, smoothing_window, poly_order)]


    # ....................................................................................
    #
    #
    @apply_publication_styles
    def plot_protein_nucleic_vector(self,
                                    seq,
                                    fragsize = 21,
                                    smoothing_window = 30,
                                    poly_order = 3,
                                    domains = [],
                                    domain_color = 'yellow',
                                    domain_alpha = 0.3,
                                    trace_width = 3,
                                    vmin = -0.8,
                                    vmax = 0.8,
                                    tic_frequency=100,
                                    cmap='PRGn',
                                    fname=None,
                                    zero_folded=True,
                                    show_grid = False,
                                    figsize=(4, 1.5),
                                    ylim = [-1.2,1.2]):
                                    
        """
        Generate a publication-ready plot of the RNA-binding profile for a protein.
        
        This function creates a figure showing the per-residue RNA interaction
        profile, with points colored by interaction strength. Regions predicted
        to be folded are shaded gray (if zero_folded=True), and custom domains
        can be highlighted.

        Parameters
        ----------
        seq : str
            The protein amino acid sequence.

        fragsize : int, optional
            Size of the sliding window for RNA interaction calculation.
            Must be odd. Default is 21.

        smoothing_window : int or False, optional
            Window size for Savitzky-Golay smoothing. Default is 30.

        poly_order : int or False, optional
            Polynomial order for smoothing. Default is 3.

        domains : list, optional
            List of [start, end] pairs for domains to highlight.
            These are shaded with domain_color. Default is [].

        domain_color : str, optional
            Color for highlighting domains. Default is 'yellow'.

        domain_alpha : float, optional
            Transparency for domain highlighting. Default is 0.3.

        trace_width : float, optional
            Size of the scatter points. Default is 3.

        vmin : float, optional
            Minimum value for color scale. Default is -0.8.

        vmax : float, optional
            Maximum value for color scale. Default is 0.8.

        cmap : str, optional
            Colormap for the trace. Default is 'PRGn'.

        fname : str, optional
            If provided, save figure to this path. Default is None.

        zero_folded : bool, optional
            If True, shade predicted folded domains in gray. Default is True.

        show_grid : bool, optional
            If True, show vertical grid lines. Default is False.

        figsize : tuple, optional
            Figure dimensions (width, height) in inches. Default is (4, 1.5).

        ylim : list, optional
            Y-axis limits [ymin, ymax]. Default is [-1.2, 1.2].

        tic_frequency : int, optional
            Spacing between x-axis tick labels. Default is 100.

        Returns
        -------
        tuple
            (fig, ax) : matplotlib Figure and Axes objects.

        Examples
        --------
        Basic RNA-binding profile plot::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            
            mf = Mpipi_frontend()
            seq = "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQ"
            
            fig, ax = mf.plot_protein_nucleic_vector(seq)
            plt.show()

        With domain highlighting::

            fig, ax = mf.plot_protein_nucleic_vector(
                seq,
                domains=[[1, 20], [30, 45]],  # Highlight these regions
                domain_color='cyan',
                domain_alpha=0.5
            )

        Save to file with custom settings::

            fig, ax = mf.plot_protein_nucleic_vector(
                seq,
                vmin=-1.0,
                vmax=1.0,
                figsize=(8, 2),
                fname='rna_binding_profile.pdf'
            )

        See Also
        --------
        protein_nucleic_vector : Get the raw data without plotting.

        """

        # get the per-residue attractive vector
        X = self.protein_nucleic_vector(seq, fragsize=fragsize, smoothing_window=smoothing_window, poly_order=poly_order)

        fig = plt.figure(figsize=figsize, dpi=450)
        ax = plt.gca()

        plt.scatter(X[0], X[1], c=X[1], s=trace_width, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.plot(X[0], X[1], 'k-',lw=0.3)

        if zero_folded:
            for d in meta.predict_disorder_domains(seq).folded_domain_boundaries:
                #ax.axvspan(d[0],d[1],linewidth=0, zorder=-10, color='w', alpha=0.5)
                ax.axvspan(d[0],d[1],linewidth=0, zorder=-12, color='grey',alpha=0.6)

        for d in domains:
            ax.axvspan(d[0],d[1],linewidth=0, zorder=-20, color=domain_color, alpha=domain_alpha)
                        
            
        plt.xlim([1,len(seq)+1])
        ax.axhline(0, color='k',lw=0.5,ls='--')

   
        # ensure axis are on top
        for spine in ax.spines.values():

            # zorder = order in which objects in a plot appear in "z" (axis coming out of the screen)
            spine.set_zorder(30)

        plt.ylabel('NA\ninteraction',fontsize=6)
        plt.yticks(fontsize=6)
        plt.ylim(ylim)
         
        xticks = [1]
        xticks.extend(list(np.arange(tic_frequency, len(seq)+1, tic_frequency)))
                  
        plt.xticks(xticks, fontsize=6)
        plt.xlabel('Residue',fontsize=6)

        if show_grid:
            ax.xaxis.grid(True, which='both', linewidth=0.2)
                  
        plt.tight_layout()

        # finally save the figure
        if fname is not None:
            plt.savefig(fname, dpi=350)

        return fig, ax


    # ....................................................................................
    #
    #
    @apply_publication_styles
    def plot_protein_peptide_vector(self,
                                    seq,
                                    peptide,
                                    fragsize = 21,
                                    smoothing_window = 30,
                                    poly_order = 3,
                                    domains = [],
                                    domain_color = 'yellow',
                                    domain_alpha = 0.3,
                                    trace_width = 3,
                                    vmin = -0.8,
                                    vmax = 0.8,
                                    tic_frequency=100,
                                    cmap='PRGn',
                                    fname=None,
                                    zero_folded=True,
                                    show_grid = False,
                                    figsize=(4, 1.5),
                                    ylim = [-1.2,1.2]):
                                    
        """
        Generate a publication-ready plot of the peptide-binding profile for a protein.
        
        This function creates a figure showing the per-residue peptide interaction
        profile, with points colored by interaction strength. Regions predicted
        to be folded are shaded gray (if zero_folded=True), and custom domains
        can be highlighted.

        Parameters
        ----------
        seq : str
            The protein amino acid sequence.

        peptide : str
            The peptide sequence to calculate binding propensity against.

        fragsize : int, optional
            Size of the sliding window for peptide interaction calculation.
            Must be odd. Default is 21.

        smoothing_window : int or False, optional
            Window size for Savitzky-Golay smoothing. Default is 30.

        poly_order : int or False, optional
            Polynomial order for smoothing. Default is 3.

        domains : list, optional
            List of [start, end] pairs for domains to highlight.
            These are shaded with domain_color. Default is [].

        domain_color : str, optional
            Color for highlighting domains. Default is 'yellow'.

        domain_alpha : float, optional
            Transparency for domain highlighting. Default is 0.3.

        trace_width : float, optional
            Size of the scatter points. Default is 3.

        vmin : float, optional
            Minimum value for color scale. Default is -0.8.

        vmax : float, optional
            Maximum value for color scale. Default is 0.8.

        cmap : str, optional
            Colormap for the trace. Default is 'PRGn'.

        fname : str, optional
            If provided, save figure to this path. Default is None.

        zero_folded : bool, optional
            If True, shade predicted folded domains in gray. Default is True.

        show_grid : bool, optional
            If True, show vertical grid lines. Default is False.

        figsize : tuple, optional
            Figure dimensions (width, height) in inches. Default is (4, 1.5).

        ylim : list, optional
            Y-axis limits [ymin, ymax]. Default is [-1.2, 1.2].

        tic_frequency : int, optional
            Spacing between x-axis tick labels. Default is 100.

        Returns
        -------
        tuple
            (fig, ax) : matplotlib Figure and Axes objects.

        Examples
        --------
        Basic peptide-binding profile plot::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            
            mf = Mpipi_frontend()
            seq = "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQ"
            peptide = "FYWFYW"
            
            fig, ax = mf.plot_protein_peptide_vector(seq, peptide)
            plt.show()

        With domain highlighting::

            fig, ax = mf.plot_protein_peptide_vector(
                seq,
                peptide,
                domains=[[1, 20], [30, 45]],  # Highlight these regions
                domain_color='cyan',
                domain_alpha=0.5
            )

        Save to file with custom settings::

            fig, ax = mf.plot_protein_peptide_vector(
                seq,
                peptide,
                vmin=-1.0,
                vmax=1.0,
                figsize=(8, 2),
                fname='peptide_binding_profile.pdf'
            )

        See Also
        --------
        protein_peptide_vector : Get the raw data without plotting.
        plot_protein_nucleic_vector : Similar function for RNA binding.

        """

        # get the per-residue peptide interaction vector
        X = self.protein_peptide_vector(seq, peptide, fragsize=fragsize, smoothing_window=smoothing_window, poly_order=poly_order)

        fig = plt.figure(figsize=figsize, dpi=450)
        ax = plt.gca()

        plt.scatter(X[0], X[1], c=X[1], s=trace_width, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.plot(X[0], X[1], 'k-',lw=0.3)

        if zero_folded:
            for d in meta.predict_disorder_domains(seq).folded_domain_boundaries:
                #ax.axvspan(d[0],d[1],linewidth=0, zorder=-10, color='w', alpha=0.5)
                ax.axvspan(d[0],d[1],linewidth=0, zorder=-12, color='grey',alpha=0.6)

        for d in domains:
            ax.axvspan(d[0],d[1],linewidth=0, zorder=-20, color=domain_color, alpha=domain_alpha)
                        
            
        plt.xlim([1,len(seq)+1])
        ax.axhline(0, color='k',lw=0.5,ls='--')

   
        # ensure axis are on top
        for spine in ax.spines.values():

            # zorder = order in which objects in a plot appear in "z" (axis coming out of the screen)
            spine.set_zorder(30)

        plt.ylabel('Peptide\ninteraction',fontsize=6)
        plt.yticks(fontsize=6)
        plt.ylim(ylim)
         
        xticks = [1]
        xticks.extend(list(np.arange(tic_frequency, len(seq)+1, tic_frequency)))
                  
        plt.xticks(xticks, fontsize=6)
        plt.xlabel('Residue',fontsize=6)

        if show_grid:
            ax.xaxis.grid(True, which='both', linewidth=0.2)
                  
        plt.tight_layout()

        # finally save the figure
        if fname is not None:
            plt.savefig(fname, dpi=350)

        return fig, ax


    # ....................................................................................
    #
    #
    def build_phase_diagram(self, seq, use_aliphatic_weighting=True, use_charge_weighting=True):
        """
        Compute Flory-Huggins phase diagram data for a homotypic system.
        
        This function calculates the binodal (coexistence curve) and spinodal
        for a protein sequence undergoing liquid-liquid phase separation (LLPS).
        It uses the sequence's epsilon value to parameterize the Flory-Huggins
        free energy and computes the phase boundaries analytically.

        The temperature is in arbitrary units (AU) related to the interaction
        strength. Higher critical temperature indicates stronger phase separation
        propensity.

        Parameters
        ----------
        seq : str
            The protein amino acid sequence.

        use_aliphatic_weighting : bool, optional
            Apply aliphatic weighting to epsilon calculation. Default is True.

        use_charge_weighting : bool, optional
            Apply charge weighting to epsilon calculation. Default is True.

        Returns
        -------
        list
            8-element list containing binodal and spinodal data:
            
            [0] : np.ndarray
                Dilute phase volume fractions (phi) for the binodal.
            
            [1] : np.ndarray
                Dense phase volume fractions (phi) for the binodal.
            
            [2] : list
                Critical point as [phi_c, T_c].
            
            [3] : np.ndarray
                Temperatures corresponding to [0] and [1].
            
            [4] : np.ndarray
                Dilute phase volume fractions for the spinodal.
            
            [5] : np.ndarray
                Dense phase volume fractions for the spinodal.
            
            [6] : list
                Spinodal critical point as [phi_c, T_c].
            
            [7] : np.ndarray
                Temperatures for the spinodal.

        Examples
        --------
        Basic usage and plotting::
        
            from finches.frontend.mpipi_frontend import Mpipi_frontend
            import matplotlib.pyplot as plt
            
            mf = Mpipi_frontend()
            seq = "FYWFYWFYWFYWFYWFYWFY"
            
            B = mf.build_phase_diagram(seq)
            
            # Plot binodal (phase boundary)
            plt.figure(figsize=(4, 3))
            plt.plot(B[0], B[3], 'b-', label='Dilute arm')
            plt.plot(B[1], B[3], 'b-', label='Dense arm')
            plt.plot(B[2][0], B[2][1], 'ro', label='Critical point')
            plt.xlabel(r'Volume fraction ($\\phi$)')
            plt.ylabel('Temperature (AU)')
            plt.legend()
            plt.title(f'Critical T = {B[2][1]:.2f}')
            plt.show()

        With spinodal::
        
            B = mf.build_phase_diagram(seq)
            
            # Binodal
            plt.plot(B[0], B[3], 'b-')
            plt.plot(B[1], B[3], 'b-')
            
            # Spinodal (metastability limit)
            plt.plot(B[4], B[7], 'r--', alpha=0.5)
            plt.plot(B[5], B[7], 'r--', alpha=0.5)

        Compare sequences::
        
            seq_wt  = "FYWFYWFYWFYWFYWFYWFY"
            seq_mut = "AYWFYWFYWFYWFYWFYWFY"  # F1A mutation
            
            B_wt = mf.build_phase_diagram(seq_wt)
            B_mut = mf.build_phase_diagram(seq_mut)
            
            print(f"WT critical T: {B_wt[2][1]:.2f}")
            print(f"Mutant critical T: {B_mut[2][1]:.2f}")
            print(f"Change in T_c: {B_mut[2][1] - B_wt[2][1]:.2f}")

        Note
        ----
        If the sequence has positive (repulsive) epsilon, it cannot phase
        separate. In this case, the function returns a minimal phase diagram
        with very low critical temperature.

        See Also
        --------
        plot_phase_diagram : Plot the phase diagram directly.
        epsilon : Get the raw epsilon value.
        
        """
        eps = self.epsilon(seq, seq, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)
        
        return epsilon_to_FHtheory.epsilon_to_phase_diagram(seq, eps)


    @apply_publication_styles
    def plot_phase_diagram(self,
                           seq,
                           use_aliphatic_weighting=True,
                           use_charge_weighting=True,
                           line_color='k',
                           line_style='-',
                           line_width=0.5,
                           xlim=None,
                           ylim=None,
                           xlog=False,
                           width=2.2,
                           height=1.2,
                           filename=None):
        """
        Generate a publication-ready Flory-Huggins phase diagram for a sequence.
        
        This function calculates the homotypic epsilon for the sequence and
        uses it to construct a phase diagram showing the binodal (coexistence curve).
        The diagram shows temperature (in arbitrary units) vs. volume fraction (phi).

        The binodal curve separates:
        
        - **One-phase region** (above the curve): homogeneous solution
        - **Two-phase region** (below the curve): phase-separated state
        
        The critical point is at the top of the binodal curve.

        Parameters
        ----------
        seq : str
            The protein amino acid sequence.

        use_aliphatic_weighting : bool, optional
            Apply aliphatic weighting. Default is True.

        use_charge_weighting : bool, optional
            Apply charge weighting. Default is True.

        line_color : str, optional
            Color of the binodal curve. Default is 'k' (black).

        line_style : str, optional
            Line style for the binodal. Default is '-' (solid).

        line_width : float, optional
            Width of the binodal line. Default is 0.5.

        xlim : tuple, optional
            X-axis (phi) limits as (min, max). Default is None (auto).

        ylim : tuple, optional
            Y-axis (T) limits as (min, max). Default is None (auto).

        xlog : bool, optional
            If True, use logarithmic x-axis. Useful for seeing the
            dilute arm of the binodal. Default is False.

        width : float, optional
            Figure width in inches. Default is 2.2.

        height : float, optional
            Figure height in inches. Default is 1.2.

        filename : str, optional
            If provided, save figure to this path. Default is None.

        Returns
        -------
        list
            [0] : tuple
                Phase diagram data from build_phase_diagram().
                See build_phase_diagram for detailed structure.
            
            [1] : matplotlib.figure.Figure
            
            [2] : matplotlib.axes.Axes

        Examples
        --------
        Basic phase diagram::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            
            mf = Mpipi_frontend()
            seq = "FYWFYWFYWFYWFYWFYWFY"  # Aromatic-rich, phase-separating
            
            data, fig, ax = mf.plot_phase_diagram(seq)
            plt.show()

        With log x-axis to see dilute arm::

            data, fig, ax = mf.plot_phase_diagram(
                seq,
                xlog=True,
                xlim=(1e-6, 1),
                line_color='blue'
            )

        Save publication figure::

            data, fig, ax = mf.plot_phase_diagram(
                seq,
                width=3,
                height=2,
                filename='phase_diagram.pdf'
            )

        Extract critical point::

            data, fig, ax = mf.plot_phase_diagram(seq)
            crit_phi, crit_T = data[2]  # [phi_c, T_c]
            print(f"Critical point: phi={crit_phi:.4f}, T={crit_T:.2f}")

        See Also
        --------
        build_phase_diagram : Get phase diagram data without plotting.
        plot_multiple_phase_diagrams : Compare multiple sequences.

        """                           
                           
        B = self.build_phase_diagram(seq, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)

        fig = plt.figure(figsize=(width, height), dpi=450)
        ax = plt.gca()
                           
        plt.plot(B[0], B[3], color=line_color, ls=line_style, lw=line_width)
        plt.plot(B[1], B[3], color=line_color, ls=line_style, lw=line_width)
        
        plt.ylabel(r'$T (AU)$', fontsize=7)
        plt.xlabel(r'$\rm\phi$', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xlog is True:
            plt.xscale('log')

        plt.tight_layout()
            
        if filename is not None:
            plt.savefig(filename, dpi=450)
            

        return [B, fig, ax]


    @apply_publication_styles
    def plot_multiple_phase_diagrams(self,
                                     seq_dict,
                                     use_aliphatic_weighting=True,
                                     use_charge_weighting=True,
                                     tc_ref=None,
                                     line_style='-',
                                     line_width=0.5,
                                     xlim=None,
                                     ylim=None,
                                     xlog=False,
                                     width=2.2,
                                     height=1.2,
                                     filename=None):
        """
        Generate a publication-ready figure comparing phase diagrams of multiple sequences.
        
        This function creates an overlay of binodal curves for multiple sequences,
        making it easy to compare their phase separation propensities. Each sequence
        is plotted with a specified color and labeled in a legend.

        Optionally, temperatures can be normalized to a reference sequence's
        critical temperature (T_c), allowing comparison of curve shapes independently
        of absolute critical temperatures.

        Parameters
        ----------
        seq_dict : dict
            Dictionary mapping sequence names to [sequence, color] pairs.
            
            Example::
            
                {
                    'WT': ['FYWFYWFYW...', 'blue'],
                    'Mutant1': ['AYWFYWFYW...', 'red'],
                    'Mutant2': ['FAWFYWFYW...', 'green']
                }

        use_aliphatic_weighting : bool, optional
            Apply aliphatic weighting. Default is True.

        use_charge_weighting : bool, optional
            Apply charge weighting. Default is True.

        tc_ref : str, optional
            Key in seq_dict for the reference sequence. If provided, all
            temperatures are normalized by this sequence's T_c, so the
            y-axis becomes T/T_c. Default is None (no normalization).

        line_style : str, optional
            Line style for all curves. Default is '-'.

        line_width : float, optional
            Line width for all curves. Default is 0.5.

        xlim : tuple, optional
            X-axis limits (min, max). Default is None (auto).

        ylim : tuple, optional
            Y-axis limits (min, max). Default is None (auto).

        xlog : bool, optional
            Use logarithmic x-axis. Default is False.

        width : float, optional
            Figure width in inches. Default is 2.2.

        height : float, optional
            Figure height in inches. Default is 1.2.

        filename : str, optional
            If provided, save figure to this path. Default is None.

        Returns
        -------
        list
            [0] : list
                List of phase diagram data tuples, one per sequence.
            
            [1] : matplotlib.figure.Figure
            
            [2] : matplotlib.axes.Axes

        Examples
        --------
        Compare wild-type and mutants::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            
            mf = Mpipi_frontend()
            
            sequences = {
                'WT': ['FYWFYWFYWFYWFYWFYWFY', 'blue'],
                'F1A': ['AYWFYWFYWFYWFYWFYWFY', 'red'],
                'Y5A': ['FYWFAWFYWFYWFYWFYWFY', 'green']
            }
            
            data, fig, ax = mf.plot_multiple_phase_diagrams(sequences)
            plt.show()

        With normalization to WT critical temperature::

            data, fig, ax = mf.plot_multiple_phase_diagrams(
                sequences,
                tc_ref='WT'  # Normalize all to WT's T_c
            )
            # Y-axis now shows T/T_c, so WT reaches 1.0

        Save publication figure::

            data, fig, ax = mf.plot_multiple_phase_diagrams(
                sequences,
                xlog=True,
                xlim=(1e-5, 1),
                width=3.5,
                height=2.5,
                filename='phase_diagram_comparison.pdf'
            )

        See Also
        --------
        plot_phase_diagram : Plot a single sequence's phase diagram.
        build_phase_diagram : Get phase diagram data without plotting.

        """
        
        
        fig = plt.figure(figsize=(width, height), dpi=450)
        ax = plt.gca()

        if tc_ref is not None:
            if tc_ref not in seq_dict:
                raise ValueError('The reference sequence is not in the dictionary.')

            local_seq = seq_dict[tc_ref][0]

            
            B_ref = self.build_phase_diagram(local_seq, use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)
            tc_ref = max(B_ref[3])
        else:
            tc_ref = 1

        # for each sequence in the dictionary
        all_phase_diagrams = []
        for k in seq_dict:
            
            B = self.build_phase_diagram(seq_dict[k][0], use_aliphatic_weighting=use_aliphatic_weighting, use_charge_weighting=use_charge_weighting)

            plt.plot(B[0], B[3]/tc_ref, color=seq_dict[k][1], ls=line_style, lw=line_width, label=k)
            plt.plot(B[1], B[3]/tc_ref, color=seq_dict[k][1], ls=line_style, lw=line_width)

            all_phase_diagrams.append(B)

        if np.isclose(tc_ref,1):
            plt.ylabel(r'$T (AU)$', fontsize=7)            
        else:
            plt.ylabel(r'$T/T_c$', fontsize=7)
            
        plt.xlabel(r'$\rm\phi$', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xlog is True:
            plt.xscale('log')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=4, frameon=False)            

        plt.tight_layout()
            
        if filename is not None:
            plt.savefig(filename, dpi=450)
            

        return [all_phase_diagrams, fig, ax]


    # ....................................................................................
    #
    #
    def dms(self,
            seq,
            amino_acids=None,
            use_aliphatic_weighting=True,
            use_charge_weighting=True,
            return_delta=False,
            show_progress=True):
        """
        Perform a deep mutational scan (DMS) on a sequence, calculating homotypic 
        epsilon values for all possible single-point mutants.

        This function systematically mutates each position in the sequence to every 
        one of the 20 standard amino acids and calculates the homotypic epsilon 
        value for each mutant sequence.

        Parameters
        ----------
        seq : str
            The input protein sequence of length n.

        amino_acids : list, optional
            List of single-letter amino acid codes to use for mutations.
            Default is None, which uses all 20 standard amino acids.

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme for the epsilon
            calculation. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme for the epsilon
            calculation. Default is True.

        return_delta : bool
            If True, return the difference (delta) between mutant and wild-type
            epsilon values (mutant - WT). Negative values indicate the mutation
            makes the sequence more self-attractive. If False, return absolute
            epsilon values. Default is False.

        show_progress : bool
            Whether to show a progress bar during calculation. Default is True.
            Recommended for longer sequences.

        Returns
        -------
        tuple
            A tuple containing:

            [0] : np.ndarray
                A 20 x n matrix where element [i, j] contains either:
                - If return_delta=False: the homotypic epsilon value for the 
                  sequence with position j mutated to amino acid i.
                - If return_delta=True: the change in epsilon (mutant - WT),
                  where negative values indicate increased self-attraction.
                Rows are ordered alphabetically by single-letter amino acid code:
                A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

            [1] : list
                List of the 20 amino acid single-letter codes in the order they 
                appear in the matrix rows: 
                ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

            [2] : np.ndarray
                Array of position indices (1-indexed) corresponding to matrix columns.

        Examples
        --------
        Basic usage::

            mf = Mpipi_frontend()
            matrix, amino_acids, positions = mf.dms("MSKGEELFT")
            
            # Get epsilon for mutating position 3 (K) to Alanine (A)
            eps_K3A = matrix[0, 2]  # Row 0 = A, Column 2 = position 3
            
            # Get all epsilon values for position 5 (E)
            all_pos5_mutants = matrix[:, 4]

        To find the most stabilizing mutation::

            matrix, aas, pos = mf.dms(seq, return_delta=True)
            min_idx = np.unravel_index(np.argmin(matrix), matrix.shape)
            best_aa = aas[min_idx[0]]
            best_pos = pos[min_idx[1]]
            print(f"Most stabilizing: {seq[min_idx[1]-1]}{best_pos}{best_aa}")

        """

        # Standard amino acids in alphabetical order
        VALID_AMINO_ACIDS = ['W', 'Y', 'F', 'I', 'L', 'V', 'M', 'A', 'G', 'S',                             
                             'T', 'N', 'Q', 'H', 'R', 'K', 'P', 'D', 'E', 'C']
        
        if amino_acids is None:
            AMINO_ACIDS = VALID_AMINO_ACIDS
        else:
            if not isinstance(amino_acids, list):
                raise ValueError('amino_acids must be a list of single-letter amino acid codes')
            
            invalid_aas = [aa for aa in amino_acids if aa not in VALID_AMINO_ACIDS]
            if invalid_aas:
                raise ValueError(f'Invalid amino acid codes found: {invalid_aas}. Valid codes are: {sorted(VALID_AMINO_ACIDS)}')
            
            AMINO_ACIDS = amino_acids


        seq_len = len(seq)
        n_amino_acids = len(AMINO_ACIDS)
        
        # Initialize the output matrix (20 amino acids x n positions)
        dms_matrix = np.zeros((n_amino_acids, seq_len))

        # Create position indices (1-indexed for biological convention)
        positions = np.arange(1, seq_len + 1)

        # Calculate total iterations for progress bar
        total_iterations = n_amino_acids * seq_len   

        # Set up iterator with optional progress bar
        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=total_iterations, desc="DMS scan")

        # Convert sequence to list for easier mutation
        seq_list = list(seq)

        # Iterate over each position
        for pos_idx in range(seq_len):
            original_aa = seq_list[pos_idx]

            # Iterate over each possible amino acid substitution
            for aa_idx, new_aa in enumerate(AMINO_ACIDS):
                
                # Create mutant sequence
                if new_aa == original_aa:
                    # No mutation - use original sequence
                    mutant_seq = seq
                else:
                    # Make the substitution
                    mutant_list = seq_list.copy()
                    mutant_list[pos_idx] = new_aa
                    mutant_seq = ''.join(mutant_list)

                # Calculate homotypic epsilon for the mutant
                eps = self.epsilon(mutant_seq, 
                                   mutant_seq,
                                   use_aliphatic_weighting=use_aliphatic_weighting,
                                   use_charge_weighting=use_charge_weighting)

                # Store in matrix
                dms_matrix[aa_idx, pos_idx] = eps

                if show_progress:
                    pbar.update(1)

        if show_progress:
            pbar.close()

        # If return_delta is True, convert to delta values (mutant - WT)
        if return_delta:
            # Calculate wild-type epsilon
            wt_epsilon = self.epsilon(seq, seq,
                                      use_aliphatic_weighting=use_aliphatic_weighting,
                                      use_charge_weighting=use_charge_weighting)
            dms_matrix = dms_matrix - wt_epsilon

        return (dms_matrix, AMINO_ACIDS, positions)


    # ....................................................................................
    #
    #
    @apply_publication_styles
    def plot_dms(self,
                 seq,
                 amino_acids=None,
                 use_aliphatic_weighting=True,
                 use_charge_weighting=True,
                 return_delta=True,
                 show_progress=True,
                 vmin=None,
                 vmax=None,
                 cmap='PRGn',
                 figsize=None,
                 tic_frequency=10,
                 show_wt_marker=True,
                 wt_marker='o',
                 wt_marker_color='black',
                 wt_marker_size=3,
                 show_title=True,
                 show_sequence=False,
                 sequence_fontsize=5,
                 fname=None):
        """
        Generate a heatmap visualization of a deep mutational scan (DMS).

        This function performs a DMS scan and visualizes the results as an 
        n x 20 heatmap where rows correspond to sequence positions and columns 
        correspond to amino acid substitutions.

        Parameters
        ----------
        seq : str
            The input protein sequence of length n.

        amino_acids : list, optional
            List of single-letter amino acid codes to use for mutations.
            Default is None, which uses all 20 standard amino acids.

        use_aliphatic_weighting : bool
            Whether to use the aliphatic weighting scheme for the epsilon
            calculation. Default is True.

        use_charge_weighting : bool
            Whether to use the charge weighting scheme for the epsilon
            calculation. Default is True.

        return_delta : bool
            If True, plot the difference (delta) between mutant and wild-type
            epsilon values (mutant - WT). Negative values (green in default cmap)
            indicate the mutation makes the sequence more self-attractive.
            If False, plot absolute epsilon values. Default is True.

        show_progress : bool
            Whether to show a progress bar during DMS calculation. Default is True.

        vmin : float, optional
            Minimum value for the color scale. Default is None (auto-determined).
            For delta values, consider using symmetric limits like -2 to 2.

        vmax : float, optional
            Maximum value for the color scale. Default is None (auto-determined).

        cmap : str
            Colormap to use for the heatmap. Default is 'PRGn' (purple-green
            diverging colormap where green = negative/attractive).

        figsize : tuple, optional
            Figure size as (width, height) in inches. Default is None, which
            auto-calculates based on sequence length.

        tic_frequency : int
            Frequency of position tick labels on the x-axis. Default is 10.

        show_wt_marker : bool
            Whether to mark wild-type amino acids at each position. Default is True.

        wt_marker : str
            Marker style for wild-type positions. Default is 'o'.

        wt_marker_color : str
            Color for wild-type markers. Default is 'black'.

        wt_marker_size : float
            Size of wild-type markers. Default is 3.

        show_title : bool
            Whether to display a title above the heatmap. Default is True.

        show_sequence : bool
            Whether to display the amino acid sequence below the heatmap,
            above the position tick labels. Default is False.

        sequence_fontsize : float
            Font size for the sequence letters when show_sequence is True.
            Default is 5.

        fname : str, optional
            Filename to save the figure. If None, the figure is displayed but
            not saved. Default is None.

        Returns
        -------
        tuple
            A tuple containing:

            [0] : matplotlib.figure.Figure
                The figure object.

            [1] : matplotlib.axes.Axes
                The axes object.

            [2] : matplotlib.image.AxesImage
                The image object from imshow.

            [3] : tuple
                The DMS results tuple (matrix, amino_acids, positions).

        Examples
        --------
        Basic usage::

            mf = Mpipi_frontend()
            fig, ax, im, dms_data = mf.plot_dms("MSKGEELFTGVVPILVELD")
            plt.show()

        With custom color scale::

            fig, ax, im, dms_data = mf.plot_dms(seq, vmin=-2, vmax=2, cmap='coolwarm')

        Save to file::

            mf.plot_dms(seq, fname='dms_heatmap.png')

        """

        # Perform DMS scan
        dms_matrix, aa_list, positions = self.dms(
            seq,
            amino_acids=amino_acids,
            use_aliphatic_weighting=use_aliphatic_weighting,
            use_charge_weighting=use_charge_weighting,
            return_delta=return_delta,
            show_progress=show_progress
        )

        # Matrix is already 20 x n (amino acids x positions), use directly
        plot_matrix = dms_matrix

        # Determine figure size if not provided
        if figsize is None:
            # Scale width with sequence length, height fixed for 20 amino acids
            width = max(6, len(seq) * 0.15)
            height = 4
            figsize = (width, height)

        # Create figure
        fig = plt.figure(figsize=figsize, dpi=350)
        ax = plt.gca()

        # Determine color scale limits
        if vmin is None and vmax is None and return_delta:
            # For delta values, use symmetric color scale
            max_abs = max(abs(np.min(plot_matrix)), abs(np.max(plot_matrix)))
            vmin = -max_abs
            vmax = max_abs

        # Create heatmap
        im = ax.imshow(plot_matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        # Set x-axis (positions)
        # Show subset of ticks based on tic_frequency
        # Start at position 1, then 10, 20, 30, etc.
        xtick_positions = [0]  # Position 1 (0-indexed)
        xtick_labels = ['1']
        # Add ticks at 10, 20, 30... (which are indices 9, 19, 29... in 0-indexed)
        for pos in range(tic_frequency, len(seq) + 1, tic_frequency):
            xtick_positions.append(pos - 1)  # Convert to 0-indexed
            xtick_labels.append(str(pos))
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=6)

        # Display sequence below heatmap if requested
        if show_sequence:
            # Create a secondary x-axis for sequence display
            ax2 = ax.secondary_xaxis('bottom')
            ax2.set_xticks(np.arange(len(seq)))
            ax2.set_xticklabels(list(seq), fontsize=sequence_fontsize, family='monospace')
            ax2.tick_params(axis='x', length=0, pad=12)  # No tick marks, pad below position labels
            # Move the position labels up a bit and add more space
            ax.tick_params(axis='x', pad=2)
            # Set xlabel on the secondary axis so it appears below the sequence
            ax2.set_xlabel('Position', fontsize=8)
        else:
            ax.set_xlabel('Position', fontsize=8)

        # Set y-axis (amino acids)
        ax.set_yticks(np.arange(len(aa_list)))
        ax.set_yticklabels(aa_list, fontsize=6)
        ax.set_ylabel('Amino Acid Substitution', fontsize=8)

        # Mark wild-type amino acids
        if show_wt_marker:
            for pos_idx, wt_aa in enumerate(seq):
                if wt_aa in aa_list:
                    aa_idx = aa_list.index(wt_aa)
                    ax.plot(pos_idx, aa_idx, wt_marker, 
                            color=wt_marker_color, 
                            markersize=wt_marker_size,
                            markeredgewidth=0.5,
                            markeredgecolor='white')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        if return_delta:
            cbar.set_label(r'$\Delta\epsilon$ (mutant - WT)', fontsize=7)
        else:
            cbar.set_label(r'$\epsilon$', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # Title
        if show_title:
            if return_delta:
                ax.set_title('DMS: Change in Homotypic Epsilon', fontsize=9)
            else:
                ax.set_title('DMS: Homotypic Epsilon', fontsize=9)

        plt.tight_layout()

        # Save if filename provided
        if fname is not None:
            plt.savefig(fname, dpi=350, bbox_inches='tight')

        return (fig, ax, im, (dms_matrix, aa_list, positions))

