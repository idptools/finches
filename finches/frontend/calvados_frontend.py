from finches.frontend.frontend_base import FinchesFrontend

# for model construction
from finches.forcefields.calvados import calvados_model
from finches import epsilon_calculation

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
            raise ValueError("CALVADOS2 cannot handle RNA ('U')")
        return func(*args, **kwargs)
    return wrapper


class CALVADOS_frontend(FinchesFrontend):
    """
    Frontend class for CALVADOS2 forcefield calculations.
    
    CALVADOS_frontend provides a high-level interface for calculating protein-protein
    interaction parameters using the CALVADOS2 coarse-grained forcefield. This class
    inherits from FinchesFrontend and adds CALVADOS-specific functionality.
    
    IMPORTANT: CALVADOS2 does NOT support RNA sequences. Any sequence containing 
    'U' (uracil) will raise a ValueError. For RNA interactions, use the Mpipi_frontend
    instead.
    
    The CALVADOS2 forcefield models interactions based on:
    - Hydrophobic/hydrophilic interactions (stickiness parameters)
    - Electrostatic interactions (pH and salt dependent)
    - Sequence-specific parameters calibrated from experimental data
    
    Example
    -------
    Basic usage for calculating epsilon between two proteins:
    
        from finches.frontend.calvados_frontend import CALVADOS_frontend
        
        # Initialize with default parameters
        cf = CALVADOS_frontend()
        
        # Or with custom solution conditions
        cf = CALVADOS_frontend(salt=0.100, pH=7.0, temperature=300)
        
        # Calculate epsilon between two sequences
        seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        seq2 = "MGSWAEFKQRLAAIKTRLQALGGSEAELAAFEK"
        eps = cf.epsilon(seq1, seq2)
        print(f"Epsilon: {eps}")
        
        # Generate an interaction figure
        fig_data = cf.interaction_figure(seq1, seq2)
    
    See Also
    --------
    Mpipi_frontend : Alternative frontend that supports both protein and RNA sequences
    FinchesFrontend : Base class with shared functionality
    
    """

    # ....................................................................................
    #
    #            
    def __init__(self, salt=0.150, pH=7.4, temperature=288):
        """
        Initialize the CALVADOS_frontend with specified solution conditions.
        
        Creates an instance of the CALVADOS2 forcefield model configured with
        the specified salt concentration, pH, and temperature. These parameters
        affect electrostatic interactions in the model.
        
        Parameters
        ----------
        salt : float, optional
            Salt concentration in molar (M). Default is 0.150 M (150 mM),
            which represents typical physiological conditions.
            
        pH : float, optional
            Solution pH. Default is 7.4 (physiological pH). The pH affects
            the charge state of ionizable residues (Asp, Glu, His, Lys, Arg).
            
        temperature : float, optional
            Temperature in Kelvin. Default is 288 K (15°C).
            
        Attributes
        ----------
        model : calvados_model
            The underlying CALVADOS2 forcefield model instance.
            
        IMC_object : InteractionMatrixConstructor
            Object for computing interaction matrices and epsilon values.
        
        Example
        -------
        Initialize with default physiological conditions:
        
            from finches.frontend.calvados_frontend import CALVADOS_frontend
            cf = CALVADOS_frontend()
            
        Initialize for low salt conditions:
        
            cf = CALVADOS_frontend(salt=0.050, pH=7.4, temperature=298)
            
        Initialize for acidic conditions:
        
            cf = CALVADOS_frontend(salt=0.150, pH=5.5, temperature=310)
        
        """
        # call superclass constructor 
        super().__init__()

        # initialize the CALVADOS forcefield object
        self.model = calvados_model('CALVADOS2', salt=salt, pH=pH, temperature=temperature)

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
                                  use_charge_weighting=True,
                                  disorder_1=True,
                                  disorder_2=True,
                                  null_shuffle=False):
        """
        Calculate the sliding-window interaction matrix between two protein sequences.
        
        This method decomposes two sequences into overlapping fragments of size
        `window_size` and calculates pairwise epsilon values between all fragment
        pairs using a sliding window approach. The result is a 2D matrix where each
        cell (i, j) represents the interaction strength between fragment i from seq1
        and fragment j from seq2.
        
        The matrix is NOT padded, so edge positions depend on window_size. The method
        returns index arrays that map matrix positions to sequence positions.
        
        IMPORTANT: CALVADOS2 does not support RNA. Sequences containing 'U' will
        raise a ValueError.

        Parameters
        ----------
        seq1 : str
            First protein amino acid sequence (single-letter codes).
            Must not contain 'U' (uracil).

        seq2 : str
            Second protein amino acid sequence (single-letter codes).
            Must not contain 'U' (uracil).

        window_size : int, optional
            Size of the sliding window for fragment analysis. Must be odd;
            will be converted to odd if even. Default is 31 residues.
            Larger windows smooth out local variations.

        use_cython : bool, optional
            If True, use the optimized Cython implementation for faster
            computation. Default is True. Only set to False for debugging.

        use_aliphatic_weighting : bool, optional
            If True, apply weighting scheme that considers clustering of
            aliphatic residues (Ala, Val, Leu, Ile, Met). Default is True.

        use_charge_weighting : bool, optional
            If True, apply weighting scheme that considers clustering of
            charged residues. Default is True.

        disorder_1 : bool, optional
            If True, compute disorder profile for seq1 using metapredict.
            If False, use uniform profile (all 1s). Default is True.

        disorder_2 : bool, optional
            If True, compute disorder profile for seq2 using metapredict.
            If False, use uniform profile (all 1s). Default is True.

        null_shuffle : bool or int, optional
            If False, compute the actual interaction matrix. If an integer,
            compute that many shuffled sequence matrices to establish a null
            distribution. Recommended: 100 shuffles if used. Default is False.

        Returns
        -------
        tuple
            A 3-element tuple containing:
            
            [0] matrix_data : tuple
                A 3-element tuple:
                - [0][0]: np.ndarray - 2D interaction matrix (epsilon values)
                - [0][1]: np.ndarray - 1D array mapping matrix row indices to seq1 positions
                - [0][2]: np.ndarray - 1D array mapping matrix col indices to seq2 positions
                
            [1] disorder_1 : np.ndarray
                Disorder profile for seq1 (values 0-1, higher = more disordered).
                All 1s if disorder_1=False.
                
            [2] disorder_2 : np.ndarray  
                Disorder profile for seq2 (values 0-1, higher = more disordered).
                All 1s if disorder_2=False.

        Example
        -------
        Calculate interaction matrix between two proteins:
        
            from finches.frontend.calvados_frontend import CALVADOS_frontend
            import numpy as np
            
            cf = CALVADOS_frontend()
            
            seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS" * 3  # 90 residues
            seq2 = "MGSWAEFKQRLAAIKTRLQALGGSEAELAAFEK" * 3  # 99 residues
            
            matrix_data, dis1, dis2 = cf.intermolecular_idr_matrix(seq1, seq2)
            
            # Unpack the matrix data
            epsilon_matrix, seq1_indices, seq2_indices = matrix_data
            
            # Find the most attractive region
            min_idx = np.unravel_index(np.argmin(epsilon_matrix), epsilon_matrix.shape)
            print(f"Most attractive at seq1 pos {seq1_indices[min_idx[0]]}, "
                  f"seq2 pos {seq2_indices[min_idx[1]]}")
            print(f"Epsilon: {epsilon_matrix[min_idx]}")
        
        Compute without disorder prediction (faster):
        
            matrix_data, _, _ = cf.intermolecular_idr_matrix(
                seq1, seq2, disorder_1=False, disorder_2=False
            )

        Raises
        ------
        ValueError
            If either sequence contains 'U' (RNA not supported by CALVADOS2).

        """
        

        
        # call the superclass function
        return super().intermolecular_idr_matrix(seq1,
                                                 seq2,
                                                 window_size=window_size,
                                                 use_cython=use_cython,
                                                 use_aliphatic_weighting=use_aliphatic_weighting,
                                                 use_charge_weighting=use_charge_weighting,
                                                 disorder_1=disorder_1,
                                                 disorder_2=disorder_2,
                                                 null_shuffle=null_shuffle)

    
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
        Calculate the epsilon (ε) interaction parameter between two protein sequences.
        
        Epsilon is a single scalar value that quantifies the overall interaction
        strength between two sequences based on the CALVADOS2 forcefield. Negative
        values indicate net attractive interactions, positive values indicate
        net repulsive interactions, and values near zero indicate neutral interactions.
        
        This is the simplest way to assess whether two proteins are likely to
        interact - just pass in their sequences and get back a single number.
        
        IMPORTANT: CALVADOS2 does not support RNA. Sequences containing 'U' will
        raise a ValueError. Use Mpipi_frontend for RNA interactions.

        Parameters
        ----------
        seq1 : str
            First protein amino acid sequence using standard single-letter codes.
            Must not contain 'U' (uracil).

        seq2 : str
            Second protein amino acid sequence using standard single-letter codes.
            Must not contain 'U' (uracil). Can be the same as seq1 for 
            self-interaction (homotypic) calculations.

        use_aliphatic_weighting : bool, optional
            If True, apply weighting that considers clustering of aliphatic
            residues (Ala, Val, Leu, Ile, Met). Clustered aliphatic residues
            contribute more strongly to hydrophobic interactions. Default is True.

        use_charge_weighting : bool, optional
            If True, apply weighting that considers clustering of charged
            residues. Adjacent charges of the same sign or opposite sign
            affect the electrostatic contribution. Default is True.

        Returns
        -------
        float
            The epsilon value for the interaction between the two sequences.
            - Negative values: attractive interaction (more negative = stronger)
            - Positive values: repulsive interaction  
            - Near zero: neutral/weak interaction
            
            Typical range: -10 to +10, though extreme sequences can exceed this.

        Example
        -------
        Calculate homotypic (self) interaction:
        
            from finches.frontend.calvados_frontend import CALVADOS_frontend
            
            cf = CALVADOS_frontend()
            seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK"
            
            # Homotypic interaction
            eps_self = cf.epsilon(seq, seq)
            print(f"Self-interaction epsilon: {eps_self:.3f}")
            
        Calculate heterotypic interaction between two different proteins:
        
            seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
            seq2 = "MGSWAEFKQRLAAIKTRLQALGGSEAELAAFEK"
            
            eps = cf.epsilon(seq1, seq2)
            if eps < -2:
                print(f"Strong attraction: {eps:.3f}")
            elif eps > 2:
                print(f"Strong repulsion: {eps:.3f}")
            else:
                print(f"Weak/neutral interaction: {eps:.3f}")
                
        Compare different solution conditions:
        
            # High salt reduces electrostatic effects
            cf_high_salt = CALVADOS_frontend(salt=0.500)
            cf_low_salt = CALVADOS_frontend(salt=0.050)
            
            seq1 = "EEEEEEEEEE"  # Highly negative
            seq2 = "KKKKKKKKKK"  # Highly positive
            
            eps_high = cf_high_salt.epsilon(seq1, seq2)
            eps_low = cf_low_salt.epsilon(seq1, seq2)
            print(f"High salt: {eps_high:.3f}, Low salt: {eps_low:.3f}")

        Raises
        ------
        ValueError
            If either sequence contains 'U' (RNA not supported by CALVADOS2).

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
                           zero_folded=True,
                           no_disorder=False,
                           null_shuffle=False,
                           plot_rectangles=None):
        """
        Generate a publication-ready interaction matrix figure between two protein sequences.
        
        Creates a comprehensive visualization showing the sliding-window interaction
        matrix as a heatmap with parallel disorder prediction tracks along the top
        and right edges. The heatmap uses a diverging colormap where purple indicates
        attractive (negative epsilon) regions and green indicates repulsive (positive
        epsilon) regions.
        
        This is the primary visualization method for understanding where along two
        sequences the strongest and weakest interactions occur.
        
        IMPORTANT: CALVADOS2 does not support RNA. Sequences containing 'U' will
        raise a ValueError.
        
        Parameters
        ----------
        seq1 : str
            First protein amino acid sequence (plotted on x-axis/horizontal).
            Must not contain 'U' (uracil).

        seq2 : str
            Second protein amino acid sequence (plotted on y-axis/vertical).
            Must not contain 'U' (uracil).

        window_size : int, optional
            Size of the sliding window for fragment analysis. Will be converted
            to an odd number if even is provided. Default is 31 residues.

        use_cython : bool, optional
            If True, use optimized Cython implementation. Default is True.
            Only set to False for debugging purposes.

        use_aliphatic_weighting : bool, optional
            If True, apply aliphatic residue clustering weights. Default is True.

        use_charge_weighting : bool, optional
            If True, apply charged residue clustering weights. Default is True.

        tic_frequency : int, optional
            Spacing of axis tick marks in residue positions. Default is 100.

        seq1_domains : list, optional
            List of [start, end] pairs defining folded domains in seq1.
            Domains are highlighted with semi-transparent overlays.
            Example: [[1, 50], [100, 150]] marks two domains.

        seq2_domains : list, optional
            List of [start, end] pairs defining folded domains in seq2.
            Example: [[25, 75], [200, 250]]

        seq1_lines : list, optional
            List of residue positions where vertical lines will be drawn.
            Useful for marking specific sites (e.g., phosphorylation sites).

        seq2_lines : list, optional
            List of residue positions where horizontal lines will be drawn.
                                                      
        vmin : float, optional
            Minimum value for the colorbar scale. Default is -7.5.
            More negative = stronger attraction (purple).

        vmax : float, optional
            Maximum value for the colorbar scale. Default is 7.5.
            More positive = stronger repulsion (green).
        
        cmap : str, optional
            Matplotlib colormap name. Default is 'PRGn' (purple-green diverging).
            Other good options: 'RdBu_r', 'coolwarm', 'seismic'.

        fname : str, optional
            If provided, save the figure to this file path. Supports formats
            like '.png', '.pdf', '.svg'. If None, figure is displayed.

        zero_folded : bool, optional
            If True, set interaction values to zero for folded domain regions
            (as defined by seq1_domains and seq2_domains). Default is True.

        no_disorder : bool, optional
            If True, hide the disorder profile tracks. Default is False.

        null_shuffle : bool or int, optional
            If False, show actual data. If an integer, use shuffled sequences
            to establish a null model. Recommended: 100 shuffles. Default is False.

        plot_rectangles : list, optional
            List of rectangle specifications to highlight regions of interest.
            Each element: [seq1_start, seq1_end, seq2_start, seq2_end, color, alpha, kwargs]
            Example: [[10, 30, 50, 80, 'red', 0.3, {}]]

        Returns
        -------
        tuple
            A 6-element tuple of matplotlib objects for customization:
            
            fig : matplotlib.figure.Figure
                The figure object.
                
            im : matplotlib.image.AxesImage
                The image object from imshow() - use for colorbar customization.
                
            ax_main : matplotlib.axes.Axes
                Main heatmap axes.
                
            ax_top : matplotlib.axes.Axes
                Top disorder profile axes.
                
            ax_right : matplotlib.axes.Axes
                Right disorder profile axes.
                
            ax_colorbar : matplotlib.axes.Axes
                Colorbar axes.

        Example
        -------
        Basic interaction figure:
        
            from finches.frontend.calvados_frontend import CALVADOS_frontend
            
            cf = CALVADOS_frontend()
            
            seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS" * 5  # 150 residues
            seq2 = "MGSWAEFKQRLAAIKTRLQALGGSEAELAAFEK" * 5  # 165 residues
            
            fig, im, ax_main, ax_top, ax_right, ax_cbar = cf.interaction_figure(
                seq1, seq2
            )
            
        Save figure with custom color scale:
        
            fig_data = cf.interaction_figure(
                seq1, seq2,
                vmin=-5, vmax=5,
                fname="interaction_map.png"
            )
            
        Highlight known domains and binding sites:
        
            fig_data = cf.interaction_figure(
                seq1, seq2,
                seq1_domains=[[10, 50], [100, 130]],  # Folded domains in seq1
                seq2_domains=[[20, 80]],               # Folded domain in seq2
                seq1_lines=[75],                       # Mark position 75 in seq1
                seq2_lines=[100, 150]                  # Mark positions in seq2
            )
            
        Customize the returned figure:
        
            fig, im, ax_main, ax_top, ax_right, ax_cbar = cf.interaction_figure(
                seq1, seq2
            )
            ax_main.set_title("Protein A vs Protein B", fontsize=14)
            fig.savefig("custom_figure.pdf", dpi=300, bbox_inches='tight')

        Raises
        ------
        ValueError
            If either sequence contains 'U' (RNA not supported by CALVADOS2).

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
                                          disorder_2=True,
                                          no_disorder=no_disorder,
                                          null_shuffle=null_shuffle,
                                          plot_rectangles=plot_rectangles)
    

    
    # ....................................................................................
    #
    #        
    def protein_nucleic_vector(seq, fragsize=31, smoothing_window=30, poly_order=3):
        """
        Calculate protein-nucleic acid interaction vector (NOT SUPPORTED in CALVADOS).
        
        This method is a stub that raises an exception. CALVADOS2 does not include
        parameters for RNA bases, so protein-RNA interaction calculations cannot
        be performed with this forcefield.
        
        For protein-RNA interaction calculations, use the Mpipi_frontend instead,
        which supports both protein-protein and protein-RNA interactions.
        
        Parameters
        ----------
        seq : str
            Protein sequence (not used - method raises exception).
            
        fragsize : int, optional
            Fragment size (not used - method raises exception). Default is 31.
            
        smoothing_window : int, optional
            Smoothing window (not used - method raises exception). Default is 30.
            
        poly_order : int, optional
            Polynomial order (not used - method raises exception). Default is 3.

        Returns
        -------
        None
            This method always raises an exception.

        Raises
        ------
        Exception
            Always raised - CALVADOS does not support RNA.

        Example
        -------
        This method will always fail - use Mpipi_frontend instead:
        
            from finches.frontend.calvados_frontend import CALVADOS_frontend
            from finches.frontend.mpipi_frontend import Mpipi_frontend
            
            # This will FAIL:
            cf = CALVADOS_frontend()
            # cf.protein_nucleic_vector(seq)  # Raises Exception!
            
            # Use Mpipi_frontend instead for RNA:
            mf = Mpipi_frontend()
            pnv = mf.protein_nucleic_vector(protein_seq, rna_type='polyU')

        See Also
        --------
        Mpipi_frontend.protein_nucleic_vector : Working implementation for protein-RNA
        
        """
        raise Exception('CALVADOS cannot currently handle RNA')
