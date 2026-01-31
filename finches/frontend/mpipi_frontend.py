# other stuff
import numpy as np

from finches import epsilon_calculation

# for model construction
from finches.forcefields.mpipi import Mpipi_model

from .frontend_base import FinchesFrontend


class Mpipi_frontend(FinchesFrontend):
    """
    Frontend class for Mpipi (GGv1) forcefield calculations.
    
    Mpipi_frontend provides a high-level interface for calculating protein-protein
    AND protein-RNA interaction parameters using the Mpipi coarse-grained forcefield.
    This class inherits from FinchesFrontend and adds Mpipi-specific functionality.
    
    KEY FEATURE: Unlike CALVADOS2, Mpipi SUPPORTS RNA sequences. RNA is represented
    using 'U' (uracil) to indicate poly-U RNA. This enables protein-RNA interaction
    calculations.
    
    The Mpipi forcefield models interactions based on:
    - Pi-pi stacking interactions (aromatic residues)
    - Cation-pi interactions
    - Electrostatic interactions (salt and dielectric dependent)
    - Hydrophobic interactions
    - Sequence-specific parameters from the Mpipi-GGv1 parameterization
    
    Example
    -------
    Basic usage for calculating epsilon between two proteins:
    
        from finches.frontend.mpipi_frontend import Mpipi_frontend
        
        # Initialize with default parameters
        mf = Mpipi_frontend()
        
        # Or with custom solution conditions
        mf = Mpipi_frontend(salt=0.100, dielectric=78.0)
        
        # Calculate epsilon between two protein sequences
        seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        seq2 = "MGSWAEFKQRLAAIKTRLQALGGSEAELAAFEK"
        eps = mf.epsilon(seq1, seq2)
        print(f"Protein-protein epsilon: {eps}")
        
    Protein-RNA interaction example:
    
        # Calculate protein-RNA interaction
        protein_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        rna_seq = "U" * 50  # 50-nucleotide poly-U RNA
        eps_rna = mf.epsilon(protein_seq, rna_seq)
        print(f"Protein-RNA epsilon: {eps_rna}")
        
        # Get per-residue protein-RNA interaction profile
        pnv = mf.protein_nucleic_vector(protein_seq, rna_type='polyU')
    
    See Also
    --------
    CALVADOS_frontend : Alternative frontend for protein-only calculations
    FinchesFrontend : Base class with shared functionality
    
    Notes
    -----
    When sequences contain 'U', disorder prediction is automatically disabled
    for that sequence since metapredict cannot predict disorder for RNA.
    
    """

    def __init__(self, salt=0.150, dielectric=80.0):
        """
        Initialize the Mpipi_frontend with specified solution conditions.
        
        Creates an instance of the Mpipi-GGv1 forcefield model configured with
        the specified salt concentration and dielectric constant. These parameters
        affect electrostatic interactions in the model.
        
        Parameters
        ----------
        salt : float, optional
            Salt concentration in molar (M). Default is 0.150 M (150 mM),
            which represents typical physiological conditions. Higher salt
            screens electrostatic interactions.
            
        dielectric : float, optional
            Dielectric constant of the solution. Default is 80.0, which is
            the approximate dielectric constant of water at 20°C. Lower
            dielectric increases the strength of electrostatic interactions.
            
        Attributes
        ----------
        model : Mpipi_model
            The underlying Mpipi-GGv1 forcefield model instance.
            
        IMC_object : InteractionMatrixConstructor
            Object for computing interaction matrices and epsilon values.
        
        Example
        -------
        Initialize with default physiological conditions:
        
            from finches.frontend.mpipi_frontend import Mpipi_frontend
            mf = Mpipi_frontend()
            
        Initialize for low salt conditions:
        
            mf = Mpipi_frontend(salt=0.050, dielectric=80.0)
            
        Initialize for organic solvent mixture (lower dielectric):
        
            mf = Mpipi_frontend(salt=0.150, dielectric=60.0)
        
        """
        # call superclass constructor
        super().__init__()

        # initialize an Mpipi forcefield object
        self.model = Mpipi_model("Mpipi_GGv1", salt=salt, dielectric=dielectric)

        # build an interaction matrix constructor object
        self.IMC_object = epsilon_calculation.InteractionMatrixConstructor(self.model)

    # functions defined in superclass listed below for clarity
    # epsilon() defined in super exclusively
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
        Calculate the sliding-window interaction matrix between two sequences.
        
        This method decomposes two sequences into overlapping fragments of size
        `window_size` and calculates pairwise epsilon values between all fragment
        pairs using a sliding window approach. The result is a 2D matrix where each
        cell (i, j) represents the interaction strength between fragment i from seq1
        and fragment j from seq2.
        
        The matrix is NOT padded, so edge positions depend on window_size. The method
        returns index arrays that map matrix positions to sequence positions.
        
        RNA SUPPORT: Sequences can contain 'U' (uracil) to represent poly-U RNA.
        When a sequence contains 'U', disorder prediction is automatically disabled
        for that sequence since metapredict cannot analyze RNA.

        Parameters
        ----------
        seq1 : str
            First sequence (protein amino acids or 'U' for RNA).
            Standard single-letter codes for proteins.

        seq2 : str
            Second sequence (protein amino acids or 'U' for RNA).
            Standard single-letter codes for proteins.

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

        disorder_1 : bool or None, optional
            If True, compute disorder profile for seq1 using metapredict.
            If False, use uniform profile (all 1s). If None (default),
            automatically set to False if seq1 contains 'U', else True.

        disorder_2 : bool or None, optional
            If True, compute disorder profile for seq2 using metapredict.
            If False, use uniform profile (all 1s). If None (default),
            automatically set to False if seq2 contains 'U', else True.

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
                All 1s if disorder_1=False or if seq1 contains 'U'.
                
            [2] disorder_2 : np.ndarray  
                Disorder profile for seq2 (values 0-1, higher = more disordered).
                All 1s if disorder_2=False or if seq2 contains 'U'.

        Example
        -------
        Calculate protein-protein interaction matrix:
        
            from finches.frontend.mpipi_frontend import Mpipi_frontend
            import numpy as np
            
            mf = Mpipi_frontend()
            
            seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS" * 3  # 90 residues
            seq2 = "MGSWAEFKQRLAAIKTRLQALGGSEAELAAFEK" * 3  # 99 residues
            
            matrix_data, dis1, dis2 = mf.intermolecular_idr_matrix(seq1, seq2)
            
            # Unpack the matrix data
            epsilon_matrix, seq1_indices, seq2_indices = matrix_data
            
            # Find the most attractive region
            min_idx = np.unravel_index(np.argmin(epsilon_matrix), epsilon_matrix.shape)
            print(f"Most attractive at seq1 pos {seq1_indices[min_idx[0]]}, "
                  f"seq2 pos {seq2_indices[min_idx[1]]}")
        
        Calculate protein-RNA interaction matrix:
        
            protein_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS" * 3
            rna_seq = "U" * 100  # 100-nt poly-U RNA
            
            # Disorder is automatically disabled for RNA sequence
            matrix_data, dis_prot, dis_rna = mf.intermolecular_idr_matrix(
                protein_seq, rna_seq
            )
            # dis_rna will be all 1s since RNA has no disorder prediction

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
        Generate a publication-ready interaction matrix figure between two sequences.
        
        Creates a comprehensive visualization showing the sliding-window interaction
        matrix as a heatmap with parallel disorder prediction tracks along the top
        and right edges. The heatmap uses a diverging colormap where purple indicates
        attractive (negative epsilon) regions and green indicates repulsive (positive
        epsilon) regions.
        
        This is the primary visualization method for understanding where along two
        sequences the strongest and weakest interactions occur.
        
        RNA SUPPORT: Sequences can contain 'U' (uracil) for poly-U RNA. When a
        sequence contains 'U', the disorder profile for that sequence is automatically
        set to uniform (all 1s) since metapredict cannot predict disorder for RNA.
        
        Parameters
        ----------
        seq1 : str
            First sequence (plotted on x-axis/horizontal). Can be protein
            (standard amino acids) or RNA ('U' characters for poly-U).

        seq2 : str
            Second sequence (plotted on y-axis/vertical). Can be protein
            (standard amino acids) or RNA ('U' characters for poly-U).

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

        linewidth : float, optional
            Width of domain boundary and marker lines. Default is 1.
                                                      
        vmin : float, optional
            Minimum value for the colorbar scale. Default is -3.
            More negative = stronger attraction (purple).

        vmax : float, optional
            Maximum value for the colorbar scale. Default is 3.
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
        Basic protein-protein interaction figure:
        
            from finches.frontend.mpipi_frontend import Mpipi_frontend
            
            mf = Mpipi_frontend()
            
            seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS" * 5  # 150 residues
            seq2 = "MGSWAEFKQRLAAIKTRLQALGGSEAELAAFEK" * 5  # 165 residues
            
            fig, im, ax_main, ax_top, ax_right, ax_cbar = mf.interaction_figure(
                seq1, seq2
            )
            
        Protein-RNA interaction figure:
        
            protein_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS" * 5
            rna_seq = "U" * 150  # 150-nt poly-U RNA
            
            # RNA sequence will have uniform disorder profile
            fig_data = mf.interaction_figure(protein_seq, rna_seq)
            
        Save figure with custom color scale:
        
            fig_data = mf.interaction_figure(
                seq1, seq2,
                vmin=-5, vmax=5,
                fname="interaction_map.png"
            )
            
        Highlight known domains and binding sites:
        
            fig_data = mf.interaction_figure(
                seq1, seq2,
                seq1_domains=[[10, 50], [100, 130]],  # Folded domains in seq1
                seq2_domains=[[20, 80]],               # Folded domain in seq2
                seq1_lines=[75],                       # Mark position 75 in seq1
                seq2_lines=[100, 150]                  # Mark positions in seq2
            )
            
        Customize the returned figure:
        
            fig, im, ax_main, ax_top, ax_right, ax_cbar = mf.interaction_figure(
                seq1, seq2
            )
            ax_main.set_title("FUS-G3BP1 Interaction", fontsize=14)
            fig.savefig("custom_figure.pdf", dpi=300, bbox_inches='tight')

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
        
