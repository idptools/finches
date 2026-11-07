"""
CALVADOS frontend for FINCHES.

Every analysis method is inherited from FinchesFrontend; this module only builds
the CALVADOS2 forcefield and states what is specific to it (no RNA parameters and
the default intermap colour range).
"""

from finches import epsilon_calculation
from finches.forcefields.calvados import CALVADOS_model

from .frontend_base import FinchesFrontend


class CALVADOS_frontend(FinchesFrontend):
    """
    Frontend class for CALVADOS2 forcefield calculations.

    CALVADOS_frontend provides a high-level interface for calculating protein-protein
    interaction parameters using the CALVADOS2 coarse-grained forcefield. All
    analysis methods are inherited from FinchesFrontend, so CALVADOS_frontend and
    Mpipi_frontend expose an identical interface.

    The CALVADOS2 forcefield models interactions based on:

    - Hydrophobic/hydrophilic interactions (stickiness parameters)
    - Electrostatic interactions (salt dependent; pH sets the histidine charge)
    - Sequence-specific parameters calibrated from experimental data

    Parameters
    ----------
    salt : float, optional
        Salt concentration in molar (M). Default is 0.150 M (150 mM),
        which represents typical physiological conditions.

    pH : float, optional
        Solution pH. Default is 7.4 (physiological pH). In CALVADOS2 the pH
        sets the (fractional) charge of histidine only; Asp, Glu, Lys and Arg
        carry their full charge at every pH.

    temperature : float, optional
        Temperature in Kelvin. Default is 288 K (15°C).

    Attributes
    ----------
    model : CALVADOS_model
        The underlying CALVADOS2 forcefield model instance.

    IMC_object : InteractionMatrixConstructor
        Object for computing interaction matrices and epsilon values.

    default_intermap_limit : float
        7.5: interaction_figure() draws the intermap on a -7.5 to +7.5 colour
        scale unless vmin/vmax are given, which suits the larger numerical range
        of CALVADOS2 epsilon values.

    Example
    -------
    Basic usage for calculating epsilon between two proteins::

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

        # Generate an interaction figure (short sequences need a small window)
        fig, im, ax_main, ax_top, ax_right, ax_cbar = cf.interaction_figure(
            seq1, seq2, window_size=11, tic_frequency=10)

    See Also
    --------
    Mpipi_frontend : Alternative frontend using the Mpipi-GG forcefield
    FinchesFrontend : Base class with shared functionality

    """

    supports_rna = False
    default_intermap_limit = 7.5

    def __init__(self, salt=0.150, pH=7.4, temperature=288):
        """
        Initialize the CALVADOS_frontend with specified solution conditions.

        Creates an instance of the CALVADOS2 forcefield model configured with
        the specified salt concentration, pH, and temperature. These parameters
        affect electrostatic interactions in the model.

        Parameters
        ----------
        salt : float, optional
            Salt concentration in molar (M). Default is 0.150 M (150 mM).

        pH : float, optional
            Solution pH. Default is 7.4. Sets the charge of histidine only.

        temperature : float, optional
            Temperature in Kelvin. Default is 288 K (15°C).

        Example
        -------
        Initialize with default physiological conditions::

            from finches.frontend.calvados_frontend import CALVADOS_frontend
            cf = CALVADOS_frontend()

        Initialize for low salt conditions::

            cf = CALVADOS_frontend(salt=0.050, pH=7.4, temperature=298)

        Initialize for acidic conditions::

            cf = CALVADOS_frontend(salt=0.150, pH=5.5, temperature=310)

        """
        # call superclass constructor
        super().__init__()

        # initialize the CALVADOS forcefield object
        self.model = CALVADOS_model(
            "CALVADOS2", salt=salt, pH=pH, temperature=temperature
        )

        # build an interaction matrix constructor object
        self.IMC_object = epsilon_calculation.InteractionMatrixConstructor(self.model)
