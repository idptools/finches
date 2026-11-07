"""
Mpipi frontend for FINCHES.

Every analysis method is inherited from FinchesFrontend; this module only builds
the Mpipi-GG forcefield and states what is specific to it (RNA support and the
default intermap colour range).
"""

from finches import epsilon_calculation
from finches.forcefields.mpipi import Mpipi_model

from .frontend_base import FinchesFrontend


class Mpipi_frontend(FinchesFrontend):
    """
    Frontend class for Mpipi (GGv1) forcefield calculations.

    Mpipi_frontend provides a high-level interface for calculating protein-protein
    interaction parameters using the Mpipi coarse-grained forcefield. All analysis
    methods are inherited from FinchesFrontend, so Mpipi_frontend and
    CALVADOS_frontend expose an identical interface.

    The Mpipi forcefield models interactions based on:

    - Pi-pi stacking interactions (aromatic residues)
    - Cation-pi interactions
    - Electrostatic interactions (salt and dielectric dependent)
    - Hydrophobic interactions
    - Sequence-specific parameters from the Mpipi-GGv1 parameterization

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

    default_intermap_limit : float
        3.0: interaction_figure() draws the intermap on a -3 to +3 colour scale
        unless vmin/vmax are given.

    Example
    -------
    Basic usage for calculating epsilon between two proteins::

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

    See Also
    --------
    CALVADOS_frontend : Alternative frontend using the CALVADOS2 forcefield
    FinchesFrontend : Base class with shared functionality
    """

    supports_rna = True
    default_intermap_limit = 3.0

    def __init__(self, salt=0.150, dielectric=80.0):
        """
        Initialize the Mpipi_frontend with specified solution conditions.

        Creates an instance of the Mpipi-GGv1 forcefield model configured with
        the specified salt concentration and dielectric constant. These parameters
        affect electrostatic interactions in the model.

        Parameters
        ----------
        salt : float, optional
            Salt concentration in molar (M). Default is 0.150 M (150 mM).

        dielectric : float, optional
            Dielectric constant of the solution. Default is 80.0.

        Example
        -------
        Initialize with default physiological conditions::

            from finches.frontend.mpipi_frontend import Mpipi_frontend
            mf = Mpipi_frontend()

        Initialize for low salt conditions::

            mf = Mpipi_frontend(salt=0.050, dielectric=80.0)

        Initialize for organic solvent mixture (lower dielectric)::

            mf = Mpipi_frontend(salt=0.150, dielectric=60.0)

        """
        # call superclass constructor
        super().__init__()

        # initialize an Mpipi forcefield object
        self.model = Mpipi_model("Mpipi_GGv1", salt=salt, dielectric=dielectric)

        # build an interaction matrix constructor object
        self.IMC_object = epsilon_calculation.InteractionMatrixConstructor(self.model)
