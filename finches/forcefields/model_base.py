"""
Base class for the forcefield models that FINCHES builds its interactions from.

Every forcefield model (Mpipi_model, CALVADOS_model, or one you write yourself)
inherits from ForcefieldModel. The InteractionMatrixConstructor only needs three
things from a model: the ALL_RESIDUES_TYPES residue groups, a CONFIGS dictionary
with the model's calibration constants, and compute_interaction_parameter(),
which turns a pair potential into the single number FINCHES uses for a residue
pair. This module defines that interface, plus the pieces of the calculation
that are the same for every model (the default distance grid and the integral
between one and three sigma), so each model only has to supply its potential.

See finches/forcefields/readme.md for a walk-through of adding a new model.

By: Alex S. Holehouse & Garrett M. Ginell
"""

from abc import ABC, abstractmethod

import numpy as np

# The 20 standard amino acids as one-letter codes. Every forcefield model covers
# these; a model that also covers nucleic acids adds them as a separate residue
# group (see ForcefieldModel.ALL_RESIDUES_TYPES).
STANDARD_AMINO_ACIDS = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)


class ForcefieldModel(ABC):
    """
    Abstract base class for forcefield models.

    Defines the interface a forcefield model must implement to be used by the
    InteractionMatrixConstructor and the rest of FINCHES, and implements the parts
    of the interaction-parameter calculation that do not depend on the model.

    Attributes
    ----------
    CONFIGS : dict
        Calibration constants for the model version, with the keys
        'charge_prefactor' and 'null_interaction_baseline'. A constant that has
        not been calibrated is stored as NaN. Subclasses set this in their
        constructor.

    version : str
        Name of the parameter set the model was built with.

    dielectric : float
        Dielectric constant of the solvent. For a model whose dielectric is a
        function of temperature (CALVADOS) this is the computed value.

    salt : float
        Monovalent salt concentration in molar (e.g. 0.150 for 150 mM).

    temperature : float
        Temperature in Kelvin.

    pH : float
        Solution pH.

    ALL_RESIDUES_TYPES : list of list of str
        The residue groups the model has parameters for. Residues within one group
        may appear together in a sequence, while a sequence may not mix residues
        from different groups.

    conditions : list of str
        Names of the attributes above that actually change the model's energies
        (e.g. ['salt', 'dielectric'] for Mpipi). The other attributes are stored
        for a uniform interface but have no effect. Condition-scanning code checks
        this list before varying an attribute.
    """

    # Class-level placeholder for configs - subclasses define their own
    CONFIGS = {}

    def __init__(
        self,
        version: str,
        dielectric: float,
        salt: float,
        temperature: float,
        pH: float,
        all_residues_types: list,
        conditions: list,
    ):
        """
        Initialize the base forcefield model.

        Parameters
        ----------
        version : str
            Name of the parameter set.

        dielectric : float
            Dielectric constant of the solvent (e.g. 80.0 for water).

        salt : float
            Monovalent salt concentration in molar (e.g. 0.150 for 150 mM).

        temperature : float
            Temperature in Kelvin.

        pH : float
            Solution pH.

        all_residues_types : list of list of str
            Residue groups the model covers, where each sublist contains
            residues allowed to occur together in a sequence.

        conditions : list of str
            Names of the environmental conditions (among 'salt', 'dielectric',
            'temperature' and 'pH') that the model's energies depend on.
        """
        self.version = version
        self.dielectric = dielectric
        self.salt = salt
        self.pH = pH
        self.temperature = temperature

        self.ALL_RESIDUES_TYPES = all_residues_types
        self.conditions = conditions

    @abstractmethod
    def compute_interaction_parameter(
        self,
        residue_1: str,
        residue_2: str,
        r=None,
        dielectric: float = None,
        salt: float = None,
    ) -> tuple:
        """
        Compute the interaction parameter between two residue types.

        This is the one method every forcefield model must implement, because it
        is what the InteractionMatrixConstructor calls to build its lookup table.
        The interaction parameter is the integral of the pair potential between
        one and three sigma (see _integrate_sigma_window()).

        Parameters
        ----------
        residue_1 : str
            One-letter code for the first residue.

        residue_2 : str
            One-letter code for the second residue.

        r : array-like, optional
            Distances in Angstroms at which to evaluate the potential. If None, a
            default grid is used (0.1 to 30 Angstroms in 0.01 Angstrom steps).

        dielectric : float, optional
            Dielectric constant. If None, the model's own value is used. Models
            whose dielectric is not a free parameter ignore this.

        salt : float, optional
            Monovalent salt concentration in molar. If None, the model's own value
            is used.

        Returns
        -------
        tuple
            A 5-element tuple:

            [0] : float
                The interaction parameter (the integral from one to three sigma).
            [1] : np.ndarray
                The pair potential (kJ/mol) at every distance in [4].
            [2] : int
                Index in [4] closest to one sigma.
            [3] : int
                Index in [4] closest to three sigma.
            [4] : np.ndarray
                The distances (Angstroms) the potential was evaluated at.

        Raises
        ------
        KeyError
            If the model has no parameters for one of the residues. The
            InteractionMatrixConstructor relies on this being a KeyError.
        """

    def compute_full_potential(
        self, residue_1: str, residue_2: str, r, dielectric=None, salt=None
    ):
        """
        Compute the full pair potential between two residue types.

        Every model in FINCHES implements this, so the potential of any model can
        be evaluated through one name. Models also keep their original,
        model-specific name (compute_full_Mpipi(), compute_full_calvados()).

        Parameters
        ----------
        residue_1 : str
            One-letter code for the first residue.

        residue_2 : str
            One-letter code for the second residue.

        r : float or array-like
            Distance(s) in Angstroms.

        dielectric : float, optional
            Dielectric constant. If None, the model's own value is used.

        salt : float, optional
            Monovalent salt concentration in molar. If None, the model's own
            value is used.

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance; a float if r is a scalar.

        Raises
        ------
        NotImplementedError
            If the model does not implement it.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_full_potential()"
        )

    def _get_default_distance_array(
        self, start: float = 0.1, stop: float = 30.0, step: float = 0.01
    ) -> np.ndarray:
        """
        Build the default distance grid for evaluating a potential.

        Parameters
        ----------
        start : float, optional
            First distance. Default is 0.1.

        stop : float, optional
            End of the grid (exclusive). Default is 30.0.

        step : float, optional
            Grid spacing. Default is 0.01.

        Returns
        -------
        np.ndarray
            np.arange(start, stop, step). The defaults are in Angstroms; a model
            that works in other units can pass its own values.
        """
        return np.arange(start, stop, step)

    @staticmethod
    def _integrate_sigma_window(energy, r_index, sigma, x=None):
        """
        Integrate a pair potential between one and three sigma.

        This is how FINCHES turns a distance-dependent potential into a single
        interaction parameter. The integration limits are the grid points closest
        to sigma and to three sigma, and the integral is the trapezoidal rule over
        the points from the first limit up to (but not including) the second.

        Parameters
        ----------
        energy : np.ndarray
            The potential evaluated at every distance in r_index.

        r_index : np.ndarray
            Distance grid used to locate the limits. Must be in the same units as
            sigma.

        sigma : float
            The pair's sigma, in the units of r_index.

        x : np.ndarray, optional
            Distance grid to integrate over, if it is in different units from
            r_index (CALVADOS locates the limits in nm but integrates over
            Angstroms). Defaults to r_index.

        Returns
        -------
        tuple
            (interaction_parameter, s1, s3): the integral as a float, and the
            indices of the one-sigma and three-sigma limits.
        """
        if x is None:
            x = r_index

        s1 = np.argmin(np.abs(sigma - r_index))
        s3 = np.argmin(np.abs(sigma * 3 - r_index))

        try:
            interaction_parameter = np.trapezoid(energy[s1:s3], x=x[s1:s3])
        except AttributeError:
            # np.trapezoid was added in NumPy 2.0; fall back to the old name
            interaction_parameter = np.trapz(energy[s1:s3], x=x[s1:s3])

        return interaction_parameter, s1, s3

    def get_valid_residues(self) -> list:
        """
        Get a flat list of every residue the model covers.

        Returns
        -------
        list of str
            One-letter codes from all residue groups in ALL_RESIDUES_TYPES.
        """
        return [res for group in self.ALL_RESIDUES_TYPES for res in group]

    def is_valid_residue(self, residue: str) -> bool:
        """
        Check whether the model covers a residue.

        Parameters
        ----------
        residue : str
            One-letter residue code.

        Returns
        -------
        bool
            True if the residue is in one of the model's residue groups.
        """
        return residue in self.get_valid_residues()

    def validate_residue_pair(self, residue_1: str, residue_2: str) -> None:
        """
        Check that the model covers both residues of a pair.

        Parameters
        ----------
        residue_1 : str
            First residue one-letter code.

        residue_2 : str
            Second residue one-letter code.

        Raises
        ------
        ValueError
            If either residue is not covered by the model.
        """
        for residue in (residue_1, residue_2):
            if not self.is_valid_residue(residue):
                raise ValueError(
                    f"Residue '{residue}' is not valid for {self.__class__.__name__}"
                )

    def __repr__(self) -> str:
        """
        Return a one-line description of the model.

        Returns
        -------
        str
            The class name, the version, and the value of each condition the
            model depends on.
        """
        conditions = ", ".join(f"{c}={getattr(self, c)}" for c in self.conditions)
        return f"{self.__class__.__name__}(version='{self.version}', {conditions})"
