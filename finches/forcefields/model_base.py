"""
Base class for forcefield models in finches.

This module defines the abstract base class that all forcefield models
(e.g., Mpipi_model, calvados_model) should inherit from to ensure a
consistent interface.

By: Alex S. Holehouse & Garrett M. Ginell
"""

from abc import ABC, abstractmethod
import numpy as np



class ForcefieldModel(ABC):
    """
    Abstract base class for forcefield models.
    
    This class defines the interface that all forcefield models must implement
    to be compatible with the InteractionMatrixConstructor and other finches
    analysis tools.
    
    Attributes
    ----------
    CONFIGS : dict
        Dictionary containing configuration information for the model,
        including 'charge_prefactor' and 'null_interaction_baseline'.
        
    version : str
        Version string identifying the specific parameterization of the model.
        
    dielectric : float
        Dielectric constant of the solvent (default 80.0 for water).
        
    salt : float
        Salt concentration in molar (e.g., 0.150 for 150 mM).
        
    ALL_RESIDUES_TYPES : list of lists
        Nested list defining which residue types can occur in the same sequence.
        Each sublist contains residues that are allowed together (e.g., 
        [['A','C',...standard AAs], ['U']] for proteins and RNA separately).
        
    conditions : list
        List of condition names that can be varied (e.g., ['salt', 'dielectric']).
    """
    
    # Class-level placeholder for configs - subclasses should define their own
    CONFIGS = {}
    
    def __init__(self, 
                 version: str,
                 dielectric: float,
                 salt: float,
                 temperature: float,
                 pH: float,
                 all_residues_types: list,
                 conditions: list):
        """
        Initialize the base forcefield model.
        
        Parameters
        ----------
        version : str
            Version string for the model parameterization.
            
        dielectric : float, optional
            Dielectric constant of the solvent. Default is 80.0 (water).
            
        salt : float, optional
            Salt concentration in molar. Default is 0.150 (150 mM).
            
        all_residues_types : list of lists, optional
            Nested list defining valid residue groupings. If None, defaults
            to standard 20 amino acids in a single group.
            
        conditions : list, optional
            List of tunable condition names; some models will allow you to
            vary environemntal conditions, and this list defines which of 
            those conditions can be varied. 
        """

        self.version = version
        self.dielectric = dielectric
        self.salt = salt
        self.pH = pH
        self.temperature = temperature
        
        self.ALL_RESIDUES_TYPES = all_residues_types            
        self.conditions = conditions
    
    @abstractmethod
    def compute_interaction_parameter(self, 
                                       residue_1: str, 
                                       residue_2: str, 
                                       r=None, 
                                       dielectric: float = None, 
                                       salt: float = None) -> tuple:
        """
        Compute the pairwise interaction parameter between two residue types.
        
        This is the core function that must be implemented by all forcefield
        models. It computes an interaction parameter based on the finite 
        integral of the potential energy between 1 and 3 sigma.
        
        Parameters
        ----------
        residue_1 : str
            One-letter code for the first residue.
            
        residue_2 : str
            One-letter code for the second residue.
            
        r : array-like, optional
            Distance array in Angstroms at which to compute the potential.
            If not provided, a default range should be used (typically
            0.1 to 30 Angstroms in 0.01 increments).
            
        dielectric : float, optional
            Dielectric constant. If None, uses the instance default.
            
        salt : float, optional
            Salt concentration in molar. If None, uses the instance default.
            
        Returns
        -------
        tuple
            Returns a tuple containing:
            
            [0] : float
                Interaction parameter (integral under the curve).
            [1] : np.ndarray
                Full pairwise potential energy vs. distance profile.
            [2] : int
                Index corresponding to one sigma in the distance array.
            [3] : int
                Index corresponding to three sigma in the distance array.
            [4] : np.ndarray
                Distance array used for the calculation.
        """
        pass
    
    def _get_default_distance_array(self, 
                                     start: float = 0.1, 
                                     stop: float = 30.0, 
                                     step: float = 0.01) -> np.ndarray:
        """
        Generate a default distance array for potential calculations.
        
        Parameters
        ----------
        start : float, optional
            Starting distance in Angstroms. Default is 0.1.
            
        stop : float, optional
            Ending distance in Angstroms. Default is 30.0.
            
        step : float, optional
            Step size in Angstroms. Default is 0.01.
            
        Returns
        -------
        np.ndarray
            Array of distances.
        """
        return np.arange(start, stop, step)
    
    def get_valid_residues(self) -> list:
        """
        Get a flat list of all valid residues across all residue groups.
        
        Returns
        -------
        list
            List of all valid residue one-letter codes.
        """
        return [res for group in self.ALL_RESIDUES_TYPES for res in group]
    
    def is_valid_residue(self, residue: str) -> bool:
        """
        Check if a residue is valid for this model.
        
        Parameters
        ----------
        residue : str
            One-letter residue code to check.
            
        Returns
        -------
        bool
            True if the residue is valid, False otherwise.
        """
        return residue in self.get_valid_residues()
    
    def validate_residue_pair(self, residue_1: str, residue_2: str) -> None:
        """
        Validate that both residues are valid for this model.
        
        Parameters
        ----------
        residue_1 : str
            First residue one-letter code.
            
        residue_2 : str
            Second residue one-letter code.
            
        Raises
        ------
        ValueError
            If either residue is not valid for this model.
        """
        if not self.is_valid_residue(residue_1):
            raise ValueError(f"Residue '{residue_1}' is not valid for {self.__class__.__name__}")
        if not self.is_valid_residue(residue_2):
            raise ValueError(f"Residue '{residue_2}' is not valid for {self.__class__.__name__}")
    
    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (f"{self.__class__.__name__}(version='{self.version}', "
                f"salt={self.salt}, dielectric={self.dielectric})")