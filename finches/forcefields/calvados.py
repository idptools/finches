"""
This Script pulls and builds the pair-wise potentials for the calvados model
see here:

https://github.com/KULL-Centre/CALVADOS

The Code below is then paired with a jupyter-notebook, and contains code
directly pulled and adapted from the calvados package.

Orignally writen by Garrett M. Ginell (2023.03.07)
Revised by ~ash 2026-01-11

######################################################################
BUILDING THE MAIN ENERGY POTENTIAL
#
# NOTES:
#  direct_coexistance uses openMM
#  single_chain uses hoomd
#
#  Lennard-Jones potential is a Ashbaugh-Hatch potential
#  Debye-Hückel potential is a Yukawa potentials
#
#
# FOR:
#
#  General SI and explanation:
#    https://www.pnas.org/doi/suppl/10.1073/pnas.2111696118/suppl_file/pnas.2111696118.sapp.pdf
#
#  Prefactors calculation:
#    (openmm)
#    https://github.com/KULL-Centre/CALVADOS/blob/main/direct_coexistence/simulate.py
#     - SEE lines 19-20
#    https://github.com/KULL-Centre/CALVADOS/blob/main/direct_coexistence/analyse.py
#     - SEE fxns genParamsLJ & genParamsDH
#
#    (hoomd)
#    https://github.com/KULL-Centre/CALVADOS/blob/main/single_chain/simulate.py
#     - SEE fxn genParams (line 18) & lines 65-73, 99-108
#
#  Potential calculation:
#    (openmm)
#    https://github.com/KULL-Centre/CALVADOS/blob/main/direct_coexistence/simulate.py
#     - SEE lines 92-105
#
#    (hoomd)
#    DH - https://github.com/joaander/hoomd-blue/blob/master/hoomd/md/EvaluatorPairYukawa.h
#    LJ - https://github.com/mphowardlab/azplugins/blob/af9bc8d63c4b45c3e79c756dc1b47b3a1ac44795/
                    azplugins/PairEvaluatorAshbaugh.h
#
#######################################################################
"""

import pickle
import numpy as np
import finches
from os.path import exists
from finches.forcefields.model_base import ForcefieldModel

VALID_AA = [
    "M",
    "G",
    "K",
    "T",
    "Y",
    "A",
    "D",
    "E",
    "V",
    "L",
    "Q",
    "W",
    "R",
    "F",
    "S",
    "H",
    "N",
    "P",
    "C",
    "I",
]

CALVADOS_CONFIGS = {}
CALVADOS_CONFIGS["CALVADOS1"] = {}
CALVADOS_CONFIGS["CALVADOS2"] = {}

CALVADOS_CONFIGS["CALVADOS1"]["charge_prefactor"] = np.nan  # not computed yet
CALVADOS_CONFIGS["CALVADOS2"]["charge_prefactor"] = 0.7  # 1.442590

CALVADOS_CONFIGS["CALVADOS1"]["null_interaction_baseline"] = np.nan  # not computed yet
CALVADOS_CONFIGS["CALVADOS2"][
    "null_interaction_baseline"
] = -0.45  # note we bump this slightly from -0.478


class CALVADOS_model(ForcefieldModel):
    def __init__(
        self,
        version="CALVADOS2",
        salt=0.150,
        pH=7.4,
        temperature=288,
        input_directory="default",
    ):
        """
        The CALVADOS_model class defines a CALVADOS_model Object which lets you calculate
        and return both individual components of the calvados forcefield potential
        alongside an 'interaction parameter' that reflects pairwise residue interactions.

        Parameters
        ---------------
        version : str
            Defines the version of the CALVADOS parameters to use for the model.
            The options here are based off of those defined in the CALVADOS data files.

            Select CALVADOS1 or CALVADOS2 to choose which stickiness parameters to use.

            Current options are: ['CALVADOS1', 'CALVADOS2']

        salt : float
            Defines the general salt concentration to build the reference model.
            This salt value tunes the electrostatic interactions. Default is 0.15 M.

        pH : float
            Defines the general pH to build the reference model. Default is 7.4.

        temperature : float
            Defines the temperature at which the forcefield model is computed.
            Functionally this really just modulates the strengths of the Yukawa
            potentials - the Ashbaugh-Hatch potentials are not temperature dependent.

        input_directory : str
            Defines the directory where the input parameter files are found.

            If 'default', the data is pulled from finches.data.calvados.

            The directory should contain a calvados_residues.pickle file with
            residue parameters. In this way, you could in principle define
            and save arbitrary parameters and then feed these into the Calvados_model
            via the input_directory keyword.

        Returns
        -------------
        Calvados_model : obj
            A finches.forcefields.Calvados_model object that can then be passed to
            the InteractionMatrixConstructor class.

        """

        # Call parent constructor - note CALVADOS uses temp-dependent dielectric
        # so we compute it later, but pass a placeholder here
        super().__init__(
            version=version,
            dielectric=80.0,  # placeholder, will be computed from temp
            salt=salt,
            temperature=temperature,
            pH=pH,
            all_residues_types=[VALID_AA],
            conditions=["salt", "pH", "temperature"],
        )

        # if 'default' is passed, use the default parameters
        if input_directory == "default":
            data_prefix = finches.get_data("calvados")
        else:
            data_prefix = input_directory

        # check files are present
        if not exists(f"{data_prefix}/calvados_residues_dict.pickle"):
            raise Exception(
                f"Using [{data_prefix}] as our data directory but no calvados_residues_dict.pickle file found"
            )

        # Load residue parameters from pickle
        # Dictionary is keyed by one-letter code with values:
        # 'three', 'one', 'MW', 'sigmas', 'q', 'CALVADOS1', 'CALVADOS2'
        with open(f"{data_prefix}/calvados_residues_dict.pickle", "rb") as fh:
            residue_dict = pickle.load(fh)

        # Build internal residue params with version-specific lambdas
        self._residue_params = {}
        for one_letter, params in residue_dict.items():
            self._residue_params[one_letter] = {
                "three": params["three"],
                "one": one_letter,
                "MW": params["MW"],
                "sigmas": params["sigmas"],
                "q": params["q"],
                "lambdas": params[version],  # Use version-specific lambda values
            }

        # Validate version
        if version not in ["CALVADOS1", "CALVADOS2"]:
            raise Exception(f"""Passed version of model unknown: {version}
                                Available versions are: [CALVADOS1, CALVADOS2]""")

        # Set precomputed forcefield config parameters
        self.CONFIGS = CALVADOS_CONFIGS[version]

        # Calculate other parameters based off of defaults in the CALVADOS model
        # line 65 single_chain/simulate.py & (line 138 of direct_coexistence/analyse.py)
        self.eps_factor = 0.2

        # line 65 single_chain/simulate.py
        self.lj_eps = 4.184 * self.eps_factor

        # line 29 of direct_coexistence/submit.py
        self.cutoff = 2.0

        # line 105 - single_chain/simulate.py & line 104 direct_coexistence/simulate.py
        self.yukawa_r_cut = 4.0

        # Generate ionic, pH, temp specific parameters
        self._genParams()

        # Build pairwise sigma and lambda maps as 2D dictionaries
        self._build_pairwise_maps()

    # .....................................................................................
    #
    def _genParams(self):
        """
        Function to generate the temperature/pH-dependent parameters for the calvados calculations -
        specifically the kappa and epsilon values for the yukawa potential, and updates the
        Histidine charge based on the pH of the solution.

        Function directly pulled from line 18 of calvados/single_chain/simulate.py
        """
        RT = 8.3145 * self.temperature * 1e-3

        # Set the charge on HIS based on the pH of the protein solution
        # NOTE this calculates the charge (q) for residue 'H' at the given pH using 1/(1+10^(pH-6))
        self._residue_params["H"]["q"] = 1.0 / (1 + 10 ** (self.pH - 6))

        # Calculate temperature-dependent dielectric
        def fepsw(T):
            return (
                5321 / T
                + 233.76
                - 0.9297 * T
                + 0.1417 * 1e-2 * T * T
                - 0.8292 * 1e-6 * T**3
            )

        epsw = fepsw(self.temperature)

        # Update instance dielectric with computed value
        self.dielectric = epsw

        lB = 1.6021766**2 / (4 * np.pi * 8.854188 * epsw) * 6.022 * 1000 / RT

        # Calculate the inverse of the Debye length
        self.yukawa_kappa = np.sqrt(8 * np.pi * lB * self.salt * 6.022 / 10)

        # Store lB*RT for yukawa_eps calculation
        self._lB_RT = lB * RT

    # .....................................................................................
    #
    def _build_pairwise_maps(self):
        """
        Build 2D dictionaries for sigma, lambda, and yukawa_eps pairwise parameters.
        These are averages of single-residue parameters following CALVADOS convention
        (from lines 70-73 of single_chain/simulate.py).


        """
        self.SIGMA_ALL = {}
        self.LAMBDA_ALL = {}
        self.YUKAWA_EPS_ALL = {}

        for r1 in VALID_AA:
            self.SIGMA_ALL[r1] = {}
            self.LAMBDA_ALL[r1] = {}
            self.YUKAWA_EPS_ALL[r1] = {}

            for r2 in VALID_AA:
                # Average sigma: (sigma_r1 + sigma_r2) / 2
                self.SIGMA_ALL[r1][r2] = (
                    self._residue_params[r1]["sigmas"]
                    + self._residue_params[r2]["sigmas"]
                ) / 2

                # Average lambda: (lambda_r1 + lambda_r2) / 2
                self.LAMBDA_ALL[r1][r2] = (
                    self._residue_params[r1]["lambdas"]
                    + self._residue_params[r2]["lambdas"]
                ) / 2

                # Yukawa epsilon: q1 * q2 * lB * RT
                self.YUKAWA_EPS_ALL[r1][r2] = (
                    self._residue_params[r1]["q"]
                    * self._residue_params[r2]["q"]
                    * self._lB_RT
                )

    # .....................................................................................
    #
    def compute_ashbaugh_hatch(self, residue_1, residue_2, r):
        """
        Function that returns the values in kJ/mol for the Ashbaugh-Hatch (LJ)
        potential associated with the pairwise interaction of the two
        passed residues.

        Parameters
        -------------
        residue_1 : str
            Must be one of the 20 valid amino acid one-letter codes

        residue_2 : str
            Must be one of the 20 valid amino acid one-letter codes

        r : int, float, array-like
            Actual distance (in Angstroms) between the beads. Can be a single
            number or a numpy array. Note: internally converted to nm for calculation.

        Returns
        ------------
        float or np.array
            Returns energy in kJ/mol that corresponds to the distance provided.

        """
        s = self.SIGMA_ALL[residue_1][residue_2]
        lam = self.LAMBDA_ALL[residue_1][residue_2]

        return ashbaugh_hatch(r * 0.1, s, lam, cutoff=self.cutoff, lj_eps=self.lj_eps)

    # .....................................................................................
    #
    def compute_yukawa(self, residue_1, residue_2, r):
        """
        Function that returns the values in kJ/mol for the Yukawa (screened Coulomb)
        potential associated with the pairwise interaction of the two
        passed residues.

        Parameters
        -------------
        residue_1 : str
            Must be one of the 20 valid amino acid one-letter codes

        residue_2 : str
            Must be one of the 20 valid amino acid one-letter codes

        r : int, float, array-like
            Actual distance (in Angstroms) between the beads. Can be a single
            number or a numpy array. Note: internally converted to nm for calculation.

        Returns
        ------------
        float or np.array
            Returns energy in kJ/mol that corresponds to the distance provided.

        """
        q = self.YUKAWA_EPS_ALL[residue_1][residue_2]

        return yukawa(r * 0.1, q, self.yukawa_kappa, yukawa_r_cut=self.yukawa_r_cut)

    # .....................................................................................
    #
    def compute_full_calvados(self, residue_1, residue_2, r):
        """
        Function that returns the values in kJ/mol for the full calvados
        potential associated with the pairwise interaction of the two
        passed residues.

        Takes two residues and an input distance, which can be a single
        value or a np.array.

        Parameters
        ----------
        residue_1 : str
            Must be one of the 20 valid amino acid one-letter codes

        residue_2 : str
            Must be one of the 20 valid amino acid one-letter codes

        r : int, float, or list/array like
            Array of distances in Angstroms at which to compute the
            calvados potential. Can be a single value or an array.

        Returns
        -------
        float or np.array
            Returns energy in kJ/mol that corresponds to the distance provided.

        """
        # Get parameters
        s = self.SIGMA_ALL[residue_1][residue_2]
        lam = self.LAMBDA_ALL[residue_1][residue_2]
        q = self.YUKAWA_EPS_ALL[residue_1][residue_2]

        # Track if input was scalar
        is_scalar = isinstance(r, (int, float))

        # Convert to numpy array and Angstroms to nm (CALVADOS uses nm internally)
        r_nm = np.atleast_1d(r) * 0.1

        # Compute energy using vectorized functions
        energies = compute_calvados_energy(
            r_nm,
            s,
            lam,
            q,
            self.yukawa_kappa,
            cutoff=self.cutoff,
            lj_eps=self.lj_eps,
            yukawa_r_cut=self.yukawa_r_cut,
        )

        # Return scalar if input was scalar
        if is_scalar:
            return energies[0] if isinstance(energies, np.ndarray) else energies
        return energies

    # .....................................................................................
    #
    def compute_interaction_parameter(
        self, residue_1, residue_2, r=None, dielectric=None, salt=None
    ):
        """
        NOTE - the name of this function must match name in other forcefield
        modules.

        Standalone function that computes pairwise interaction parameter
        between two residue types based on the finite integral between 1
        and 3 sigma.

        Parameters
        --------------
        residue_1 : str
            Must be one of the 20 valid amino acid one-letter codes

        residue_2 : str
            Must be one of the 20 valid amino acid one-letter codes

        r : array-like
            Actual distance (in Angstroms) between the beads. Can be a single
            number or a numpy array. If not provided uses 0.1 to 30 in
            increments of 0.01 Angstroms.

        dielectric : float
            Not used for CALVADOS (temperature-dependent). Included for interface
            compatibility with base class.

        salt : float
            Not used in this method (salt is set at initialization). Included for
            interface compatibility with base class.

        Returns
        -----------
        tuple
            Returns a tuple with several values:
            [0] - float -  interaction parameter (integral under sigma to 3*sigma)
            [1] - np.array - full pairwise potential energy vs. distance profile
            [2] - int - index for one sigma (in Angstroms)
            [3] - int - index for 3 sigma (in Angstroms)
            [4] - np.array - distance array (in Angstroms)

        """
        if r is None:
            # Default range in nm, then convert
            r_nm = np.arange(0.01, 3, 0.001)
        else:
            # Convert Angstroms to nm
            r_nm = np.asarray(r) * 0.1

        # Convert to Angstroms for output and compute_full_calvados()
        r_angstroms = r_nm * 10

        # Determine sigma bounds (note sigma is in nm in CALVADOS)
        sig1 = self.SIGMA_ALL[residue_1][residue_2]
        sig3 = self.SIGMA_ALL[residue_1][residue_2] * 3

        # Get index in distance-dependent energy that matches 1 sigma
        # (note both r_nm and sig in nm)
        s1 = np.argmin(np.abs(sig1 - r_nm))

        # Get index of 3*sigma
        s3 = np.argmin(np.abs(sig3 - r_nm))

        # Calculate the combined energy vector of the range
        combo = self.compute_full_calvados(residue_1, residue_2, r_angstroms)

        # Take the numerical finite integral between 1 and 3 sigma to calculate
        # an interaction parameter
        try:
            interaction_param = np.trapezoid(combo[s1:s3], x=r_angstroms[s1:s3])
        except AttributeError:
            interaction_param = np.trapz(combo[s1:s3], x=r_angstroms[s1:s3])

        return (interaction_param, combo, s1, s3, r_angstroms)


######################################
# non - parameter specific functions #
######################################


# .....................................................................................
#
def ashbaugh_hatch(r, s, lam, cutoff=2.0, lj_eps=4.184 * 0.2):
    """
    Compute the Ashbaugh-Hatch potential energy at distance r (vectorized).

    Parameters
    ----------
    r : float or array-like
        Distance in nanometers

    s : float
        Sigma parameter (nm)

    lam : float
        Lambda parameter (stickiness)

    cutoff : float
        Cutoff distance in nm used to shift the potential (default 2.0)

    lj_eps : float
        Epsilon parameter for LJ potential (kJ/mol)

    Returns
    -------
    float or array
        Energy in kJ/mol
    """
    r = np.atleast_1d(r)
    rc = cutoff
    eps = lj_eps

    shift = (s / rc) ** 12 - (s / rc) ** 6

    # Precompute (s/r)^6 and (s/r)^12 for all r values
    s_over_r_6 = (s / r) ** 6
    s_over_r_12 = s_over_r_6 * s_over_r_6

    # Vectorized step function: x_out = 1 where r >= 2^(1/6)*s, else 0
    # This is equivalent to: step(r - 2^(1/6)*s)
    threshold = 2 ** (1 / 6) * s
    mask = r >= threshold  # True where we use y, False where we use z

    # Compute both branches for all r values
    # y = 4*eps*lam*((s/r)^12 - (s/r)^6 - shift)  -- attractive regime
    # z = 4*eps*((s/r)^12 - (s/r)^6 - lam*shift) + eps*(1-lam)  -- repulsive regime
    y = 4 * eps * lam * (s_over_r_12 - s_over_r_6 - shift)
    z = 4 * eps * (s_over_r_12 - s_over_r_6 - lam * shift) + eps * (1 - lam)

    # Select y where mask is True, z where mask is False
    result = np.where(mask, y, z)

    return result[0] if len(result) == 1 else result


# .....................................................................................
#
def yukawa(r, q, yukawa_kappa, yukawa_r_cut=4.0):
    """
    Compute the Yukawa (screened Coulomb) potential energy at distance r (vectorized).

    Parameters
    ----------
    r : float or array-like
        Distance in nanometers

    q : float
        Charge product prefactor (q1*q2*lB*RT)

    yukawa_kappa : float
        Inverse Debye length

    yukawa_r_cut : float
        Cutoff distance (nm)

    Returns
    -------
    float or array
        Energy in kJ/mol
    """
    r = np.atleast_1d(r)

    # Already vectorized - just compute shift once and apply to all r
    shift = np.exp(-yukawa_kappa * yukawa_r_cut) / yukawa_r_cut
    result = q * (np.exp(-yukawa_kappa * r) / r - shift)

    return result[0] if len(result) == 1 else result


# .....................................................................................
#
def compute_calvados_energy(
    r, s, lam, q, yukawa_kappa, cutoff=2.0, lj_eps=4.184 * 0.2, yukawa_r_cut=4.0
):
    """
    Compute the full calvados potential energy between two residues (vectorized).

    NOTE - to ensure we can use the native CALVADOS parameters and implementation,
    distance here (r) must be in nanometers as opposed to Angstroms. We use Angstroms
    for consistency with finches everywhere else, but the forcefield code that in
    principle is "internal" operates in whatever units the native forcefield uses.

    Note this combines an Ashbaugh-Hatch potential with a Yukawa potential. The
    Ashbaugh-Hatch potential is a Lennard-Jones potential with a shifted cutoff,
    and the Yukawa potential is a screened Coulombic potential.

    Parameters
    --------------
    r : float or array-like
        Distance in nanometers between two residues. Can be scalar or array.

    s : float
        Sigma parameter for the LJ potential

    lam : float
        Lambda parameter for the LJ potential

    q : float
        Charge parameter for the Yukawa potential

    yukawa_kappa : float
        Kappa parameter for the Yukawa potential

    cutoff : float
        Cutoff distance for the LJ potential

    lj_eps : float
        Epsilon parameter for the LJ potential

    yukawa_r_cut : float
        Cutoff distance for the Yukawa potential

    Returns
    -----------
    float or array
        The total potential energy between the two residues at
        the given distance(s).

    """
    ah = ashbaugh_hatch(r, s, lam, cutoff=cutoff, lj_eps=lj_eps)
    yu = yukawa(r, q, yukawa_kappa, yukawa_r_cut=yukawa_r_cut)

    return yu + ah


# Backwards compatibility alias
calvados_model = CALVADOS_model
