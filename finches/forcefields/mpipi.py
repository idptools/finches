"""
The Mpipi forcefield for FINCHES.

Implements the Mpipi coarse-grained forcefield of Joseph et al. (2021): a
Wang-Frenkel short-range potential plus a Debye-Huckel screened Coulomb potential
between one-bead-per-residue particles. Mpipi_model reads a parameter set,
evaluates these potentials for any residue pair, and turns them into the
interaction parameters FINCHES builds its interaction matrices from. The
stand-alone potentials (wang_frenkel(), coulomb() and harmonic()) are also exposed
at module level.

Three parameter sets are available (see Mpipi_model):

* 'Mpipi_GGv1' (default): the Mpipi-GG re-parameterization used throughout
  FINCHES, which covers the 20 amino acids and the RNA nucleotide U.
* 'Mpipi_original': the unmodified Joseph et al. parameters, amino acids only.
* 'OLD_Mpipi_GGv1': a legacy snapshot of the GGv1 parameters kept only so that
  old results can be reproduced.

Reference
---------
Joseph, J. A., Reinhardt, A., Aguirre, A., Chew, P. Y., Russell, K. O.,
Espinosa, J. R., Garaizar, A., & Collepardo-Guevara, R. (2021). Physics-driven
coarse-grained model for biomolecular phase separation with near-quantitative
accuracy. Nature Computational Science, 1(11), 732-743.

By: Alex S. Holehouse & Garrett M. Ginell
"""

import pickle
from os.path import exists

import numpy as np

import finches
from finches.forcefields.model_base import ForcefieldModel, STANDARD_AMINO_ACIDS

# residues Mpipi can cover: the 20 amino acids and, for the GGv1 parameters, the
# RNA nucleotide U (poly-U)
VALID_AA = list(STANDARD_AMINO_ACIDS)
VALID_RNA = ["U"]

# Calibration constants per version. The charge prefactor was fitted as described
# in the FINCHES paper and then rounded to 0.20 for every version (the unrounded
# fits are given in the comments); the null interaction baseline makes poly(GS)
# score zero against itself.
MPIPI_CONFIGS = {
    "Mpipi_original": {
        "charge_prefactor": 0.20,  # 0.184890
        "null_interaction_baseline": -0.066265,
    },
    "Mpipi_GGv1": {
        "charge_prefactor": 0.20,  # 0.216145
        "null_interaction_baseline": -0.128533,
    },
    "OLD_Mpipi_GGv1": {
        "charge_prefactor": 0.20,  # 0.216145
        "null_interaction_baseline": -0.128533,
    },
}

# The five parameter pickles each version is read from, as (prefix, suffix) around
# the parameter name: e.g. ('', '_ggv1') gives sigma_ggv1.pickle.
_PARAMETER_NAMES = ("sigma", "epsilon", "nu", "mu", "charge")
_PARAMETER_FILE_PATTERN = {
    "Mpipi_original": ("", ""),
    "Mpipi_GGv1": ("", "_ggv1"),
    "OLD_Mpipi_GGv1": ("old_", "_ggv1"),
}


class Mpipi_model(ForcefieldModel):
    """
    The Mpipi forcefield as a FINCHES forcefield model.

    Holds one Mpipi parameter set and evaluates the Wang-Frenkel and Coulomb pair
    potentials from it. Pass an instance to InteractionMatrixConstructor (or use
    finches.frontend.mpipi_frontend.Mpipi_frontend, which does this for you).

    Attributes
    ----------
    SIGMA_ALL, EPSILON_ALL, NU_ALL, MU_ALL : dict of dict
        Pairwise Wang-Frenkel parameters, indexed as [residue_1][residue_2]:
        sigma in Angstroms, epsilon (well depth) in kJ/mol, and the
        dimensionless exponents nu and mu.

    CHARGE_ALL : dict
        Per-residue charge in units of the elementary charge.

    CONFIGS : dict
        Calibration constants for the version (see MPIPI_CONFIGS).

    ALL_RESIDUES_TYPES : list of list of str
        The residue groups the loaded parameter set covers (see ForcefieldModel).

    conditions : list of str
        ['salt', 'dielectric']. The temperature and pH attributes are set to 300 K
        and 7.4 for a uniform interface but have no effect on Mpipi energies.
    """

    def __init__(
        self,
        version="Mpipi_GGv1",
        input_directory="default",
        dielectric=80.0,
        salt=0.150,
    ):
        """
        Load an Mpipi parameter set.

        Parameters
        ----------
        version : str, optional
            Which parameter set to load. One of:

            * 'Mpipi_GGv1' (default): the Mpipi-GG re-parameterization (Ginell &
              Holehouse) used throughout FINCHES. Covers the 20 amino acids and
              the RNA nucleotide U. It also contains parameters for ten
              context-dependent aliphatic codes (a, l, m, i, v and b, o, x, y, z;
              see finches.data.Mpipi.mPiPi_GGv1_modification_fxns), which FINCHES
              does not use: its aliphatic weighting is applied to the interaction
              matrix instead.
            * 'Mpipi_original': the unmodified Joseph et al. 2021 parameters,
              which cover the 20 amino acids only.
            * 'OLD_Mpipi_GGv1': a legacy snapshot of the GGv1 parameters from when
              they were rebuilt on every initialization. Among the 20 amino acids
              and U it differs from 'Mpipi_GGv1' in the epsilons of the 15
              aliphatic-aliphatic pairs (A, I, L, M, V), the P-U sigma, and the
              exponents of every U pair, which are swapped (mu = 1, nu = 3 rather
              than mu = 3, nu = 1). It also has no U-to-aliphatic-code entries.
              Kept for reproducibility only; do not use it for new work.

        input_directory : str, optional
            Directory to read the parameter pickles from. Default 'default', which
            uses the files shipped in finches/data/Mpipi.

            To use your own parameters, point this at a directory holding five
            pickled dictionaries whose names depend on the version: for
            'Mpipi_original' they are

            * sigma.pickle: 2D dictionary, [R1][R2] gives sigma (Angstroms).
            * epsilon.pickle: 2D dictionary, [R1][R2] gives epsilon (kJ/mol).
            * nu.pickle: 2D dictionary, [R1][R2] gives nu.
            * mu.pickle: 2D dictionary, [R1][R2] gives mu.
            * charge.pickle: 1D dictionary, [R1] gives the residue's charge.

            For 'Mpipi_GGv1' the same five files carry a _ggv1 suffix
            (sigma_ggv1.pickle and so on), and for 'OLD_Mpipi_GGv1' an old_
            prefix as well (old_sigma_ggv1.pickle and so on). The 2D dictionaries
            must be symmetric and cover every pair of the 20 amino acids; if they
            also cover U, U is added as an RNA residue group.

        dielectric : float, optional
            Dielectric constant of the solvent. Default 80.0 (water).

        salt : float, optional
            Monovalent salt concentration in molar, used for the Debye screening
            of the Coulomb potential. Default 0.150 (150 mM).

        Raises
        ------
        ValueError
            If version is not one of the options above.

        FileNotFoundError
            If one of the parameter pickles is missing from the directory.
        """
        if version not in MPIPI_CONFIGS:
            raise ValueError(
                f"Unrecognized version [{version}] passed to Mpipi_model. Must be "
                f"one of {list(MPIPI_CONFIGS)}"
            )

        if input_directory == "default":
            data_prefix = finches.get_data("Mpipi")
        else:
            data_prefix = input_directory

        # check all five files are present before reading any of them
        prefix, suffix = _PARAMETER_FILE_PATTERN[version]
        paths = {
            name: f"{data_prefix}/{prefix}{name}{suffix}.pickle"
            for name in _PARAMETER_NAMES
        }
        for path in paths.values():
            if not exists(path):
                raise FileNotFoundError(
                    f"Using [{data_prefix}] as our data directory but no "
                    f"{path.split('/')[-1]} file found"
                )

        params = {}
        for name, path in paths.items():
            with open(path, "rb") as fh:
                params[name] = pickle.load(fh)

        self.SIGMA_ALL = params["sigma"]
        self.EPSILON_ALL = params["epsilon"]
        self.NU_ALL = params["nu"]
        self.MU_ALL = params["mu"]
        self.CHARGE_ALL = params["charge"]

        # RNA is only a valid residue group if this parameter set covers it
        # (Mpipi_original does not)
        residue_groups = [list(VALID_AA)]
        if all(r in self.SIGMA_ALL for r in VALID_RNA):
            residue_groups.append(list(VALID_RNA))

        super().__init__(
            version=version,
            dielectric=dielectric,
            salt=salt,
            temperature=300,
            pH=7.4,
            all_residues_types=residue_groups,
            conditions=["salt", "dielectric"],
        )

        # set precomputed forcefield calibration constants
        self.CONFIGS = MPIPI_CONFIGS[version]

    # .....................................................................................
    #
    def compute_wang_frenkel(self, residue_1, residue_2, r):
        """
        Wang-Frenkel potential between two residues.

        Parameters
        ----------
        residue_1 : str
            One-letter code of the first residue.

        residue_2 : str
            One-letter code of the second residue.

        r : float or array-like
            Distance(s) between the beads in Angstroms.

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance; a float if r is a scalar.

        Raises
        ------
        KeyError
            If the parameter set has no entry for the pair.
        """
        return wang_frenkel(
            r,
            self.SIGMA_ALL[residue_1][residue_2],
            self.EPSILON_ALL[residue_1][residue_2],
            self.MU_ALL[residue_1][residue_2],
            self.NU_ALL[residue_1][residue_2],
        )

    # .....................................................................................
    #
    def compute_coulomb(self, residue_1, residue_2, r, dielectric=None, salt=None):
        """
        Debye-Huckel screened Coulomb potential between two residues.

        Parameters
        ----------
        residue_1 : str
            One-letter code of the first residue.

        residue_2 : str
            One-letter code of the second residue.

        r : float or array-like
            Distance(s) between the beads in Angstroms.

        dielectric : float, optional
            Dielectric constant. If None (default), the model's dielectric is used.

        salt : float, optional
            Monovalent salt concentration in molar. If None (default), the
            model's salt concentration is used.

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance.

        Raises
        ------
        KeyError
            If the parameter set has no charge for one of the residues.
        """
        if salt is None:
            salt = self.salt

        if dielectric is None:
            dielectric = self.dielectric

        return coulomb(
            self.CHARGE_ALL[residue_1],
            self.CHARGE_ALL[residue_2],
            r,
            dielectric=dielectric,
            salt=salt,
        )

    def compute_colulomb(self, residue_1, residue_2, r, dielectric=None, salt=None):
        """
        Alias of compute_coulomb(), kept under its original (misspelt) name.

        Parameters
        ----------
        residue_1, residue_2, r, dielectric, salt
            As for compute_coulomb().

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance.
        """
        return self.compute_coulomb(
            residue_1, residue_2, r, dielectric=dielectric, salt=salt
        )

    # .....................................................................................
    #
    def compute_full_potential(
        self, residue_1, residue_2, r, dielectric=None, salt=None
    ):
        """
        Full Mpipi pair potential: Wang-Frenkel plus screened Coulomb.

        Parameters
        ----------
        residue_1 : str
            One-letter code of the first residue.

        residue_2 : str
            One-letter code of the second residue.

        r : float or array-like
            Distance(s) between the beads in Angstroms.

        dielectric : float, optional
            Dielectric constant. If None (default), the model's dielectric is used.

        salt : float, optional
            Monovalent salt concentration in molar. If None (default), the
            model's salt concentration is used.

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance.

        Raises
        ------
        KeyError
            If the parameter set has no entry for the pair.
        """
        wf = self.compute_wang_frenkel(residue_1, residue_2, r)
        c = self.compute_coulomb(
            residue_1, residue_2, r, dielectric=dielectric, salt=salt
        )
        return wf + c

    def compute_full_Mpipi(self, residue_1, residue_2, r, dielectric=None, salt=None):
        """
        Alias of compute_full_potential(), kept under its original name.

        Parameters
        ----------
        residue_1, residue_2, r, dielectric, salt
            As for compute_full_potential().

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance.
        """
        return self.compute_full_potential(
            residue_1, residue_2, r, dielectric=dielectric, salt=salt
        )

    # ................................................................................
    #
    def compute_interaction_parameter(
        self, residue_1, residue_2, r=None, dielectric=None, salt=None
    ):
        """
        Interaction parameter between two residue types.

        The full Mpipi potential is evaluated on a distance grid and integrated
        between the grid points closest to one and three sigma (see
        ForcefieldModel._integrate_sigma_window()). Three sigma is also the
        Wang-Frenkel cutoff.

        Parameters
        ----------
        residue_1 : str
            One-letter code of the first residue.

        residue_2 : str
            One-letter code of the second residue.

        r : array-like, optional
            Distance grid in Angstroms. If None (default), 0.1 to 30 Angstroms in
            0.01 Angstrom steps, which comfortably spans three sigma for every
            residue pair.

        dielectric : float, optional
            Dielectric constant. If None (default), the model's dielectric is used.

        salt : float, optional
            Monovalent salt concentration in molar. If None (default), the
            model's salt concentration is used.

        Returns
        -------
        tuple
            [0] : float, the interaction parameter (kJ/mol times Angstroms).
            [1] : np.ndarray, the full potential (kJ/mol) at every distance in [4].
            [2] : int, index in [4] closest to one sigma.
            [3] : int, index in [4] closest to three sigma.
            [4] : np.ndarray, the distance grid in Angstroms.

        Raises
        ------
        KeyError
            If the parameter set has no entry for the pair.
        """
        if r is None:
            r = self._get_default_distance_array()

        # the docstring promises array-like input, so make sure a plain list works
        r = np.asarray(r, dtype=float)

        combo = self.compute_full_potential(
            residue_1, residue_2, r, dielectric=dielectric, salt=salt
        )

        interaction_param, s1, s3 = self._integrate_sigma_window(
            combo, r, self.SIGMA_ALL[residue_1][residue_2]
        )

        return (interaction_param, combo, s1, s3, r)


######################################
# non - parameter specific functions #
######################################


# ................................................................................
#
#
def harmonic(r, mode="protein", k=8.03, ideal_length=None):
    """
    Harmonic bond potential, equation (2) of the Mpipi paper.

    E = (k / 2) (r - r_ref)^2, with k = 8.03 J mol^-1 pm^-2 and r_ref = 381 pm
    (protein) or 500 pm (RNA), as in Joseph et al. 2021. Not used by FINCHES
    itself, which only needs non-bonded interactions, but provided for
    completeness.

    Parameters
    ----------
    r : float or array-like
        Bond length(s) in Angstroms.

    mode : str, optional
        'protein' (equilibrium length 3.81 Angstroms) or 'nucleic' (5.00
        Angstroms). Default 'protein'. Ignored if ideal_length is given.

    k : float, optional
        Spring constant in J mol^-1 pm^-2. Default 8.03.

    ideal_length : float, optional
        Equilibrium bond length in Angstroms, overriding the value set by mode.

    Returns
    -------
    float or np.ndarray
        Energy in kJ/mol.

    Raises
    ------
    ValueError
        If mode is not 'protein' or 'nucleic' and no ideal_length is given.
    """

    # define the ideal bond length; if we passed an ideal_length
    # in this gets used and we ignore other things.
    if ideal_length is not None:
        r_ref = ideal_length

    # if not, use one of the two default values defined in the paper
    elif mode == "protein":
        r_ref = 3.81
    elif mode == "nucleic":
        r_ref = 5.00
    else:
        raise ValueError(
            f"Invalid mode [{mode}] passed to harmonic(); must be 'protein' or "
            "'nucleic'"
        )

    # calculate (r - r_ref)^2 and convert from Angstroms to picometers
    distance_deviation_in_pm = np.power(100 * (np.asarray(r, dtype=float) - r_ref), 2)

    # the /1000 is so we go from J to kJ
    return 0.5 * k * distance_deviation_in_pm / 1000


# ................................................................................
#
#
def coulomb(qi, qj, rij, dielectric=80.0, salt=0.150):
    """
    Coulomb potential with Debye-Huckel screening.

    A stand-alone potential that does not use any Mpipi parameters. The Debye
    screening length is 3.06 / sqrt(salt) Angstroms (salt in molar), the room-
    temperature value for a monovalent salt in water; see
    https://minds.wisconsin.edu/bitstream/handle/1793/79225/1923-debye-huckel-theory-2020-braus-translation-with-preface.pdf
    for a translation of the original paper in which the 3.06 appears. At 150 mM
    this gives 7.90 Angstroms, against the 7.95 Angstroms used in the Mpipi paper.

    Parameters
    ----------
    qi : float
        Charge of the first bead in units of the elementary charge.

    qj : float
        Charge of the second bead in units of the elementary charge.

    rij : float or array-like
        Distance(s) between the beads in Angstroms.

    dielectric : float, optional
        Dielectric constant of the medium. Default 80.0 (water).

    salt : float, optional
        Monovalent salt concentration in molar. Default 0.150. A value of 0 turns
        screening off.

    Returns
    -------
    float or np.ndarray
        Energy in kJ/mol at each distance.

    Notes
    -----
    Avogadro's number is taken as 6.023e23 rather than 6.02214e23. The Mpipi
    calibration constants were fitted with this value, so it is left as is (the
    difference is 0.013%).
    """

    # distances as a float array so a list works as well as a scalar or array
    rij = np.asarray(rij, dtype=float)

    # inverse Debye length in 1/Angstrom
    kappa = np.sqrt(salt) / 3.06

    # define some constants
    C = 1.602176634e-19  # elementary charge (Coulombs)
    Na = 6.023e23  # Avogadro's number (see Notes)

    # converts charge from elemental units into Coulombs and then into C mol, and
    # divides by 4 pi e0 - this is in units of J m mol-1
    conversion_constant = (C * C * Na) / (np.pi * 4 * 8.854187812799999e-12)

    # convert A to meters
    rij_m = rij * 1e-10

    # Debye damping factor (unitless). If salt=0 this is 1 and screening is off,
    # otherwise it is between 1 and 0. rij stays in Angstroms here to match kappa
    DH = np.exp(-kappa * rij)

    energy_in_J = (conversion_constant * qi * qj) / (dielectric * rij_m) * DH

    return energy_in_J / 1000


# ................................................................................
#
#
def wang_frenkel(r, sigma_ij, epsilon_ij, mu_ij=2, nu_ij=1):
    """
    Wang-Frenkel potential.

    A stand-alone potential that does not use any Mpipi parameters. The cutoff
    R_ij is three sigma: the potential goes smoothly to zero there and is zero
    beyond it.

    Parameters
    ----------
    r : float or array-like
        Distance(s) between the beads in Angstroms.

    sigma_ij : float
        Size parameter in Angstroms: the distance at which the potential crosses
        zero. Should be positive.

    epsilon_ij : float
        Well depth in kJ/mol. Should be positive.

    mu_ij : float, optional
        Exponent that sets the width and steepness of the well. Default 2.

    nu_ij : float, optional
        Second exponent that sets the width and steepness of the well. Default 1.

    Returns
    -------
    float or np.ndarray
        Energy in kJ/mol at each distance; a float if r is a scalar.
    """

    # distances as a float array so a list works as well as a scalar or array
    r = np.asarray(r, dtype=float)

    # define R_ij - this is the cutoff distance of the potential
    R_ij = 3 * sigma_ij

    alpha_ij_term1 = 2 * nu_ij * np.power(R_ij / sigma_ij, 2 * mu_ij)

    alpha_ij_term2 = (2 * nu_ij + 1) / (
        2 * nu_ij * (np.power(R_ij / sigma_ij, 2 * mu_ij) - 1)
    )

    alpha_ij = alpha_ij_term1 * np.power(alpha_ij_term2, 2 * nu_ij + 1)

    main_term1 = epsilon_ij * alpha_ij

    main_term2 = np.power(sigma_ij / r, 2 * mu_ij) - 1

    main_term3 = np.power(np.power(R_ij / r, 2 * mu_ij) - 1, 2 * nu_ij)

    energy = main_term1 * main_term2 * main_term3

    # Evaluating the closed form past the cutoff does not give zero (the
    # (R_ij/r)^(2mu) - 1 factor changes sign and the energy tends to
    # -epsilon*alpha at large r), so truncate here.
    energy = np.where(r <= R_ij, energy, 0.0)

    # hand back a plain float if a single distance was passed
    if energy.ndim == 0:
        return float(energy)

    return energy
