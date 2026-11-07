"""
The CALVADOS forcefield for FINCHES.

Implements the CALVADOS coarse-grained forcefield (Tesei et al. 2021; Tesei &
Lindorff-Larsen 2022): an Ashbaugh-Hatch short-range potential, whose per-residue
stickiness (lambda) is the thing CALVADOS fits, plus a Debye-Huckel (Yukawa)
screened Coulomb potential between one-bead-per-residue particles. CALVADOS_model
reads the residue parameters, evaluates these potentials for any residue pair,
and turns them into the interaction parameters FINCHES builds its interaction
matrices from. The stand-alone potentials (ashbaugh_hatch(), yukawa() and
compute_calvados_energy()) are also exposed at module level.

The code follows the CALVADOS implementation (https://github.com/KULL-Centre/CALVADOS)
as closely as possible, including its units: the stand-alone potentials work in
nanometres, as CALVADOS does, while the CALVADOS_model methods take distances in
Angstroms like every other part of FINCHES and convert internally. Salt, pH and
temperature enter through the Yukawa term only: the Debye length depends on salt
and temperature, the dielectric constant is a fitted function of temperature, and
the histidine charge is set from the pH. The Ashbaugh-Hatch term does not depend
on any of them.

Sources in the CALVADOS repository that this module follows:

* the general SI and explanation:
  https://www.pnas.org/doi/suppl/10.1073/pnas.2111696118/suppl_file/pnas.2111696118.sapp.pdf
* parameter generation: single_chain/simulate.py (genParams, line 18, and lines
  65-73 and 99-108), and direct_coexistence/analyse.py (genParamsLJ, genParamsDH)
* the potentials: direct_coexistence/simulate.py (lines 92-105), the HOOMD Yukawa
  evaluator (hoomd/md/EvaluatorPairYukawa.h) and the azplugins Ashbaugh-Hatch
  evaluator (azplugins/PairEvaluatorAshbaugh.h)

Originally written by Garrett M. Ginell (2023-03-07); revised by Alex S. Holehouse
(2026-01-11).
"""

import pickle
from os.path import exists

import numpy as np

import finches
from finches.forcefields.model_base import ForcefieldModel, STANDARD_AMINO_ACIDS

# CALVADOS covers the 20 amino acids only (no nucleic acids)
VALID_AA = list(STANDARD_AMINO_ACIDS)

# Calibration constants per version. CALVADOS1 has not been calibrated, so its
# constants are NaN and an InteractionMatrixConstructor built on it will ask for
# them explicitly. The CALVADOS2 null interaction baseline was moved slightly from
# the poly(GS) value of -0.478 (see the FINCHES paper); the unrounded charge
# prefactor fit is given in the comment.
CALVADOS_CONFIGS = {
    "CALVADOS1": {
        "charge_prefactor": np.nan,
        "null_interaction_baseline": np.nan,
    },
    "CALVADOS2": {
        "charge_prefactor": 0.7,  # 1.442590
        "null_interaction_baseline": -0.45,
    },
}


class CALVADOS_model(ForcefieldModel):
    """
    The CALVADOS forcefield as a FINCHES forcefield model.

    Holds the CALVADOS residue parameters for one version and evaluates the
    Ashbaugh-Hatch and Yukawa pair potentials from them. Pass an instance to
    InteractionMatrixConstructor (or use
    finches.frontend.calvados_frontend.CALVADOS_frontend, which does this for you).

    Salt, pH and temperature can be changed after construction by assigning to
    the salt, pH and temperature attributes: the condition-dependent Yukawa
    parameters are rebuilt automatically the next time a potential is evaluated.
    An InteractionMatrixConstructor built on the model has to rebuild its own
    lookup table afterwards (its _update_lookup_dict() method).

    Attributes
    ----------
    SIGMA_ALL : dict of dict
        Pairwise bead diameter in nm, the mean of the two residues' values,
        indexed as [residue_1][residue_2].

    LAMBDA_ALL : dict of dict
        Pairwise stickiness, the mean of the two residues' lambdas for the chosen
        version.

    YUKAWA_EPS_ALL : dict of dict
        Pairwise Yukawa prefactor q1 * q2 * lB * RT in kJ/mol nm, for the current
        conditions.

    yukawa_kappa : float
        Inverse Debye length in 1/nm for the current salt and temperature.

    dielectric : float
        Dielectric constant of water at the current temperature, computed from
        the CALVADOS fit; not an independent parameter.

    lj_eps : float
        Ashbaugh-Hatch well depth, 0.2 kcal/mol expressed in kJ/mol (0.8368).

    cutoff : float
        Ashbaugh-Hatch cutoff, 2.0 nm.

    yukawa_r_cut : float
        Yukawa cutoff, 4.0 nm.

    conditions : list of str
        ['salt', 'pH', 'temperature'].
    """

    def __init__(
        self,
        version="CALVADOS2",
        salt=0.150,
        pH=7.4,
        temperature=288,
        input_directory="default",
    ):
        """
        Load the CALVADOS residue parameters and build the pair parameters.

        Parameters
        ----------
        version : str, optional
            Which stickiness (lambda) parameters to use: 'CALVADOS1' or
            'CALVADOS2'. Default 'CALVADOS2'. CALVADOS1 has no FINCHES calibration
            constants, so it can only be used with an explicitly supplied charge
            prefactor and null interaction baseline.

        salt : float, optional
            Monovalent salt concentration in molar. Default 0.150 (150 mM).

        pH : float, optional
            Solution pH. It sets the histidine charge, 1 / (1 + 10^(pH - 6)); no
            other residue's charge depends on it. Default 7.4.

        temperature : float, optional
            Temperature in Kelvin. It sets the dielectric constant of water and
            the thermal energy in the Bjerrum length, so it changes only the
            Yukawa term. Default 288.

        input_directory : str, optional
            Directory to read the residue parameters from. Default 'default',
            which uses the file shipped in finches/data/calvados.

            To use your own parameters, point this at a directory holding a
            calvados_residues_dict.pickle file: a pickled dictionary keyed by
            one-letter amino acid code, where each value is itself a dictionary
            with the keys 'three' (three-letter code), 'MW' (molecular weight),
            'sigmas' (bead diameter in nm), 'q' (charge in units of the elementary
            charge) and one stickiness (lambda) entry per version name
            ('CALVADOS1', 'CALVADOS2').

        Raises
        ------
        ValueError
            If version is not 'CALVADOS1' or 'CALVADOS2'.

        FileNotFoundError
            If calvados_residues_dict.pickle is missing from the directory.
        """

        # Call parent constructor. CALVADOS computes its dielectric from the
        # temperature (in _genParams), so the value passed here is a placeholder
        super().__init__(
            version=version,
            dielectric=80.0,  # placeholder, will be computed from temp
            salt=salt,
            temperature=temperature,
            pH=pH,
            all_residues_types=[VALID_AA],
            conditions=["salt", "pH", "temperature"],
        )

        # Validate version before it is used to index the residue parameters below,
        # otherwise an unknown version raises a bare KeyError instead of this message
        if version not in CALVADOS_CONFIGS:
            raise ValueError(
                f"Unrecognized version [{version}] passed to CALVADOS_model. Must be "
                f"one of {list(CALVADOS_CONFIGS)}"
            )

        # if 'default' is passed, use the default parameters
        if input_directory == "default":
            data_prefix = finches.get_data("calvados")
        else:
            data_prefix = input_directory

        # check files are present
        if not exists(f"{data_prefix}/calvados_residues_dict.pickle"):
            raise FileNotFoundError(
                f"Using [{data_prefix}] as our data directory but no calvados_residues_dict.pickle file found"
            )

        # Load residue parameters from pickle. The dictionary is keyed by one-letter
        # code, with values holding 'three', 'MW', 'sigmas', 'q' and a lambda per
        # version ('CALVADOS1', 'CALVADOS2')
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

        # Set precomputed forcefield config parameters
        self.CONFIGS = CALVADOS_CONFIGS[version]

        # Ashbaugh-Hatch well depth: 0.2 kcal/mol, in kJ/mol
        # (line 65 single_chain/simulate.py & line 138 of direct_coexistence/analyse.py)
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
        Build the condition-dependent parameters of the Yukawa potential.

        Sets the histidine charge from the pH, the dielectric constant of water
        from the temperature (the CALVADOS fit), the Bjerrum length, and from it
        the inverse Debye length for the current salt. Adapted from genParams in
        CALVADOS single_chain/simulate.py (line 18).

        The conditions the parameters were built for are recorded in
        self._condition_state, so that _sync_conditions() can rebuild them if
        salt, pH or temperature are changed on the object after construction.
        Call _build_pairwise_maps() afterwards to propagate the new charges and
        Bjerrum length into YUKAWA_EPS_ALL.

        Returns
        -------
        None
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

        # Store the Bjerrum length (nm) so kappa can be recomputed for other salts
        self._lB = lB

        # Calculate the inverse of the Debye length
        self.yukawa_kappa = self._kappa_for_salt(self.salt)

        # Store lB*RT for yukawa_eps calculation
        self._lB_RT = lB * RT

        # Remember which conditions these parameters correspond to
        self._condition_state = (self.salt, self.pH, self.temperature)

    # .....................................................................................
    #
    def _kappa_for_salt(self, salt):
        """
        Inverse Debye length (in 1/nm) for a monovalent salt concentration, using the
        Bjerrum length computed for the current temperature.

        Parameters
        ----------
        salt : float
            Monovalent salt concentration in molar.

        Returns
        -------
        float
            Inverse Debye length in 1/nm.

        """
        return np.sqrt(8 * np.pi * self._lB * salt * 6.022 / 10)

    # .....................................................................................
    #
    def _sync_conditions(self):
        """
        Rebuild the condition-dependent parameters if the conditions have changed.

        The condition-scanning code (e.g. build_SALT_dependent_phase_diagrams)
        changes conditions by assigning to model.salt, model.pH or
        model.temperature and then rebuilding the interaction lookup, so the
        Yukawa parameters must follow those attributes rather than being frozen at
        construction. Every method that evaluates the Yukawa term calls this
        first.

        Returns
        -------
        None
        """
        if (self.salt, self.pH, self.temperature) != self._condition_state:
            self._genParams()
            self._build_pairwise_maps()

    # .....................................................................................
    #
    def _build_pairwise_maps(self):
        """
        Build the pairwise parameter dictionaries from the per-residue parameters.

        Sigma and lambda of a pair are the means of the two residues' values, and
        the Yukawa prefactor is q1 * q2 * lB * RT, following CALVADOS
        single_chain/simulate.py (lines 70-73). Populates SIGMA_ALL, LAMBDA_ALL
        and YUKAWA_EPS_ALL.

        Returns
        -------
        None
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
        Ashbaugh-Hatch (shifted, truncated Lennard-Jones) potential between two
        residues.

        Parameters
        ----------
        residue_1 : str
            One-letter code of the first residue.

        residue_2 : str
            One-letter code of the second residue.

        r : float or array-like
            Distance(s) between the beads in Angstroms (converted to nm
            internally).

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance; a float if r is a scalar.

        Raises
        ------
        KeyError
            If either residue is not one of the 20 amino acids.
        """
        s = self.SIGMA_ALL[residue_1][residue_2]
        lam = self.LAMBDA_ALL[residue_1][residue_2]

        return ashbaugh_hatch(
            np.asarray(r, dtype=float) * 0.1,
            s,
            lam,
            cutoff=self.cutoff,
            lj_eps=self.lj_eps,
        )

    # .....................................................................................
    #
    def compute_yukawa(self, residue_1, residue_2, r, salt=None):
        """
        Yukawa (Debye-Huckel screened Coulomb) potential between two residues.

        Parameters
        ----------
        residue_1 : str
            One-letter code of the first residue.

        residue_2 : str
            One-letter code of the second residue.

        r : float or array-like
            Distance(s) between the beads in Angstroms (converted to nm
            internally).

        salt : float, optional
            Monovalent salt concentration in molar. If None (default), the
            model's salt concentration is used.

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance; a float if r is a scalar.

        Raises
        ------
        KeyError
            If either residue is not one of the 20 amino acids.
        """
        self._sync_conditions()

        q = self.YUKAWA_EPS_ALL[residue_1][residue_2]
        kappa = self.yukawa_kappa if salt is None else self._kappa_for_salt(salt)

        return yukawa(
            np.asarray(r, dtype=float) * 0.1, q, kappa, yukawa_r_cut=self.yukawa_r_cut
        )

    # .....................................................................................
    #
    def compute_full_potential(
        self, residue_1, residue_2, r, dielectric=None, salt=None
    ):
        """
        Full CALVADOS pair potential: Ashbaugh-Hatch plus Yukawa.

        Parameters
        ----------
        residue_1 : str
            One-letter code of the first residue.

        residue_2 : str
            One-letter code of the second residue.

        r : float or array-like
            Distance(s) between the beads in Angstroms (converted to nm
            internally).

        dielectric : float, optional
            Ignored. CALVADOS computes its dielectric constant from the
            temperature; the argument exists for a uniform interface across
            forcefield models.

        salt : float, optional
            Monovalent salt concentration in molar. If None (default), the
            model's salt concentration is used.

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance; a float if r is a scalar.

        Raises
        ------
        KeyError
            If either residue is not one of the 20 amino acids.
        """
        # make sure the Yukawa parameters match the current salt/pH/temperature
        self._sync_conditions()

        # Get parameters
        s = self.SIGMA_ALL[residue_1][residue_2]
        lam = self.LAMBDA_ALL[residue_1][residue_2]
        q = self.YUKAWA_EPS_ALL[residue_1][residue_2]
        kappa = self.yukawa_kappa if salt is None else self._kappa_for_salt(salt)

        # Track if input was scalar
        is_scalar = np.ndim(r) == 0

        # Convert to numpy array and Angstroms to nm (CALVADOS uses nm internally)
        r_nm = np.atleast_1d(r) * 0.1

        energies = compute_calvados_energy(
            r_nm,
            s,
            lam,
            q,
            kappa,
            cutoff=self.cutoff,
            lj_eps=self.lj_eps,
            yukawa_r_cut=self.yukawa_r_cut,
        )

        # Return scalar if input was scalar
        if is_scalar:
            return float(energies[0])
        return energies

    def compute_full_calvados(self, residue_1, residue_2, r, salt=None):
        """
        Alias of compute_full_potential(), kept under its original name.

        Parameters
        ----------
        residue_1, residue_2, r, salt
            As for compute_full_potential().

        Returns
        -------
        float or np.ndarray
            Energy in kJ/mol at each distance.
        """
        return self.compute_full_potential(residue_1, residue_2, r, salt=salt)

    # .....................................................................................
    #
    def compute_interaction_parameter(
        self, residue_1, residue_2, r=None, dielectric=None, salt=None
    ):
        """
        Interaction parameter between two residue types.

        The full CALVADOS potential is evaluated on a distance grid and integrated
        between the grid points closest to one and three sigma (see
        ForcefieldModel._integrate_sigma_window()). The limits are located on the
        grid in nm, as CALVADOS sigmas are in nm, and the integral is taken over
        Angstroms, like every other FINCHES interaction parameter.

        Parameters
        ----------
        residue_1 : str
            One-letter code of the first residue.

        residue_2 : str
            One-letter code of the second residue.

        r : array-like, optional
            Distance grid in Angstroms. If None (default), 0.1 to 30 Angstroms in
            0.01 Angstrom steps (built in nm as 0.01 to 3 nm in 0.001 nm steps).

        dielectric : float, optional
            Ignored. CALVADOS computes its dielectric constant from the
            temperature; the argument exists for a uniform interface across
            forcefield models.

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
            If either residue is not one of the 20 amino acids.
        """
        # make sure the Yukawa parameters match the current salt/pH/temperature
        self._sync_conditions()

        if r is None:
            # default grid in nm (sigmas are in nm), converted below
            r_nm = self._get_default_distance_array(0.01, 3, 0.001)
        else:
            # Convert Angstroms to nm
            r_nm = np.asarray(r, dtype=float) * 0.1

        # Convert to Angstroms for output and compute_full_potential()
        r_angstroms = r_nm * 10

        combo = self.compute_full_potential(
            residue_1, residue_2, r_angstroms, salt=salt
        )

        # limits located in nm, integral taken over Angstroms
        interaction_param, s1, s3 = self._integrate_sigma_window(
            combo, r_nm, self.SIGMA_ALL[residue_1][residue_2], x=r_angstroms
        )

        return (interaction_param, combo, s1, s3, r_angstroms)


######################################
# non - parameter specific functions #
######################################


# .....................................................................................
#
def ashbaugh_hatch(r, s, lam, cutoff=2.0, lj_eps=4.184 * 0.2):
    """
    Ashbaugh-Hatch potential (vectorized).

    A Lennard-Jones potential whose attractive part is scaled by the stickiness
    lambda, shifted so that it is zero at the cutoff and truncated (zero) beyond
    it, as in the CALVADOS simulations.

    Parameters
    ----------
    r : float or array-like
        Distance(s) in nanometers.

    s : float
        Sigma (bead diameter) in nanometers.

    lam : float
        Stickiness lambda, between 0 and 1.

    cutoff : float, optional
        Cutoff distance in nanometers. Default 2.0.

    lj_eps : float, optional
        Well depth in kJ/mol. Default 0.2 kcal/mol (0.8368 kJ/mol).

    Returns
    -------
    float or np.ndarray
        Energy in kJ/mol at each distance; a float if r is a scalar.
    """
    is_scalar = np.ndim(r) == 0
    r = np.atleast_1d(r)
    rc = cutoff
    eps = lj_eps

    shift = (s / rc) ** 12 - (s / rc) ** 6

    # Precompute (s/r)^6 and (s/r)^12 for all r values
    s_over_r_6 = (s / r) ** 6
    s_over_r_12 = s_over_r_6 * s_over_r_6

    # the potential uses the attractive branch beyond the LJ minimum at 2^(1/6) s
    # and the repulsive branch inside it
    threshold = 2 ** (1 / 6) * s
    mask = r >= threshold

    # y = 4*eps*lam*((s/r)^12 - (s/r)^6 - shift)  -- attractive regime
    # z = 4*eps*((s/r)^12 - (s/r)^6 - lam*shift) + eps*(1-lam)  -- repulsive regime
    y = 4 * eps * lam * (s_over_r_12 - s_over_r_6 - shift)
    z = 4 * eps * (s_over_r_12 - s_over_r_6 - lam * shift) + eps * (1 - lam)

    result = np.where(mask, y, z)

    # The shift makes the potential zero at the cutoff, but past it the closed
    # form turns slightly positive again; CALVADOS truncates it there
    result = np.where(r <= rc, result, 0.0)

    return float(result[0]) if is_scalar else result


# .....................................................................................
#
def yukawa(r, q, yukawa_kappa, yukawa_r_cut=4.0):
    """
    Yukawa (Debye-Huckel screened Coulomb) potential (vectorized).

    Shifted so that it is zero at the cutoff and truncated (zero) beyond it, as
    in the CALVADOS simulations.

    Parameters
    ----------
    r : float or array-like
        Distance(s) in nanometers.

    q : float
        Pair prefactor q1 * q2 * lB * RT in kJ/mol nm.

    yukawa_kappa : float
        Inverse Debye length in 1/nm.

    yukawa_r_cut : float, optional
        Cutoff distance in nanometers. Default 4.0.

    Returns
    -------
    float or np.ndarray
        Energy in kJ/mol at each distance; a float if r is a scalar.
    """
    is_scalar = np.ndim(r) == 0
    r = np.atleast_1d(r)

    shift = np.exp(-yukawa_kappa * yukawa_r_cut) / yukawa_r_cut
    result = q * (np.exp(-yukawa_kappa * r) / r - shift)

    # zero at the cutoff by construction; truncated beyond it
    result = np.where(r <= yukawa_r_cut, result, 0.0)

    return float(result[0]) if is_scalar else result


# .....................................................................................
#
def compute_calvados_energy(
    r, s, lam, q, yukawa_kappa, cutoff=2.0, lj_eps=4.184 * 0.2, yukawa_r_cut=4.0
):
    """
    Full CALVADOS pair potential: Ashbaugh-Hatch plus Yukawa (vectorized).

    Like the two potentials it combines, this works in nanometres, the units the
    native CALVADOS parameters are in. The CALVADOS_model methods take Angstroms,
    like the rest of FINCHES, and convert before calling it.

    Parameters
    ----------
    r : float or array-like
        Distance(s) in nanometers.

    s : float
        Sigma (bead diameter) in nanometers.

    lam : float
        Stickiness lambda.

    q : float
        Yukawa pair prefactor q1 * q2 * lB * RT in kJ/mol nm.

    yukawa_kappa : float
        Inverse Debye length in 1/nm.

    cutoff : float, optional
        Ashbaugh-Hatch cutoff in nanometers. Default 2.0.

    lj_eps : float, optional
        Ashbaugh-Hatch well depth in kJ/mol. Default 0.8368 (0.2 kcal/mol).

    yukawa_r_cut : float, optional
        Yukawa cutoff in nanometers. Default 4.0.

    Returns
    -------
    float or np.ndarray
        Energy in kJ/mol at each distance; a float if r is a scalar.
    """
    ah = ashbaugh_hatch(r, s, lam, cutoff=cutoff, lj_eps=lj_eps)
    yu = yukawa(r, q, yukawa_kappa, yukawa_r_cut=yukawa_r_cut)

    return yu + ah


# Backwards compatibility alias: the class was originally called calvados_model
calvados_model = CALVADOS_model
