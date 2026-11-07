"""
Functions that turn the original Mpipi parameters into the Mpipi-GG (GGv1) set.

These are the steps build_MpipiGG_v1.ipynb applies, in this order, to the
original Mpipi pickles to produce the *_ggv1.pickle files that
finches.forcefields.mpipi.Mpipi_model loads by default:

1. ADD_RNA_U_ALL() - add the RNA nucleotide U.
2. CORRECT_aliphatic_aliphatic_ALL() - rebuild the aliphatic-aliphatic epsilons
   from hydrophobicity and fix the I-I and I-V mu.
3. CREATE_new_aliphatic_residues_ALL() - add ten context-dependent aliphatic
   residue codes.
4. SCALE_aliphatic_group_EPSILON() - scale the epsilons of those codes.
5. ENLARGE_Proline_SIGMA() - enlarge every proline sigma.
6. STRENGTHEN_small_polar_EPSILON() - strengthen G-G and G-S.
7. WEAKEN_Aromatic_Charge_EPSILON() - weaken aromatic-charge contacts.

Every function modifies the dictionaries it is given in place and also returns
them. Run in this order on the original pickles, they reproduce the shipped
GGv1 pickles exactly, so their behaviour (including the details documented in
each docstring) is the definition of the shipped parameters and should not be
changed without regenerating the pickles.

The context-dependent aliphatic codes are carried in the GGv1 pickles but are
not used by FINCHES, which applies its aliphatic weighting to the interaction
matrix instead (see finches.parsing_aminoacid_sequences).

By: Garrett M. Ginell & Alex S. Holehouse (updated 2024-01-13)
"""

from . import defined_RNA_params

# the ten context-dependent aliphatic codes created by
# CREATE_new_aliphatic_residues_ALL(), mapped to the residue they derive from:
# lower case (a, l, m, i, v) for an aliphatic with one aliphatic neighbour, and
# b, o, x, y, z for one with two
ALIPHATIC_CONTEXT_CODES = {
    "a": "A",
    "l": "L",
    "m": "M",
    "i": "I",
    "v": "V",
    "b": "A",
    "o": "L",
    "x": "M",
    "y": "I",
    "z": "V",
}


## -----------------------------------------------------------
#
#
def STRENGTHEN_small_polar_EPSILON(EPSILON_ALL):
    """
    Strengthen the G-G and G-S epsilons.

    Each is increased by 119% of its value, i.e. multiplied by 2.19.

    Parameters
    ----------
    EPSILON_ALL : dict
        Pairwise epsilons in kJ/mol, indexed as EPSILON_ALL[R1][R2]. Modified in
        place.

    Returns
    -------
    dict
        EPSILON_ALL, updated.
    """

    # 119%
    local_scalar = 1.19

    # Increasing G:G and G:S epsilon to +119%
    for i in ["G"]:
        for j in ["G", "S"]:
            # i.e. G-G
            if i == j:
                EPSILON_ALL[i][j] = EPSILON_ALL[i][j] + EPSILON_ALL[i][j] * local_scalar

            # i.e. G-S
            else:
                EPSILON_ALL[i][j] = EPSILON_ALL[i][j] + EPSILON_ALL[i][j] * local_scalar
                EPSILON_ALL[j][i] = EPSILON_ALL[j][i] + EPSILON_ALL[j][i] * local_scalar

    return EPSILON_ALL


## -----------------------------------------------------------
#
#
def WEAKEN_Aromatic_Charge_EPSILON(EPSILON_ALL):
    """
    Weaken the aromatic-charge epsilons.

    Every pair of Y, F or W with R, E or D is reduced by 60% of its value, i.e.
    multiplied by 0.4. Lysine is left alone, as its aromatic interactions were
    already weak enough.

    Parameters
    ----------
    EPSILON_ALL : dict
        Pairwise epsilons in kJ/mol, indexed as EPSILON_ALL[R1][R2]. Modified in
        place.

    Returns
    -------
    dict
        EPSILON_ALL, updated.
    """

    # 60%
    local_scalar = 0.60

    for i in ["Y", "F", "W"]:
        for j in ["R", "E", "D"]:
            EPSILON_ALL[i][j] = EPSILON_ALL[i][j] - EPSILON_ALL[i][j] * local_scalar
            EPSILON_ALL[j][i] = EPSILON_ALL[j][i] - EPSILON_ALL[j][i] * local_scalar

    return EPSILON_ALL


## -----------------------------------------------------------
#
#
def ENLARGE_Proline_SIGMA(SIGMA_ALL):
    """
    Enlarge every sigma involving proline by 33%.

    Every P-X sigma (and P-P) is multiplied by 1.33, where X is every other key
    of SIGMA_ALL at the time of the call. In the build pipeline that includes U
    and the context-dependent aliphatic codes, so P-U is enlarged too.

    Parameters
    ----------
    SIGMA_ALL : dict
        Pairwise sigmas in Angstroms, indexed as SIGMA_ALL[R1][R2]. Modified in
        place.

    Returns
    -------
    dict
        SIGMA_ALL, updated.
    """

    # 33%
    local_scalar = 0.33

    # P:X sigma +33% for every residue code except proline itself (handled
    # below). This used to be hardcoded to the 20 amino acids; it now covers
    # every code present so it is applied universally.
    for i in SIGMA_ALL.keys():
        if i != "P":
            SIGMA_ALL["P"][i] = SIGMA_ALL["P"][i] + SIGMA_ALL["P"][i] * local_scalar
            SIGMA_ALL[i]["P"] = SIGMA_ALL[i]["P"] + SIGMA_ALL[i]["P"] * local_scalar

    # tuning P:P sigma +33%
    SIGMA_ALL["P"]["P"] = SIGMA_ALL["P"]["P"] + SIGMA_ALL["P"]["P"] * local_scalar

    return SIGMA_ALL


## -----------------------------------------------------------
#
#
def CORRECT_aliphatic_aliphatic_ALL(CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL):
    """
    Rebuild the aliphatic-aliphatic epsilons from hydrophobicity.

    In the original Mpipi parameters the epsilon of a pair of aliphatic residues
    (A, L, M, I, V) falls as the pair becomes more hydrophobic. This replaces all
    25 aliphatic-aliphatic epsilons with a value that rises with hydrophobicity:

        epsilon = 0.00838733 * (KD_1 + KD_2) + 0.07555277 / 4.3

    (roughly 0.0084 X + 0.0176 kJ/mol), where KD is the Kyte-Doolittle
    hydrophobicity of each residue. The slope is the magnitude of the slope of a
    linear fit of the original aliphatic epsilons against summed hydrophobicity.
    It also sets mu to 2.0 for I-I and I-V. Nothing else is changed.

    Parameters
    ----------
    CHARGE_ALL : dict
        Per-residue charges. Returned unchanged.

    MU_ALL : dict
        Pairwise Wang-Frenkel mu exponents. Modified in place.

    NU_ALL : dict
        Pairwise Wang-Frenkel nu exponents. Returned unchanged.

    EPSILON_ALL : dict
        Pairwise epsilons in kJ/mol. Modified in place.

    SIGMA_ALL : dict
        Pairwise sigmas in Angstroms. Returned unchanged.

    Returns
    -------
    tuple of dict
        (CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL).
    """

    # Fit from original aliphatic epsilons vs. sum of hydrophobicity scale values for
    # each pair of aliphatic residues
    linear_fit = [-0.00838733, 0.07555277]

    # original Mpipi shows a reduction in eps as pariwise hydrophobicity increases, but
    # this is not what we want. We want to increase eps as hydrophobicity increases, so
    # we take the absolute value of the slope.
    corrected_dependency = abs(linear_fit[0])

    # intercept of that fit
    intercept = linear_fit[1]

    # scales the fit's intercept onto the epsilon scale used in the Mpipi
    # force field
    alpha = 4.3

    # Kyte-Doolitle hydrophobicty scale per AA
    AA_KD_scale = {
        "A": 1.80,
        "C": 2.50,
        "D": -3.50,
        "E": -3.50,
        "F": 2.80,
        "G": -0.40,
        "H": -3.20,
        "I": 4.50,
        "K": -3.90,
        "L": 3.80,
        "M": 1.90,
        "N": -3.50,
        "P": -1.60,
        "Q": -3.50,
        "R": -4.50,
        "S": -0.80,
        "T": -0.70,
        "V": 4.20,
        "W": -0.90,
        "Y": -1.30,
    }

    # scaling epsilon based on Kyte-doolittle Hydrophobisity
    for a1 in ["A", "L", "M", "I", "V"]:
        for a2 in ["A", "L", "M", "I", "V"]:
            # sum of hydrophobicity scale values
            KDSSUM = AA_KD_scale[a1] + AA_KD_scale[a2]

            # new epsilon value
            NEWEPS = (KDSSUM * corrected_dependency) + (intercept / alpha)

            # update epsilon values (NOTE this loops over self pairs
            # multiple times, but this is not a problem because we're not basing
            # NEWPS on the original value, but rather the sum of the hydrophobicity
            # scale values
            EPSILON_ALL[a1][a2] = NEWEPS
            EPSILON_ALL[a2][a1] = NEWEPS

    # FIX MU for I:I and I:V such that it is 2.0
    for i in ["I"]:
        for j in ["V", "I"]:
            MU_ALL[i][j] = 2.0
            MU_ALL[j][i] = 2.0

    return CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL


## -----------------------------------------------------------
#
#
def CREATE_new_aliphatic_residues_ALL(
    CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL
):
    """
    Add ten context-dependent aliphatic residue codes.

    Each aliphatic residue gets two new codes, one for when it has one aliphatic
    neighbour (a, l, m, i, v) and one for when it has two (b, o, x, y, z); see
    ALIPHATIC_CONTEXT_CODES. Every new code starts as an exact copy of the
    residue it derives from: its row of every pairwise dictionary is copied, its
    column in every other residue's row is filled in, and its charge is zero.
    SCALE_aliphatic_group_EPSILON() then scales their epsilons.

    Parameters
    ----------
    CHARGE_ALL : dict
        Per-residue charges. Modified in place.

    MU_ALL : dict
        Pairwise Wang-Frenkel mu exponents. Modified in place.

    NU_ALL : dict
        Pairwise Wang-Frenkel nu exponents. Modified in place.

    EPSILON_ALL : dict
        Pairwise epsilons in kJ/mol. Modified in place.

    SIGMA_ALL : dict
        Pairwise sigmas in Angstroms. Modified in place.

    Returns
    -------
    tuple of dict
        (CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL).
    """

    all_aligroup_conversions = ALIPHATIC_CONTEXT_CODES

    # define the new residues we're going to create
    new_residues = list(all_aligroup_conversions.keys())

    # for each of the new residues (a,l,...z)
    for i in new_residues:
        # get the real residue name associated with this rescode
        a1 = all_aligroup_conversions[i]

        # set default values for new residue to match the base residue
        EPSILON_ALL[i] = EPSILON_ALL[a1].copy()
        SIGMA_ALL[i] = SIGMA_ALL[a1].copy()
        MU_ALL[i] = MU_ALL[a1].copy()
        NU_ALL[i] = NU_ALL[a1].copy()

        # set default charge to zero
        CHARGE_ALL[i] = 0.0

    # every residue code, now including the new ones added above
    all_residues = list(EPSILON_ALL.keys())

    # for each of the new residues (a,l,...z)
    # add aliphatic group residues to other residue dicts
    for i in new_residues:
        # get the real residue name associated with this rescode
        a1 = all_aligroup_conversions[i]

        # for ALL residue codes (both base codes and new codes)
        for a2 in all_residues:
            # for pairs where BOTH residues are new residues (i.e. at end of the a2 list)
            if a2 in new_residues:
                EPSILON_ALL[a2][i] = EPSILON_ALL[a2][a1]
                SIGMA_ALL[a2][i] = SIGMA_ALL[a2][a1]
                MU_ALL[a2][i] = MU_ALL[a2][a1]
                NU_ALL[a2][i] = NU_ALL[a2][a1]

            # for pairs where only 1 of the 2 residues (i) is a new residue
            else:
                EPSILON_ALL[a2][i] = EPSILON_ALL[i][a2]
                SIGMA_ALL[a2][i] = SIGMA_ALL[i][a2]
                MU_ALL[a2][i] = MU_ALL[i][a2]
                NU_ALL[a2][i] = NU_ALL[i][a2]

    return CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL


# ---------------------------------------------------------
def SCALE_aliphatic_group_EPSILON(EPSILON_ALL):
    """
    Scale the epsilons of the context-dependent aliphatic codes.

    The codes come in three groups: the plain aliphatic residues (grp0: A, L, M,
    I, V), those with one aliphatic neighbour (grp1: a, l, m, i, v) and those
    with two (grp2: b, o, x, y, z). The intent was to leave any pair involving a
    grp0 residue unchanged, scale grp1 pairs (with grp1 or grp2) by 1.5, and scale
    grp2-grp2 pairs by 3.

    Because each loop visits both (i, j) and (j, i) and writes the scaled value to
    both entries, a pair of two different codes from the same group is scaled
    twice. What the function actually produces, and what the shipped GGv1 pickles
    contain, relative to the epsilon of the underlying residues, is:

    * any pair with a grp0 residue (e.g. A-a): x 1.0
    * a grp1 code with itself (a-a): x 1.5
    * two different grp1 codes (a-l): x 2.25
    * a grp1 code with a grp2 code (a-b): x 1.5
    * a grp2 code with itself (b-b): x 3.0
    * two different grp2 codes (b-o): x 9.0

    FINCHES does not use these codes (see the module docstring), so this has no
    effect on FINCHES results; it is documented so the parameter set is described
    exactly.

    Parameters
    ----------
    EPSILON_ALL : dict
        Pairwise epsilons in kJ/mol, including the context-dependent codes.
        Modified in place.

    Returns
    -------
    dict
        EPSILON_ALL, updated.
    """

    # scaling associated with one residue in grp0
    Ali0_scaler = 0

    # scaling when one residue is in grp1 and the other is in grp1 or grp2
    Ali1_scaler = 0.5

    # scaling when both residues are in grp2
    Ali2_scaler = 2

    # grp0 scaling
    # Aligroup0 : Aligroup0, Aligroup1, and Aligroup2
    for i in ["A", "L", "M", "I", "V"]:
        for j in [
            "A",
            "L",
            "M",
            "I",
            "V",
            "a",
            "l",
            "m",
            "i",
            "v",
            "b",
            "o",
            "x",
            "y",
            "z",
        ]:
            # calculate epsilon after scaling
            NEWEPS = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j] * Ali0_scaler)

            EPSILON_ALL[i][j] = NEWEPS
            EPSILON_ALL[j][i] = NEWEPS

    # grp1 scaling
    # Aligroup1 : Aligroup1 and Aligroup2
    for i in ["a", "l", "m", "i", "v"]:
        for j in ["a", "l", "m", "i", "v", "b", "o", "x", "y", "z"]:
            # calculate epsilon after scaling
            NEWEPS = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j] * Ali1_scaler)

            EPSILON_ALL[i][j] = NEWEPS
            EPSILON_ALL[j][i] = NEWEPS

    # grp2 scaling
    # Aligroup2 : Aligroup2
    for i in ["b", "o", "x", "y", "z"]:
        for j in ["b", "o", "x", "y", "z"]:
            # calculate epsilon after scaling
            NEWEPS = EPSILON_ALL[i][j] + (EPSILON_ALL[i][j] * Ali2_scaler)

            EPSILON_ALL[i][j] = NEWEPS
            EPSILON_ALL[j][i] = NEWEPS

    return EPSILON_ALL


# ----------------------------------------------------------------------------------------
def ADD_RNA_U_ALL(CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL):
    """
    Add the RNA nucleotide U to the parameter set.

    Sets U's charge (-0.75) and every amino acid-U and U-U parameter from the
    values in finches.data.Mpipi.defined_RNA_params, which were taken from the
    Mpipi LAMMPS input. This has to run first, on dictionaries that hold only
    the 20 amino acids.

    Parameters
    ----------
    CHARGE_ALL : dict
        Per-residue charges in units of the elementary charge. Modified in place.

    MU_ALL : dict
        Pairwise Wang-Frenkel mu exponents. Modified in place.

    NU_ALL : dict
        Pairwise Wang-Frenkel nu exponents. Modified in place.

    EPSILON_ALL : dict
        Pairwise epsilons in kJ/mol. Modified in place.

    SIGMA_ALL : dict
        Pairwise sigmas in Angstroms. Modified in place.

    Returns
    -------
    tuple of dict
        (CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL).

    Raises
    ------
    ValueError
        If SIGMA_ALL already holds anything other than the 20 amino acids.
    """

    AAs = [
        "P",
        "E",
        "D",
        "R",
        "K",
        "I",
        "V",
        "L",
        "T",
        "H",
        "M",
        "G",
        "A",
        "C",
        "S",
        "N",
        "Q",
        "F",
        "Y",
        "W",
    ]

    for i in SIGMA_ALL:
        if i not in AAs:
            raise ValueError(
                "SIGMA_ALL contains something beyond the standard 20 amino acids; "
                "ADD_RNA_U_ALL() must be the first update applied to the parameters."
            )

    # update charge as defined in the RNA params
    for r in defined_RNA_params.CHARGE:
        CHARGE_ALL[r] = defined_RNA_params.CHARGE[r]

    # update existing parameters for amino acids with AA:RNA interaction
    for a1 in defined_RNA_params.CHARGE:
        for a2 in AAs:
            # parameters for this amino acid with U, in the order
            # [partner, epsilon, sigma, nu, mu] (see defined_RNA_params.HEADER)
            l_params = defined_RNA_params.PARAMS[a2]

            EPSILON_ALL[a2][a1] = l_params[1]
            SIGMA_ALL[a2][a1] = l_params[2]

            # nu and mu are in that order in the table (an older version of
            # this code had them swapped; see Mpipi_model's OLD_Mpipi_GGv1)
            NU_ALL[a2][a1] = l_params[3]
            MU_ALL[a2][a1] = l_params[4]

    # create new parameters for U:everything
    for a1 in defined_RNA_params.CHARGE:
        EPSILON_ALL[a1] = {}
        SIGMA_ALL[a1] = {}
        MU_ALL[a1] = {}
        NU_ALL[a1] = {}

        for a2, l_params in defined_RNA_params.PARAMS.items():
            EPSILON_ALL[a1][a2] = l_params[1]
            SIGMA_ALL[a1][a2] = l_params[2]

            NU_ALL[a1][a2] = l_params[3]
            MU_ALL[a1][a2] = l_params[4]

    return CHARGE_ALL, MU_ALL, NU_ALL, EPSILON_ALL, SIGMA_ALL
