"""
Flory-Huggins phase diagrams from FINCHES epsilon values.

This module turns a mean-field homotypic interaction parameter (epsilon) into a
Flory-Huggins phase diagram for a chain of the same length, using the
analytical solution to the Flory-Huggins model from

    Qian, D., Michaels, T. C. T., & Knowles, T. P. J. (2022). Analytical
    Solution to the Flory-Huggins Model. Journal of Physical Chemistry
    Letters, 13(33), 7853-7860.

The Flory-Huggins code itself lives in finches.analytical_fh: backend.py is the
code released with the paper (https://github.com/KnowlesLab-Cambridge/FloryHuggins)
and floryhuggins.py is our wrapper around it.

How epsilon becomes a phase diagram
-----------------------------------
For a chain of N residues we scan chi over a fixed range, compute the binodal
and spinodal at each chi, and convert chi to a temperature with

    chi = delta_eps / T,   i.e.   T = delta_eps / chi

where delta_eps = -epsilon / N is the per-residue contact energy (the sign flips
because a negative FINCHES epsilon is attractive, whereas a positive chi is).
The temperature is therefore in the units of epsilon with kB = 1: it is a
relative scale for comparing sequences, not a temperature in Kelvin. Note that
epsilon = 0 maps onto chi = 0 (an athermal solvent) with no reference offset,
so only the shape of a phase diagram and its position relative to other
sequences' diagrams are meaningful.

The build_*_dependent_phase_diagrams functions repeat this over a list of
solution conditions (salt, pH or dielectric), rebuilding the forcefield at each
condition.

By : Garrett M. Ginell & Alex S. Holehouse
2023-08-18
"""

import numpy as np

from finches.epsilon_stateless import get_sequence_epsilon_value
from finches.analytical_fh import floryhuggins

# the chi range scanned when building a phase diagram, and the number of grid
# points used for the binodal and spinodal
_CHI_MAX = 0.8
_N_BINODAL_POINTS = 50000
_N_SPINODAL_POINTS = 10000


## ---------------------------------------------------------------------------
##
def _check_condition(X_class, condition):
    """
    Check that the forcefield behind an InteractionMatrixConstructor actually
    lets us vary a given environmental condition.

    Every forcefield model declares the conditions it can vary in its
    ``parameters.conditions`` list (e.g. Mpipi: salt and dielectric; CALVADOS:
    salt, pH and temperature). Simply having the attribute is not enough - the
    base class sets salt, pH, dielectric and temperature on every model, so
    e.g. ``parameters.pH`` exists on an Mpipi model even though nothing in
    Mpipi depends on it. Varying a condition the model ignores would silently
    return the same phase diagram for every value, so we raise instead.

    Parameters
    ----------
    X_class : InteractionMatrixConstructor
        The constructor whose ``parameters`` model is being varied.

    condition : str
        Name of the condition attribute ('salt', 'pH', 'dielectric', ...).

    Raises
    ------
    ValueError
        If the model does not list ``condition`` as something it can vary. The
        message names the condition.
    """
    supported = getattr(X_class.parameters, "conditions", None)
    if supported is None:
        supported = []

    if condition not in supported:
        raise ValueError(
            f'Condition "{condition}" NOT FOUND - Input parameter model '
            f"{X_class.parameters.version} of passed X_class does not support "
            f'"{condition}" as a variable condition. Conditions this model can '
            f"vary are: {list(supported)}"
        )


## ---------------------------------------------------------------------------
##
def _set_condition(X_class, condition, value):
    """
    Set one environmental condition on the forcefield behind an
    InteractionMatrixConstructor and rebuild everything downstream of it.

    Mpipi reads salt and dielectric at call time, so setting the attribute and
    rebuilding the lookup table is enough. CALVADOS derives its Debye screening
    length and the histidine charge from salt, pH and temperature when the
    model is built (``_genParams`` and ``_build_pairwise_maps``), so those
    have to be regenerated first or the rebuilt lookup table just reproduces
    the old conditions. We call those hooks when the model has them.

    Parameters
    ----------
    X_class : InteractionMatrixConstructor
        The constructor whose ``parameters`` model is being varied.

    condition : str
        Name of the condition attribute ('salt', 'pH', 'dielectric', ...).

    value : float
        New value for the condition.

    Returns
    -------
    None
    """
    setattr(X_class.parameters, condition, value)

    for hook in ("_genParams", "_build_pairwise_maps"):
        if hasattr(X_class.parameters, hook):
            getattr(X_class.parameters, hook)()

    # rebuild the pairwise interaction lookup with the new conditions
    X_class._update_lookup_dict()


## ---------------------------------------------------------------------------
##
def _build_condition_dependent_phase_diagrams(
    seq, X_class, condition, condition_list, prefactor, null_interaction_baseline
):
    """
    Shared implementation for the build_*_dependent_phase_diagrams functions.

    For each value in ``condition_list`` the condition is set on the model and
    the interaction lookup is rebuilt; the homotypic epsilon is computed, and
    the phase diagram is built from that same epsilon, so the two outputs
    always agree. The model is returned to its original condition afterwards,
    even if a condition fails part way through.

    Parameters
    ----------
    seq : str
        Input sequence.

    X_class : InteractionMatrixConstructor
        Constructor whose model is re-initialized for each condition.

    condition : str
        Name of the condition attribute to vary ('salt', 'pH', 'dielectric').

    condition_list : list
        Values the condition is set to, one at a time.

    prefactor : float or None
        Charge prefactor passed through to get_sequence_epsilon_value(). If
        None the constructor's own charge_prefactor is used.

    null_interaction_baseline : float or None
        Baseline passed through to get_sequence_epsilon_value(). If None the
        constructor's own null_interaction_baseline is used.

    Returns
    -------
    list
        [condition_list, {condition value: phase diagram},
        {condition value: epsilon}], where each phase diagram is the 8-element
        list returned by epsilon_to_phase_diagram().

    Raises
    ------
    ValueError
        If the model behind X_class cannot vary ``condition``.
    """
    _check_condition(X_class, condition)

    out_diagrams = {}
    out_epsilons = {}

    base_value = getattr(X_class.parameters, condition)

    try:
        # iterate conditions to get epsilon values and phase diagrams at each one
        for i in condition_list:
            _set_condition(X_class, condition, i)

            out_epsilons[i] = get_sequence_epsilon_value(
                seq,
                seq,
                X_class,
                charge_prefactor=prefactor,
                null_interaction_baseline=null_interaction_baseline,
                use_charge_weighting=True,
                use_aliphatic_weighting=True,
            )
            out_diagrams[i] = epsilon_to_phase_diagram(seq, out_epsilons[i])
    finally:
        # always put the model back the way we found it, even if a condition
        # blew up part way through
        _set_condition(X_class, condition, base_value)

    return [condition_list, out_diagrams, out_epsilons]


## ---------------------------------------------------------------------------
##
def build_DIELECTRIC_dependent_phase_diagrams(
    seq, X_class, condition_list, prefactor=None, null_interaction_baseline=None
):
    """
    Build a phase diagram for a sequence at each of a list of dielectric
    constants.

    For each dielectric constant the forcefield is rebuilt, the homotypic
    epsilon is computed, and a phase diagram is built from that epsilon (see
    epsilon_to_phase_diagram()). The model is returned to its original
    dielectric constant afterwards.

    Parameters
    ----------
    seq : str
        Input sequence.

    X_class : InteractionMatrixConstructor
        The constructor to use. The underlying model must list 'dielectric' in
        its ``parameters.conditions`` (Mpipi does; CALVADOS does not, because
        it derives the dielectric from temperature).

    condition_list : list of float
        Dielectric constants, each set in turn as
        ``X_class.parameters.dielectric``.

    prefactor : float, optional
        Charge prefactor used when computing epsilon. If None (default) the
        constructor's own charge_prefactor is used.

    null_interaction_baseline : float, optional
        Baseline used to split attractive and repulsive interactions when
        computing epsilon. If None (default) the constructor's own
        null_interaction_baseline is used.

    Returns
    -------
    list
        A list with three elements:

        [0] - the input condition_list
        [1] - dict mapping each dielectric constant to its phase diagram (the
              8-element list returned by epsilon_to_phase_diagram())
        [2] - dict mapping each dielectric constant to its epsilon

    Raises
    ------
    ValueError
        If the model behind X_class cannot vary its dielectric constant.
    """
    return _build_condition_dependent_phase_diagrams(
        seq, X_class, "dielectric", condition_list, prefactor, null_interaction_baseline
    )


## ---------------------------------------------------------------------------
##
def build_PH_dependent_phase_diagrams(
    seq, X_class, condition_list, prefactor=None, null_interaction_baseline=None
):
    """
    Build a phase diagram for a sequence at each of a list of pH values.

    For each pH the forcefield is rebuilt, the homotypic epsilon is computed,
    and a phase diagram is built from that epsilon (see
    epsilon_to_phase_diagram()). The model is returned to its original pH
    afterwards.

    Parameters
    ----------
    seq : str
        Input sequence.

    X_class : InteractionMatrixConstructor
        The constructor to use. The underlying model must list 'pH' in its
        ``parameters.conditions`` (CALVADOS does, via the histidine charge;
        Mpipi does not).

    condition_list : list of float
        pH values, each set in turn as ``X_class.parameters.pH``.

    prefactor : float, optional
        Charge prefactor used when computing epsilon. If None (default) the
        constructor's own charge_prefactor is used.

    null_interaction_baseline : float, optional
        Baseline used to split attractive and repulsive interactions when
        computing epsilon. If None (default) the constructor's own
        null_interaction_baseline is used.

    Returns
    -------
    list
        A list with three elements:

        [0] - the input condition_list
        [1] - dict mapping each pH to its phase diagram (the 8-element list
              returned by epsilon_to_phase_diagram())
        [2] - dict mapping each pH to its epsilon

    Raises
    ------
    ValueError
        If the model behind X_class cannot vary pH.
    """
    return _build_condition_dependent_phase_diagrams(
        seq, X_class, "pH", condition_list, prefactor, null_interaction_baseline
    )


## ---------------------------------------------------------------------------
##
def build_SALT_dependent_phase_diagrams(
    seq, X_class, condition_list, prefactor=None, null_interaction_baseline=None
):
    """
    Build a phase diagram for a sequence at each of a list of salt
    concentrations.

    For each salt concentration the forcefield is rebuilt, the homotypic
    epsilon is computed, and a phase diagram is built from that epsilon (see
    epsilon_to_phase_diagram()). The model is returned to its original salt
    concentration afterwards.

    Parameters
    ----------
    seq : str
        Input sequence.

    X_class : InteractionMatrixConstructor
        The constructor to use. The underlying model must list 'salt' in its
        ``parameters.conditions`` (both Mpipi and CALVADOS do).

    condition_list : list of float
        Salt concentrations in M, each set in turn as
        ``X_class.parameters.salt``.

    prefactor : float, optional
        Charge prefactor used when computing epsilon. If None (default) the
        constructor's own charge_prefactor is used.

    null_interaction_baseline : float, optional
        Baseline used to split attractive and repulsive interactions when
        computing epsilon. If None (default) the constructor's own
        null_interaction_baseline is used.

    Returns
    -------
    list
        A list with three elements:

        [0] - the input condition_list
        [1] - dict mapping each salt concentration to its phase diagram (the
              8-element list returned by epsilon_to_phase_diagram())
        [2] - dict mapping each salt concentration to its epsilon

    Raises
    ------
    ValueError
        If the model behind X_class cannot vary salt.
    """
    return _build_condition_dependent_phase_diagrams(
        seq, X_class, "salt", condition_list, prefactor, null_interaction_baseline
    )


## ---------------------------------------------------------------------------
##
def return_phase_diagram(seq, X_class):
    """
    Build the Flory-Huggins phase diagram for a sequence.

    Computes the homotypic epsilon of ``seq`` with ``X_class`` (with charge and
    aliphatic weighting, and the constructor's own charge prefactor and null
    baseline) and passes it to epsilon_to_phase_diagram(), which does the
    conversion to a phase diagram; see that function for how, and for what the
    returned values mean.

    Parameters
    ----------
    seq : str
        Input sequence.

    X_class : InteractionMatrixConstructor
        The constructor used to compute the homotypic epsilon.

    Returns
    -------
    list
        The 8-element list returned by epsilon_to_phase_diagram().
    """
    eps_real = get_sequence_epsilon_value(
        seq,
        seq,
        X_class,
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
    )

    return epsilon_to_phase_diagram(seq, eps_real)


## ---------------------------------------------------------------------------
##
def _chi_to_temperature(scan, delta_eps):
    """
    Convert a chi scan from finches.analytical_fh.floryhuggins into temperatures.

    Parameters
    ----------
    scan : tuple
        The (chis, low, high, phi_c, chi_c) tuple returned by
        floryhuggins.calculate_binodal() or calculate_spinodal().

    delta_eps : float
        Per-residue contact energy (positive = attractive), in the units of
        epsilon with kB = 1.

    Returns
    -------
    list
        [low, high, [phi_c, T_c], T], where T = delta_eps / chi is a numpy array
        matching the entries of ``low`` and ``high``.
    """
    chis, low, high, phi_c, chi_c = scan
    temperatures = delta_eps * (1 / np.array(chis))
    critical_temperature = (1 / chi_c) * delta_eps
    return [low, high, [phi_c, critical_temperature], temperatures]


## ---------------------------------------------------------------------------
##
def epsilon_to_phase_diagram(seq, epsilon):
    """
    Build a Flory-Huggins phase diagram from an epsilon value.

    The sequence is used only for its length N. The steps are:

    1. A non-negative (repulsive, or exactly zero) epsilon is replaced by
       -0.01, i.e. very weakly attractive. The conversion to temperature below
       assumes a contact energy that cannot be repulsive: a positive epsilon
       would give negative temperatures and an inverted diagram, and an
       epsilon of exactly zero would put every temperature at zero. The result
       is a diagram at very low temperature, i.e. one that says "does not
       phase separate at any sensible temperature".

    2. The per-residue contact energy is delta_eps = -epsilon / N. The chain
       length is divided out because Flory-Huggins theory already accounts for
       it through N, and the sign flips because a negative epsilon is
       attractive whereas a positive chi is.

    3. The binodal (analytic_binodal mode, 50,000 grid points) and spinodal
       (10,000 grid points) are computed for a chain of length N over chi in
       [0.5, 0.8), and each chi is converted to a temperature
       T = delta_eps / chi.

    The temperature is in the units of epsilon with kB = 1. It is a relative
    scale for comparing sequences with each other, not a temperature in
    Kelvin, because epsilon = 0 maps onto chi = 0 (athermal solvent) with no
    reference offset. Note also that the chi range is fixed, so the critical
    chi of short chains, 0.5 * (1 + 1/sqrt(N))**2, can lie above it: sequences
    shorter than 15 residues come back with empty binodal and spinodal lists.

    Parameters
    ----------
    seq : str
        Input sequence. Only its length is used, to set the polymer length N
        in the Flory-Huggins calculation.

    epsilon : float
        Mean-field homotypic interaction parameter for the sequence, as
        returned by get_sequence_epsilon_value(). Negative values are
        attractive.

    Returns
    -------
    list
        A list with eight elements. The binodal and spinodal lists hold one
        entry per chi grid point above the critical chi:

        [0] - list of dilute-phase volume fractions on the binodal
        [1] - list of dense-phase volume fractions on the binodal
        [2] - [critical volume fraction, critical temperature]
        [3] - np.ndarray of the temperatures matching [0] and [1]
        [4] - list of low-concentration volume fractions on the spinodal
        [5] - list of high-concentration volume fractions on the spinodal
        [6] - [critical volume fraction, critical temperature], from the
              spinodal scan (identical to [2], since both come from the same
              critical point)
        [7] - np.ndarray of the temperatures matching [4] and [5]
    """
    # see step 1 of the docstring for why a non-negative epsilon is clamped
    if epsilon >= 0:
        epsilon = -0.01

    # per-residue contact energy; negative epsilon = positive (attractive) delta_eps
    delta_eps = -epsilon / len(seq)

    binodal = floryhuggins.calculate_binodal(
        len(seq), "analytic_binodal", n_points=_N_BINODAL_POINTS, chi_max=_CHI_MAX
    )
    spinodal = floryhuggins.calculate_spinodal(
        len(seq), n_points=_N_SPINODAL_POINTS, chi_max=_CHI_MAX
    )

    return _chi_to_temperature(binodal, delta_eps) + _chi_to_temperature(
        spinodal, delta_eps
    )
