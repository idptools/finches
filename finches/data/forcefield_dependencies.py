"""
Calibration of forcefield-dependent constants for the InteractionMatrixConstructor.

The null interaction baseline can be recomputed here from any forcefield. The
charge prefactor cannot be derived automatically - it involves judgement about
how strong local charge effects should be - and is calibrated in
finches/forcefields/calibration/prefactor_calibration.ipynb.

By: Garrett M. Ginell & Alex S. Holehouse
"""

from scipy.optimize import root_scalar

from finches import epsilon_stateless


## ------------------------------------------------------------------------------
##
def get_null_interaction_baseline(
    X_model, lower_end=-10.0, upper_end=10.0, alternative_sequence=None
):
    """
    Calibrate the null interaction baseline so that poly(GS) has an epsilon of zero.

    We assume a GS repeat is the right reference for "no net interaction" and
    solve for the baseline b at which its epsilon against itself (with no charge
    or aliphatic weighting) is zero. Because epsilon is linear in b the root is
    unique. The GS assumption is itself an approximation, and the shipped
    CALVADOS2 baseline includes a small empirical correction to the value this
    returns.

    Parameters
    ----------
    X_model : finches.epsilon_calculation.InteractionMatrixConstructor
        Constructor for the forcefield to calibrate.

    lower_end : float, optional
        Lower end of the bracket searched for the baseline. Default -10.0.

    upper_end : float, optional
        Upper end of the bracket searched for the baseline. Default 10.0.

    alternative_sequence : str, optional
        Sequence to calibrate against instead of the default 400-residue GS
        repeat. Default None.

    Returns
    -------
    float
        The baseline b at which the reference sequence has zero epsilon.

    Raises
    ------
    ValueError
        If the root is not bracketed by [lower_end, upper_end], or the solver
        does not converge.

    """

    if alternative_sequence is not None:
        seq = alternative_sequence
    else:
        seq = "GS" * 200

    def f(null_interaction_baseline):
        return epsilon_stateless.get_sequence_epsilon_value(
            seq,
            seq,
            X_model,
            null_interaction_baseline=null_interaction_baseline,
            use_charge_weighting=False,
            use_aliphatic_weighting=False,
        )

    result = root_scalar(f, bracket=[lower_end, upper_end])
    if not result.converged:
        raise ValueError(
            f"Null interaction baseline calibration did not converge: {result.flag}"
        )

    return result.root
