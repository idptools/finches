"""
User-facing wrappers around the two-component Flory-Huggins solutions in
:mod:`finches.analytical_fh.backend`.

The backend implements the analytical spinodal, the Ginzburg-Landau binodal,
the self-consistent (iterative) binodal, and the closed-form analytical
binodal from

    Qian, D., Michaels, T. C. T., & Knowles, T. P. J. (2022).
    Analytical Solution to the Flory-Huggins Model.
    J. Phys. Chem. Lett. 13(33), 7853-7860.

The backend functions operate on a single chi value (or a pre-filtered array
of chi values) and raise if chi is below the critical value. The two functions
here scan a chi range for a polymer of length ``L``, quietly skip the
sub-critical region, and return lists ready for plotting a chi/phi phase
diagram. These are the functions used by
:mod:`finches.epsilon_to_FHtheory`.

A reminder on units: chi = eps / (kB T), where eps is the site-to-site contact
energy in Flory-Huggins theory (larger eps = stronger attraction). This means
a chi/phi diagram can be turned into a temperature-like diagram by plotting
1/chi vs. phi, since 1/chi = T * (kB / eps).
"""

from . import backend as FH
import numpy as np


def calculate_binodal(
    L, mode="analytic_binodal", chi_min=0.5, chi_max=2.0, n_points=500
):
    """
    Compute the binodal of a length-L polymer over a range of chi values.

    The chi range [chi_min, chi_max) is sampled on a grid of ``n_points``
    evenly spaced values (chi_max itself is excluded).
    For each chi, the backend function selected by ``mode`` is called; chi
    values below the critical chi (where the backend raises a ``ValueError``)
    are skipped, so the returned lists can be shorter than ``n_points``.

    The "mode" selector lets you choose how the binodal is calculated. For
    completeness all three approaches described in the paper are offered,
    although in practice 'analytic_binodal' (the point of the paper) should
    be fine for basically all cases.

    Parameters
    ----------
    L : int
        Length of the polymer, i.e. the number of lattice sites it occupies
        (N in the paper). Must be >= 1.

    mode : str
        How the binodal is computed. One of:

        * 'analytic_binodal' (default) - closed-form solution (eq. 34/36).
          Stable across the whole phase diagram.
        * 'binodal' - self-consistent Newton-Raphson iteration (5 iterations,
          improved map). Slightly more accurate near the critical point, but
          per-chi scalar evaluation makes it slower, and for large L and
          large chi the intermediate exponentials can overflow to NaN.
        * 'GL_binodal' - Ginzburg-Landau expansion about the critical point.
          Only meaningful very close to chi_c; away from it the dilute branch
          goes negative and the dense branch exceeds 1.

    chi_min : float
        Minimum chi value to scan. Values below 0.5 can never give phase
        separation (chi_c -> 0.5 as L -> infinity), so there is no point
        going lower.

    chi_max : float
        Maximum chi value scanned (exclusive).

    n_points : int
        Number of chi grid points between chi_min and chi_max.

    Returns
    -------
    tuple
        A tuple with 5 elements:

        0 : list of chi values for which a binodal point was obtained
        1 : list of dilute-phase volume fractions (phi-) at those chi values
        2 : list of dense-phase volume fractions (phi+) at those chi values
        3 : float, the critical volume fraction phi_c
        4 : float, the critical interaction parameter chi_c

    Raises
    ------
    Exception
        If ``mode`` is not one of the three recognised strings.

    See Also
    --------
    calculate_spinodal : Same scan for the spinodal.
    finches.analytical_fh.backend : The underlying implementations.
    """

    # check we passed in a valid mode
    if mode not in ["binodal", "analytic_binodal", "GL_binodal"]:
        raise Exception("mode must be one of 'binodal','analytic_binodal','GL_binodal'")

    # map mode selector to a specific function. Note all three have the same input
    # signature.
    if mode == "binodal":
        fx = FH.binodal
    elif mode == "analytic_binodal":
        fx = FH.analytic_binodal
    elif mode == "GL_binodal":
        fx = FH.GL_binodal
    else:
        raise Exception("UH OH...")

    dense = []
    dilute = []
    chis = []

    # chi between chi_min and chi max. Backend functions raise a ValueError
    # for sub-critical chi, which we skip.
    for chi in np.linspace(chi_min, chi_max, n_points, endpoint=False):
        # get
        try:
            x = fx(chi, L)
            dense.append(x[0])
            dilute.append(x[1])
            chis.append(chi)
        except ValueError:
            pass

    # get critical point/conc info
    c = FH.critical(L)

    return (chis, dilute, dense, c[0], c[1])


def calculate_spinodal(L, chi_min=0.5, chi_max=2.0, n_points=500):
    """
    Compute the spinodal of a length-L polymer over a range of chi values.

    Uses the exact analytical spinodal expression (eq. 2 of Qian et al. 2022)
    via :func:`finches.analytical_fh.backend.spinodal`. The chi range
    [chi_min, chi_max) is sampled on a grid of ``n_points`` evenly spaced
    values (chi_max itself is excluded). Chi values
    below the critical chi are skipped, so the returned lists can be shorter
    than ``n_points``.

    Parameters
    ----------
    L : int
        Length of the polymer, i.e. the number of lattice sites it occupies
        (N in the paper). Must be >= 1.

    chi_min : float
        Minimum chi value to scan. Values below 0.5 can never give phase
        separation.

    chi_max : float
        Maximum chi value scanned (exclusive).

    n_points : int
        Number of chi grid points between chi_min and chi_max.

    Returns
    -------
    tuple
        A tuple with 5 elements:

        0 : list of chi values for which a spinodal point was obtained
        1 : list of low-concentration spinodal volume fractions at those chi values
        2 : list of high-concentration spinodal volume fractions at those chi values
        3 : float, the critical volume fraction phi_c
        4 : float, the critical interaction parameter chi_c

    See Also
    --------
    calculate_binodal : Same scan for the binodal.
    """
    dense = []
    dilute = []
    chis = []

    # chi between chi_min and chi max. spinodal() raises a ValueError for
    # sub-critical chi, which we skip.
    for chi in np.linspace(chi_min, chi_max, n_points, endpoint=False):
        # get
        try:
            x = FH.spinodal(chi, L)
            dense.append(x[0])
            dilute.append(x[1])
            chis.append(chi)
        except ValueError:
            pass

    # get critical point/conc info
    c = FH.critical(L)

    return (chis, dilute, dense, c[0], c[1])
