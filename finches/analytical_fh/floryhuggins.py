"""
User-facing wrappers around the two-component Flory-Huggins solutions in
:mod:`finches.analytical_fh.backend`.

The backend implements the analytical spinodal, the Ginzburg-Landau binodal,
the self-consistent (iterative) binodal, and the closed-form analytical
binodal from

    Qian, D., Michaels, T. C. T., & Knowles, T. P. J. (2022).
    Analytical Solution to the Flory-Huggins Model.
    J. Phys. Chem. Lett. 13(33), 7853-7860.

The backend functions take a chi value (or an array of chi values) and raise
if chi is below the critical value. The two functions here scan a chi range
for a polymer of length ``L``, quietly skip the sub-critical region, and
return lists ready for plotting a chi/phi phase diagram. These are the
functions used by :mod:`finches.epsilon_to_FHtheory`.

A reminder on units: chi = eps / (kB T), where eps is the site-to-site contact
energy in Flory-Huggins theory (larger eps = stronger attraction). This means
a chi/phi diagram can be turned into a temperature-like diagram by plotting
1/chi vs. phi, since 1/chi = T * (kB / eps).
"""

import numpy as np

from . import backend as FH

# the binodal solvers calculate_binodal() can dispatch to; all three share the
# (chi, n) signature
_BINODAL_MODES = {
    "analytic_binodal": FH.analytic_binodal,
    "binodal": FH.binodal,
    "GL_binodal": FH.GL_binodal,
}


def _scan_chi(fx, L, chi_min, chi_max, n_points):
    """
    Evaluate a backend solver over a grid of chi values for a length-L polymer.

    The grid is ``n_points`` evenly spaced values over [chi_min, chi_max), and
    the solver is called once on the whole grid. The backend drops chi values
    at or below the critical chi, so the returned lists only cover the
    two-phase region.

    Parameters
    ----------
    fx : callable
        A backend solver with the signature ``fx(chi_array, n)`` that returns
        ``[phi_high, phi_low, chi_kept]`` for array input.

    L : int or float
        Polymer length (number of lattice sites).

    chi_min : float
        Lower end of the chi grid (inclusive).

    chi_max : float
        Upper end of the chi grid (exclusive).

    n_points : int
        Number of grid points.

    Returns
    -------
    tuple
        ``(chis, low, high, phi_c, chi_c)``: lists of the chi values kept and of
        the low- and high-concentration branch at each, followed by the
        critical volume fraction and critical chi as floats. The lists are empty
        if no chi in the grid is above the critical chi.
    """
    phi_c, chi_c = FH.critical(L)
    chi_grid = np.linspace(chi_min, chi_max, n_points, endpoint=False)

    try:
        high, low, chis = fx(chi_grid, L)
    except ValueError:
        # no grid point is above the critical chi
        return ([], [], [], phi_c, chi_c)

    return (list(chis), list(low), list(high), phi_c, chi_c)


def calculate_binodal(
    L, mode="analytic_binodal", chi_min=0.5, chi_max=2.0, n_points=500
):
    """
    Compute the binodal of a length-L polymer over a range of chi values.

    The chi range [chi_min, chi_max) is sampled on a grid of ``n_points``
    evenly spaced values (chi_max itself is excluded), and the backend solver
    selected by ``mode`` is evaluated on the whole grid at once. Chi values at
    or below the critical chi cannot give phase separation and are skipped, so
    the returned lists can be shorter than ``n_points``, or empty.

    The "mode" selector lets you choose how the binodal is calculated. For
    completeness all three approaches described in the paper are offered.
    'analytic_binodal' (the point of the paper) is the default and is what
    FINCHES uses, but read the accuracy notes below before relying on the
    absolute values for long chains.

    Parameters
    ----------
    L : int or float
        Length of the polymer, i.e. the number of lattice sites it occupies
        (N in the paper). Must be >= 1.

    mode : str, optional
        How the binodal is computed. One of:

        * 'analytic_binodal' (default) - closed-form solution (eq. 34/36).
          Stable across the whole phase diagram, but an approximation: for
          L <= 100 the dense branch is within ~1-8% of the exact common-
          tangent solution, but for L ~ 1000 it drifts to 25-30% once chi
          is more than ~10% above chi_c, and the (exponentially small)
          dilute branch is off by tens of percent near chi_c and by orders
          of magnitude at large L*chi.
        * 'binodal' - self-consistent Newton-Raphson iteration (5 iterations,
          improved map). Converges to the exact common-tangent solution, but
          for large L and large chi the intermediate exponentials can
          overflow to NaN.
        * 'GL_binodal' - Ginzburg-Landau expansion about the critical point.
          Only meaningful very close to chi_c; away from it the dilute branch
          goes negative and the dense branch exceeds 1.

    chi_min : float, optional
        Minimum chi value to scan (inclusive). Default 0.5. Values below 0.5
        can never give phase separation (chi_c -> 0.5 as L -> infinity), so
        there is no point going lower.

    chi_max : float, optional
        Maximum chi value scanned (exclusive). Default 2.0.

    n_points : int, optional
        Number of chi grid points between chi_min and chi_max. Default 500.

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
    ValueError
        If ``mode`` is not one of the three recognised strings.

    See Also
    --------
    calculate_spinodal : Same scan for the spinodal.
    finches.analytical_fh.backend : The underlying implementations.
    """
    if mode not in _BINODAL_MODES:
        raise ValueError(
            "mode must be one of 'binodal', 'analytic_binodal', 'GL_binodal'; "
            f"got {mode!r}"
        )

    return _scan_chi(_BINODAL_MODES[mode], L, chi_min, chi_max, n_points)


def calculate_spinodal(L, chi_min=0.5, chi_max=2.0, n_points=500):
    """
    Compute the spinodal of a length-L polymer over a range of chi values.

    Uses the exact analytical spinodal expression (eq. 2 of Qian et al. 2022)
    via :func:`finches.analytical_fh.backend.spinodal`. The chi range
    [chi_min, chi_max) is sampled on a grid of ``n_points`` evenly spaced
    values (chi_max itself is excluded). Chi values at or below the critical
    chi are skipped, so the returned lists can be shorter than ``n_points``,
    or empty.

    Parameters
    ----------
    L : int or float
        Length of the polymer, i.e. the number of lattice sites it occupies
        (N in the paper). Must be >= 1.

    chi_min : float, optional
        Minimum chi value to scan (inclusive). Default 0.5. Values below 0.5
        can never give phase separation.

    chi_max : float, optional
        Maximum chi value scanned (exclusive). Default 2.0.

    n_points : int, optional
        Number of chi grid points between chi_min and chi_max. Default 500.

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
    return _scan_chi(FH.spinodal, L, chi_min, chi_max, n_points)
