"""

Library for 2-component Flory-Huggins theory.

This module implements the analytical self-consistent solution for binodal
concentrations of the two-component Flory-Huggins phase separation model,
as described in:

    Qian, D., Michaels, T.C.T., & Knowles, T.P.J. (2022).
    "Analytical Solution to the Flory-Huggins Model"
    J. Phys. Chem. Lett. 13, 7853-7860.
    https://doi.org/10.1021/acs.jpclett.2c01986

The Flory-Huggins model describes liquid-liquid phase separation (LLPS)
driven by a competition between entropy and interaction energy. The free
energy density is given by:

    f(ϕ) = (ϕ/N)ln(ϕ) + (1-ϕ)ln(1-ϕ) + χ·ϕ·(1-ϕ)

where:
    - ϕ is the polymer volume fraction
    - N is the polymer chain length (number of lattice sites occupied)
    - χ (chi) is the Flory-Huggins interaction parameter

Key concepts:
    - Spinodal: Boundary between locally stable/unstable regions (f''(ϕ) = 0)
    - Binodal: Boundary between globally stable/unstable regions (common tangent)
    - Critical point: Where dense and dilute phases coincide

This library provides:
    - Exact analytical solutions for spinodal concentrations
    - Self-consistent iterative solutions for binodal concentrations
    - Closed-form analytical approximations for binodal concentrations

All functions take the interaction parameter as ``x`` (chi) and the chain
length as ``n`` (N). Every function accepts either a single chi or an array
of chi values; the array path silently drops chi <= chi_c while a scalar
chi <= chi_c raises ``ValueError``. For a scalar, ``spinodal`` and
``analytic_binodal`` return ``[phi_dense, phi_dilute]`` while ``GL_binodal``
and ``binodal`` return ``[phi_dense, phi_dilute, chi]``; for an array every
function returns ``[phi_dense_array, phi_dilute_array, chi_array]``. These
return conventions are kept from the original release.

Author: Daoyuan Qian
Date created: 23 March 2022

Originally released as FH.py in
https://github.com/KnowlesLab-Cambridge/FloryHuggins. Detailed documentation,
boundary-case fixes and consolidation of the scalar and array code paths by
Alex Holehouse (2026).

"""

import numpy as np

# message raised for a chi that cannot give phase separation. Kept verbatim from the
# original release because callers match on it.
_NO_LLPS = "interaction strength too small - no LLPS!"


def help():
    """
    Print a short summary of the public functions in this module.

    Kept from the original KnowlesLab-Cambridge/FloryHuggins release, where it
    was the only documentation. Note that, as a module-level function, it
    shadows Python's built-in help() only when imported by name (e.g.
    ``from finches.analytical_fh.backend import *``); ``backend.help()`` is the
    intended use. The per-function docstrings are the authoritative
    documentation.

    Returns
    -------
    None
    """
    print("Functions in finches.analytical_fh.backend:\n")
    print("    critical(n=1): the critical point, returned as [phi_c, chi_c]\n")
    print(
        "    spinodal(x, n=1): exact spinodal; [phi_dense, phi_dilute] for a scalar "
        "chi, [phi_dense, phi_dilute, chi] for an array\n"
    )
    print(
        "    GL_binodal(x, n=1): Ginzburg-Landau binodal, [phi_dense, phi_dilute, chi]\n"
    )
    print(
        "    binodal(x, n=1, iteration=5, UseImprovedMap=True): self-consistent "
        "binodal after the specified number of iterations, "
        "[phi_dense, phi_dilute, chi]\n"
    )
    print(
        "    analytic_binodal(x, n=1): closed-form binodal; [phi_dense, phi_dilute] "
        "for a scalar chi, [phi_dense, phi_dilute, chi] for an array\n"
    )


# .....................................................................................
#
#
def _supercritical_chi(x, x_c):
    """
    Validate chi against the critical chi and drop sub-critical values.

    This is the single place where every function in the module decides which
    chi values can give phase separation, so that the scalar and array code
    paths always agree: only chi strictly greater than chi_c survives (at
    chi = chi_c the two branches meet at phi_c, which critical() returns).

    Parameters
    ----------
    x : float, int, list, tuple or np.ndarray
        The chi value(s) passed by the caller.

    x_c : float
        The critical chi for the chain length being considered.

    Returns
    -------
    tuple
        ``(chi, is_scalar)``. For scalar input ``chi`` is returned unchanged; for
        array-like input it is a numpy array containing only the values
        strictly greater than ``x_c``, in their original order.

    Raises
    ------
    ValueError
        If a scalar chi is <= x_c, or no element of an array is > x_c (including
        an empty array).
    """
    if np.ndim(x) == 0:
        if x > x_c:
            return x, True
        raise ValueError(_NO_LLPS)

    x = np.asarray(x)
    if x.size == 0 or np.max(x) <= x_c:
        raise ValueError(_NO_LLPS)
    return x[x > x_c], False


# .....................................................................................
#
#
def critical(n=1):
    """
    Calculate the critical point for Flory-Huggins phase separation.

    The critical point is where the dense and dilute phases coincide, representing
    the minimum interaction strength (χ) required for phase separation to occur.

    From equation (3) in Qian et al. 2022:

        χ_c = (1/2) x (1 + 1/√N)²
        ϕ_c = 1 / (1 + √N)

    Physical interpretation:
        - For N=1 (symmetric case): χ_c = 2, ϕ_c = 0.5
        - As N → ∞: χ_c → 0.5, ϕ_c → 0 (longer polymers phase separate more easily)
        - The critical concentration ϕ_c shifts toward dilute as chain length increases

    Parameters
    ----------
    n : int or float, optional
        Polymer chain length (number of lattice sites occupied by one polymer).
        FINCHES uses one lattice site per residue; strictly a residue is larger
        than a water-sized lattice site, so this is a lower bound. Default is 1
        (symmetric solute-solvent case).

    Returns
    -------
    np.ndarray
        Array with 2 elements:
        [0] - ϕ_c: Critical volume fraction (concentration at critical point)
        [1] - χ_c: Critical interaction parameter (minimum χ for phase separation)

    Examples
    --------
    >>> critical(n=1)  # Symmetric case
    array([0.5, 2. ])

    >>> critical(n=100)  # Long polymer: lower χ needed, more dilute critical point
    array([0.09090909, 0.605     ])

    Notes
    -----
    The critical interaction parameter χ_c represents the threshold: phase separation
    only occurs when χ > χ_c. The critical volume fraction ϕ_c is the concentration
    at the top of the phase diagram (the "apex" of the binodal curve).

    """
    x_c = 0.5 * np.power(1.0 + 1.0 / np.sqrt(n), 2)
    phi_c = 1.0 / (1.0 + np.sqrt(n))
    return np.array([phi_c, x_c])


# .....................................................................................
#
#
def spinodal(x, n=1):
    """
    Calculate the spinodal concentrations for given interaction strength(s).

    The spinodal defines the boundary between locally stable and locally unstable
    regions. It is obtained analytically by finding where the second derivative
    of the free energy equals zero: f''(ϕ) = 0.

    From equation (2) in Qian et al. 2022:

        ϕ_spi± = (1/2 - γ/(4χ)) ± √[(1/2 - γ/(4χ))² - 1/(2χN)]

    where γ = 1 - 1/N.

    Physical interpretation:
        - Inside the spinodal: the system is unstable and will spontaneously
          phase separate via spinodal decomposition
        - Between spinodal and binodal: metastable region where nucleation is required
        - The spinodal concentrations have POWER-LAW scaling at large χ:
          ϕ_spi⁻ ~ 1/(2χN) for large χ (equation 5 in paper)
        - This is qualitatively different from the EXPONENTIAL scaling of the binodal

    Parameters
    ----------
    x : float, int, list, tuple or np.ndarray
        The Flory-Huggins interaction parameter χ (chi). Can be a single value
        or an array of values. Must be greater than χ_c for phase separation.

    n : int or float, optional
        Polymer chain length (number of lattice sites). Default is 1.

    Returns
    -------
    np.ndarray
        If x is a single value: array of shape (2,) with [ϕ_dense, ϕ_dilute].
        If x is array-like: array of shape (3, n_valid) with
        [ϕ_dense_array, ϕ_dilute_array, valid_χ_array].

        Only χ values strictly greater than χ_c are included in the array
        output (at χ = χ_c the two branches collapse onto ϕ_c; use critical()
        for that point).

    Raises
    ------
    ValueError
        If a scalar χ is <= χ_c, or every χ in an array is <= χ_c (no phase
        separation possible).

    Examples
    --------
    >>> spinodal(3.0, n=1)  # Single χ value: [dense, dilute]
    array([0.78867513, 0.21132487])

    >>> spinodal([2.0, 2.5, 3.0], n=1)  # χ = 2.0 = χ_c is dropped
    array([[0.7236068 , 0.78867513],
           [0.2763932 , 0.21132487],
           [2.5       , 3.        ]])

    Notes
    -----
    The spinodal is always bounded within 0 < ϕ < 1, unlike the Ginzburg-Landau
    binodal approximation which can enter unphysical regions at large χ.

    """
    x_c = critical(n)[1]
    x, is_scalar = _supercritical_chi(x, x_c)

    # γ = 1 - 1/N, zero for N=1 (symmetric case); equation 2
    gamma = 1.0 - 1.0 / n
    t1 = 1.0 / 2.0 - gamma / (4.0 * x)
    t2 = np.sqrt(np.power(t1, 2) - 1.0 / (2.0 * x * n))

    if is_scalar:
        return np.array([t1 + t2, t1 - t2])
    return np.array([t1 + t2, t1 - t2, x])


# .....................................................................................
#
#
def GL_binodal(x, n=1):
    """
    Calculate the Ginzburg-Landau (GL) approximate binodal concentrations.

    The Ginzburg-Landau approximation is obtained by expanding the free energy
    around the critical point to fourth order in δϕ = ϕ - ϕ_c. This gives a
    simple analytical form that is accurate near the critical point but diverges
    at large χ.

    From equation (8) in Qian et al. 2022:

        ϕ_GL± = ϕ_c ± √[3(χ - χ_c) / (2χ_c² × √N)]

    Physical interpretation:
        - This is a classic second-order phase transition form
        - Valid near criticality where δχ = χ - χ_c is small
        - At large χ, enters unphysical regions (ϕ < 0 or ϕ > 1)
        - Used as the initial guess for the self-consistent iteration

    The GL approximation fails at large χ because:
        - The dilute branch can become negative (ϕ < 0)
        - The dense branch can exceed 1 (ϕ > 1)
        - These are the "gray zones" shown in Figure 1B,C of the paper

    Parameters
    ----------
    x : float, int, list, tuple or np.ndarray
        The Flory-Huggins interaction parameter χ (chi). Can be a single value
        or an array of values. Must be greater than χ_c for phase separation.

    n : int or float, optional
        Polymer chain length (number of lattice sites). Default is 1.

    Returns
    -------
    np.ndarray
        If x is a single value: array [ϕ_dense, ϕ_dilute, χ].
        If x is array-like: array of shape (3, n_valid) with
        [ϕ_dense_array, ϕ_dilute_array, valid_χ_array], where only χ > χ_c
        are kept.

    Raises
    ------
    ValueError
        If a scalar χ is <= χ_c, or every χ in an array is <= χ_c.

    Notes
    -----
    Despite its limitations at large χ, the GL binodal is crucial as the starting
    point for the self-consistent iteration. The self-consistent solution corrects
    the GL binodal to give accurate results across all χ values.

    See Also
    --------
    binodal : Self-consistent iterative solution (more accurate)
    analytic_binodal : Closed-form analytical solution

    """
    phi_c, x_c = critical(n)
    x, _ = _supercritical_chi(x, x_c)

    # GL binodal: ϕ± = ϕ_c ± √[3(χ - χ_c) / (2χ_c² × √N)]
    t2 = np.sqrt(3.0 * (x - x_c) / (2.0 * np.power(x_c, 2) * np.sqrt(n)))
    return np.array([phi_c + t2, phi_c - t2, x])


# .....................................................................................
#
#
def binodal(x, n=1, iteration=5, UseImprovedMap=True):
    """
    Calculate binodal concentrations using the self-consistent iterative method.

    The binodal defines the boundary between globally stable single-phase and
    two-phase regions. This function uses a contractive mapping (fixed-point
    iteration) approach starting from the Ginzburg-Landau approximation.

    The method is based on the self-consistent equations derived in Qian et al. 2022.

    For N=1 (symmetric case), the map is (equation 10):

        g(ϕ) = 1 / (1 + exp(-2χϕ + χ))

    For general N, the 2D map uses (equations 18-19):

        ϕ+ = (1 - e^(-y)) / (1 - e^(-N(x-y)) × e^(-y))
        ϕ- = (1 - e^(-y)) / (e^(N(x-y)) - e^(-y))

    where x = 2χ(ϕ+ - ϕ-) and y = γ(ϕ+ - ϕ-) + χ(ϕ+² - ϕ-²).

    The improved map (UseImprovedMap=True) uses Newton-Raphson acceleration
    via the Jacobian matrix to achieve faster convergence (equation 39):

        H(ϕ) = ϕ + (1 - J)^(-1) × (G(ϕ) - ϕ)

    Key insight from the paper:
        - The dilute phase binodal has EXPONENTIAL scaling: ϕ- ~ exp(-Nχ)
        - This explains why LLPS spans orders of magnitude in concentration
        - Spinodal has only power-law scaling: ϕ_spi- ~ 1/(2χN)

    Parameters
    ----------
    x : float, int, list, tuple or np.ndarray
        The Flory-Huggins interaction parameter χ (chi). Can be a single value
        or an array of values. Must be greater than χ_c for phase separation.

    n : int or float, optional
        Polymer chain length (number of lattice sites). Must be >= 1. Default
        is 1.

    iteration : int, optional
        Number of self-consistent iterations to perform. Must be >= 0; 0
        returns the Ginzburg-Landau starting guess. Default is 5, which with
        the improved map converges to machine precision wherever the
        iteration does not overflow.

    UseImprovedMap : bool, optional
        If True (default), use Newton-Raphson improved iteration for faster
        convergence. If False, use the simple fixed-point iteration.

    Returns
    -------
    np.ndarray
        [ϕ_dense, ϕ_dilute, χ] for a scalar χ, or the same three rows as arrays
        (shape (3, n_valid), only χ > χ_c kept) for array-like χ.
        For N=1: returns [ϕ+, 1-ϕ+, χ] due to symmetry.
        For N>1: returns [ϕ+, ϕ-, χ].

    Raises
    ------
    ValueError
        If ``iteration`` is negative, ``n`` < 1, a scalar χ is <= χ_c, or every
        χ in an array is <= χ_c.

    Examples
    --------
    >>> binodal(3.0, n=1)  # N=1, χ=3
    array([0.92927982, 0.07072018, 3.        ])

    >>> binodal(1.5, n=100)  # long polymer: an exponentially dilute branch
    array([8.58497948e-01, 1.01201942e-27, 1.50000000e+00])

    Notes
    -----
    The convergence is controlled by |g'(ϕ)| near the fixed point:
        - |g'(ϕ)| < 1 gives stable convergence
        - Near criticality, |g'(ϕ)| ≈ 1, so convergence is slower
        - At large χ, |g'(ϕ)| ≈ 0, so convergence is fast

    The improved map accelerates convergence especially near the critical point
    where the simple map is slow.

    Two numerical limits apply to the dilute branch, which is exponentially
    small in Nχ. With too few iterations the Newton step can overshoot to a
    tiny negative value (e.g. -5e-26 for N=100, χ=1.5 after 3 iterations), and
    for very large Nχ the intermediate exponentials overflow and the result is
    NaN, or a tiny negative value even after 5 iterations (e.g. N=5000,
    χ=0.7). Use analytic_binodal() in that regime.

    See Also
    --------
    GL_binodal : Initial guess (Ginzburg-Landau approximation)
    analytic_binodal : Closed-form analytical solution
    spinodal : Spinodal boundary calculation

    """
    if iteration < 0:
        raise ValueError(f"iteration must be >= 0, got {iteration}")
    if n < 1:
        raise ValueError("chain length n must be >= 1")

    if n == 1:
        # =====================================================================
        # SYMMETRIC CASE (N = 1)
        # =====================================================================
        # For N=1 the free energy is symmetric under ϕ → 1-ϕ, so the binodal
        # condition simplifies to f'(ϕ) = 0, giving the self-consistent map
        # g(ϕ) = 1/(1 + exp(-2χϕ + χ))
        pp, _, xx = GL_binodal(x)

        for _ in range(iteration):
            ee = np.exp(-2 * xx * pp + xx)
            if UseImprovedMap:
                # Newton-Raphson improved map (equation 31):
                # h(ϕ) = ϕ + (g(ϕ) - ϕ) / (1 - g'(ϕ))
                pp = (2.0 * xx * pp * ee - 1.0 - ee) / (2.0 * xx * ee - (1.0 + ee) ** 2)
            else:
                # simple fixed-point map (equation 10)
                pp = 1 / (1 + ee)

        # for N=1, ϕ- = 1 - ϕ+ by symmetry
        return np.array([pp, 1 - pp, xx])

    # =========================================================================
    # GENERAL CASE (N > 1)
    # =========================================================================
    # Solve the coupled equations for ϕ+ and ϕ- with the 2D map G(ϕ) of
    # equations 18-19, seeded from the Ginzburg-Landau binodal.
    gamma = 1.0 - 1.0 / n
    p1, p2, xx = GL_binodal(x, n=n)

    for _ in range(iteration):
        # exponents from equation 17:
        # x = 2χ(ϕ+ - ϕ-), y = γ(ϕ+ - ϕ-) + χ(ϕ+² - ϕ-²)
        # a = exp(-x), b = exp(-y), c = (a/b)^N
        a = np.exp(-2.0 * xx * (p1 - p2))
        b = np.exp(-gamma * (p1 - p2) - xx * (np.power(p1, 2) - np.power(p2, 2)))
        c = np.power(a / b, n)

        # the fixed-point map G(ϕ) (equations 18-19)
        g1 = (1.0 - b) / (1.0 - c * b)
        g2 = (1.0 - b) / (np.power(b / a, n) - b)

        if not UseImprovedMap:
            # simple fixed-point iteration: ϕ_new = G(ϕ)
            p1, p2 = g1, g2
            continue

        # Newton-Raphson improved map (equations 38-39):
        # H(ϕ) = ϕ + (1 - J)^(-1) × (G(ϕ) - ϕ), with J = ∂G/∂ϕ
        # partial derivatives of ln(a) and ln(b) with respect to ϕ+ (1) and ϕ- (2)
        d1lna = -2.0 * xx
        d1lnb = -gamma - xx * 2.0 * p1
        d2lna = 2.0 * xx
        d2lnb = gamma + xx * 2.0 * p2

        # Jacobian matrix elements J_ij = ∂g_i/∂ϕ_j
        j11 = (
            g1**2
            * (
                -d1lnb * b * (1 - c) / (1 - b) ** 2
                + n * (d1lna - d1lnb) * c * b / (1 - b)
            )
            - 1
        )
        j21 = g1**2 * (
            -d2lnb * b * (1 - c) / (1 - b) ** 2 + n * (d2lna - d2lnb) * c * b / (1 - b)
        )
        j12 = (j11 + 1) * c + g1 * n * c * (d1lna - d1lnb)
        j22 = j21 * c + g1 * n * c * (d2lna - d2lnb) - 1

        # invert the 2x2 system with Cramer's rule
        detj = j11 * j22 - j12 * j21
        p1, p2 = (
            p1 + (-(g1 - p1) * j22 + (g2 - p2) * j21) / detj,
            p2 + (-(g2 - p2) * j11 + (g1 - p1) * j12) / detj,
        )

    return np.array([p1, p2, xx])


# .....................................................................................
#
#
def analytic_binodal(x, n=1):
    """
    Compute binodal concentrations using the closed-form analytical solution.

    This function implements equations 34-36 from Qian et al. (2022), which provide
    an explicit analytical expression for the binodal without requiring iteration.
    The solution is derived by solving the transcendental equations at the level
    of the auxiliary variable z (equations 25-28), which decouples the fixed-point
    equations into a single self-consistent equation.

    **Mathematical Framework:**

    For N = 1 (symmetric case), the solution is (equation 34):

        ϕ± = 1 / (1 + exp(∓A))

        where A = χ × tanh(χ × √(3(χ - 2)/8))

    For N ≠ 1 (asymmetric case), the solution uses scaled variables (equation 36):

        α = N^(1/4)           (scaling parameter)
        Δ = (χ - χ_c) / χ_c   (reduced distance from critical point)

        ϕ+ = (1 - exp(-X)) / (1 - exp(-Y))
        ϕ- = (1 - exp(+X)) / (1 - exp(+Y))

    where X and Y are functions of hyperbolic cotangent terms:

        coth(A) = 1/tanh((1 + Δ/α²) × √(3Δ) / α)
        coth(B) = 1/tanh((1 + Δα²) × √(3Δ) × α)

    **Physical Interpretation:**

    The analytical solution captures several important physical features:

    1. **Exponential Scaling of Dilute Phase**: ϕ- ~ exp(-Nχ), i.e.
       exponential suppression at large N or χ. This is fundamentally
       different from the spinodal power-law scaling.

    2. **Asymmetry in Phase Behavior**: For N > 1, the dense and dilute
       phases respond differently to changes in χ, encoded in the asymmetric
       dependence on α = N^(1/4).

    3. **Critical Point Behavior**: The solution smoothly approaches the
       critical concentrations as χ → χ_c from above.

    **Advantages over Iterative Methods:**

    - No iteration required - direct evaluation
    - Numerically stable across the entire phase diagram
    - No convergence issues at extreme parameter values
    - Computationally efficient for vectorized calculations

    **Limitations:**

    - Only valid for χ > χ_c (within the two-phase region)
    - Raises ValueError if χ is below the critical value
    - It is an approximation. Against the exact common-tangent solution
      (which binodal() converges to) the dense branch is within ~1-8% for
      N <= 100, but for N ~ 1000 it is only accurate right at χ_c and is
      25-30% off by χ ≈ 1.3 χ_c (40-50% for N ~ 5000). The exponentially
      small dilute branch is off by 10-30% near χ_c and by orders of
      magnitude at large Nχ: the exp(-Nχ) scaling is captured, the
      prefactor is not.

    Parameters
    ----------
    x : float, int, list, tuple or np.ndarray
        The Flory-Huggins interaction parameter χ (chi). Can be a scalar
        or array. Array values <= χ_c are dropped; a scalar <= χ_c raises.

    n : int or float, optional
        The polymer chain length / asymmetry parameter N. Default is 1.
        The solvent is assumed to have N_solvent = 1, making n = N_polymer.

    Returns
    -------
    numpy.ndarray
        For scalar input: [ϕ+, ϕ-]
        For array input: [ϕ+_array, ϕ-_array, χ_array], shape (3, n_valid)

        where ϕ+ is the dense phase concentration and ϕ- is the dilute
        phase concentration.

    Raises
    ------
    ValueError
        If a scalar χ is <= χ_c, or every χ in an array is <= χ_c (no phase
        separation possible).

    Examples
    --------
    >>> analytic_binodal(3.0, n=1)
    array([0.94539891, 0.05460109])

    >>> analytic_binodal(200 / 300, n=100)
    array([0.30077676, 0.00406758])

    See Also
    --------
    binodal : Iterative computation using self-consistent fixed-point maps.
    critical : Compute the critical point (χ_c, ϕ_c).

    References
    ----------
    Qian, D., Michaels, T.C.T., Knowles, T.P.J. (2022). "Analytical Solution to
    the Flory-Huggins Model". J. Phys. Chem. Lett. 13, 7853-7860.
    DOI: 10.1021/acs.jpclett.2c01986

    Key equations:

    - Equation 34: Symmetric (N=1) closed-form solution
    - Equation 36: Asymmetric (N≠1) closed-form solution
    - Equations 25-28: Auxiliary variable approach
    """
    x_c = critical(n)[1]
    x, is_scalar = _supercritical_chi(x, x_c)

    if n == 1:
        # Symmetric case (N=1): Equation 34
        # ϕ± = 1 / (1 + exp(∓A)), where A = χ × tanh(χ × √(3(χ-2)/8))
        A = x * np.tanh(x * np.sqrt(3 * (x - 2) / 8))
        pp = 1 / (1 + np.exp(-A))  # Dense phase (ϕ+)
        pm = 1 / (1 + np.exp(+A))  # Dilute phase (ϕ-)

    else:
        # Asymmetric case (N≠1): Equation 36
        # α = N^(1/4) - scaling parameter that captures asymmetry
        a = n**0.25
        # Δ = (χ - χ_c) / χ_c - reduced distance from critical point
        D = (x - x_c) / x_c

        # symmetric and antisymmetric combinations of α
        c = (a + 1 / a) / 2  # cosh-like: (α + 1/α)/2
        s = (a - 1 / a) / 2  # sinh-like: (α - 1/α)/2

        # hyperbolic cotangent terms from equations 35-36, which encode the
        # asymmetry between dense and dilute phases
        cothA = 1 / np.tanh((1 + D / a**2) * np.sqrt(3 * D) / a)
        cothB = 1 / np.tanh((1 + D * a**2) * np.sqrt(3 * D) * a)

        prefactor = c / (cothA + cothB)

        # exponents X and Y in ϕ± = (1 - exp(∓X)) / (1 - exp(∓Y))
        numerator_exp = 8 * prefactor * (s / a**2 + (1 + D) * prefactor * cothB / a**2)
        denominator_exp = (
            8
            * prefactor
            * (
                s * (1 / a**2 - a**2)
                + (1 + D) * prefactor * (cothB / a**2 + a**2 * cothA)
            )
        )

        pp = (1 - np.exp(-numerator_exp)) / (1 - np.exp(-denominator_exp))  # dense
        pm = (1 - np.exp(+numerator_exp)) / (1 - np.exp(+denominator_exp))  # dilute

    if is_scalar:
        return np.array([pp, pm])
    return np.array([pp, pm, x])
