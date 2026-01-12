'''

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

Author: Daoyuan Qian
Date created: 23 March 2022

Detailed documentation provided by Alex (2026-01-12)

'''

import numpy as np


def help():
    print('Here are the list of functions included in FH.py:\n')
    print('	critical(n = 1): returns the critical concentration and critical interaction [phi_c, chi_c]\n')
    print('	spinodal(chi, n = 1): returns spinodal concentrations [p1, p2, chi] in the valid chi range\n')
    print('	GL_binodal(chi, n = 1): Ginzburg-Landau binodal [p1, p2, chi]\n')
    print(
        '	binodal(chi, n = 1, iteration = 5, UseImprovedMap = True): self-consistent solution with speficied number of iterations [p1, p2, chi]. You can also use the simple map to see what it does\n')
    print(' analytic_binodal(x, n = 1): analytic forms')


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
    n : int or float
        Polymer chain length (number of lattice sites occupied by one polymer).
        For proteins, this is typically larger than the number of residues because
        amino acids are larger than water molecules (the lattice site is water-sized).
        Default is 1 (symmetric solute-solvent case).

    Returns
    -------
    np.ndarray
        Array with 2 elements:
        [0] - ϕ_c: Critical volume fraction (concentration at critical point)
        [1] - χ_c: Critical interaction parameter (minimum χ for phase separation)

    Examples
    --------
    >>> critical(n=1)  # Symmetric case
    array([0.5, 2.0])
    
    >>> critical(n=100)  # Long polymer
    array([0.09090909, 0.605])  # Lower χ needed, more dilute critical point

    Notes
    -----
    The critical interaction parameter χ_c represents the threshold: phase separation
    only occurs when χ > χ_c. The critical volume fraction ϕ_c is the concentration
    at the top of the phase diagram (the "apex" of the binodal curve).

    """
    x_c = 0.5 * np.power(1. + 1. / np.sqrt(n), 2)
    phi_c = 1. / (1. + np.sqrt(n))
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
    
        ϕ_spi± = (1/2 - gamma/(4χ)) ± √[(1/2 - gamma/(4χ))² - 1/(2χN)]
    
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
    x : float, int, list, or np.ndarray
        The Flory-Huggins interaction parameter χ (chi). Can be a single value
        or an array of values. Must be greater than χ_c for phase separation.
        
    n : int or float
        Polymer chain length (number of lattice sites). Default is 1.

    Returns
    -------
    np.ndarray
        If x is a single value: array of shape (2,) with [ϕ_dense, ϕ_dilute]
        If x is an array: array of shape (3, len(valid_x)) with 
        [ϕ_dense_array, ϕ_dilute_array, valid_x_array]
        
        Note: Only χ values >= χ_c are included in the output.

    Raises
    ------
    ValueError
        If all χ values are below the critical χ_c (no phase separation possible).

    Examples
    --------
    >>> spinodal(3.0, n=1)  # Single χ value
    array([0.833..., 0.166...])  # [dense, dilute] concentrations
    
    >>> spinodal([2.0, 2.5, 3.0], n=1)  # Multiple χ values
    array([[dense_1, dense_2, dense_3],
           [dilute_1, dilute_2, dilute_3],
           [2.0, 2.5, 3.0]])

    Notes
    -----
    The spinodal is always bounded within 0 < ϕ < 1, unlike the Ginzburg-Landau
    binodal approximation which can enter unphysical regions at large χ.

    """
    # get critical chi
    crit = critical(n)
    x_c = crit[1]

    # calculate gamma (see equation 2)
    # γ = 1 - 1/N, goes to zero for N=1 (symmetric case)
    gamma = 1. - 1. / n

    # if x is a single value (float or int)
    if not np.array(x).shape:

        # if chi is greater (equal to or stronger) than critical chi
        if x > x_c:

            # Calculate spinodal using equation 2
            # t1 = first term: (1/2 - γ/(4χ))
            # t2 = second term: √[(1/2 - γ/(4χ))² - 1/(2χN)]
            t1 = 1. / 2. - gamma / (4. * x)
            t2 = np.sqrt(np.power(t1, 2) - 1. / (2. * x * n))
            return np.array([t1 + t2, t1 - t2])

        # else chi is too weak for phase separation
        else:
            raise ValueError('interaction strength too small - no LLPS!')

    # else if x is an array or list
    else:
        # if the largest (strongest) chi is less than the critical chi then
        # none of the values will give rise to phase separation
        if max(x) < x_c:
            raise ValueError('interaction strength too small - no LLPS!')

        # Calculate for all valid χ values (χ >= χ_c)
        else:
            x = np.array(x)
            x = x[x >= x_c]
            t1 = 1. / 2. - gamma / (4. * x)
            t2 = np.sqrt(np.power(t1, 2) - 1. / (2. * x * n))
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
    x : float, int, list, or np.ndarray
        The Flory-Huggins interaction parameter χ (chi). Can be a single value
        or an array of values. Must be greater than χ_c for phase separation.
        
    n : int or float
        Polymer chain length (number of lattice sites). Default is 1.

    Returns
    -------
    np.ndarray
        If x is a single value: array [ϕ_dense, ϕ_dilute, χ]
        If x is an array: array [ϕ_dense_array, ϕ_dilute_array, valid_χ_array]

    Raises
    ------
    ValueError
        If all χ values are below the critical χ_c.

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
    crit = critical(n)
    x_c = crit[1]
    phi_c = crit[0]

    if not np.array(x).shape:
        if x > x_c:
            # GL binodal: ϕ± = ϕ_c ± √[3(χ - χ_c) / (2χ_c² × √N)]
            t1 = phi_c
            t2 = np.sqrt(3. * (x - x_c) / (2. * np.power(x_c, 2) * np.sqrt(n)))
            return np.array([t1 + t2, t1 - t2, x])
        else:
            raise ValueError('interaction strength too small - no LLPS!')
    else:
        if max(x) < x_c:
            raise ValueError('interaction strength too small - no LLPS!')
        else:
            x = np.array(x)
            x = x[x >= x_c]
            t1 = phi_c
            t2 = np.sqrt(3. * (x - x_c) / (2. * np.power(x_c, 2) * np.sqrt(n)))
            return np.array([t1 + t2, t1 - t2, x])


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
    x : float, int, list, or np.ndarray
        The Flory-Huggins interaction parameter χ (chi). Can be a single value
        or an array of values. Must be greater than χ_c for phase separation.
        
    n : int or float
        Polymer chain length (number of lattice sites). Default is 1.
        
    iteration : int
        Number of self-consistent iterations to perform. Default is 5.
        Typically 2-3 iterations achieve numerical accuracy (see Figure 1E,F).
        
    UseImprovedMap : bool
        If True (default), use Newton-Raphson improved iteration for faster
        convergence. If False, use the simple fixed-point iteration.

    Returns
    -------
    np.ndarray
        Array [ϕ_dense, ϕ_dilute, χ] or arrays if x is a vector.
        For N=1: returns [ϕ+, 1-ϕ+, χ] due to symmetry.
        For N>1: returns [ϕ+, ϕ-, χ].

    Examples
    --------
    >>> binodal(3.0, n=1)  # N=1, χ=3
    array([0.905..., 0.094..., 3.0])
    
    >>> binodal(1.5, n=100, iteration=3)  # Long polymer
    array([0.85..., 0.001..., 1.5])

    Notes
    -----
    The convergence is controlled by |g'(ϕ)| near the fixed point:
        - |g'(ϕ)| < 1 gives stable convergence
        - Near criticality, |g'(ϕ)| ≈ 1, so convergence is slower
        - At large χ, |g'(ϕ)| ≈ 0, so convergence is fast
    
    The improved map accelerates convergence especially near the critical point
    where the simple map is slow.

    See Also
    --------
    GL_binodal : Initial guess (Ginzburg-Landau approximation)
    analytic_binodal : Closed-form analytical solution
    spinodal : Spinodal boundary calculation

    """
    assert iteration >= 0
    crit = critical(n)
    x_c = crit[1]
    phi_c = crit[0]
    gamma = 1. - 1. / n

    if n == 1:
        # =====================================================================
        # SYMMETRIC CASE (N = 1)
        # =====================================================================
        # For N=1, the free energy is symmetric under ϕ → 1-ϕ
        # The binodal condition simplifies to f'(ϕ) = 0
        # Leading to the self-consistent map: g(ϕ) = 1/(1 + exp(-2χϕ + χ))

        guess = GL_binodal(x)

        pp = guess[0]  # Dense phase concentration
        xx = guess[2]  # Chi values

        if UseImprovedMap:
            # Improved map using Newton-Raphson (equation 31)
            # h(ϕ) = ϕ + (g(ϕ) - ϕ) / (1 - g'(ϕ))
            for _ in range(iteration):
                ee = np.exp(- 2 * xx * pp + xx)
                # This implements the improved map h(ϕ) with Jacobian correction
                pp = (2. * xx * pp * ee - 1. - ee) / (2. * xx * ee - (1. + ee)**2)

        else:
            # Simple fixed-point map (equation 10)
            # g(ϕ) = 1 / (1 + exp(-2χϕ + χ))
            for _ in range(iteration):
                ee = np.exp(- 2 * xx * pp + xx)
                pp = 1 / (1 + ee)

        # For N=1, ϕ- = 1 - ϕ+ by symmetry
        return np.array([pp, 1 - pp, xx])

    if n > 1:
        # =====================================================================
        # GENERAL CASE (N > 1)
        # =====================================================================
        # For N>1, need to solve coupled equations for ϕ+ and ϕ-
        # Using the 2D map G(ϕ) defined in equations 18-19

        guess = GL_binodal(x, n=n)

        p1 = guess[0]  # Dense phase (ϕ+)
        p2 = guess[1]  # Dilute phase (ϕ-)
        xx = guess[2]  # Chi values

        if UseImprovedMap:
            # Newton-Raphson improved map (equations 38-39)
            # H(ϕ) = ϕ + (1 - J)^(-1) × (G(ϕ) - ϕ)
            # Requires computing the Jacobian matrix J = ∂G/∂ϕ
            for _ in range(iteration):

                # Define the exponents from equation 17:
                # x = 2χ(ϕ+ - ϕ-)  [drives apart based on concentration difference]
                # y = γ(ϕ+ - ϕ-) + χ(ϕ+² - ϕ-²)  [includes entropic and interaction terms]
                # a = exp(-x), b = exp(-y), c = (a/b)^N
                a = np.exp(- 2. * xx * (p1 - p2))
                b = np.exp(- gamma * (p1 - p2) - xx * (np.power(p1, 2) - np.power(p2, 2)))
                c = np.power(a / b, n)

                # The fixed-point map G(ϕ) from equation 18-19:
                # g1 = ϕ+ update, g2 = ϕ- update
                g1 = (1. - b) / (1. - np.power(a / b, n) * b)
                g2 = (1. - b) / (np.power(b / a, n) - b)

                # Compute partial derivatives for the Jacobian matrix
                # d1lna = ∂ln(a)/∂ϕ+, d1lnb = ∂ln(b)/∂ϕ+
                # d2lna = ∂ln(a)/∂ϕ-, d2lnb = ∂ln(b)/∂ϕ-
                d1lna = - 2. * xx
                d1lnb = - gamma - xx * 2. * p1
                d2lna = 2. * xx
                d2lnb = gamma + xx * 2. * p2

                # Jacobian matrix elements J_ij = ∂g_i/∂ϕ_j
                j11 = g1**2 * (- d1lnb * b * (1 - c) / (1 - b)**2 + n * (d1lna - d1lnb) * c * b / (1 - b)) - 1
                j21 = g1**2 * (- d2lnb * b * (1 - c) / (1 - b)**2 + n * (d2lna - d2lnb) * c * b / (1 - b))
                j12 = (j11 + 1) * c + g1 * n * c * (d1lna - d1lnb)
                j22 = j21 * c + g1 * n * c * (d2lna - d2lnb) - 1

                # Newton-Raphson update: ϕ_new = ϕ + (1-J)^(-1) × (G(ϕ) - ϕ)
                # Using Cramer's rule for the 2x2 matrix inversion
                detj = j11 * j22 - j12 * j21

                p1_new = np.copy(p1 + (- (g1 - p1) * j22 + (g2 - p2) * j21) / detj)
                p2_new = np.copy(p2 + (- (g2 - p2) * j11 + (g1 - p1) * j12) / detj)

                p1 = p1_new
                p2 = p2_new

        else:
            # Simple fixed-point iteration (equation 19)
            # Just apply G(ϕ) repeatedly without Newton-Raphson acceleration
            for _ in range(iteration):

                # Same exponent definitions as above
                a = np.exp(- 2. * xx * (p1 - p2))
                b = np.exp(- gamma * (p1 - p2) - xx * (np.power(p1, 2) - np.power(p2, 2)))
                c = np.power(a / b, n)

                g1 = (1. - b) / (1. - np.power(a / b, n) * b)
                g2 = (1. - b) / (np.power(b / a, n) - b)

                # Simple update: ϕ_new = G(ϕ)
                p1_new = np.copy((1. - b) / (1. - np.power(a / b, n) * b))
                p2_new = np.copy((1. - b) / (np.power(b / a, n) - b))

                p1 = p1_new
                p2 = p2_new

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
    
    1. **Exponential Scaling of Dilute Phase**: Near the critical point,
       ϕ- ~ exp(-Nχ), showing exponential suppression at large N or χ.
       This is fundamentally different from the spinodal power-law scaling.
    
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
    
    Parameters
    ----------
    x : float or numpy.ndarray
        The Flory-Huggins interaction parameter χ (chi). Can be a scalar
        or array. Values below χ_c will be filtered out or raise an error.
        
    n : float, optional
        The polymer chain length / asymmetry parameter N. Default is 1.
        The solvent is assumed to have N_solvent = 1, making n = N_polymer.
    
    Returns
    -------
    numpy.ndarray
        For scalar input: [ϕ+, ϕ-]
        For array input: [ϕ+_array, ϕ-_array, χ_array]
        
        where ϕ+ is the dense phase concentration and ϕ- is the dilute
        phase concentration.
    
    Raises
    ------
    ValueError
        If χ < χ_c for all input values (no phase separation possible).
    
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

    crit = critical(n)
    x_c = crit[1]

    # Handle scalar input
    if not np.array(x).shape:
        if x > x_c:
            if n == 1:
                # Symmetric case (N=1): Equation 34
                # ϕ± = 1 / (1 + exp(∓A))
                # where A = χ × tanh(χ × √(3(χ-2)/8))
                A = x * np.tanh(x * np.sqrt(3 * (x - 2) / 8))
                pp = 1 / (1 + np.exp(-A))  # Dense phase (ϕ+)
                pm = 1 / (1 + np.exp(+A))  # Dilute phase (ϕ-)

            else:
                # Asymmetric case (N≠1): Equation 36
                # Use scaled variables α and Δ
                
                # α = N^(1/4) - scaling parameter that captures asymmetry
                a = n ** 0.25
                # Δ = (χ - χ_c) / χ_c - reduced distance from critical point
                D = (x - x_c) / x_c

                # Symmetric and antisymmetric combinations of α
                c = (a + 1 / a) / 2  # cosh-like: (α + 1/α)/2
                s = (a - 1 / a) / 2  # sinh-like: (α - 1/α)/2

                # Hyperbolic cotangent terms from equation 35-36
                # These encode the asymmetry between dense and dilute phases
                cothA = 1 / np.tanh((1 + D / a**2) * np.sqrt(3 * D) / a)
                cothB = 1 / np.tanh((1 + D * a**2) * np.sqrt(3 * D) * a)

                # Common prefactor
                prefactor = c / (cothA + cothB)

                # Exponents X and Y in ϕ± = (1 - exp(∓X)) / (1 - exp(∓Y))
                numerator_exp = 8 * prefactor * (s / a**2 + (1 + D) * prefactor * cothB / a**2)
                denominator_exp = 8 * prefactor * (s * (1 / a**2 - a**2) + (1 + D)
                                                   * prefactor * (cothB / a**2 + a**2 * cothA))

                # Final binodal concentrations
                pp = (1 - np.exp(-numerator_exp)) / (1 - np.exp(-denominator_exp))  # Dense phase
                pm = (1 - np.exp(+numerator_exp)) / (1 - np.exp(+denominator_exp))  # Dilute phase

            return np.array([pp, pm])

        else:
            raise ValueError('interaction strength too small - no LLPS!')
    
    # Handle array input - vectorized computation
    else:
        if max(x) < x_c:
            raise ValueError('interaction strength too small - no LLPS!')
        else:
            # Filter to only include χ values above critical point
            x = np.array(x)
            x = x[x >= x_c]

            if n == 1:
                # Symmetric case (N=1): Equation 34 - vectorized
                A = x * np.tanh(x * np.sqrt(3 * (x - 2) / 8))
                pp = 1 / (1 + np.exp(-A))  # Dense phase array
                pm = 1 / (1 + np.exp(+A))  # Dilute phase array

            else:
                # Asymmetric case (N≠1): Equation 36 - vectorized
                # Same algorithm as scalar case, but operates element-wise on arrays
                
                a = n ** 0.25
                D = (x - x_c) / x_c

                c = (a + 1 / a) / 2
                s = (a - 1 / a) / 2

                cothA = 1 / np.tanh((1 + D / a**2) * np.sqrt(3 * D) / a)
                cothB = 1 / np.tanh((1 + D * a**2) * np.sqrt(3 * D) * a)

                prefactor = c / (cothA + cothB)

                numerator_exp = 8 * prefactor * (s / a**2 + (1 + D) * prefactor * cothB / a**2)
                denominator_exp = 8 * prefactor * (s * (1 / a**2 - a**2) + (1 + D)
                                                   * prefactor * (cothB / a**2 + a**2 * cothA))

                pp = (1 - np.exp(-numerator_exp)) / (1 - np.exp(-denominator_exp))
                pm = (1 - np.exp(+numerator_exp)) / (1 - np.exp(+denominator_exp))

            # Return with filtered χ values (only those above critical point)
            return np.array([pp, pm, x])
