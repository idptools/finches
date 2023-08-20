'''

Library for 2-component Flory-Huggings theory.

Author: Daoyuan Qian
Date created: 23 March 2022


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
    Calculates the critical point for a given polymer length.

    Equation (3) in the paper.

    I don't understand why this works or where these functional
    forms come from?

    Parameters
    -----------------
    n : int
        Polymer length.


    Returns
    ---------------
    np.array of l=2
        Returns an numpy array with 2 elements
        [0] - volume fraction at critical point
        [1] - chi at critical point

    """

    x_c = 0.5 * np.power(1. + 1. / np.sqrt(n), 2)
    phi_c = 1. / (1. + np.sqrt(n))
    return np.array([phi_c, x_c])


# .....................................................................................
#
#
def spinodal(x, n=1):
    """
    Returns the spinodal curve solved analytically based on the instability from the
    second derivative of free energy with respect to polymer volume fraction.

    Parameters
    ----------------
    x : float/int or list/array
        Value or vector of values for chis (recommend only using single values
        otherwise cannot relate chi (x) to the spinodal values

    n : int
        Polymer length.

    Returns
    -----------
    np.array len = 2
        If a single float/int is passed as chi, then a 2-element array
        or float is returned with the [dense, dilute] concentrations
        of the spinodal

    """

    # get critical chi
    crit = critical(n)
    x_c = crit[1]

    # calculate gamma (see equation 2)
    gamma = 1. - 1. / n

    # if x is a single value (float or int)
    if not np.array(x).shape:

        # if chi is greater (equal to or stronger) than critical chi
        if x > x_c:

            # calculate chi as per equation 2 (t1 and t2 are the
            # two terms)
            t1 = 1. / 2. - gamma / (4. * x)
            t2 = np.sqrt(np.power(t1, 2) - 1. / (2. * x * n))
            return np.array([t1 + t2, t1 - t2])

        # else chi is too weak
        else:
            raise ValueError('interaction strength too small - no LLPS!')

    # else if x is an array or list
    else:
        # if the largest (strongest) chi is less than the critical chi then
        # none of the values will give rise to phase separation, so reject
        # this premise! FWIW raising an exception here is bad - this isn't
        # an 'error' it's just meaning there is no spinodal
        if max(x) < x_c:
            raise ValueError('interaction strength too small - no LLPS!')

        # else calculate equation 2 for each value where chi is stronger than
        # critical chi - note this looses information as to which chi values
        # are used, which makes the output here basically unuseable if
        # len(return) != len(x).
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
    GL = Ginzerberg-Landau

    """

    crit = critical(n)
    x_c = crit[1]
    phi_c = crit[0]

    if not np.array(x).shape:
        if x > x_c:
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

    assert iteration >= 0
    crit = critical(n)
    x_c = crit[1]
    phi_c = crit[0]
    gamma = 1. - 1. / n

    if n == 1:

        guess = GL_binodal(x)

        pp = guess[0]
        xx = guess[2]

        if UseImprovedMap:

            for _ in range(iteration):
                ee = np.exp(- 2 * xx * pp + xx)
                pp = (2. * xx * pp * ee - 1. - ee) / (2. * xx * ee - (1. + ee)**2)

        else:

            for _ in range(iteration):
                ee = np.exp(- 2 * xx * pp + xx)
                pp = 1 / (1 + ee)

        return np.array([pp, 1 - pp, xx])

    if n > 1:

        guess = GL_binodal(x, n=n)

        p1 = guess[0]
        p2 = guess[1]
        xx = guess[2]

        if UseImprovedMap:

            for _ in range(iteration):

                a = np.exp(- 2. * xx * (p1 - p2))
                b = np.exp(- gamma * (p1 - p2) - xx * (np.power(p1, 2) - np.power(p2, 2)))
                c = np.power(a / b, n)

                g1 = (1. - b) / (1. - np.power(a / b, n) * b)
                g2 = (1. - b) / (np.power(b / a, n) - b)

                d1lna = - 2. * xx
                d1lnb = - gamma - xx * 2. * p1
                d2lna = 2. * xx
                d2lnb = gamma + xx * 2. * p2

                j11 = g1**2 * (- d1lnb * b * (1 - c) / (1 - b)**2 + n * (d1lna - d1lnb) * c * b / (1 - b)) - 1
                j21 = g1**2 * (- d2lnb * b * (1 - c) / (1 - b)**2 + n * (d2lna - d2lnb) * c * b / (1 - b))
                j12 = (j11 + 1) * c + g1 * n * c * (d1lna - d1lnb)
                j22 = j21 * c + g1 * n * c * (d2lna - d2lnb) - 1

                detj = j11 * j22 - j12 * j21

                p1_new = np.copy(p1 + (- (g1 - p1) * j22 + (g2 - p2) * j21) / detj)
                p2_new = np.copy(p2 + (- (g2 - p2) * j11 + (g1 - p1) * j12) / detj)

                p1 = p1_new
                p2 = p2_new

        else:

            for _ in range(iteration):

                a = np.exp(- 2. * xx * (p1 - p2))
                b = np.exp(- gamma * (p1 - p2) - xx * (np.power(p1, 2) - np.power(p2, 2)))
                c = np.power(a / b, n)

                g1 = (1. - b) / (1. - np.power(a / b, n) * b)
                g2 = (1. - b) / (np.power(b / a, n) - b)

                p1_new = np.copy((1. - b) / (1. - np.power(a / b, n) * b))
                p2_new = np.copy((1. - b) / (np.power(b / a, n) - b))

                p1 = p1_new
                p2 = p2_new

        return np.array([p1, p2, xx])


# .....................................................................................
#
#
def analytic_binodal(x, n=1):

    crit = critical(n)
    x_c = crit[1]

    if not np.array(x).shape:
        if x > x_c:
            if n == 1:
                pp = 1 / (1 + np.exp(-x * np.tanh(x * np.sqrt(3 * (x - 2) / 8))))
                pm = 1 / (1 + np.exp(-x * np.tanh(-x * np.sqrt(3 * (x - 2) / 8))))

            else:

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

            return np.array([pp, pm])

        else:
            raise ValueError('interaction strength too small - no LLPS!')
    else:
        if max(x) < x_c:
            raise ValueError('interaction strength too small - no LLPS!')
        else:
            x = np.array(x)
            x = x[x >= x_c]

            if n == 1:
                pp = 1 / (1 + np.exp(-x * np.tanh(x * np.sqrt(3 * (x - 2) / 8))))
                pm = 1 / (1 + np.exp(-x * np.tanh(-x * np.sqrt(3 * (x - 2) / 8))))

            else:

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

            return np.array([pp, pm, x])
