from . import backend as FH
import numpy as np



def calculate_binodal(L, mode='analytic_binodal', chi_min=0.5, chi_max=2.0, n_points=500):    
    """    
    Wrapper function for computing binodal using different Flory-Huggins
    implementations introduced in the paper 

    Qian, D., Michaels, T. C. T., & Knowles, T. P. J. (2022). 
    Analytical Solution to the Flory-Huggins Model. Journal of 
    Physical Chemistry Letters, 13(33), 7853–7860.

    Specifically, this function lets you pass a polymer length
    (L) a mode (discussed below) and then a range of chi values
    which will be scanned over between chi_min and chi_max with
    n_points evenly spaced between those two values.

    It then returns a phase diagram in chi/phi space. 

    Recall that:

    chi - Flory Huggins interaction strength, and 
    
        chi = eps / (kB * T)

    Where eps = site-to-site contact in Flory Huggins theory 
                (lager eps = strong attractive interactions)

    kB = Boltzmann's constant 

    T = Temperature (in K)

    This means you can in principle convert a chi/phi diagram
    into T (like) vs. phi diagram by plotting 1/chi vs. phi,
    in that 1/chi = T * (kB/eps), so
    # T = eps/(chi*kB)

    The "mode" selector lets you choose how the binodal is
    calculated. For completeness, the three modes described
    at depth in the paper are offered, although in reality
    sticking with the 'analytic_binodal' (which is the point
    of the paper) should for basically all cases be totally
    fine. 

        
    Parameters
    ----------------
    L : int
        Length of the polymer in question
        
    mode : str
        Selector for how to compute, must be one of
        'binodal','analytic_binodal','GL_binodal. In general
        the 'binodal' or 'analytic_binodal' should work well - 
        analytic binodal is a more stable implementation that
        comes from the paper.
        
    chi_min : float
        Minimum chi value to use (lower than 0.5 will never give 
        phase separation
    
    chi_max : float
        Maximum chi value used

    n_points : int
        Number of points between

    Returns
    ---------------
    tuple
        Returns a tuple with 5 elements

        0 : the list of chi values used to get biondal data
        1 : the list of dilute phase concentrations in volume fraction (phi) 
        2 : the list of dense phase concentrations in volume fraction (phi) 
        3 : the critical point volume fraction (phi)
        4 : the critical point chi

    
    """
    
    # check we passed in  av
    if mode not in ['binodal','analytic_binodal','GL_binodal']:
        raise Exception("mode must be one of 'binodal','analytic_binodal','GL_binodal'")
        
        
    # map mode selector to a specific function. Note all three have the same input
    # signature.
    if mode == 'binodal':
        fx = FH.binodal
    elif mode == 'analytic_binodal':
        fx = FH.analytic_binodal
    elif mode == 'GL_binodal':
        fx = FH.GL_binodal
    else:
        raise Exception('UH OH...')

    dense = []
    dilute = []
    chis = []

    # calculate stepsize in chi
    chi_step = (chi_max-chi_min)/n_points

    # chi between chi_min and chi max
    for chi in np.arange(chi_min, chi_max, chi_step):

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
    Wrapper function for computing spinodal using the analytical 
    expression implemented in in the paper:

    Qian, D., Michaels, T. C. T., & Knowles, T. P. J. (2022). 
    Analytical Solution to the Flory-Huggins Model. Journal of 
    Physical Chemistry Letters, 13(33), 7853–7860.

    Specifically, this function lets you pass a polymer length
    (L) and then a range of chi values
    which will be scanned over between chi_min and chi_max with
    n_points evenly spaced between those two values
    
        

    Parameters
    ----------------
    L : int
        Length of the polymer in question
                
    chi_min : float
        Minimum chi value to use (lower than 0.5 will never give 
        phase separation
    
    chi_max : float
        Maximum chi value used

    n_points : int
        Number of points between

    Returns
    ---------------
    tuple
        Returns a tuple with 5 elements

        0 : the list of chi values used to get biondal data
        1 : the list of dilute phase concentrations in volume fraction (phi) 
        2 : the list of dense phase concentrations in volume fraction (phi) 
        3 : the critical point volume fraction (phi)
        4 : the critical point chi

    
    """
    dense = []
    dilute = []
    chis = []

    # calculate stepsize in chi
    chi_step = (chi_max-chi_min)/n_points

    # chi between chi_min and chi max
    for chi in np.arange(chi_min, chi_max, chi_step):

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

        
        
