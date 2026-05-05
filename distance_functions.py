# -*- coding: utf-8 -*-
"""
distance_functions.py.

Created on Tue Jun 10 11:35:46 2025
v1.0

@author: sane

Distance functions between PDFs. All distance functions must satisfy the properties:
    1. d[p(x,t),π(x;T)] > 0 for all p;
    2. d[p(x,t_2),π(x;T)] <= d[p(x,t_1),π(x;T)] if t_2 > t_1 (monotonic nonincreasing-ness with relaxation);
    3. d[π(x;T_h),π(x;T_c)] > d[π(x;T_w),π(x;T_c)] if T_h > T_w (monotonic increasingness with temperature)
    
These are optimised to take in vectors of histograms and operate on the bins axis.
"""

import numpy as np
import scipy
import numba

def L1(pdf_1, pdf_2, dx, axis=None):
    """
    Find sum |pdf_2-pdf_1|_i. Does NOT calculate the integral; inputs must be converted to PMFs before function is called.

    Parameters
    ----------
    pdf_1 : 2D vector of numerics
        Some probability density vector (this will not fail if pdf_1 is not a probability mass but the intended use case is for PMFs).
    pdf_2 : 2D vector of numerics of same shape as pdf_1
        Second probability density vector.
    dx: float
        Distance between bins
    axis : int, optional
        If pdf_1 is multi-dimensional, the axis to sum along. The default is None.

    Returns
    -------
    float
        sum |pdf_2-pdf_1|_i. If this is less than zero or greater than two, pdf_1 and pdf_2 are not correctly normalised.

    """
    # pdf_1 and pdf_2 must be at least two-dimensional arrays
    assert pdf_1.shape[axis]==pdf_2.shape[axis], f"Mismatch between pdf_1 {np.array(pdf_1).shape} and pdf_2 {np.array(pdf_2).shape}"
    return np.sum(np.abs(pdf_2-pdf_1), axis=axis)*dx

def Lp_constructor(p):
    """Returns a p-norm."""
    def Lp(pdf_1, pdf_2, dx, axis=None):
        """
        Find sum |pdf_2-pdf_1|^p_i. Does NOT calculate the integral; inputs must be converted to PMFs before function is called.
    
        Parameters
        ----------
        pdf_1 : 2D vector of numerics
            Some probability density vector (this will not fail if pdf_1 is not a probability mass but the intended use case is for PMFs).
        pdf_2 : 2D vector of numerics of same shape as pdf_1
            Second probability density vector.
        dx: float
            Distance between bins
        axis : int, optional
            If pdf_1 is multi-dimensional, the axis to sum along. The default is None.
        p : int, optional
            The p in the p-norm. The default is 2.
    
        Returns
        -------
        float
            sum |pdf_2-pdf_1|_i. If this is less than zero or greater than two, pdf_1 and pdf_2 are not correctly normalised.
    
        """
        # pdf_1 and pdf_2 must be at least two-dimensional arrays
        assert pdf_1.shape[axis]==pdf_2.shape[axis], f"Mismatch between pdf_1 {np.array(pdf_1).shape} and pdf_2 {np.array(pdf_2).shape}"
        return dx*(np.sum(np.abs(pdf_2-pdf_1)**p, axis=axis)**(1/p))
    return Lp

@numba.njit
def _helper_2D(arr_1, arr_2, epsilon):
    # Avoids issues with taking the log of 0.
    out_arr = np.zeros_like(arr_1)
    for i, subarr_1 in enumerate(arr_1):
        for j, element_1 in enumerate(subarr_1):
            element_2 = arr_2[i,j]
            if element_1 == 0:
                out_arr[i,j] = 0
            else:
                if element_2 != 0:
                    out_arr[i,j] = element_1*np.log(element_1)-element_1*np.log(element_2)
                else:
                    out_arr[i,j] = element_1*np.log(element_1/epsilon)
    return out_arr

@numba.njit
def _helper_3D(arr_1, arr_2, epsilon):
    # Avoids issues with taking the log of 0.
    out_arr = np.zeros_like(arr_1)
    for i, subarr_1 in enumerate(arr_1):
        for j, subsubarr_1 in enumerate(subarr_1):
            for k, element_1 in enumerate(subsubarr_1):
                element_2 = arr_2[i,j,k]
                if element_1 == 0:
                    out_arr[i,j,k] = 0
                else:
                    if element_2 != 0:
                        out_arr[i,j,k] = element_1*np.log(element_1)-element_1*np.log(element_2)
                    else:
                        out_arr[i,j,k] = element_1*np.log(element_1/epsilon)
    return out_arr
    
def kullback_leibler(pdf_1, pdf_2, dx, axis=None, epsilon=None):
    """
    Take the entropic distance between pdf_1 and pdf_2. Not implemented to fullest potential yet.

    Parameters
    ----------
    pdf_1 : 2D vector of numerics
        Some probability density vector (this will not fail if pdf_1 is not a probability density but the intended use case is for PDFs).
    pdf_2 : 2D vector of numerics of same shape as pdf_1
        Second probability density vector.
    dx: float
        Distance between bins
    axis : int, optional
        If pdf_1 is multi-dimensional, the axis to sum along. The default is None.
    epsilon : float, optional
        Some small pseudo-count probability to avoid issues with taking log(0)

    Returns
    -------
    vector of floats
        KL distance between pdf_1 and pdf_2.

    """
    if epsilon is None:
        epsilon = 1/(pdf_1.shape[axis]*dx)
        print(epsilon)
    
    log_ratio = np.where(np.isfinite(np.log(pdf_1)-np.log(pdf_2)), np.log(pdf_1)-np.log(pdf_2), epsilon*np.ones_like(pdf_1))
    
    # if len(pdf_1.shape) == 2:
    #     helper = _helper_2D
    # elif len(pdf_1.shape) == 3:
    #     helper = _helper_3D
    # else:
    #     raise NotImplementedError
    return np.sum(pdf_1*log_ratio, axis=axis)*dx # If an element in pdf_1 is zero, pdf_1*log(epsilon) = 0.
def KL_constructor(epsilon):
    return lambda pdf_1, pdf_2, dx, axis: kullback_leibler(pdf_1, pdf_2, dx=dx, axis=axis, epsilon=epsilon)

def kolmogorov_smirnov(pdf_1, pdf_2, dx, axis=None):
    """
    Calculate the Kolmogorov-Smirnov statistic for histograms pdf_1 and pdf_2.

    Parameters
    ----------
    pdf_1 : vector of numerics
        Probability density vector.
    pdf_2 : vector of numerics of same shape as pdf_1
        Another probability density vector.
    dx: float
        Distance between bins
    axis : int, optional
        axis to cumsum along. The default is None.

    Returns
    -------
    vector of numerics
        Kolmogorov-Smirnov statistic between both histograms.

    """
    CDF1, CDF2 = np.cumsum(pdf_1*dx, axis=axis), np.cumsum(pdf_2*dx, axis=axis) # Compute the CDFs
    return np.max(np.abs(CDF2-CDF1), axis=axis)

def KS_empirical(data, CDF, x):
    """
    Use the empirical CDF generated from data to compare it to a proper CDF. Uses scipy's methods. CDF is an array --- we generate an interpolating function to pass to scipy.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    CDF : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    CDF_interpolated = lambda t: np.interp(t, x, CDF)
    return CDF_interpolated

def entropic_distance(pdf_1, pdf_2, dx, axis=None, energies=None, k_BT_c=1):
    def log_helper(pdf):
        if pdf <= 0:
            return 0
        return pdf*np.log(pdf)
    log_helper = np.vectorize(log_helper)
    E_1 = energies*pdf_1
    E_2 = energies*pdf_2
    return ((E_1-E_2)/k_BT_c + log_helper(pdf_1) - log_helper(pdf_2)).sum(axis=axis)*dx
def entropic_distance_loader(bins, potential, k_BT_c=1, num_temps=3):
    energies_array = potential.U(bins)[np.newaxis, :]
    energies = np.repeat(energies_array, num_temps, axis=0)[...,np.newaxis]
    return lambda pdf_1, pdf_2, dx, axis: entropic_distance(pdf_1, pdf_2, dx=dx, axis=axis, energies=energies, k_BT_c=k_BT_c)

def asymmetry_monotone(pdf, pdf2, dx, distance_function = L1, axis=None):
    """Asymmetry monotone defined by Teza and Moroderm. pdf2 is not really necessary as an argument but it's needed so that this distance function is consistent with previous formats."""
    pdf_rev = np.flip(pdf, axis=axis) # Flip along the axis of integration
    G_p = 0.5*(pdf+pdf_rev) # Symmetrised 'twirled' PDF
    
    return distance_function(pdf, G_p, dx=dx, axis=axis)
def asymmetry_monotone_loader(distance_function):
    return lambda pdf, pdf2, dx, axis: asymmetry_monotone(pdf, pdf2, dx, axis=axis, distance_function=distance_function)

def symmetrised_pdf(pdf, axis=None):
    pdf_rev = np.flip(pdf, axis=axis) # Flip along the axis of integration
    G_p = 0.5*(pdf+pdf_rev) # Symmetrised 'twirled' PDF
    return G_p