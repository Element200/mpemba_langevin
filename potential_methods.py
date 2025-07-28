# -*- coding: utf-8 -*-
"""
potential_methods.py.

Created on Wed Jun 11 10:22:16 2025
v1.2

@author: sane
Creates classes that can be used to add a bunch of handy attributes to various kinds of potentials.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sympy as sym
import pandas as pd

class Potential(object):
    """Takes in a functional form of a potential and defines a bunch of handy functions."""
    
    def __init__(self):
        x = sym.symbols('x') # Define symbols for numeric computation
        self.F_0 = sym.lambdify(x, sym.diff(-self.U_0(x))) # Precompute the derivative and store it as a lambda function. Precomputing is very important for speed.
        
    def U(self, x):
        """
        Vectorised potential function, with maximum slope constraints.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).

        Returns
        -------
        numeric or vector of numerics
            Potential evaluated at positions. Essential for functioning of later code.

        """
        # vectorised potential function
        return self.U_0(x)
        # Using branchless programming (i.e. doing the above instead of using if statements) makes the code less readable but nearly 10x faster.
    def F(self, x):
        """
        Compute the force, -U'(x).

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).

        Returns
        -------
        Numeric or vector of numerics
            Force at positions. Essential for functioning of later code.

        """
        return self.F_0(x)
    def plot_potential(self, plot_range=None):
        """
        Plot the potential.

        Parameters
        ----------
        plot_range : sorted 1d vector of numerics
            Mesh of initial positions.

        Returns
        -------
        None. 

        """
        if plot_range is None:
            plot_range = np.linspace(self.x_min, self.x_max, 50)
        plt.plot(plot_range, self.U(plot_range), 'g')
        return None
    def plot_force(self, plot_range=None):
        """
        Plot the force.

        Parameters
        ----------
        plot_range : sorted 1d vector of numerics
            Mesh of initial positions.

        Returns
        -------
        None.

        """
        if plot_range is None:
            plot_range = np.linspace(self.x_min, self.x_max, 50)
        plt.plot(plot_range, self.F(plot_range), 'c')
        return None
    def _boltzmann_unnormalised(self, x, k_BT):
        # unnormalised, unvectorised boltzmann distro at time zero
        return np.exp(-self.U(x)/k_BT)
        # If k_BT is array-like, generate an array of shape len(x),len(k_BT)
    def boltzmann(self, x, k_BT, integration_bounds=None):
        """
        Vectorised, normalised boltzmann distro. Implicitly sets p(x outside bounds) = 0 by integrating over x_min, x_max instead of -infty,infty.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s). If this is a vector, all entries must be evenly spaced.
        k_BT : numeric
            Temperature times the Boltzmann constant k_B. Usually we choose units so that k_B*T_0 = 1 for some T_0.
        integration_bounds : iterable of length 2, optional
            Region to integrate over to find the normalisation constant Z. The default is [x_min, x_max].

        Returns
        -------
        Numeric or vector of numerics
            e^{-U(x)/k_BT}/Z, where Z = int_{x_min}^{x_max} e^{-U(x)/k_BT}dx.

        """
        if integration_bounds is None:
            integration_bounds = [self.x_min, self.x_max]
        Z = scipy.integrate.quad_vec(lambda x: self._boltzmann_unnormalised(x, k_BT), *integration_bounds)[0]
        func = np.vectorize(lambda x, k_BT: self._boltzmann_unnormalised(x, k_BT)/Z) # Normalise the PDF
        return func(x, k_BT)
    def boltzmann_PMF(self, x, k_BT, **kwargs):
        """
        Normalise the PDF again over the sum of the values of x because of an annoying issue in np.random.choice where the probability vector has to sum to exactly 1 with no tolerance for floating point error.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).
        k_BT : numeric
            Temperature.
        **kwargs : key-word arguments
            Key-word args for boltzmann().

        Returns
        -------
        Numeric or vector of numerics
            PDF evaluated at x.

        """
        f = self.boltzmann(x, k_BT, **kwargs)
        return f/f.sum()
    def boltzmann_CDF(self, x, k_BT, **kwargs):
        """
        Calculate the CDF of the boltzmann distribution at temperature k_BT. Since this is done numerically, we just cumsum the boltzmann_array.

        Parameters
        ----------
        x : vector of numerics
            Positions to evaluate the CDF at.
        k_BT : numeric
            Temperature.
        **kwargs : key-word arguments
            Key-word args for boltzmann().

        Returns
        -------
        Vector of numerics
            CDF evaluated at x.

        """
        PDF = self.boltzmann_PMF(x, k_BT, **kwargs)/self.dx
        CDF = np.cumsum(PDF)*self.dx
        return CDF


class BoundedForcePotential(Potential):
    """Takes in a functional form of a potential and defines a bunch of handy functions."""
    
    def __init__(self, *args, **kwargs):
        # cls.__init__(self, *args, **kwargs) # Initialise the parent class so we can use the variables defined there. args and kwargs are stuff (eg. relevant parameters) to be passed up to the parent class.
        super().__init__(self, *args, **kwargs)
        x = sym.symbols('x') # Define symbols for numeric computation
        self.F_0 = sym.lambdify(x, sym.diff(-self.U_0(x))) # Precompute the derivative and store it as a lambda function. Precomputing is very important for speed.
        def _newton_raphson_solver(f, x_0, max_iterations = 100, dx = 1e-6, return_error=False, tolerance = 1e-5, debug_mode=False):
            # Returns a zero of f given initial condition (x_0,f(x_0)). Modified from standard techniques to fail more gracefully in certain edge cases.
            x_n = x_0
            for i in range(max_iterations):
                f_prime_n = (f(x_n+dx)-f(x_n))/dx
                if f_prime_n == 0:
                    raise ZeroDivisionError
                x_n = x_n - f(x_n)/f_prime_n
                if debug_mode: print(f(x_n))
            return_error = np.abs(f(x_n)) > tolerance 
            if return_error:
                raise ValueError("Convergence failed!", x_n, f(x_n))
            return x_n
        self.x_l = _newton_raphson_solver(lambda x: self.F_0(x) - self.F_left, self.x_min)
        self.x_r = _newton_raphson_solver(lambda x: self.F_0(x) + self.F_right, self.x_max)
        # self.x_l and self.x_r definitions assume that the force applied at any x at any point in time is strictly not greater than F(x,0). Relaxing this assumption will greatly slow down the code as it will require x_l and x_r to be recomputed at each timestep.
     
    def U(self, x):
        """
        Vectorised potential function, with maximum slope constraints.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).

        Returns
        -------
        numeric or vector of numerics
            Potential evaluated at positions. Essential for functioning of later code.

        """
        # vectorised potential function
        # self.get_slope_boundaries() # Generate x_l and x_r
        in_well = (x >= self.x_l) & (x <= self.x_r)
        left_of_well = (x < self.x_l)
        right_of_well = (x > self.x_r)
        return in_well*self.U_0(x)+left_of_well*(self.U_0(self.x_l)-self.F_left*(x-self.x_l))+right_of_well*(self.U_0(self.x_r)+self.F_right*(x-self.x_r))
        # Using branchless programming (i.e. doing the above instead of using if statements) makes the code less readable but nearly 10x faster.
    def F(self, x):
        """
        Compute the force, -U'(x).

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).

        Returns
        -------
        Numeric or vector of numerics
            Force at positions. Essential for functioning of later code.

        """
        in_well = (x >= self.x_l) & (x <= self.x_r)
        left_of_well = (x < self.x_l)
        right_of_well = (x > self.x_r)
        return in_well*(self.F_0(x))+left_of_well*self.F_left-right_of_well*self.F_right