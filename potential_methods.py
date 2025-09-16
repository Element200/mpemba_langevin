# -*- coding: utf-8 -*-
"""
potential_methods.py.

Created on Wed Jun 11 10:22:16 2025
v1.4

@author: sane
Creates classes that can be used to add a bunch of handy attributes to various kinds of potentials.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sympy as sym


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
        y = sym.symbols("y")
        return sym.lambdify(y, self.U_0(y))(x) # Can cause issues if U is called repeatedly
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
        func = lambda x, k_BT: self._boltzmann_unnormalised(x, k_BT)/Z # Normalise the PDF
        if type(k_BT) == np.ndarray:
            k_BT = k_BT[np.newaxis, :] # So that we can get a 2D output
            x = x[:,np.newaxis]
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
        return f/f.sum(axis=0)
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
        CDF = np.cumsum(PDF, axis=0)*self.dx
        return CDF
    
    def right_half_probability_ratio(self, k_BTs, use_infinite_domain = True):
        """
        Compute the probability mass on the right hand side of the barrier.

        Parameters
        ----------
        k_BTs : vector of numerics
            Vector containing the temperatures to compute at.
        use_infinite_domain : bool, optional
            Integrate over the real line. If this is set to False, the bounds of the potential will be used instead. The default is True.

        Raises
        ------
        ValueError
            If the barrier calculation fails.

        Returns
        -------
        vector of numerics
            Probability ratios for each temperature.

        """
        
        x_barrier = scipy.optimize.fsolve(self.F_0, x0=0) # Find the barrier position (should be close to zero)
        if np.abs(x_barrier) > self.x_well:
            raise ValueError("Failed barrier calculation")
        if use_infinite_domain:
            Z = scipy.integrate.quad_vec(lambda x: self._boltzmann_unnormalised(x, k_BTs), -np.inf, np.inf)[0]
            return scipy.integrate.quad_vec(lambda x: self._boltzmann_unnormalised(x, k_BTs)/Z, x_barrier, np.inf)[0]
        else:
            Z = scipy.integrate.quad_vec(lambda x: self._boltzmann_unnormalised(x, k_BTs), self.x_min, self.x_max)[0]
            return scipy.integrate.quad_vec(lambda x: self._boltzmann_unnormalised(x, k_BTs)/Z, x_barrier, self.x_max)[0]
        
    
    def enclosed_probability_mass(self, k_BT):
        """
        Find out how much of the probability is enclosed within [x_min, x_max] (ideally we want most of the probability to live inside this domain).

        Parameters
        ----------
        k_BT : numeric
            Temperature to calculate the ratio at.
        
        Returns
        -------
        numeric: Ratio of probability enclosed with [x_min, x_max] to probability enclosed within the whole real line, at temperature k_BT.

        """
        Z = scipy.integrate.quad(lambda x: self._boltzmann_unnormalised(x, k_BT), -np.inf, np.inf)[0]
        return scipy.integrate.quad_vec(lambda x: self._boltzmann_unnormalised(x, k_BT)/Z, self.x_min, self.x_max)[0]
    
    def grima_newman_discretisation(self, x):
        """
        Construct the transition matrix using the discretisation of the FP operator from Grima and Newman (2004, PhysRev-E).

        Parameters
        ----------
        x : vector of numerics
            EVENLY SPACED positions which p is evaluated on. S will have dimensions of len(x)*len(x).

        Returns
        -------
        len(x)*len(x) matrix.

        """
        U_vals = self.U(x)
        dU = (U_vals-np.roll(U_vals, 1))[1:]
        diag_plus = np.exp(dU/2)
        diag_minus = np.exp(-dU/2)
        diag_0 = -np.array([0, *diag_plus])-np.array([*diag_minus, 0])
        S = scipy.sparse.diags([diag_minus, diag_0, diag_plus], [-1,0,1]) # Assumes force is not varying with time
        return S
    def W_matrix(self, x, D, h=lambda t: 1, t=None):
        """
        Construct a transition matrix from a NON-TIME VARYING potential.

        Parameters
        ----------
        x : vector of numerics
            EVENLY SPACED positions which p is evaluated on. W will have dimensions of len(x)*len(x)
        D : numeric
            Bath diffusivity
        h : function, optional
            Scaling function for time-varying diffusivity. The default is lambda t: 1.
        t : numeric, optional
            Time to evaluate at, if h is not None.
            
        Returns
        -------
        W
            Sparse matrix of dimensions len(x)*len(x) that acts as a Fokker-Planck operator.

        """
        S = self.grima_newman_discretisation(x)
        dx = x[1]-x[0] # Assumes uniform spacing
        return h(t)*D*S/dx**2
    
    def a_k(self, p_0, k=2, x=None, imaginary_tolerance = 1e-4):
        """
        Get the coefficient of the eigenfunction v_k, assuming an initial state characterised by p_0. We compute eigenfunctions by taking the eigenvectors of the W-matrix we define using the Grima-Newman method.

        Parameters
        ----------
        p_0 : vector of numerics
            Initial probability state.
        k : int, optional
            The index of the eigenfunction to compute. The default is 2.
        x : vector of numerics, optional
            Mesh to compute the eigenvector over. The default is len(p_0) evenly spaced numbers between x_max and x_min. This function will fail if len(x) != len(p_0)
        imaginary_tolerance : numeric, optional
            Maximum allowable size of the imaginary component of the eigenvalue. Theoretically it should be zero, so if it's too high that's an indication that something is wrong.

        Returns
        -------
        numeric
            Coefficient of the eigenvector v_k.

        """
        if x is None:
            x = np.linspace(self.x_min, self.x_max, len(p_0))
        assert len(x) == len(p_0), "Size mismatch between x and p_0."
        S = self.grima_newman_discretisation(x).toarray() # If S is not too huge, ndarrays should be fine
        # We don't need the W-matrix at all since W = scale_factor*S, which will only affect eigenvalues, which we don't care about anyway
        eigenvals, left_eigenvecs, right_eigenvecs = scipy.linalg.eig(S, right=True, left=True)
        abs_eigenvals_index = np.abs(eigenvals).argsort()
        
        eigenvals_sorted = eigenvals[abs_eigenvals_index]
        right_eigenvecs_sorted = right_eigenvecs[:,abs_eigenvals_index]
        left_eigenvecs_sorted = left_eigenvecs[:,abs_eigenvals_index]
        
        right_eigenvec_k = right_eigenvecs_sorted[:,k-1].real # k-1 because of 0-indexing. Take the real part because all eigenvectors should be purely real anyway
        left_eigenvec_k = left_eigenvecs_sorted[:,k-1].real
        
        eigenval_k = eigenvals_sorted[k-1]
        if np.abs(eigenval_k.imag) > imaginary_tolerance:
            raise ValueError("Eigenvectors were not correctly computed")
        
        return (left_eigenvec_k @ p_0)/(left_eigenvec_k @ right_eigenvec_k) # Numerical integration, computes [<u_k|p_0>/<u_k|v_k>    

    def a_k_boltzmannIC(self, k_BT, x=None, n_x=100, **kwargs_for_a_k):
        """
        Compute a_k given p_0 is a Boltzmann distro with temperature k_BT.

        Parameters
        ----------
        k_BT : numeric
            Temperature of the Boltzmann initial condition.
        x : vector of numerics, optional
            Grid to use to generate probability vectors. The default is None.
        n_x : int, optional
            If x is unspecified, the number of elements in the grid. The default is 100.

        Returns
        -------
        None.

        """
        if x is None:
            x = np.linspace(self.x_min, self.x_max, n_x)
        pi_0 = self.boltzmann_PMF(x, k_BT)
        return self.a_k(pi_0, x=x, **kwargs_for_a_k)
    
    def infer_fast_mpemba_effect(self, k_BT_max, n_T = 100, method='a_2', tolerance=1e-3, use_infinite_domain=False):
        """
        Check if a fast Mpemba effect exists and if so, find the temperature at which it occurs.

        Parameters
        ----------
        k_BT_max : numeric
            Highest temperature to check.
        n_T : int
            Number of temperatures.
        method : str ('a_2' or 'P')
            Which method to use to determine whether a fast Mpemba effect occurs.

        Returns
        -------
        Temperature at which Mpemba effect occurs, if any.

        """
        k_BTs = np.logspace(0, np.log(k_BT_max)/np.log(10), n_T)        
        
        if method == 'a_2':
            a_2s = self.a_k_boltzmannIC(k_BTs, n_x=500)
            arr = a_2s
        elif method == 'P':
            right_probability_ratios = self.right_half_probability_ratio(k_BTs, use_infinite_domain=use_infinite_domain)
            arr = right_probability_ratios
        else:
            raise ValueError("Only using a_2 and probability ratios can a strong Mpemba effect be determined.")
        closeToTarget = np.isclose(arr[1:], arr[0]*np.ones_like(arr[1:]), atol=tolerance)
        fastMpembaEffectOccurs = closeToTarget.any() # arr[0]*np.ones_like(arr) is a string of zeros if arr is the a_2 values and a string of the initial probability values if arr is the P-ratios. 
        # Exclude arr[0] because that will trivially always be True.
        if fastMpembaEffectOccurs:
            print("Strong Mpemba effect detected!")
            return k_BTs[1:][closeToTarget]
        else:
            print("No strong Mpemba effect detected.")
            return None
            
        

class UnboundedForcePotential(Potential):
    """Trivial class to avoid massive overheads from symbolic computation by inheriting directly from Potential."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        x = sym.symbols('x') # Define symbols for numeric computation
        self.F_0 = sym.lambdify(x, sym.diff(-self.U_0(x))) # Precompute the derivative and store it as a lambda function. Precomputing is very important for speed.
    def U(self, x):
        """Return U_0(x)."""
        return self.U_0(x)
    def F(self, x):
        """Return F_0(x)."""
        return self.F_0(x)

class BoundedForcePotential(Potential):
    """Takes in a functional form of a potential and defines a bunch of handy functions."""
    
    def __init__(self, *args, **kwargs):
        # cls.__init__(self, *args, **kwargs) # Initialise the parent class so we can use the variables defined there. args and kwargs are stuff (eg. relevant parameters) to be passed up to the parent class.
        super().__init__()
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
    
# def F_flat(x, t, x_bounds=(-2,2), F_max=50):
#     return np.piecewise(x, [x<x_bounds[0],(x>=x_bounds[0])&(x<=x_bounds[1]),x>x_bounds[1]], [F_max,0,-F_max])

class FlatBoundedPotential(Potential):
    """Flat potential for testing laser."""   

    def __init__(self, x_r, x_l=None, F_max=50):
        self.x_r = x_r
        self.F_max = F_max
        if x_l is None: self.x_l = -self.x_r
        super().__init__()
    def U_0(self, x):
        """
        Flat potential with boundaries.

        Parameters
        ----------
        x : symbolic
            symbol for Potential to differentiate using sympy.

        Returns
        -------
        U_0(x) : symbolic
            Potential evaluated at x.

        """
        return sym.Piecewise((self.F_max*(x-self.x_r),x>=self.x_r),(-self.F_max*(x-self.x_l), x<=self.x_l), (0,True))
    def __str__(self):
        """
        Print all the variables we use in the potential.

        Returns
        -------
        outstring : str
            String representation of AsymmetricDoubleWellPotential.

        """
        variables = self.__dict__
        outstring = "FLAT POTENTIAL WITH FINITE MAXIMUM SLOPES\n"
        for var in variables:
            if not callable(variables[var]):
                outstring += f"{var} : {variables[var]}\n"
            else:
                try:
                    x = sym.symbols("x")
                    outstring += f"{var}({x}) : {variables[var](x)}\n"
                except TypeError:
                    outstring += f"{var} : {variables[var]}\n"
        return outstring
    
    def __repr__(self):
        """
        Do the same schtick as __str__ but now we don't have to make a print call.

        Returns
        -------
        outstring : str
            String representation of AsymmetricDoubleWellPotential.

        """
        return self.__str__()
        
    