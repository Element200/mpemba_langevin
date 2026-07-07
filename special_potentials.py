# -*- coding: utf-8 -*-
"""
special_potentials.py.

Created on Tue Sep 2 11:37:51 2025
v1.0

@author: sane
Special potentials that we're interested in -- Asymmetric Double-well potentials, bounded potentials, all that jazz.
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

import sys
import os
directory = os.path.dirname(__file__)
sys.path.append(directory)
# This assumes that all of the mpemba files are in the same directory
try:
    import potential_methods
except ImportError:
    print("Something went wrong when importing dependencies! Make sure local imports are in the same directory as this file")
    raise ImportError
import sympy as sym

class AsymmetricDoubleWellPotential(potential_methods.BoundedForcePotential):
    """Basic asymmetric double-well potential that we use for Mpemba simulations. Any potential can be defined here: simply define all of the relevant parameters in __init__ and the form of the potential in U_0. Inheriting from BoundedForcePotential will add all other necessary methods."""
    
    def __init__(self, E_barrier=2, E_tilt=1.3, x_well=0.5, x_min=-1, x_max=2, F_left=50, force_asymmetry=1):
        """Define parameters used in the potential."""
        self.E_barrier = E_barrier
        self.E_tilt = E_tilt
        self.x_well = x_well
        self.x_min = x_min
        self.x_max = x_max
        self.mesh = lambda n: np.linspace(self.x_min, self.x_max, n)
        self.F_left = F_left
        self.F_right = force_asymmetry*F_left
        super().__init__() # Initialise the parent class *after* useful variables are defined so that it knows what variables to use
        return None
    
    def U_0(self, x, t=None):
        """
        Generate basic potential (without bounded derivatives) evaluated at x.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).
        t : numeric
            Time to evaluate potential at

        Returns
        -------
        Numeric or vector of numerics
            Potential evaluated at x at time t. For now this is time-independent.

        """
        return self.E_barrier*(1-2*(x/self.x_well)**2 + (x/self.x_well)**4) - self.E_tilt*(x/self.x_well)/2
    
    def __str__(self):
        """
        Print all the variables we use in the potential.

        Returns
        -------
        outstring : str
            String representation of AsymmetricDoubleWellPotential.

        """
        variables = self.__dict__
        outstring = "ASYMMETRIC DOUBLE-WELLED QUARTIC POTENTIAL WITH FINITE MAXIMUM SLOPES\n"
        for var in variables:
            if not callable(variables[var]):
                outstring += f"{var} : {variables[var]}\n"
            else:
                outstring += "\n"
                # if var == 'mesh':
                #     outstring += "\n"
                # else:
                #     x = sym.symbols("x")
                #     outstring += f"{var}({x}) : {variables[var](x)}\n"
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
    
class AsymmetricDoubleWellPotential_noWalls(potential_methods.UnboundedForcePotential):
    """Basic asymmetric double-well potential that we use for Mpemba simulations. Any potential can be defined here: simply define all of the relevant parameters in __init__ and the form of the potential in U_0. Inheriting from BoundedForcePotential will add all other necessary methods."""
    
    def __init__(self, E_barrier=2, E_tilt=1.3, x_well=0.5, x_min=-1, x_max=2, F_left=50, force_asymmetry=1):
        """Define parameters used in the potential."""
        self.E_barrier = E_barrier
        self.E_tilt = E_tilt
        self.x_well = x_well
        self.x_min = x_min
        self.x_max = x_max
        self.mesh = lambda n: np.linspace(self.x_min, self.x_max, n)
        self.F_left = F_left
        self.F_right = force_asymmetry*F_left
        super().__init__() # Initialise the parent class *after* useful variables are defined so that it knows what variables to use
        return None
    
    def U_0(self, x, t=None):
        """
        Generate basic potential (without bounded derivatives) evaluated at x.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).
        t : numeric
            Time to evaluate potential at

        Returns
        -------
        Numeric or vector of numerics
            Potential evaluated at x at time t. For now this is time-independent.

        """
        return self.E_barrier*(1-2*(x/self.x_well)**2 + (x/self.x_well)**4) - self.E_tilt*(x/self.x_well)/2
    
    def __str__(self):
        """
        Print all the variables we use in the potential.

        Returns
        -------
        outstring : str
            String representation of AsymmetricDoubleWellPotential.

        """
        variables = self.__dict__
        outstring = "ASYMMETRIC DOUBLE-WELLED QUARTIC POTENTIAL WITH FINITE MAXIMUM SLOPES\n"
        for var in variables:
            if not callable(variables[var]):
                outstring += f"{var} : {variables[var]}\n"
            else:
                outstring += "\n"
                # if var == 'mesh':
                #     outstring += "\n"
                # else:
                #     x = sym.symbols("x")
                #     outstring += f"{var}({x}) : {variables[var](x)}\n"
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

class BumpyAsymmetricDoubleWellPotential(potential_methods.Potential):
    """Asymmetric double-well potential with odd-polynomial 'bump'. Kind of, sort of inherits from AsymmetricDoubleWellPotential, but direct inheritance made my life hard so I'm just doing it this way."""
    
    def __init__(self, E_barrier=2, E_tilt=1.3, x_well=0.5, x_min=-1, x_max=2, F_left=50, force_asymmetry=1, b=1, n=1, x_s=0.5):
        self.E_barrier = E_barrier
        self.E_tilt = E_tilt
        self.x_well = x_well
        self.x_min = x_min
        self.x_max = x_max
        self.mesh = lambda n: np.linspace(self.x_min, self.x_max, n)
        self.F_left = F_left
        self.F_right = force_asymmetry*F_left
        self.b = b
        self.n = n
        self.x_s = x_s
        self.a = self.F_left/((2*n+1)*x_s**(2*n) + b)
        self.unbumpy = AsymmetricDoubleWellPotential(E_barrier=E_barrier, E_tilt=E_tilt, x_well=x_well, x_min=x_min, x_max=x_max, F_left=F_left, force_asymmetry=1) # This is where the "inheritance" comes from
        self.x_l = self.unbumpy.x_l
        self.x_r = self.unbumpy.x_r
        self.x_d = ((self.F_right/self.a - b)/(2*n+1))**(0.5/n) # Godawful analytic solution for where the 'bumpy' potential has slope F_right
        super().__init__()
    def B(self, x):
        """Bump for the potential."""
        return self.a*(x**(2*self.n+1)+self.b*x)
    def U_0(self, x):
        """Utterly godforsaken potential definition to use to generate a potential with a bumpy shelf."""
        return sym.Piecewise((self.unbumpy.U_0(self.x_l)-self.F_left*(x-self.x_l), (x<=self.x_l)), (self.unbumpy.U_0(x), (x<=self.x_r)&(x>=self.x_l)), 
                              (self.unbumpy.U_0(self.x_r)-self.B(-self.x_s)+self.B(x-self.x_r-self.x_s), (x>=self.x_r) & (x<=self.x_r+self.x_s+self.x_d)),
                              (self.unbumpy.U_0(self.x_r)-self.B(-self.x_s)+self.B(self.x_d)+self.F_right*(x-self.x_r-self.x_s-self.x_d), (x>=self.x_r+self.x_s+self.x_d)))
    
    def __str__(self):
        """
        Print all the variables we use in the potential.

        Returns
        -------
        outstring : str
            String representation of BumpyAsymmetricDoubleWellPotential.

        """
        variables = self.__dict__
        outstring = "ASYMMETRIC DOUBLE-WELLED QUARTIC POTENTIAL WITH FINITE MAXIMUM SLOPES AND CUBIC 'BUMP'\n"
        for var in variables:
            if not callable(variables[var]):
                outstring += f"{var} : {variables[var]}\n"
            else:
                x = sym.symbols("x")
                if var == 'F_0':
                    outstring += f"F_0 : {-sym.diff(self.U_0(x), x)}\n"
                elif var == 'mesh':
                    outstring += "\n"
                else:
                    outstring += f"{var}({x}) : {variables[var](x)}\n"
        return outstring
    
    def __repr__(self):
        """Do the same schtick as __str__ but now we don't have to make a print call."""
        return self.__str__()
    
        

class AsymmetricTripleWellPotential(potential_methods.BoundedForcePotential):
    """Basic asymmetric double-well potential that we use for Mpemba simulations. Any potential can be defined here: simply define all of the relevant parameters in __init__ and the form of the potential in U_0. Inheriting from BoundedForcePotential will add all other necessary methods."""
    
    def __init__(self, E_barrier=10, E_rise=0.8, E_tilt=1, x_well=0.5, x_min=-5, x_max=5, F_left=50, force_asymmetry=1):
        """Define parameters used in the potential."""
        self.E_barrier = E_barrier
        self.E_tilt = E_tilt
        self.E_rise = E_rise
        self.x_well = x_well
        self.x_min = x_min
        self.x_max = x_max
        self.mesh = lambda n: np.linspace(self.x_min, self.x_max, n)
        self.F_left = F_left
        self.F_right = force_asymmetry*F_left
        super().__init__() # Initialise the parent class *after* useful variables are defined so that it knows what variables to use
        return None
    
    def U_0(self, x, t=None):
        """
        Generate basic potential (without bounded derivatives) evaluated at x.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).
        t : numeric
            Time to evaluate potential at

        Returns
        -------
        Numeric or vector of numerics
            Potential evaluated at x at time t. For now this is time-independent.

        """
        return self.E_barrier*(1-2*(x/self.x_well)**2 + (x/self.x_well)**4)*(x/self.x_well)**2 - self.E_rise*(x/self.x_well)**2/2 +self.E_tilt*(x/self.x_well)/2
    
    def __str__(self):
        """
        Print all the variables we use in the potential.

        Returns
        -------
        outstring : str
            String representation of AsymmetricDoubleWellPotential.

        """
        variables = self.__dict__
        outstring = "TRIPLE-WELLED QUARTIC POTENTIAL WITH FINITE MAXIMUM SLOPES\n"
        for var in variables:
            if not callable(variables[var]):
                outstring += f"{var} : {variables[var]}\n"
            else:
                outstring += "\n"
                # if var == 'mesh':
                #     outstring += "\n"
                # else:
                #     x = sym.symbols("x")
                #     outstring += f"{var}({x}) : {variables[var](x)}\n"
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
    
class AnalyticAsymmetricDoubleWellPotential(potential_methods.UnboundedForcePotential):
    def __init__(self, E_barrier=2, E_tilt=1.3, F_left=50, x_well=0.5, x_min=-10, x_max=10, force_asymmetry = 1, n=1, n2=1, optimise=True, x_0=1, x_1=0, x_l=-1, x_r=1):
        self.E_barrier = E_barrier
        self.x_well = x_well
        self.E_tilt = E_tilt
        self.F_left = F_left
        self.F_right = force_asymmetry*F_left
        self.force_asymmetry = force_asymmetry
        self.x_min=x_min
        self.x_max=x_max
        self.n = n
        comparisonPotential = AsymmetricDoubleWellPotential(E_barrier=self.E_barrier, E_tilt=self.E_tilt, F_left=self.F_left, force_asymmetry=self.force_asymmetry, x_min=self.x_min, x_max=self.x_max)
        self.comparisonPotential = comparisonPotential
        
        y = sym.symbols('y')
        k = sym.symbols('k')
        X_1 = sym.symbols('x_1')
        X_l = sym.symbols('x_l')
        X_r = sym.symbols('x_r')
        unboundedPotential = lambda y: self.E_barrier*(1-(y/self.x_well)**2)**2 - 0.5*self.E_tilt*y/self.x_well
        # 
        rect_n = lambda y, k, x_1: sym.exp(-((y-x_1)/k)**(2*n2))
        rect_n_lambda = sym.lambdify([y, k, X_1], rect_n(y, k, X_1))
        unboundedForce = sym.lambdify(y, sym.diff(-unboundedPotential(y), y))
        rect_n_diff = sym.lambdify([y, k, X_1], sym.diff(-rect_n(y, k, X_1), y))
        force_func = lambda x, k, x_1: unboundedForce(x)*rect_n_lambda(x, k, x_1)+unboundedPotential(x)*rect_n_diff(x, k, x_1)
        const_slope_left = lambda x, x_l: sym.log(1+sym.exp(-n*self.F_left*(x-x_l)))/n
        const_slope_right = lambda x, x_r: sym.log(1+sym.exp(n*self.F_right*(x-x_r)))/n
        bounds_func = lambda x, x_l, x_r, k, x_1: (const_slope_left(x, x_l)+const_slope_right(x, x_r))*(1-rect_n(x, k, x_1))
        bounds_func_diff = sym.lambdify([y, X_l, X_r, k, X_1], sym.diff(-bounds_func(y, X_l, X_r, k, X_1), y))
        # const_slope_right = lambda x, x_r: np.log(1+np.exp(n*self.F_right*(x-x_r)))/n
        if optimise:
            x = np.linspace(-3, 3, 10000)
            optimising_func = lambda x, k, x_1, x_l, x_r: force_func(x, k, x_1)+bounds_func_diff(x, x_l, x_r, k, x_1)
            
            popt, pcov = scipy.optimize.curve_fit(optimising_func, xdata=x, ydata=comparisonPotential.F(x), p0 = np.array([1, 0, -1, 1]), )
            if (np.diag(pcov)/popt**2).sum() < 4:    
                self.k, self.x_1, self.x_l, self.x_r = popt
            else:
                plt.plot(x, force_func(x, popt[0], popt[1])+const_slope_left(x, -1)+const_slope_right(x, 1))
                plt.plot(x, comparisonPotential.F(x))
                raise ValueError(popt, pcov)
            print(popt)
            print(pcov)
        else:
            self.k, self.x_1, self.x_l, self.x_r = x_0, x_1, x_l, x_r
        
        self.rect_n = lambda x: rect_n(x, self.k, self.x_1)
        self.unboundedPotential = unboundedPotential
        self.bounds_func = lambda x: bounds_func(x, self.x_l, self.x_r, self.k, self.x_1)
        # self.const_slope_left = lambda x: const_slope_left_lambda(x, self.x_l)
        # self.const_slope_right = lambda x: const_slope_right_lambda(x, self.x_r)
        # self.const_slope_right = lambda x: sym.log(1+sym.exp(n*self.F_right*(x-self.x_r)))/n
        super().__init__()
        return None

    def U_0(self, x):        
        return self.bounds_func(x) + self.rect_n(x)*self.unboundedPotential(x)
        
class SimpleQuartic(potential_methods.BoundedForcePotential):
    """Basic asymmetric double-well potential that we use for Mpemba simulations. Any potential can be defined here: simply define
    all of the relevant parameters in __init__ and the form of the potential in U_0. Inheriting from BoundedForcePotential will add all other necessary methods."""
    
    def __init__(self, E_barrier=10, E_tilt=0, x_min=-5, x_max=5, F_left=50, force_asymmetry=1):
        """Define parameters used in the potential."""
        self.E_barrier = E_barrier
        self.x_min = x_min
        self.x_max = x_max
        self.mesh = lambda n: np.linspace(self.x_min, self.x_max, n)
        self.F_left = F_left
        self.F_right = force_asymmetry*F_left
        super().__init__() # Initialise the parent class *after* useful variables are defined so that it knows what variables to use
        return None
    
    def U_0(self, x, t=None):
        """
        Generate basic potential (without bounded derivatives) evaluated at x.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).
        t : numeric
            Time to evaluate potential at

        Returns
        -------
        Numeric or vector of numerics
            Potential evaluated at x at time t. For now this is time-independent.

        """
        return self.E_barrier*x**4
    
    def __str__(self):
        """
        Print all the variables we use in the potential.

        Returns
        -------
        outstring : str
            String representation of AsymmetricDoubleWellPotential.

        """
        variables = self.__dict__
        outstring = "SIMPLE QUARTIC POTENTIAL WITH FINITE MAXIMUM SLOPES\n"
        for var in variables:
            if not callable(variables[var]):
                outstring += f"{var} : {variables[var]}\n"
            else:
                outstring += "\n"
                # if var == 'mesh':
                #     outstring += "\n"
                # else:
                #     x = sym.symbols("x")
                #     outstring += f"{var}({x}) : {variables[var](x)}\n"
        return outstring
    
    def __repr__(self):
        """Do the same schtick as __str__ but now we don't have to make a print call."""
        return self.__str__()

class BoundedHarmonicPotential(potential_methods.BoundedForcePotential):
    """Simple harmonic potential for comparison with double-well potentials."""
    
    def __init__(self, k, F_left=50, force_asymmetry=1, x_min=-5, x_max=5):
        self.k = k
        self.F_left = F_left
        self.F_right = force_asymmetry*self.F_left
        self.force_asymmetry = force_asymmetry
        self.x_min=x_min
        self.x_max=x_max
        super().__init__()
    def U_0(self, x):
        """Unbounded potential definition."""
        return 0.5*self.k*x**2
    def __str__(self):
        """I'm sure you can look up what __str__ does in classes."""
        variables = self.__dict__
        outstring = "SIMPLE QUARTIC POTENTIAL WITH FINITE MAXIMUM SLOPES\n"
        for var in variables:
            if not callable(variables[var]):
                outstring += f"{var} : {variables[var]}\n"
            else:
                outstring += "\n"
                # if var == 'mesh':
                #     outstring += "\n"
                # else:
                #     x = sym.symbols("x")
                #     outstring += f"{var}({x}) : {variables[var](x)}\n"
        return outstring
    def __repr__(self):
        """Do the same schtick as __str__."""
        return self.__str__
    
class SimpleHarmonic(potential_methods.UnboundedForcePotential):
    """Simple harmonic potential for comparison with double-well potentials."""
    
    def __init__(self, k, x_min=-5, x_max=5):
        self.k = k
        self.x_min=x_min
        self.x_max=x_max
        super().__init__()
    def U_0(self, x):
        """Unbounded potential definition."""
        return 0.5*self.k*x**2
    def __str__(self):
        """I'm sure you can look up what __str__ does in classes."""
        variables = self.__dict__
        outstring = "SIMPLE QUARTIC POTENTIAL WITH FINITE MAXIMUM SLOPES\n"
        for var in variables:
            if not callable(variables[var]):
                outstring += f"{var} : {variables[var]}\n"
            else:
                outstring += "\n"
                # if var == 'mesh':
                #     outstring += "\n"
                # else:
                #     x = sym.symbols("x")
                #     outstring += f"{var}({x}) : {variables[var](x)}\n"
        return outstring
    def __repr__(self):
        """Do the same schtick as __str__."""
        return self.__str__
# def polynomial_interpolator_with_derivatives(X, Y, derivatives=None):
#     if derivatives is None:
#         derivatives = np.zeros_like(Y)
#     N = len(X)
#     assert len(Y) == N
#     Y_matching_matrix = np.array([X**k for k in range(2*N)]).T
#     derivative_matching_matrix = np.array([k*X**(k-1) if k > 0 else np.zeros_like(X) for k in range(2*N)]).T
#     M = np.stack([Y_matching_matrix, derivative_matching_matrix], axis=0)
#     output_vector = np.stack([Y, derivatives], axis=1)
#     return output_vector

def optimise_parameters(initial_params, param_constraints, hard_constraints, num_iters = 20):
    potential = BumpyAsymmetricDoubleWellPotential(**initial_params, **hard_constraints)
    k_BTs = np.logspace(0,3,50)
    a_2s = potential.a_k_boltzmannIC(k_BTs, k=2)
    k_BT_h = k_BTs[a_2s==0][0]
    for i in range(num_iters):
        a_2s = potential.a_k_boltzmannIC(k_BTs, k=2)
        k_BT = k_BTs[a_2s==0]
        if k_BT:
            k_BT_h = k_BT
    
    
    