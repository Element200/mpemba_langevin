# -*- coding: utf-8 -*-
"""
special_potentials.py.

Created on Tue Sep 2 11:37:51 2025
v1.0

@author: sane
Special potentials that we're interested in -- Asymmetric Double-well potentials, bounded potentials, all that jazz.
"""
import numpy as np

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
                x = sym.symbols("x")
                outstring += f"{var}({x}) : {variables[var](x)}\n"
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
        self.F_left = F_left
        self.F_right = force_asymmetry*F_left
        self.b = b
        self.n = n
        self.x_s = x_s
        self.a = self.F_left/((2*n+1)*x_s**(2*n) + b)
        self.unbumpy = AsymmetricDoubleWellPotential(E_barrier=E_barrier, E_tilt=E_tilt, x_well=x_well, x_min=x_min, x_max=x_max, F_left=F_left, force_asymmetry=1)
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
                    outstring += f"F_0 : {-sym.diff(self.U_0(x), x)}"
                else:
                    outstring += f"{var}({x}) : {variables[var](x)}\n"
        return outstring
    
    def __repr__(self):
        """
        Do the same schtick as __str__ but now we don't have to make a print call.

        Returns
        -------
        outstring : str
            String representation of BumpyAsymmetricDoubleWellPotential.

        """
        return self.__str__()
        