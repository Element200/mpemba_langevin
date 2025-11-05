# -*- coding: utf-8 -*-
"""
fokker_planck.py.

Created on Fri Aug 15 11:13:00 2025
v1.2

@author: sane
Analytic methods to solve the FPE (surprisingly I couldn't find any good libraries to do this for arbitrary F(x, t)).
"""

import numpy as np
from tqdm import tqdm
import scipy
import mpemba
import numba

@numba.njit
def RK4_step(W, p, dt):
    """Runge-Kutta 4th order integration step for the master equation dp/dt=W@p."""
    k_1 = (W @ p) # RK4 integration
    k_2 = W @ (p+k_1*dt/2)
    k_3 = W @ (p+k_2*dt/2)
    k_4 = W @ (p+k_3*dt)
    return (p + (k_1 + 2*k_2 + 2*k_3 + k_4)*dt/6)

@numba.njit
def euler_step(W, p, dt):
    """Euler integration step for the master equation dp/dt=W@p. This is faster than RK4 but slightly less accurate."""
    return p + (W @ p)*dt

D = 150

potential = mpemba.special_potentials.BumpyAsymmetricDoubleWellPotential(x_max=10,x_min=-10, E_barrier=2, E_tilt=0.5, force_asymmetry = 0.4, b=-0.0, n=3)
bumpy = mpemba.special_potentials.BumpyAsymmetricDoubleWellPotential(x_max=5,x_min=-5, E_barrier=2, E_tilt=0.5, force_asymmetry = 0.4, b=-0.0, n=1)


@numba.njit
def fokker_planck_core(p_0, S, dx, dt, steps, saving_timestep, D, error_tolerance=1e-3, h=numba.njit(lambda t:np.ones_like(t)), integrator=euler_step):
    """Numba-optimised Fokker-Planck integration."""
    p = p_0.copy()
    saved = np.zeros((*p_0.shape, steps//saving_timestep), dtype=p_0.dtype)
    W = D*S/dx**2
    t_vals = np.linspace(0, steps*dt, steps)
    h_vals = h(t_vals) # Precompute outside the hot loop for speed
    for i in range(steps):
        p = integrator(W*h_vals[i], p, dt) # euler is faster than RK4 and the accuracy difference is not that big
        if (i % saving_timestep == 0):
            saved[..., i//saving_timestep] = p.copy()
            Z = p.sum() * dx
            if not np.isclose(Z, 1.0, atol=error_tolerance): # We check that Z ~= 1 every so often so that we can check that our integration hasn't screwed up and kill the loop if it has.
                raise ValueError(f"Integration failure! Z={Z}")
    return saved



def fokker_planck_solver(p_0, potential, D, t_max, dt=1e-6, h=lambda t: 1, saving_timestep=1000, error_tolerance=1e-3):
    """
    Solve the FPE for time-independent potential U(x) = potential.U(x) and diffusivity D. You can also specify a diffusivity protocol.

    Parameters
    ----------
    p_0 : vector of numerics
        Initial probability vector.
    potential : potential_methods.Potential object
        Potential to use.
    D : numeric
        Diffusivity.
    t_max : numeric
        Time interval to integrate over, [0,t_max].
    dt : numeric, optional
        Timestep. If this is too large, the integrator will fail. The default is 1e-6.
    h : function, optional
        Diffusion protocol. The default is lambda t: 1.
    saving_timestep : int, optional
        Since the integrator will produce p(t) for each t_max//dt timestep, it will use an unwieldy amount of memory. saving_timestep controls the interval of saving an intermediate probability vector to manage the amount of memory used. The default is 1000.
    error_tolerance : numeric
        Every timestep, the integrator checks to see if p is normalised. If it is not, the integrator will fail. Error_tolerance is the amount of wiggle room p is allowed to have.

    Raises
    ------
    ValueError
        If the integration fails (tested by the probability vector not being correctly normalised), a ValueError is raised.

    Returns
    -------
    2D vector of numerics (*p_0.shape, t_max//saving_timestep)
        p(t) if t//dt is a multiple of saving_timestep.

    """
    width = potential.x_max-potential.x_min
    dx = width/len(p_0)
    x = np.linspace(potential.x_min, potential.x_max, len(p_0))
    S = potential.grima_newman_discretisation(x=x)
    steps = int(t_max//dt)
    out = fokker_planck_core(p_0, S, dx, dt, steps, saving_timestep, D, error_tolerance=error_tolerance)
    return np.array(out)

def analytic_solution(k_BTs, potential, D, n_x = 500, t_max = 0.1, dt=1e-6, resolution = 1e-5, error_tolerance=1e-3):
    """
    Analytically solves the FPE given p(x,0) is a boltzmann distro with temperatures in k_BTs.

    Parameters
    ----------
    k_BTs : Vector of numerics
        Vector containing the initial temperatures.
    potential : child of Potential class
        Potential to use.
    D : Numeric
        Diffusivity.
    n_x : int, optional
        Mesh size. The default is 500.
    t_max : Numeric, optional
        Length of time to integrate over. The default is 0.1.
    dt : numeric, optional
        Timestep for integration. The default is 1e-6.
    resolution : numeric, optional
        Timestep to save probability vectors at. The default is 1e-5.
    error_tolerance : numeric
        Every timestep, the integrator checks to see if p is normalised. If it is not, the integrator will fail. Error_tolerance is the amount of wiggle room p is allowed to have.

    Returns
    -------
    Vector of probabilities of shape (len(k_BTs), n_x, t_max//dt)
        Probabilities at each timestep, for each temperature.

    """
    p_outs = []
    x = np.linspace(potential.x_min, potential.x_max, n_x)
    for k_BT in tqdm(k_BTs):
        p_0 = potential.boltzmann(x, k_BT)
        p_outs.append(fokker_planck_solver(p_0, potential, D, t_max=t_max, dt=dt, saving_timestep=int(resolution//dt), error_tolerance=error_tolerance))
    return np.array(p_outs)

def quasistatic_limit(potential, h, times, k_BT_b=1, n_x=500):
    h_vals = h(times)
    k_BTs = (k_BT_b*h_vals)[np.newaxis, :]
    mesh = np.linspace(potential.x_min, potential.x_max, n_x)
    boltzmann_distros = potential.boltzmann(mesh, k_BTs)
    return boltzmann_distros

def analytical_distances_to_boltzmann(p_trajectories, potential, distance_function=mpemba.distance_functions.L1, axis=1, k_BT_b=1):
    """
    Compute the distance between the PDFs generated by the FPE solver above and the true Boltzmann distro.

    Parameters
    ----------
    p_trajectories : 3D (num_temps,num_bins,timesteps) vector of numerics
        Vector of p at each timestep.
    potential : child of Potential
        Potential used to generate p_trajectories.
    distance_function : function, optional
        Distance function to compute. The default is mpemba.distance_functions.L1.
    axis : int, optional
        Axis to integrate along. The default is 1.
    k_BT_b : Numeric, optional
        Bath temperature. The default is 1.

    Returns
    -------
    2D vector of numerics (num_temps*timesteps)
        Distances at each timestep, for each set of trajectories.

    """
    x = np.linspace(potential.x_min, potential.x_max, p_trajectories.shape[1])
    dx = x[1]-x[0]
    pi = potential.boltzmann(x, k_BT_b)
    pi_structured = np.repeat(np.reshape(pi, (1,*pi.shape, 1)), p_trajectories.shape[0], axis=0)
    return distance_function(p_trajectories, pi_structured, dx=dx, axis=axis)