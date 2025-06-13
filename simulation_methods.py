# -*- coding: utf-8 -*-
"""
simulation_methods.py.

Created on Wed Jun 11 14:42:20 2025
v1

@author: sane
Contains methods to integrate the Langevin equation, effectively simulating what the setup defined in Kumar and Bechhoefer (2020) does
"""

import numpy as np
from tqdm import tqdm
import pandas as pd
import xarray as xr

dt = 1e-5
expt_length = 6e-2
F = lambda x: -x
x_min = -1
x_max = 3
D = 170 # px^2/s
k_BT_b = 1
gamma = k_BT_b/D #0.02
E_barrier,E_tilt, F_left, x_well, force_asymmetry = 2,1.3,50,0.5,1 # for testing purposes

def langevin_simulation(x_0, dt=dt, gamma=gamma, expt_length = expt_length, force = F, temperature_function=lambda t: k_BT_b):
    """
    Integrate (the Ito way) the SDE dx = F(x)/gamma * dt + D(x,t)*eta(t), where eta is a normally distributed random number, from t=0 to t=expt_length.

    Parameters
    ----------
    x_0 : numeric or vector of numerics
        Initial position(s) (at t=0). The shape will determine the number of output trajectories.
    dt : numeric, optional
        Time interval. The default is 1e-5.
    gamma : numeric, optional
        Viscosity. The default is roughly 5.88e-3.
    expt_length : numeric, optional
        Time interval to integrate over. The default is 6e-2.
    force : vectorised function, optional
        Vectorised function, deterministic term. The default is F(x)=-x.
    temperature_function : function, optional
        For time-varying quenches, the function describing how the bath temperature evolves. The default is lambda t: k_BT_b, which is an instantaneous quench.

    Returns
    -------
    x : 2D vector of shape (expt_length//dt, *x_0.shape). 
        If x_0 is 1D, this is a 2D array with the first axis corresponding to the timestep and the second axis corresponding to the 'particle number' in the trajectory. If x_0 is a single numeric, x will only have one useful dimension corresponding to the timestep. x is a vector of all of the trajectories we get from stochastically integrating the Langevin equation, and forms an ensemble that we can plug into the Ensemble object defined earlier.

    """
    # if type(x_0) != np.array:
    #     x_0 = np.array([x_0])
    timesteps = round(expt_length/dt)
    x_i = x_0
    x = np.zeros((*x_0.shape, timesteps)) # Preallocating space and mutating the array is more efficient than appending data
    x[...,0] = x_0
    t = np.arange(0,expt_length, dt)
    D = temperature_function(t)/gamma
    thermal_fluctuation_std = np.sqrt(2*D*dt)
    stochastic_displacement = np.random.normal(0,thermal_fluctuation_std, size=(*x_0.shape, timesteps)) # Precalculate the noise terms we will use in the integral -- speeds up code. One array of noise (same shape as x_0) is needed per timestep. 
    # measurement_noise = np.zeros_like(stochastic_displacement) # We assume no measurement noise. Uncomment this and replace it with some Gaussian-like term for measurement noise (just setting it equal to zeros wastes memory and speed)
    for i in tqdm(range(1, timesteps)):
        deterministic_displacement = force(x_i)*dt/gamma
        x_i = x_i + (deterministic_displacement + stochastic_displacement[..., i])# + measurement_noise[:,i]) # Uncomment to add measurement noise

        x[...,i] = x_i # 'Append' to the preallocated array
    return x

def run_mpemba_simulations(k_BTs, num_particles, potential, quench_protocol = lambda t: k_BT_b, num_allowed_initial_positions=100_000,dt=1e-5, expt_length=1e-1):
    """
    Simulate the experiment in Bechhoefer and Kumar (2020) by choosing an initial position from a Boltzmann distribution and integrating the Langevin equation that it corresponds to.

    Parameters
    ----------
    k_BTs : pandas array/dict of {temperature_label: temperature},
        eg. {'h': 1000, 'w': 12, 'c': 1}.
    num_particles : int
        Number of particles in the ensemble (preferably >1000).
    potential : Asymmetric_DoubleWell_WithMaxSlope object
        In principle this could also be another potential object that has the right methods.
    quench_protocol : TYPE, optional
        DESCRIPTION. The default is lambda k_BT: k_BT.
    num_allowed_initial_positions : int, optional
        The size of the array we'll draw initial positions from. Ideally this should be large. The default is 100_000.
    dt : numeric, optional
        Timestep for Ito integration. The default is 1e-5.
    expt_length : numeric, optional
        Interval to integrate over. The default is 1e-1.

    Returns
    -------
    xr.DataArray with coordinates T (temperatures), n (particle number), t (time)
        An ensemble containing the particle trajectories, and appropriately labelled. This can be passed into an Ensemble object later.

    """
    # potential_params ordering: [E_barrier, E_tilt, F_left, x_well, x_min,x_max,force_asymmetry]
    if type(k_BTs) == dict:
        k_BTs = pd.Series(k_BTs) # Convert for stability
    active_range = np.linspace(x_min, x_max, num_allowed_initial_positions)
    results = []
    initial_distro = xr.DataArray([potential.boltzmann_array(active_range,k_BT) for k_BT in k_BTs], coords=(k_BTs.index, active_range), dims=(['T','p'])) 
    # (len(k_BTs) x num_allowed_initial_positions) array to draw positions from

    times = np.arange(0,expt_length,dt)
    for k_BT_label in k_BTs.index:
        x = np.random.choice(active_range, num_particles, p=initial_distro.loc[k_BT_label]) # draw initial position from its probability distribution
        results.append(langevin_simulation(x, force=potential.F, temperature_function=quench_protocol, expt_length=expt_length))
    return xr.DataArray(results, coords = [('T', k_BTs.index), ('n', np.arange(0,num_particles,1)),('t', times)])
