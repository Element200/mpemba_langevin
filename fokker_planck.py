# -*- coding: utf-8 -*-
"""
fokker_planck.py.

Created on Fri Aug 15 11:13:00 2025
v1.0

@author: sane
Analytic methods to solve the FPE (surprisingly I couldn't find any good libraries to do this for arbitrary F(x)).
"""

import numpy as np
from tqdm import tqdm
import scipy
import mpemba

# Parameters
x_min = -1 
x_max = 6
N = 500           # Number of grid points
dx = (x_max-x_min) / N        # Spatial resolution
dt = 1e-6       # Time step
T = 1e-1           # Total time
steps = int(T / dt)

D = 100       # Diffusion coefficient

# Spatial grid
x = np.linspace(x_min,x_max, N)

# User-defined potential function U(x)
potential = mpemba.special_potentials.AsymmetricDoubleWellPotential(x_max=x_max)
bumpy = mpemba.special_potentials.BumpyAsymmetricDoubleWellPotential(x_max=x_max, force_asymmetry = 0.3, b=0.1)

U = bumpy.U
U_vals = U(x)
p_0 = np.exp(-U(x)/100) # hot boltzmann distro
p_f = np.exp(-U(x)/1) # cold boltzmann distro
p_f /= p_f.sum()*dx # normalise
p_0 /= p_0.sum()*dx

dU = (U_vals-np.roll(U_vals, 1))[1:]

diag_plus = np.exp(dU/2)
diag_minus = np.exp(-dU/2)
diag_0 = -(np.pad(diag_plus, (1,0), mode='constant', constant_values=(0,0))+np.pad(diag_minus, (0,1), mode='constant', constant_values=(0,0)))

S = scipy.sparse.diags([diag_minus, diag_0, diag_plus], [-1,0,1])

p = p_0
Z_vals = []
intermediate_val = 3_000
in_range = lambda x, L, epsilon: (x>=L-epsilon) and (x<=L+epsilon)
p_mid=np.zeros_like(p)
# for i in tqdm(range(steps)):
#     W = S*D/dx**2
#     if i == intermediate_val:
#         p_mid = np.copy(p)
#     k_1 = (W @ p) # RK4 integration
#     k_2 = W @ (p+k_1*dt/2)
#     k_3 = W @ (p+k_2*dt/2)
#     k_4 = W @ (p+k_3*dt)
#     # if not in_range(change.sum(), 0, 1e-3):
#     #     break
#     p = p + (k_1 + 2*k_2 + 2*k_3 + k_4)*dt/6
#     Z = p.sum()*dx
#     if not in_range(Z, 1, 1e-3):
#         print("INTEGRATION FAILURE!")
#         break
#     Z_vals.append(Z)

def fokker_planck_solver(p_0, potential, D, t_max, dt=1e-6, h=lambda t: 1, saving_timestep=1000):
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
    # U_vals = potential.U(x)
    # p_0 /= p_0.sum()*dx
    # dU = (U_vals-np.roll(U_vals, 1))[1:]

    # diag_plus = np.exp(dU/2)
    # diag_minus = np.exp(-dU/2)
    # diag_0 = -(np.array([0, *diag_plus])+np.array([*diag_minus, 0]))
    # # diag_0 = -(np.pad(diag_plus, (1,0), mode='constant', constant_values=(0,0))+np.pad(diag_minus, (0,1), mode='constant', constant_values=(0,0)))
    # S = scipy.sparse.diags([diag_minus, diag_0, diag_plus], [-1,0,1]) # Assumes force is not varying with time
    
    p = p_0
    p_out = []
    for i in tqdm(range(int(t_max//dt))):
        W = potential.W_matrix(x=x, D=D, h=h, t=i*dt)
        k_1 = (W @ p) # RK4 integration
        k_2 = W @ (p+k_1*dt/2)
        k_3 = W @ (p+k_2*dt/2)
        k_4 = W @ (p+k_3*dt)
        p = p + (k_1 + 2*k_2 + 2*k_3 + k_4)*dt/6
        if i % saving_timestep==0: p_out.append(p)
        Z = p.sum()*dx
        if not in_range(Z, 1, 1e-3): # The probability must integrate to one (to at most 0.1% error)
            raise ValueError("Integration failure!")
    return np.array(p_out)

def analytic_solution(k_BTs, potential, D, n_x = 500, t_max = 0.1, dt=1e-6, resolution = 1e-4):
    """
    

    Parameters
    ----------
    k_BTs : TYPE
        DESCRIPTION.
    potential : TYPE
        DESCRIPTION.
    D : TYPE
        DESCRIPTION.
    n_x : TYPE, optional
        DESCRIPTION. The default is 500.
    t_max : TYPE, optional
        DESCRIPTION. The default is 0.1.
    dt : TYPE, optional
        DESCRIPTION. The default is 1e-6.
    resolution : TYPE, optional
        DESCRIPTION. The default is 1e-4.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    p_outs = []
    x = np.linspace(potential.x_min, potential.x_max, n_x)
    for k_BT in k_BTs:
        p_0 = potential.boltzmann(x, k_BT)
        p_outs.append(fokker_planck_solver(p_0, potential, D, t_max=t_max, dt=1e-6, saving_timestep=int(resolution//dt)))
    return np.array(p_outs), np.arange(0, t_max+resolution, resolution)

def analytical_distances(p_trajectories, t, distance_function=mpemba.distance_functions.L1):
    return distance_function()

# plt.plot(x, p_0, 'r')
# plt.plot(x, p_mid, 'orange')
# plt.plot(x, p, 'g')
# print()
# print("L1 wrt p_f:", np.abs(p-p_f).sum()*dx)
# print("L1 wrt p_0:", np.abs(p-p_0).sum()*dx)
# print(Z_vals[-10:])
