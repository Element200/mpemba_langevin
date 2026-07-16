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
import numba
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm

directory = os.path.dirname(__file__)

import sys
sys.path.append(directory)
# This assumes that all of the mpemba files are in the same directory
try:
    import mpemba
except ImportError:
    print("Something went wrong when importing dependencies! Make sure local imports are in the same directory as this file")
    raise ImportError

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

# potential = mpemba.special_potentials.BumpyAsymmetricDoubleWellPotential(x_max=5,x_min=-5, E_barrier=3, E_tilt=0.5, force_asymmetry = 0.4, b=-0.05, n=1)
# potential = mpemba.special_potentials.BumpyAsymmetricDoubleWellPotential(x_max=5,x_min=-5, E_barrier=2, E_tilt=1, force_asymmetry = 0.2, b=0, n=1, x_s=0.1)
# bumpy = mpemba.special_potentials.BumpyAsymmetricDoubleWellPotential(x_max=7,x_min=-5, E_barrier=3, E_tilt=0.7, force_asymmetry = 0.2, b=-0.05, n=1)
bumpy = mpemba.special_potentials.BumpyAsymmetricDoubleWellPotential(x_max=5,x_min=-5, E_barrier=2, E_tilt=0.5, force_asymmetry = 0.4, b=-0.0, n=1)
potential = mpemba.special_potentials.AsymmetricDoubleWellPotential(x_min=-4, x_max=6, E_barrier=2, E_tilt=0.4, force_asymmetry=0.5)
other_potential = mpemba.special_potentials.AsymmetricDoubleWellPotential(x_min=-6, x_max=6, E_barrier=2, E_tilt=0.6, force_asymmetry=0.3) # Also has decent results

# k_BTs = np.logspace(0, np.log(15)/np.log(10), 200)
# a_2s = potential.a_k_boltzmannIC(k_BTs, n_x=1000)
# bumpy_a_2s = bumpy.a_k_boltzmannIC(k_BTs, n_x=1000)
# plt.semilogx(k_BTs, np.abs(np.array([a_2s, bumpy_a_2s]).T))

@numba.njit(cache=True, fastmath=False)
def fokker_planck_core(p_0, S, dx, dt, steps, saving_timestep, D, error_tolerance=1e-3, h_vals=None):
    n = p_0.shape[0]

    p = p_0.copy()
    p_new = np.empty_like(p)

    saved = np.zeros((n, steps // saving_timestep), dtype=p.dtype)

    if h_vals is None:
        h_vals = np.ones(steps, dtype=p.dtype)

    prefactor = D / (dx * dx)

    save_idx = 0

    S_plus = np.diag(S, k=1)
    S_minus = np.diag(S, k=-1)
    print(S_plus.max()*prefactor*dt, S_minus.max()*prefactor*dt)
    for t in range(steps):

        h = h_vals[t]
        scale = h * prefactor
        inv_h = 1.0 / h

        diag_plus = S_plus**inv_h
        diag_minus = S_minus**inv_h
        diag_0 = -np.array([0, *diag_plus]) - np.array([*diag_minus, 0])
        # W is a sparse matrix, and it slows down code tremendously to compute a full matrix-vector product over such large scales. diag_0 is an array one bigger than diag_plus or diag_minus
        
        wp = diag_0[0]*p[0] + diag_plus[0]*p[1]
        p_new[0] = p[0] + dt * scale * wp # left BC
        
        for i in range(1, n - 1):
            wp = diag_minus[i-1]*p[i-1] + diag_0[i]*p[i] + diag_plus[i]*p[i+1]
            p_new[i] = p[i] + dt*scale*wp

        # Right BC
        wp = diag_minus[n-2]*p[n-2] + diag_0[n-1]*p[n-1]
        p_new[n - 1] = p[n - 1] + dt*scale*wp

        p, p_new = p_new, p
        if t % saving_timestep == 0:
            Z = p.sum() * dx # Check normalisation
            if not (np.abs(Z - 1.0) < error_tolerance):
                raise ValueError("Probability conservation failed", Z, t)
            saved[:, save_idx] = p
            save_idx += 1
    return saved

def fokker_planck_core_deprecated(p_0, S, dx, dt, steps, saving_timestep, D, error_tolerance=1e-3, h_vals=None, integrator=euler_step):
    """Numba-optimised Fokker-Planck integration."""
    p = p_0.copy()
    saved = np.zeros((*p_0.shape, steps//saving_timestep), dtype=p_0.dtype)
    t_vals = np.arange(steps)*dt
    if h_vals is None:
        h_vals = np.ones_like(t_vals)
    for i in range(steps):
        # p_prev = np.copy(p)
        diag_plus = np.diag(S, k=1)**(1/h_vals[i])
        diag_minus = np.diag(S, k=-1)**(1/h_vals[i])
        diag_0 = -np.array([0, *diag_plus])-np.array([*diag_minus, 0])
        W = np.diag(diag_plus, k=+1) + np.diag(diag_minus, k=-1) + np.diag(diag_0, k=0) # Directly raising S to a power causes bugs because of the negative numbers on the diagonals.
        W *= h_vals[i]*D/dx**2
        p = integrator(W, p, dt) # euler is faster than RK4 and the accuracy difference is not that big
        
        # if np.isnan(p).any():
        #     raise ValueError("Probability vector contains NaNs", i, np.isnan(p).sum(), p_prev)
        if (i % saving_timestep == 0):
            Z = p.sum() * dx
            if not np.isclose(Z, 1.0, atol=error_tolerance):
                raise ValueError("Probability conservation failed! Z=", Z, "t=", i)
            saved[..., i//saving_timestep] = p.copy()
            
    return saved



def fokker_planck_solver(p_0, potential, D, t_max, dt=np.float64(1e-6), h=lambda t: np.ones_like(t), saving_timestep=1000, error_tolerance=1e-3, integrator=euler_step, x=None):
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
    
    if x is None:
        x = np.linspace(potential.x_min, potential.x_max, len(p_0))
        width = potential.x_max-potential.x_min
        dx = width/len(p_0)
    else:
        dx = x[1]-x[0]
    S = potential.grima_newman_discretisation(x=x)
    steps = int(t_max//dt)
    t_vals = np.arange(steps)*dt
    h_vals = h(t_vals)
    
    out = fokker_planck_core(p_0, S, dx, dt, steps, saving_timestep, D, error_tolerance=error_tolerance, h_vals=h_vals)
    return np.array(out)

def analytic_solution(k_BTs, potential, D, n_x = 500, t_max = 0.1, dt=1e-6, resolution = 1e-5, error_tolerance=1e-3, integrator=euler_step, FPE_solver=fokker_planck_solver, quench_protocol = mpemba.quench_methods.InstantaneousQuench(), k_BT_b=1):
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
        quench_protocol.set_a(k_BT/k_BT_b-1)
        p_outs.append(FPE_solver(p_0, potential, D, t_max=t_max, dt=dt, saving_timestep=int(resolution//dt), error_tolerance=error_tolerance, integrator=integrator, h=quench_protocol.h))
    return np.array(p_outs)

def quasistatic_limit(potential, h, times, k_BT_b=1, n_x=500):
    h_vals = h(times)
    k_BTs = (k_BT_b*h_vals)[np.newaxis, :]
    mesh = np.linspace(potential.x_min, potential.x_max, n_x)
    boltzmann_distros = potential.boltzmann(mesh, k_BTs)
    return boltzmann_distros

def analytical_distances_to_boltzmann(p_trajectories, potential, distance_function=mpemba.distance_functions.L1, axis=1, k_BT_b=1, x=None):
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
    if x is None:
        x = np.linspace(potential.x_min, potential.x_max, p_trajectories.shape[1])
    dx = x[1]-x[0]
    pi = potential.boltzmann(x, k_BT_b)
    pi_structured = np.repeat(np.reshape(pi, (1,*pi.shape, 1)), p_trajectories.shape[0], axis=0)
    return distance_function(p_trajectories, pi_structured, dx=dx, axis=axis)

# if __name__ == '__main__':
#     quench = mpemba.quench_methods.ExponentialQuench(None, 1e-3)
#     p_outs = analytic_solution([22,4,1], potential0, D=150, n_x=500, t_max=6e-2, FPE_solver=fokker_planck_solver_instQuench, dt=1e-5, error_tolerance=1e-2, quench_protocol=quench)
#     plt.semilogx(np.arange(6e-2//1e-5)*1e-5, analytical_distances_to_boltzmann(p_outs, potential0).T)

def compare_to_analytic_solution(ensemble, D, distance_function=mpemba.distance_functions.L1, plot=True, **kwargs_for_analytic_soln):
    k_BTs = ensemble.temperatures[ensemble.temperatures != 1] # Don't waste time computing D[pi_c, pi_c] = 0
    pdfs = analytic_solution(k_BTs, ensemble.potential, D, t_max = float(np.max(ensemble.times)), **kwargs_for_analytic_soln)
    distances = analytical_distances_to_boltzmann(pdfs, ensemble.potential)
    if plot:
        # plt.semilogx(ensemble.times, distances.T)
        ensemble.plot_distances()
        plt.plot(ensemble.times, np.where(distances.T>=1e-2, distances.T, np.nan*np.ones_like(distances.T)))
    return pdfs, distances

def animate_pdfs(x, pdfs, interval=200, repeat=False):
    """
    Animate an array of probability density functions.

    Parameters
    ----------
    x : 1D numpy array
        The x-axis values.
    pdfs : 2D numpy array
        Array of PDFs with shape (n_frames, len(x)).
    interval : int
        Delay between frames in milliseconds.
    repeat : bool
        Whether the animation should repeat.
    """
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    # Set axis limits
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(pdfs), np.max(pdfs))
    ax.set_xlabel("x")
    ax.set_ylabel("PDF")
    ax.set_title("PDF Evolution")

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        y = pdfs[frame]
        line.set_data(x, y)
        ax.set_title(f"Frame {frame+1}/{len(pdfs)}")
        return line,

    anim = FuncAnimation(
        fig,
        update,
        frames=len(pdfs),
        init_func=init,
        interval=interval,
        blit=False,
        repeat=repeat
    )

    plt.show()
    return anim

def animate_pdfs_(x, pdfs, interval=200, repeat=False, dt=1e-5, frame_decay_const=100):
    """
    Animate an array of probability density functions.
    using logarithmic time mapping.

    Early PDFs evolve slowly; later ones evolve faster.
    """
    num_animated_frames = len(pdfs)
    expt_length = num_animated_frames*dt
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(pdfs), np.max(pdfs))
    ax.set_xlabel("x")
    ax.set_ylabel("p(x,0)")
    ax.set_title("PDF")

    # Precompute logarithmic frame mapping
    # Linear parameter in [0, 1]

    exponential_frame_func = lambda x, x_0: (expt_length/dt)*(np.exp(x/x_0)-1)/(np.exp(num_animated_frames/x_0)-1) # Function to pick out a large number of frames in the beginning and a small number of frames in the end
    frame_indices = np.rint(exponential_frame_func(np.arange(0,num_animated_frames+1, 1), frame_decay_const)).astype('int')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        pdf_index = frame_indices[frame]
        y = pdfs[pdf_index]
        line.set_data(x, y)
        ax.set_label(f"p(x,{frame*dt})")
        return line,

    anim = FuncAnimation(
        fig,
        update,
        frames=num_animated_frames,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=repeat
    )

    plt.show()
    return anim
