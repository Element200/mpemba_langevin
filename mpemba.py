import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
import sympy as sym
import matplotlib.animation as animation

dt = 1e-5
expt_length = 6e-2
F = lambda x: -x
x_min = -1
x_max = 3
D = 170 # px^2/s
k_BT_b = 1
gamma = k_BT_b/D #0.02

# class BoundedForcePotential(object):
#     """Harmonic potential with max. slopes at some limits"""
#     def __init__(self, function):
#         self.function = function
#     def F(self, x):
#         return 

class Potential(object):
    """Takes in a functional form of a potential and defines a bunch of handy functions"""
    def __init__(self, U_0):
        self.U_0 = U_0
        x = sym.symbols('x')
        self.F_0 = sym.lambdify(x, sym.diff(-U_0(x)))
    def F(x, x_l, x_r):
        return None

class Asymmetric_DoubleWell_WithMaxSlope(object):
    """Object for generating standard double-well potential but also retrieving various parameters from the function. This is temporarily defined as its own class, and should be redefined to inherit from a more basic class in the future."""
    
    def __init__(self, E_barrier, E_tilt, F_left, x_well=0.5,x_min=x_min,x_max=x_max, force_asymmetry=1):
        """
        Define initial parameters for the well.

        Parameters
        ----------
        E_barrier : numeric
            Barrier height: controls "hopping time".
        E_tilt : numeric
            Asymmetry between wells: controls probability mass ratio between wells at equilibrium.
        F_left : numeric
            The asymptotic force on the left side of the well.
        x_well : numeric, optional
            Distance from the origin of both wells. The default is 0.5.
        x_min : numeric, optional
            Distance of the left side of the "box". The default is -1.
        x_max : numeric, optional
            distance of the right side of the "box". The default is 3.
        force_asymmetry : numeric, optional
            Ratio of the left and right forces. The default is 1 -- i.e. the asymptotic forces on both sides are equal.

        Returns
        -------
        None.

        """
        self.E_barrier = E_barrier
        self.E_tilt = E_tilt
        self.x_well = x_well
        self.F_left = F_left
        self.F_right = force_asymmetry*F_left
        self.x_min = x_min
        self.x_max = x_max
        self.x_l, self.x_r = self.get_slope_boundaries()
        return None
    def _newton_raphson_solver(self, f, x_0, max_iterations = 100, dx = 1e-6, return_error=False, tolerance = 1e-5, debug_mode=False):
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
            raise ValueError(x_n, f(x_n))
        return x_n
    def U_0(self, x):
        """
        Generate basic potential (without bounded derivatives) evaluated at x.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).

        Returns
        -------
        Numeric or vector of numerics
            Potential evaluated at x.

        """
        return self.E_barrier*(1-2*(x/self.x_well)**2 + (x/self.x_well)**4) - self.E_tilt*(x/self.x_well)/2
    def dU_0dx(self, x):
        """
        Evaluate the derivative of U_0 at x.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).

        Returns
        -------
        Numeric or vector of numerics
            Derivative of potential evaluated at x.

        """
        return self.E_barrier*(-4*x/self.x_well**2 + 4*x**3/self.x_well**4) - self.E_tilt/(2*self.x_well)
    def get_slope_boundaries(self, **kwargs):
        """
        Compute the points where the slope of U_0 is F_left and F_right.

        Returns
        -------
        x_l
            Left boundary.
        x_r
            Right boundary.

        """
        self.x_l = self._newton_raphson_solver(lambda x: self.dU_0dx(x) + self.F_left, self.x_min, **kwargs)
        self.x_r = self._newton_raphson_solver(lambda x: self.dU_0dx(x) - self.F_right, self.x_max, **kwargs)
        return self.x_l, self.x_r
    
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
            Potential evaluated at positions.

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
            Force at positions.

        """
        in_well = (x >= self.x_l) & (x <= self.x_r)
        left_of_well = (x < self.x_l)
        right_of_well = (x > self.x_r)
        return in_well*(-self.dU_0dx(x))+left_of_well*self.F_left-right_of_well*self.F_right
    def plot_potential(self, plot_range):
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
        plt.plot(plot_range, self.U(plot_range), 'g')
    def plot_force(self, plot_range):
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
        plt.plot(plot_range, self.F(plot_range), 'c')
    def _boltzmann_unnormalised(self, x, k_BT):
        # unnormalised, unvectorised boltzmann distro
        return np.exp(-self.U(x)/k_BT)
        # If k_BT is array-like, generate an array of shape len(x),len(k_BT)
    def boltzmann(self, x, k_BT, integration_bounds=None):
        """
        Vectorised, normalised boltzmann distro. Implicitly sets p(x outside bounds) = 0 by integrating over x_min, x_max instead of -infty,infty.

        Parameters
        ----------
        x : numeric or vector of numerics
            Position(s).
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
        func = np.vectorize(lambda x, k_BT: self._boltzmann_unnormalised(x, k_BT)/Z)
        return func(x, k_BT)
    def boltzmann_array(self,x,k_BT, **kwargs):
        """
        Generate normalised boltzmann array (Asymmetric_DoubleWell_WithMaxSlope.boltzmann IS normalised, but has floating point error that will cause numpy.random to complain).

        Parameters
        ----------
        x : 1D array-like
            Array of positions. Must be 1D for now.
        k_BT : Numeric
            Temperature.

        Returns
        -------
        Numeric or vector of numerics
            Normalised PDF in array format.

        """
        # arr MUST be 1D (for now!)
        arr = self.boltzmann(x,k_BT, **kwargs)
        return arr/arr.sum()

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

def L1(vec1, vec2, axis=None):
    """
    Find sum |vec2-vec1|_i. Does NOT calculate the integral; inputs must be converted to PMFs before function is called.

    Parameters
    ----------
    vec1 : 2D vector of numerics
        Some probability mass vector (this will not fail if vec1 is not a probability mass but the intended use case is for PMFs).
    vec2 : 2D vector of numerics of same shape as vec1
        Second probability mass vector.
    axis : int, optional
        If vec1 is multi-dimensional, the axis to sum along. The default is None.

    Returns
    -------
    float
        sum |vec2-vec1|_i. If this is less than zero or greater than two, vec1 and vec2 are not correctly normalised.

    """
    # vec1 and vec2 must be at least two-dimensional arrays
    assert vec1.shape[axis]==vec2.shape[axis], f"Mismatch between vec1 {np.array(vec1).shape} and vec2 {np.array(vec2).shape}"
    return np.sum(np.abs(vec2-vec1), axis=axis)



def kullback_leibler(vec1, vec2):
    """
    Take the entropic distance between vec1 and vec2. Not implemented to fullest potential yet.

    Parameters
    ----------
    vec1 : 2D vector of numerics
        Some probability mass vector (this will not fail if vec1 is not a probability mass but the intended use case is for PMFs).
    vec2 : 2D vector of numerics of same shape as vec1
        Second probability mass vector.

    Returns
    -------
    float
        KL distance between vec1 and vec2.

    """
    def _helper(element):
        # Avoids issues with taking the log of 0.
        if element == 0:
            return 0
        else:
            return np.log(element)
    log_ish = np.vectorize(_helper)
    assert len(vec1) == len(vec2)
    return np.sum(vec1*log_ish(vec1) - vec1*log_ish(vec2))

class Ensemble(object):
    """Contains methods to do useful things with trajectory data obtained either experimentally or via simulations."""
    
    def __init__(self, data, potential, dt=1e-5, expt_length=6e-2, temperatures={'h':1000,'w':12,'c':1}):
        """
        Initialise the object with relevant parameters.

        Parameters
        ----------
        ensemble : xarray.DataArray of dimension = 3, with axis 0=temperature (generally, h, w, or c), axis 1 = particle number, axis 2=time
            The x(t) trajectories obtained from integrating the Langevin equation (via simulation or experiment) for each temperature. The time axis should contain the times.
        potential : Potential object
            Should contain methods to calculate the potential and force. Will fail if potential and force are not vectorised.
        temperature: dict
            dictionary of temperature_name:temperature_value (eg. 'h':1000)

        Returns
        -------
        None.

        """
        # ensemble: 
        # COORD NAMES: 'T', 'n', 't'
        # temperatures: a dictionary mapping temp name to temp value, eg {'h': 1000, 'w': 12, 'c': 1}
        # potential: needs to be a vectorised function
        self.ensemble = data
        self.potential = potential
        self.temperatures = temperatures # This needs to be changed so that we get it from the data itself
        self.num_temperatures = data.shape[0] # The first axis must be the number of temperatures
        self.N = data.shape[1] # The second axis must be the number of trials
        self.dt = dt
        self.expt_length = data.shape[-1]*dt # The last axis must be the timesteps
        self.times = data['t']
        self.bins, self.heights = None, None
        
        return None

    def get_histograms(self, x_max=None, x_min=None, num_bins=100):
        """
        Histogram over ensemble number. If self.bins and self.heights are already generated, return them; this will only do work if self.bins and self.heights are None and None.

        Parameters
        ----------
        x_max : numeric, optional
            Right end of the box. If None, x_max will be pulled from the potential definition. The default is None.
        x_min : numeric, optional
            Left end of the box. If None, x_min will be pulled from the potential definition. The default is None.
        bins : preferably int, but arrays also work; optional
            Number of bins in each histogram. The default is 100.

        Returns
        -------
        bins
            1D array (size=bins-1) of all of the bins.
        heights
            2D array(size = (expt_length//dt, num_temperatures)) of the histograms for each timestep.

        """
        if x_min is None:
            x_min = self.potential.x_min
        if x_max is None:
            x_max = self.potential.x_max
        
        # returns an array of histograms 
        if not (self.heights is None and self.bins is None):
            return self.bins, self.heights # Don't do all this work if you've already done it before
        
        # binned_active_domain = np.linspace(x_min, x_max, bins) # In principle you should use binned_active_domains for computations but it's possible that the particles leave the box, in which case histogramming should be done over the full range of possible positions to avoid bugs.
        self.global_min = np.min(self.ensemble)
        self.global_max = np.max(self.ensemble)
        binned_active_range = np.linspace(self.global_min, self.global_max, num_bins)
        self.dx = binned_active_range[1]-binned_active_range[0]

        heights = np.apply_along_axis(lambda data: np.histogram(data, bins=binned_active_range, density=True)[0], axis=1, arr=self.ensemble)
        
        self.bins, self.heights = binned_active_range[1:]-self.dx/2, heights

        return binned_active_range[:-1]-self.dx/2, heights # Centre the bins before returning them so that they can be plotted properly. Also throw away the first element so that the array have the same first dimensional-shape

    def get_distances(self, eqbm_boltzmann_distro=None, distance_function=L1):
        """
        Get the distance of the ensemble to equilibrium, defined by eqbm_boltzmann_distro.

        Parameters
        ----------
        eqbm_boltzmann_distro : 1D array, size = len(self.bins), optional
            PMF (*not* PDF) of the Boltzmann distribution at the cold temperature. The size of
        the number of bins, otherwise taking the distance WILL fail. The default is the boltzmann array for the temperature corresponding to temperature 'c'. (If 'c' is not in the list of temperatures, this must be explicitly defined.)
        distance_function: function, optional
            The distance function to use. This must take in two arguments and an optional argument called 'axis'. The default is L1.

        Returns
        -------
        distances : Array of shape (timesteps, num_temperatures)
            Distances at each timestep, for each temperature.

        """
        bins, heights = self.get_histograms() # Generate the variables we need
        if eqbm_boltzmann_distro is None:
            eqbm_boltzmann_distro = self.potential.boltzmann_array(bins, k_BT=self.temperatures['c'])
            eqbm_boltzmann_distro = np.repeat(np.reshape(eqbm_boltzmann_distro, (1,*eqbm_boltzmann_distro.shape)), self.num_temperatures, axis=0)
            eqbm_boltzmann_distro = np.reshape(eqbm_boltzmann_distro, (*eqbm_boltzmann_distro.shape, 1))
        distances = distance_function(heights*self.dx, eqbm_boltzmann_distro, axis=1) # L1 takes in probability *mass* functions, not densities, so we divide by dx
        return distances
    
    def plot_sample_trajectories(self, num_trajectories = 4, temp='h'):
        """
        Plot a bunch of randomly selected trajectories.

        Parameters
        ----------
        num_trajectories : int, optional
            Number of randomly selected trajectories to plot. The default is 4.
        temp : str, optional
            Temperature key to plot from. The default is 'h'.

        Returns
        -------
        None.

        """
        _ = self.get_histograms() # generate global_min and global_max
        plt.plot(range(int(self.expt_length//self.dt)+1), self.ensemble.loc[temp, np.random.choice(self.N, num_trajectories),:].T)
        plt.plot([0,self.expt_length/self.dt],[[self.potential.x_min,self.potential.x_max], [self.potential.x_min, self.potential.x_max]], 'r')
        plt.plot([0,self.expt_length/self.dt],[[self.global_min,self.global_max], [self.global_min,self.global_max]], 'g')
        plt.show()
        return None

    def gut_checks(self, init=0, mid=341, end=6000, plot_init=False, plot_end=False, num_bins=100):
        """
        Plot a bunch of things just to verify that everything's working properly. Also return the variables we plot just in case we want to manipulate them somehow.

        Parameters
        ----------
        init : int, optional
            Timestep corresponding to the first timestep. The default is 0.
        mid : int, optional
            Timestep corresponding to some middle timestep (since relaxation is exponential this should be very shortly after init). The default is 341.
        end : int, optional
            The final timestep, when the system has fully equilibriated. The default is 6000.
        plot_init : bool, optional
            Plot the predicted Boltzmann distribution at t=0
        plot_fin : bool, optional
            Plot the predicted Boltzmann distribution at t -> inf

        Returns
        -------
        bins : 1D array
            The bins we get by histogramming.
        list : list of three 1D arrays
            Heights for each timestep.
        """
        plt.close()
        bins, heights = self.get_histograms(num_bins=num_bins)
        heights_init = heights[...,init]
        heights_mid = heights[...,mid]
        heights_end = heights[...,end]
        fig, ax = plt.subplots(self.num_temperatures)
        keys = list(self.temperatures.keys())
        if self.num_temperatures == 1:
            ax = [ax] # Annoyingly, plt.subplots(1) does not yield a list of axes with one element.
        for i in range(self.num_temperatures):
            # ax[i].title("T =", self.ensemble['T'][i])
            # print(self.dx)
            ax[i].bar(bins, heights_init[i,:], color = 'red', width = self.dx, label=f"t={init}"*(i==0), alpha=0.3) # Multiplying by (i==0) ensures we don't get redundant labels
            ax[i].bar(bins, heights_mid[i,:], color = 'orange', width = self.dx, label=f"t={mid}"*(i==0), alpha=0.3)
            ax[i].bar(bins, heights_end[i,:], color = 'green', width = self.dx, label=f"t={end}"*(i==0), alpha=0.3)
            if plot_init:
                heights_init_pred = self.potential.boltzmann_array(bins, k_BT=self.temperatures[keys[i]])/self.dx
                chi_squared = np.sum((heights_init[i,:] - heights_init_pred)**2 / heights_init_pred)
                print(f"Initial histogram, T = {self.temperatures[keys[i]]}: chi^2 =", chi_squared)
                ax[i].plot(bins, heights_init_pred, 'red', label=r"$\pi(x;T_0)$"*(i==0)) # fr-strings are both formatted strings and raw strings, apparently
            if plot_end:
                heights_end_pred = self.potential.boltzmann_array(bins, k_BT=self.temperatures['c'])/self.dx # Final distro is always the cold one
                ax[i].plot(bins, heights_end_pred, 'blue', label=r"$\pi(x;T_c)$"*(i==0))
                chi_squared = np.sum((heights_end[i,:] - heights_end_pred)**2 / heights_end_pred)
                print(f"Final histogram, T = {self.temperatures[keys[i]]}: chi^2 =", chi_squared)
            ax[i].set_ylabel(r"$p(x,t)$")
        if i == self.num_temperatures-1:
            fig.legend()
        plt.tight_layout()
        plt.xlabel("$x$")

        plt.show()
        return bins, [heights_init, heights_mid, heights_end]
    
    def animate(self, T='h', num_bins=200, num_animated_frames = 500):
        new_binned_active_range = np.linspace(self.potential.x_min, self.potential.x_max, num_bins)
        new_binned_initial_distro = self.potential.boltzmann(new_binned_active_range, self.temperatures[T])
        new_binned_final_distro = self.potential.boltzmann(new_binned_active_range, self.temperatures[T])
        
        fig, ax = plt.subplots()
        bins, all_heights = self.get_histograms(num_bins=num_bins)
        patches = ax.bar(bins, all_heights[0,:,0]/self.N, width=bins[1]-bins[0]) # FIX THIS: WE SET THE AXIS OF THE HISTOGRAMS TO PLOT TO ALWAYS BE 0,:,0
        ax_height = np.max(new_binned_final_distro + 0.02)
        ax.set_ylim(0,ax_height)
        # temperature_function= lambda t: (temperature[T]-T_b)*np.exp(-t/tau)+T_b
        
        set_const_height = True
        set_moving_height = False
        plot_analytic_solution = False
        
        # animated_frames = [i for i in range(20)] + [i for i in range(20,60, 2)] + [i for i in range(60,360, 5)] + [i for i in range(360,3960, 10)] + [i for i in range(3960,30000, 100)]
        # Keep it small for efficient rendering
        exponential_frame_func = lambda x, x_0: (expt_length/dt)*(np.exp(x/x_0)-1)/(np.exp(num_animated_frames/x_0)-1)
        animated_frames = np.rint(exponential_frame_func(np.arange(0,num_animated_frames+1, 1), 100)).astype('int')
        # Time in the video will be on a log scale, primarily for rendering efficiency
        
        ax.set_xlim((-1,3))
        ax.set_xlabel("x")
        ax.set_ylabel("p(x,t)")
        ax.plot(new_binned_active_range, new_binned_initial_distro, 'r')
        ax.plot(new_binned_active_range, new_binned_final_distro, 'g')
        # Update function for the animation
        def update(frame_number):
            # Get the data for the current frame
            # ax.clear()
            heights, _ = all_heights[0,:,frame_number], bins
            # Update the histogram data
            for i in range(len(patches)):
                patches[i].set_height(heights[i])#, width=bins[1]-bins[0], color='y', alpha=0.7)
            ax.set_title(f"t = {frame_number*dt*1e6 : .0f} μs")
            if set_moving_height:
                if frame_number % 100 == 0:
                    ax.set_ylim(0,ax_height)
            elif set_const_height:
                ax.set_ylim(0,ax_height)
            # for i in range(len(bins)):
            #     patches[i].set_height(heights[i])
            return patches
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=animated_frames, interval=10, blit=True, cache_frame_data=True)
        plt.show()
        return ani
    
    # def verify_histograms(self):
    #     bins, heights = self.get_histograms()
    #     heights_init = heights[..., 0]
    #     heights_end = heights[...,-1]
    #     heights_init_pred, heights_end_pred = [], []
    #     for T in self.temperatures:
    #         heights_init_pred.append(self.potential.boltzmann_array(bins, self.temperatures[T]))
    #         chisq = ((heights_init[i, :]-heights_init_pred[i])**2/heights_init_pred[i]).sum()
    #         print(chisq)
    #     return None
    
    def average_energy(self, normalise_by_final=False, normalise_timesteps=100):
        """
        Compute the ENSEMBLE average energy for each timestep, for each temperature.

        Parameters
        ----------
        normalise_by_final : bool, optional
            If true, rescale all the energies so that the equilibrium average energy E_eq=1. The default is False.
        normalise_timesteps : int, optional
            If normalise_by_final is True, compute the equilibrium average energy by normalising over the last few timesteps. This variable sets the number of timesteps to normalise over --- making it bigger would make the initial estimate of E_eq less noisy but runs the risk of averaging over non-stochastic behaviour if it's too big. The default is 100.

        Returns
        -------
        energy_av: array of shape (timesteps, num_temperatures)
            The ensemble average energy for each timestep, for each temperature.

        """
        energy = self.potential.U(self.ensemble)
        energy_av = energy.mean(dim='n') # Take ensemble average, not time average
        if normalise_by_final:
            final_energy_av = energy_av[-1,-normalise_timesteps:].mean(dim='t')
            return energy_av/final_energy_av # Return <E>/E_eq instead of just <E>
        return energy_av

    def export_raw_data(self, filename):
        """
        **UNIMPLEMENTED** Convert the ensemble data into a netCDF file.

        Parameters
        ----------
        filename : str
            The name of the file to export to.

        Returns
        -------
        None.

        """
        pass # TODO

def sample_from_distro(x_range, distro, num_particles):
    """
    Choose the initial position from a distribution. This just uses numpy's builtin methods but in principle this could be its own inverse transform sampling protocol.

    Parameters
    ----------
    x_range : 1D array
        (Large) array of possible initial positions to choose from.
    distro : 1D array with same size as x_range
        Probability mass function to choose x_range from.
    num_particles : int
        Number of samples to draw.

    Returns
    -------
    1D array
        Array of initial positions, randomly selected according to distro.

    """
    return np.random.choice(x_range, num_particles, p=distro)

E_barrier,E_tilt, F_left, x_well, force_asymmetry = 2,1.3,50,0.5,1 # for testing purposes
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

def chunk_splitter(dataframe, dt=1e-5):
    """
    Turn one giant dataarray of particle trajectories into an xarray DataArray that stores each individual particle's trajectory in a new dimension --- this gives us something we can work with.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The raw data with column names set appropriately. This code will fail if certain column names do not exist.

    Returns
    -------
    chunks : xarray.DataArray with dims-=[n,t,col]
        DataArray of dimensions (n, t, 2) containing each 'chunk' --- a trajectory for a single particle. col, the third dimension, contains the position data in the 0th column and state data in the 1st column. On further cleaning, the state data can be thrown away so that we save some more memory. We throw away time data (redundant; changed to an index in the xarray) and voltage data (which we don't ever actually use) to save some memory.

    """
    # Each chunk contains one particle's trajectory
    # To save memory, we will get rid of the time column (assuming all time columns are identical)
    masked_data = dataframe[dataframe['state']==2]
    forward = dataframe[1:].reset_index(drop=True)
    pivots = dataframe[(forward['state']==0) & (dataframe['state']==2)].index + 1# This is a bit complex
    chunks = []
    for i in range(len(pivots)-1):
        chunks.append(masked_data.loc[pivots[i]:pivots[i+1]])
    chunks = np.array(chunks)
    times = chunks[0,:,1]*dt # Assumes all time columns are identical
    return xr.DataArray(chunks[...,0], dims=['n','t'], coords={'t':times}) # Throw away the voltage and time data (time is redundant between all chunks) and also state data (because we filtered it so it's all =2)

def extract_file_data(filenames, protocol_time, dt=1e-5, column_names=['x','t','V','state'], temperatures = ['h','w','c']):
    """
    Convert experimental file data into an ensemble with the same structure as that of the simulation.

    Parameters
    ----------
    filenames : list of str
        List of filenames to process.
    protocol_time : numeric
        Length of the protocol (in seconds).
    dt : numeric, optional
        Timestep in seconds. Must be less than the length of the protocol The default is 1e-5.
    column_names : , list of str
        Names of the columns in the output data. The default is ['x','t','V','state'].
    temperatures : list of str, optional
        Key names for the initial temperatures at which the protocol happens, eg. 'h', 'w', 'c' for 'hot','warm', 'cold' respectively. The default is ['h','w','c'].

    Returns
    -------
    array : xarray.DataArray
        Array with appropriate dimensions in with the same structure as after the simulation, so that further analysis on either data is identical.

    """
    # 
    data = []
    chunks = {}
    n_min = np.inf # Minimum number of particles in an array. Initially set to infinity because that's much higher than any experiment will realistically produce
    for filename in filenames:
        chunks[filename] = chunk_splitter(pd.read_table(filename, names=column_names, usecols=['x','t','state']))
        if chunks[filename].shape[0] < n_min:
            n_min = int(chunks[filename].shape[0]) # Once this for loop is done running, n_min will hold the lowest number of particles in the dataarray
    for filename in filenames:
        data.append(chunks[filename].loc[:n_min, ...]) # This ensures that data will contain fixed-length data along the n dimension as well
    array = xr.DataArray(data, dims=['T','n','t'], coords={'T':temperatures, 't': np.arange(0,protocol_time,dt)})
    return array

def run_mpemba_analysis(filenames, potential, protocol_time=7e-2, dt=1e-5, column_names = ['x','t','V','state'], temperatures=['h','w','c'], gut_checks=True):
    """
    High-level functions to run all of the basic analysis defined here.

    Parameters
    ----------
    filenames : list of str
        List of filenames to process.
    potential : Potential object
        The potential used by the experiment or simulation.
    protocol_time : numeric, optional
        Length of the procol, in seconds. The default is 7e-2.
    dt : numeric, optional
        Timestep in seconds. The default is 1e-5.
    column_names : List of str, optional
        Names of the columns. The default is ['x','t','V','state'].
    temperatures : List of str, optional
        Keys for the temperatures. The default is ['h','w','c'].
    gut_checks : bool, optional
        Plot sample histograms to check whether everything is working. The default is True.

    Returns
    -------
    times : numpy.array
        Mesh of times from 0 to expt_length in steps of dt.
    distances : xarray.DataArray
        Distances from equilibrium, for each timestep, for each temperature.
    energies : xarray.DataArray
        Average energy at each timestep.

    """
    # I've only defined Asymmetric_DoubleWell_WithMaxSlope but in principle any class with the right keywords can be defined
    data = extract_file_data(filenames, protocol_time=protocol_time, dt=dt, column_names=column_names, temperatures=temperatures)
    # data has the form we want of an ensemble
    ensemble = Ensemble(data, potential)
    if gut_checks:
        ensemble.gut_checks()
    distances = ensemble.get_distances()
    energies = ensemble.get_energies()
    times = ensemble.times
    return times, distances, energies

if __name__ == '__main__':
    import os
    os.chdir("..")
    temperatures = {'h': 1000, 'w': 12, 'c': 1}
    potential = Asymmetric_DoubleWell_WithMaxSlope(2, 1.3, 50, x_min=-1, x_max=3)
    data = run_mpemba_simulations(temperatures, 1000, potential)
    ensemble = Ensemble(data, potential, expt_length=1e-1, temperatures=temperatures)
    plt.plot(ensemble.ensemble.t, ensemble.get_distances().T)