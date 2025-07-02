"""
mpemba.py.

v1
@author: sane
Contains high-level functions to perform analysis of Mpemba experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
from tqdm import tqdm
import sympy as sym

# LOCAL IMPORTS
import distance_functions
import potential_methods
import simulation_methods
import file_processing

dt = 1e-5
expt_length = 6e-2
F = lambda x: -x
x_min = -1
x_max = 3
D = 170 # px^2/s
k_BT_b = 1
gamma = k_BT_b/D #0.02

def in_range(x, lower, upper):
    """Find whether  lower <= x <= upper."""
    return (x <= upper) & (x >= lower) 

@potential_methods.BoundedForcePotential
class AsymmetricDoubleWellPotential:
    """Basic asymmetric double-well potential that we use for Mpemba simulations. Any potential can be defined here: simply define all of the relevant parameters in __init__ and the form of the potential in U_0. Decorating with @BoundedForcePotential will add all other necessary methods."""
    
    def __init__(self, E_barrier=2, E_tilt=1.3, x_well=0.5, x_min=-1, x_max=2, F_left=50, force_asymmetry=1):
        """Define parameters used in the potential."""
        self.E_barrier = E_barrier
        self.E_tilt = E_tilt
        self.x_well = x_well
        self.x_min = x_min
        self.x_max = x_max
        self.F_left = F_left
        self.F_right = force_asymmetry*F_left
        return None
    
    def U_0(self, x):
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
        variables = self.__dict__
        outstring = ""
        for var in variables:
            if not callable(variables[var]):
                outstring += f"{var} : {variables[var]}\n"
            else:
                x = sym.symbols("x")
                outstring += f"{var}({x}) : {variables[var](x)}"
        return outstring

    

class Ensemble(object):
    """Contains methods to do useful things with trajectory data obtained either experimentally or via simulations (the class is deliberately agnostic as to where the data comes from)."""
    
    def __init__(self, data, potential):
        """
        Initialise the object with relevant parameters.

        Parameters
        ----------
        data : xarray.DataArray of dimension = 3, with axis 0=temperature (generally, h, w, or c), axis 1 = particle number, axis 2=time
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
        self.data = data
        self.potential = potential
        self.temperatures = self.data['T'].to_numpy() # This needs to be changed so that we get it from the data itself
        self.num_temperatures = data.shape[0] # The first axis must be the number of temperatures
        self.N = data.shape[1] # The second axis must be the number of trials
        self.dt = float(self.data.t[1]-self.data.t[0])
        self.expt_length = data.shape[-1]*dt # The last axis must be the timesteps
        self.times = data['t']
        self.bins, self.heights = None, None
        self.distances = None
        self.p_values = None
        
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
        self.global_min = np.min(self.data)
        self.global_max = np.max(self.data)
        binned_active_range = np.linspace(self.global_min, self.global_max, num_bins)
        self.dx = binned_active_range[1]-binned_active_range[0]

        heights = np.apply_along_axis(lambda data: np.histogram(data, bins=binned_active_range, density=True)[0], axis=1, arr=self.data)
        
        self.bins, self.heights = binned_active_range[1:]-self.dx/2, heights

        return binned_active_range[:-1]+self.dx/2, heights # Centre the bins before returning them so that they can be plotted properly. Also throw away the first element so that the array have the same first dimensional-shape
    
    def get_CDFs(self, axis = 1, **kwargs):
        """
        Generate the cumulative density function at each point in time (by cumsumming the histogram).

        Parameters
        ----------
        axis : TYPE, optional
            DESCRIPTION. The default is 1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        bins : TYPE
            DESCRIPTION.
        CDFs : TYPE
            DESCRIPTION.

        """
        bins, heights = self.get_histograms(**kwargs)
        CDFs = self.dx*heights.cumsum(axis=axis)
        return bins, CDFs

    def get_distances(self, eqbm_boltzmann_distro=None, distance_function=distance_functions.L1):
        """
        Get the distance of the ensemble to equilibrium, defined by eqbm_boltzmann_distro.

        Parameters
        ----------
        eqbm_boltzmann_distro : 1D array, size = len(self.bins), optional
            PMF (*not* PDF) of the Boltzmann distribution at the cold temperature. The size of
        the number of bins, otherwise taking the distance WILL fail. The default is the boltzmann array for the temperature corresponding to temperature 'c'. (If 'c' is not in the list of temperatures, this must be explicitly defined.)
        distance_function: function, optional
            The distance function to use. This must take in three arguments and an optional argument called 'axis'. The default is L1.

        Returns
        -------
        distances : Array of shape (timesteps, num_temperatures)
            Distances at each timestep, for each temperature.

        """
        bins, heights = self.get_histograms() # Generate the variables we need
        if eqbm_boltzmann_distro is None:
            eqbm_boltzmann_distro = self.potential.boltzmann_array(bins, k_BT=1)/self.dx # Turn PMF into PDF
            eqbm_boltzmann_distro = np.repeat(np.reshape(eqbm_boltzmann_distro, (1,*eqbm_boltzmann_distro.shape)), self.num_temperatures, axis=0)
            eqbm_boltzmann_distro = np.reshape(eqbm_boltzmann_distro, (*eqbm_boltzmann_distro.shape, 1))
        distances = distance_function(heights, eqbm_boltzmann_distro, self.dx, axis=1)
        return distances
    
    def get_noise_floor(self, eqbm_boltzmann_distro=None, distance_function=distance_functions.L1, final_averaging_window=1000):
        """
        We define the average noise floor to be the time-averaged noise in the distance function over the last few timesteps.

        Parameters
        ----------
        eqbm_boltzmann_distro : array, optional
            Passed to get_distances(). The default is None.
        distance_function : function, optional
            Distance function passed to get_distances(). The default is distance_functions.L1.
        final_averaging_window : int, optional
            Number of timesteps to average over. The default is 1000.

        Returns
        -------
        average_noise_floor : array of floats
            Noise floor for each temperatures.
        average_noise_amplitude : array of floats
            Noise amplitude (standard deviation) for each temperature.

        """
        distances = self.get_distances(eqbm_boltzmann_distro=eqbm_boltzmann_distro, distance_function=distance_function)
        average_noise_floor = distances[...,-final_averaging_window:].mean(axis=1)
        average_noise_amplitude=distances[...,-final_averaging_window:].std(axis=1)
        
        return average_noise_floor, average_noise_amplitude
    
    def time_to_noise_floor(self, eqbm_boltzmann_distro=None, distance_function = distance_functions.L1, threshold=0.02):
        distances = self.get_distances(eqbm_boltzmann_distro=eqbm_boltzmann_distro, distance_function=distance_function)
        return np.array([self.data['t'][distances[i,:]<=threshold][0].to_numpy() for i in range(self.data.shape[0]-1)])
        
        
    
    def signal_noise_ratio(self, eqbm_boltzmann_distro=None, distance_function=distance_functions.L1, final_averaging_window=1000):
        """
        We define the signal-noise ratio to be the ratio of the initial distance to equilibrium to the noise floor.

        Parameters
        ----------
        eqbm_boltzmann_distro : array, optional
            Passed to get_distances(). The default is None.
        distance_function : function, optional
            Distance function passed to get_distances(). The default is distance_functions.L1.
        final_averaging_window : int, optional
            Number of timesteps to average over. The default is 1000.

        Returns
        -------
        average_noise_floor : array of floats
            Noise floor for each temperatures.
        average_noise_amplitude : array of floats
            Noise amplitude (standard deviation) for each temperature.

        """
        distances = self.get_distances(eqbm_boltzmann_distro=eqbm_boltzmann_distro, distance_function=distance_function)
        average_noise_floor, average_noise_amplitude = self.get_noise_floor(eqbm_boltzmann_distro=eqbm_boltzmann_distro, distance_function=distance_function,final_averaging_window=final_averaging_window)
        
        return distances[...,0]/average_noise_floor # SNR := d[p(x,0),π(x;T)]/d[p(x,t>t_eq),π(x;T)]
    
    def find_crossings(self, epsilon=0.01, eqbm_boltzmann_distro=None, distance_function=distance_functions.L1, final_averaging_window=1000):
        distances = self.get_distances(eqbm_boltzmann_distro=eqbm_boltzmann_distro, distance_function=distance_function)
        average_noise_floor, average_noise_amplitude = self.get_noise_floor(eqbm_boltzmann_distro=eqbm_boltzmann_distro, distance_function=distance_function, final_averaging_window=final_averaging_window)
        distances_h, distances_w, _ = distances
        distances_above_noise_floor_h = distances_h[distances_h>=(average_noise_floor+5*average_noise_amplitude)[0]] # Crossings after the noise floor won't tell us very much
        distances_above_noise_floor_w = distances_w[distances_h>=(average_noise_floor+5*average_noise_amplitude)[0]] # That's not a typo in the index --we want distances_above_noise_floor_w to have the same shape as distances_above_noise_floor_h
        # We want the distance to be 5 sigmas above the noise floor to guarantee (1 in 2e6 chance of being wrong) that we're not looking at noise.
        possible_crossings = distances_above_noise_floor_h[in_range(distances_above_noise_floor_h, distances_above_noise_floor_w-epsilon,distances_above_noise_floor_w+epsilon)]
        times_of_possible_crossings = self.times[distances_h>=(average_noise_floor+5*average_noise_amplitude)[0]][in_range(distances_above_noise_floor_h, distances_above_noise_floor_w-epsilon,distances_above_noise_floor_w+epsilon)]
        return possible_crossings, times_of_possible_crossings
        
    def get_average_energy(self, normalise_by_final=False, normalise_timesteps=100):
        """
        Compute the ENSEMBLE average energy for each timestep, for each temperature. This is not a distance function.

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
        energy = self.potential.U(self.data)
        energy_av = energy.mean(dim='n') # Take ensemble average, not time average
        if normalise_by_final:
            final_energy_av = energy_av[-1,-normalise_timesteps:].mean(dim='t')
            return energy_av/final_energy_av # Return <E>/E_eq instead of just <E>
        return energy_av
    
    def p_vals(self):
        if self.p_values is None: # Only generate this variable if it's not generated already
            ref = np.repeat(self.data[-1,:,0].to_numpy()[np.newaxis, :], self.data.shape[0], axis=0)[...,np.newaxis]
            self.p_values = scipy.stats.kstest(self.data, ref, axis=1).pvalue
        return self.p_values
    
    def t_eq(self, *args, threshold=None, **kwargs):
        p_values = self.p_vals()
        if threshold is None:
            threshold = p_values[-1,...].mean()-5*p_values[-1,...].std() # 'Noise amplitude' of cold p_values
            print(threshold)
        
        
        return np.array([self.data['t'][p_values[i,:]>=threshold][0].to_numpy() for i in range(self.data.shape[0]-1)]) # This list comprehension will run quickly because the number of temperatures is generally small. Exclude the last (cold) temperature because this will trivially be 0.
        
    
        
    
    def plot_sample_trajectories(self, num_trajectories = 4, temp=None):
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
        if temp is None: temp = self.data['T'][0]
        _ = self.get_histograms() # generate global_min and global_max
        plt.plot(range(int(self.expt_length//self.dt)+1), self.data.loc[temp, np.random.choice(self.N, num_trajectories),:].T)
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
        if self.num_temperatures == 1:
            ax = [ax] # Annoyingly, plt.subplots(1) does not yield a list of axes with one element.
        for i in range(self.num_temperatures):
            # ax[i].title("T =", self.data['T'][i])
            # print(self.dx)
            ax[i].bar(bins, heights_init[i,:], color = 'red', width = self.dx, label=f"t={init}"*(i==0), alpha=0.3) # Multiplying by (i==0) ensures we don't get redundant labels
            ax[i].bar(bins, heights_mid[i,:], color = 'orange', width = self.dx, label=f"t={mid}"*(i==0), alpha=0.3)
            ax[i].bar(bins, heights_end[i,:], color = 'green', width = self.dx, label=f"t={end}"*(i==0), alpha=0.3)
            if plot_init:
                heights_init_pred = self.potential.boltzmann_array(bins, k_BT=self.temperatures[i])/self.dx
                chi_squared = np.sum((heights_init[i,:] - heights_init_pred)**2 / heights_init_pred)
                print(f"Initial histogram, T = {self.temperatures[i]}: chi^2 =", chi_squared)
                ax[i].plot(bins, heights_init_pred, 'red', label=r"$\pi(x;T_0)$"*(i==0)) # fr-strings are both formatted strings and raw strings, apparently
            if plot_end:
                heights_end_pred = self.potential.boltzmann_array(bins, k_BT=self.temperatures['c'])/self.dx # Final distro is always the cold one
                ax[i].plot(bins, heights_end_pred, 'blue', label=r"$\pi(x;T_c)$"*(i==0))
                chi_squared = np.sum((heights_end[i,:] - heights_end_pred)**2 / heights_end_pred)
                print(f"Final histogram, T = {self.temperatures[i]}: chi^2 =", chi_squared)
            ax[i].set_ylabel(r"$p(x,t)$")
        if i == self.num_temperatures-1:
            fig.legend()
        plt.tight_layout()
        plt.xlabel("$x$")

        plt.show()
        return bins, [heights_init, heights_mid, heights_end]
    
    def animate(self, T=None, num_bins=30, num_animated_frames = 500, set_const_height = True, use_log_time=True, frame_decay_const = 100, max_left=-1, max_right=3):
        """
        Animate p(x,t)

        Parameters
        ----------
        T : int, optional
            The temperature to use. The default is the hot temperature.
        num_bins : int, optional
            Number of bins to use in the histogram. The default is 200.
        num_animated_frames : int, optional
            The number of frames to animate. Keep it small for efficient rendering. The default is 500.
        set_const_height : bool, optional
            Stop matplotlib from resizing the plot when the histogram gets taller or shorter. The default is True.
        use_log_time : bool, optional
            Use logarithmic time to animate the . The default is True.
        frame_decay_const : int, optional
            Controls how fast the exponential frame rate goes. The default is 100.

        Returns
        -------
        matplotlib animation
            Animation of p(x,t).

        """
        if T is None: T = float(self.data['T'].max())
        binned_active_range = np.linspace(self.potential.x_min, self.potential.x_max, num_bins)
        fine_active_range = np.linspace(self.potential.x_min, self.potential.x_max, num_bins*10)
        binned_initial_distro = self.potential.boltzmann(fine_active_range, T)
        binned_final_distro = self.potential.boltzmann(fine_active_range, 1) # Cold temperature is always 1
        num_times = len(self.data.t)
        
        fig, ax = plt.subplots()
        bins, all_heights = self.get_histograms(num_bins=num_bins)
        patches = ax.bar(bins, all_heights[0,:,0]/self.N, width=bins[1]-bins[0])
        ax_height = np.max(binned_final_distro + 0.02)
        ax.set_ylim(0,ax_height)
        
        exponential_frame_func = lambda x, x_0: (self.expt_length/self.dt)*(np.exp(x/x_0)-1)/(np.exp(num_animated_frames/x_0)-1) # Function to pick out a large number of frames in the beginning and a small number of frames in the end
        if use_log_time:
            animated_frames = np.rint(exponential_frame_func(np.arange(0,num_animated_frames+1, 1), frame_decay_const)).astype('int')
        else:
            animated_frames = np.array(range(num_times)[::len(num_times)//num_animated_frames])
        # Time in the video will be on a log scale, primarily for rendering efficiency
        
        ax.set_xlim((max_left,max_right))
        ax.set_xlabel("x")
        ax.set_ylabel("p(x,t)")
        ax.plot(fine_active_range, binned_initial_distro, 'r')
        ax.plot(fine_active_range, binned_final_distro, 'g')
        ax.set_title(r"t = 0 $\mu$s")
        
        # Update function for the animation
        def update(frame_number):
            # Get the data for the current frame
            heights, _ = all_heights[0,:,frame_number-1], bins
            # Update the histogram data
            for i in range(len(patches)):
                patches[i].set_height(heights[i])
            ax.set_title(fr"t = {10*frame_number} $\mu$s")
            if not set_const_height:
                if frame_number % 50 == 0:
                    ax.set_ylim(0,ax_height) # Adjust the axis height every fifty frames
            elif set_const_height:
                ax.set_ylim(0,ax_height)
            return patches
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=animated_frames, interval=10, blit=False, cache_frame_data=True)
        plt.show()
        return ani

    def export_raw_data(self, filename, extension = ".csv"):
        """
        Convert the ensemble data into as many csv files as there are temperatures.

        Parameters
        ----------
        filename : str
            The name of the file to export to. Each file name will have the temperature data on it
        extension : str, optional
            The filetype. Default is csv.

        Returns
        -------
        None.

        """
        for T in tqdm(self.temperatures):
            df_T = self.data.loc[T].to_pandas().T # Transpose for cleanliness
            filename_T = filename + "_T=" + str(T) + ".csv" # Construct the full filename
            df_T.to_csv(filename_T)
         # TODO

def run_mpemba_analysis(filenames, potential, protocol_time=7e-2, dt=1e-5, column_names = ['x', 't', 'V', 'state', 'drift'], temperatures=[1000,12,1], gut_checks=True):
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
    data = file_processing.extract_file_data(filenames, protocol_time=protocol_time, dt=dt, column_names=column_names, temperatures=temperatures)
    # data has the form we want of an ensemble
    ensemble = Ensemble(data, potential)
    if gut_checks:
        ensemble.gut_checks()
    distances = ensemble.get_distances()
    energies = ensemble.get_average_energy()
    times = ensemble.data.t
    fig2, ax2 = plt.subplots(2)
    ax2[0].loglog(times, distances.T)
    ax2[1].loglog(times, energies.T)
    return times, distances, energies


def backwards_cumulative_max(data, axis=0):
    """
    Calculate the cumulative maximum of 'data', starting at the end and moving to the beginning. Eg, cummax([1,3,9,3,4,6,1,3,6,3,1,2]) = [9,9,9,6,6,6,6,6,6,3,2,2]

    Parameters
    ----------
    data : vector of numerics
        Data to find the cumulative max of
    axis : int or None, optional
        Axis to cummax over. The default is 0.

    Returns
    -------
    cummaxed_data : vector of numerics (same shape as data)
        Data with the cummax operation applied to it.

    """
    reverse = np.flip(data, axis=axis) # Use flip instead of slicing tricks so that we can specify the axis
    cummaxed_reverse = np.maximum.accumulate(reverse, axis=axis)
    cummaxed_data = np.flip(cummaxed_reverse, axis=axis) # After cummaxing the reversed data, flip it again so the ordering is correct
    return cummaxed_data

if __name__ == '__main__':
    import os
    os.chdir("..")