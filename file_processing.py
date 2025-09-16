# -*- coding: utf-8 -*-
"""
file_processing.py.

Created on Wed Jun 11 14:46:01 2025
v1.4

@author: sane
Contains methods to convert file data into an object that the Ensemble class in mpemba.py can operate on.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import scipy

def chunk_splitter(dataframe, dt=1e-5, state_to_extract='protocol', state_phase_lag=0, states = {'calibration' : 0, 'positioning' : 1, 'protocol' : 2}):
    """
    Turn one giant dataarray of particle trajectories into an xarray DataArray that stores each individual particle's trajectory in a new dimension --- this gives us something we can work with.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The raw data with column names set appropriately. This code will fail if certain column names do not exist.
    state_to_extract : str, optional
        state to extract. The code will throw away the other states to save memory.
        If state_to_extract represents a state with non fixed-length data, this code will break.
    state_phase_lag : int, optional
        If the state resets at the moment the time variable is also reset to 0, this is equal to zero. If it lags by 1, set this to one, etc. The default is 0 (i.e. assuming that the state variable resets properly with time).
    states : dict, optional
        Dictionary with key-value pairs explaining state codes. state_to_extract must be one of the keys. This code will break if you try to extract a state with uneven size (eg. the positioning state).
            
        

    Returns
    -------
    chunks : xarray.DataArray with dims-=[n,t,col]
        DataArray of dimensions (n, t, 2) containing each 'chunk' --- a trajectory for a single particle. col, the third dimension, contains the position data in the 0th column and state data in the 1st column. On further cleaning, the state data can be thrown away so that we save some more memory. We throw away time data (redundant; changed to an index in the xarray) and voltage data (which we don't ever actually use) to save some memory.

    """
    # Each chunk contains one particle's trajectory
    # To save memory, we will get rid of the time column (assuming all time columns are identical)
    masked_data = dataframe[dataframe['state']==states[state_to_extract]]
    forward = dataframe[1:].reset_index(drop=True)
    pivots = dataframe[(forward['state']==states['calibration']) & (dataframe['state']==states['protocol'])].index + 1 - state_phase_lag
    chunks = []
    expected_length = len(masked_data.loc[pivots[0]:pivots[1]])
    for i in range(len(pivots)-1):
        chunk = masked_data.loc[pivots[i]:pivots[i+1]]
        if len(chunk) == expected_length:
            chunks.append(chunk) # Avoid adding a trajectory if there's some weirdness in the chunk splitting that causes nonuniformity.

    chunks = np.array(chunks)
    data_cols = list(dataframe.columns)
    cols_to_extract = {col: data_cols.index(col) for col in data_cols if (not col in ['t','state'])} # We extract all of the columns except for time and state
    t_index = data_cols.index('t') # Get index of the time column
    times = chunks[0,:,t_index]*dt # Assumes all time columns are identical
    return xr.Dataset(data_vars={col: (['n','t'],chunks[...,cols_to_extract[col]]) for col in cols_to_extract.keys()}, coords={'t':times}) # Throw away the voltage and time data (time is redundant between all chunks) and also state data (because we filtered it so it's all =2)

def chunk_splitter_v2(dataframe, temperatures, dt=1e-5, state_to_extract='protocol', state_phase_lag=0, states = {'calibration' : 0, 'positioning' : 1, 'protocol' : 2}):
    """
    Turn one giant dataarray of particle trajectories into an xarray DataArray that stores each individual particle's trajectory in a new dimension --- this gives us something we can work with. Updated to work with interwoven temperature data.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The raw data with column names set appropriately. This code will fail if certain column names do not exist.
    state_to_extract : str, optional
        state to extract. The code will throw away the other states to save memory.
        If state_to_extract represents a state with non fixed-length data, this code will break.
    state_phase_lag : int, optional
        If the state resets at the moment the time variable is also reset to 0, this is equal to zero. If it lags by 1, set this to one, etc. The default is 0 (i.e. assuming that the state variable resets properly with time).
    states : dict, optional
        Dictionary with key-value pairs explaining state codes. state_to_extract must be one of the keys. This code will break if you try to extract a state with uneven size (eg. the positioning state).
            
        

    Returns
    -------
    chunks : xarray.DataArray with dims-=[n,t,col]
        DataArray of dimensions (n, t, 2) containing each 'chunk' --- a trajectory for a single particle. col, the third dimension, contains the position data in the 0th column and state data in the 1st column. On further cleaning, the state data can be thrown away so that we save some more memory. We throw away time data (redundant; changed to an index in the xarray) and voltage data (which we don't ever actually use) to save some memory.

    """
    # Each chunk contains one particle's trajectory
    # To save memory, we will get rid of the time column (assuming all time columns are identical)
    masked_data = dataframe[dataframe['state']==states[state_to_extract]]
    forward = dataframe[1:].reset_index(drop=True)
    pivots = dataframe[(forward['state']==states['calibration']) & (dataframe['state']==states['protocol'])].index + 1 - state_phase_lag
    chunks = []
    expected_length = len(masked_data.loc[pivots[0]:pivots[1]])
    for i in range(len(pivots)-1):
        chunk = masked_data.loc[pivots[i]:pivots[i+1]]
        if len(chunk) == expected_length:
            chunks.append(chunk) # Avoid adding a trajectory if there's some weirdness in the chunk splitting that causes nonuniformity.

    chunks = np.array(chunks)
    data_cols = list(dataframe.columns)
    cols_to_extract = {col: data_cols.index(col) for col in data_cols if (not col in ['t','state', 'T'])} # We extract all of the columns except for time and state
    t_index = data_cols.index('t') # Get index of the time column
    T_index = data_cols.index('T')
    num_protocols = np.inf
    split_chunks = []
    for T in T_index:
        split_chunk = chunks[chunks[...,data_cols.index('T')]==T]
        split_chunks.append(split_chunk)
        if num_protocols > len(split_chunk):
            num_protocols = len(split_chunk)
    adjusted_chunks = np.array([split_chunks[:num_protocols, ...]])
    times = adjusted_chunks[0,:,t_index]*dt # Assumes all time columns are identical
    return xr.Dataset(data_vars={col: (['T', 'n','t'],adjusted_chunks[...,cols_to_extract[col]]) for col in cols_to_extract.keys()}, coords={'t':times, 'T': temperatures}) # Throw away the voltage and time data (time is redundant between all chunks) and also state data (because we filtered it so it's all =2)

def extract_file_data(filenames, protocol_time, dt=1e-5, column_names=['x','t','drift','state','x0'], cols_to_extract= ['x'], temperatures = [1000,12,1]):
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
        Names of the columns in the output data. The default is ['x','t','drift','state', 'x0'].
    temperatures : list of numerics, optional
        The initial temperatures, sorted *in descending order*. The default is [1000,10,1].

    Returns
    -------
    array : xarray.DataArray
        Array with appropriate dimensions with the same structure as after the simulation, so that further analysis on either data is identical.

    """
    if not 1 in temperatures:
        raise ValueError("No reference temperature!")
    temperatures = sorted(temperatures, reverse = True) # Sort temperatures in descending order
    chunks = {}
    n_min = np.inf # Minimum number of particles in an array. Initially set to infinity because that's much higher than any experiment will realistically produce
    for filename in filenames:
        chunks[filename] = chunk_splitter(pd.read_table(filename, names=column_names, usecols=[*cols_to_extract,'t','state']))
        n = int(chunks[filename]['n'][-1]) # Number of particles in chunks['filename']
        if n < n_min:
            n_min = int(chunks[filename]['n'][-1]) # Once this for loop is done running, n_min will hold the lowest number of particles in the dataarray
    array = []
    for var in chunks[filenames[0]].data_vars:
        array.append(xr.concat([chunks[filename][var].loc[:n_min,...].assign_coords(t=np.arange(0,protocol_time,dt)) for filename in filenames], pd.Index(temperatures, name='T')))
    # array = xr.DataArray(data, dims=['T','n','t'], coords={'T':temperatures, 't': np.arange(0,protocol_time,dt)})
    return array

def extract_file_data_v2(filename, protocol_time, dt=1e-5, column_names=['x','t','drift','state','x0','T'], cols_to_extract= ['x'], temperatures = [1000,12,1]):
    """
    Extract file data from new version of protocol that interweaves temperatures.

    Parameters
    ----------
    filename : str
        directory+filename of data.
    protocol_time : numeric
        Length of the protocol.
    dt : numeric, optional
        Timestep (must be less than protocol_time, obviously). The default is 1e-5.
    column_names : vector of str, optional
        Columns in the data. The length of this must match the number of columns in the file. The default is ['x','t','drift','state','x0','T'].
    cols_to_extract : vector of str, optional
        A subset of column_names -- this is the data you want to extract. The default is ['x'].
    temperatures : vector of numeric, optional
        Temperatures of the data, scaled by one of the temperatures. At least one of the temperatures must be 1. The default is [1000,12,1].

    Returns
    -------
    array : xarray.DataArray.
        Array with appropriate dimensions with the same structure as after the simulation, so that further analysis on either data is identical.
    
    """
    if not 1 in temperatures:
        raise ValueError("No reference temperature!")
    temperatures = sorted(temperatures, reverse = True) # Sort temperatures in descending order
    chunks = {}
    n_min = np.inf # Minimum number of particles in an array. Initially set to infinity because that's much higher than any experiment will realistically produce
    chunks = chunk_splitter(pd.read_table(filename, names=column_names, usecols=[*cols_to_extract,'t','state','T'])) # We need 't' and 'state' to properly split the data; we need 'T' to figure out how to split the data further
    array = xr.DataArray(chunks)
    # for var in chunks[filenames[0]].data_vars:
    #     array.append(xr.concat([chunks[filename][var].loc[:n_min,...].assign_coords(t=np.arange(0,protocol_time,dt)) for filename in filenames], pd.Index(temperatures, name='T')))
    # array = xr.DataArray(data, dims=['T','n','t'], coords={'T':temperatures, 't': np.arange(0,protocol_time,dt)})
    return array

def load_brownian_data(filenames, column_names=['x','t','drift','state','F','x0','T'], cols_to_extract=['x']):
    """
    Load Brownian calibration data and extract the position data.

    Parameters
    ----------
    filenames : list of str
        filenames of brownian data.
    column_names : list of str, optional
        column names of the data. The length of this list must match the number of columns. The default is ['x','t','drift','state','F','x0'].
    cols_to_extract : list of str, optional
        list of useful data. The default is ['x'], and I'm not sure why you'd want to extract anything else but in principle you can.

    Returns
    -------
    pd.DataFrame of trajectories. 
        All trajectories are listed by index -- x0, x1, etc.

    """
    tables = []
    length = np.inf # Because any length is less than infinity
    for i in range(len(filenames)):
        table = pd.read_table(filenames[i], names = column_names, usecols=cols_to_extract).rename(columns={'x':f'x{i}'})
        trial_length = table[~np.isnan(table.loc[:,f'x{i}'])].shape[0]
        if trial_length < length: 
            length = trial_length # Find the smallest data length so that we don't have nans in the output
        tables.append(table)
    data = pd.concat(tables, axis=1)
    return data.iloc[:length, :] # To avoid nan values if importing unevenly sized data

class BrownianEnsemble(object):
    def __init__(self, data, dt=1e-5):
        self.data = data
        self.dt=dt
    def get_PSD(self, noverlap=50):
        f, P = scipy.signal.welch(self.data, fs=1/self.dt, noverlap=noverlap, axis=0, nperseg=1e6)

def fit_aliased_lorentzian(f, P, dt=1e-5, noverlap=50):
    def _aliased_lorentzian(f, c, dx, dt=dt):
        return (dt*dx**2)/(1+c**2-2*c*np.cos(2*np.pi*f*dt))
    popt, pcov = scipy.optimize.curve_fit(_aliased_lorentzian, xdata=f, ydata=P, p0=[1,0.05])
    c, dx = popt
    f_nyq = 1/dt/2 #f_s/2
    f_c = -f_nyq*np.log(c)/np.pi
    D = (dx**2)/(1-c**2)*2*np.pi*f_c
    return f_c, D*noverlap/(noverlap-2) # Correct for systematic errors due to LSQ fits -- see Norrelykke and Flyvbjerg (2003)

def aliased_lorentzian(f, f_c, D, dt=1e-5):
    f_nyq = 1/dt/2
    c = np.exp(-np.pi*f_c/f_nyq)
    dx = np.sqrt((1-c**2)*D/(2*np.pi*f_c))
    return (dt*dx**2)/(1+c**2-2*c*np.cos(2*np.pi*f*dt))

def boltzmann_fit(x, num_bins=20):
    heights, bins = np.histogram(x, bins=num_bins, density=True)
    dx = bins[1]-bins[0]
    x_range = bins[:-1]+dx/2 # Centre the bins
    popt, pcov = scipy.optimize.curve_fit(lambda x, k: np.sqrt(k/(2*np.pi))*np.exp(-0.5*k*x**2), xdata=x_range, ydata=heights)
    return popt

def equipartition(x):
    sigma_x = np.std(x)
    return 1/sigma_x**2 # Assumes no measurement noise

def load_processed_data(filenames, temperatures=[1000,12,1]):
    """
    Load trajectory data that has already been processed to remove the junk. Can also load processed simulation data. Do not use this to import raw data. All of the time columns must be the same and the file format must be a csv with the first row corresponding to the ensemble index and the first column corresponding to the time.

    Parameters
    ----------
    filenames : list of str
        Names of processed files.
    temperatures : list of numerics
        Temperatures corresponding to each file (in the same order).

    Returns
    -------
    xr.DataArray
        Processed data as an xarray

    """
    assert len(filenames) == len(temperatures), "Mismatch between number of files and number of temperatures!"
    data = []
    for i in tqdm(range(len(filenames))):
        data.append(pd.read_csv(filenames[i], header = 0, index_col=0).T)
    t = data[-1].columns # Assumes the time columns are all the same
    return xr.DataArray(data, dims=('T', 'n', 't'), coords = {'T': temperatures, 't':t})