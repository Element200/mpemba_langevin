# -*- coding: utf-8 -*-
"""
file_processing.py.

Created on Wed Jun 11 14:46:01 2025
v1

@author: sane
Contains methods to convert file data into an object that the Ensemble class in mpemba.py can operate on.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

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
        Dictionary with key-value pairs explaining state codes. state_to_extract must be one of the keys. This code will break if 
            
        

    Returns
    -------
    chunks : xarray.DataArray with dims-=[n,t,col]
        DataArray of dimensions (n, t, 2) containing each 'chunk' --- a trajectory for a single particle. col, the third dimension, contains the position data in the 0th column and state data in the 1st column. On further cleaning, the state data can be thrown away so that we save some more memory. We throw away time data (redundant; changed to an index in the xarray) and voltage data (which we don't ever actually use) to save some memory.

    """
    # Each chunk contains one particle's trajectory
    # To save memory, we will get rid of the time column (assuming all time columns are identical)
    masked_data = dataframe[dataframe['state']==states[state_to_extract]]
    forward = dataframe[1:].reset_index(drop=True)
    pivots = dataframe[(forward['state']==states['calibration']) & (dataframe['state']==states['protocol'])].index + 1 - state_phase_lag # Assumes 'calibration' is the first state and 'protocol' is the last.
    chunks = []
    for i in range(len(pivots)-1):
        chunks.append(masked_data.loc[pivots[i]:pivots[i+1]])
    chunks = np.array(chunks)
    data_cols = list(dataframe.columns)
    cols_to_extract = {col: data_cols.index(col) for col in data_cols if (not col in ['t','state'])} # We extract all of the columns except for time and state
    t_index = data_cols.index('t') # Get index of the time column
    times = chunks[0,:,t_index]*dt # Assumes all time columns are identical
    return xr.Dataset(data_vars={col: (['n','t'],chunks[...,cols_to_extract[col]]) for col in cols_to_extract.keys()}, coords={'t':times}) # Throw away the voltage and time data (time is redundant between all chunks) and also state data (because we filtered it so it's all =2)

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
        Array with appropriate dimensions in with the same structure as after the simulation, so that further analysis on either data is identical.

    """
    chunks = {}
    n_min = np.inf # Minimum number of particles in an array. Initially set to infinity because that's much higher than any experiment will realistically produce
    for filename in filenames:
        chunks[filename] = chunk_splitter(pd.read_table(filename, names=column_names, usecols=[*cols_to_extract,'t','state']))
        n = int(chunks[filename]['n'][-1]) # Nymber of particles in chunks['filename']
        if n < n_min:
            n_min = int(chunks[filename]['n'][-1]) # Once this for loop is done running, n_min will hold the lowest number of particles in the dataarray
    # data_array = {var: xr.zeros.zeros((len(filenames), n_min, int(protocol_time//dt))) for var in chunks[filenames[0]].data_vars}
    # for i in range(len(filenames)):
    #     filename = filenames[i]
    #     for var in chunks[filename].data_vars:
    #         data_array[var][i,...] = chunks[filename][var].loc[:n_min, ...] # This ensures that data will contain fixed-length data along the n dimension as well
    array = []
    for var in chunks[filenames[0]].data_vars:
        array.append(xr.concat([chunks[filename][var].loc[:n_min,...].assign_coords(t=np.arange(0,protocol_time,dt)) for filename in filenames], pd.Index(temperatures, name='T')))
    # array = xr.DataArray(data, dims=['T','n','t'], coords={'T':temperatures, 't': np.arange(0,protocol_time,dt)})
    return array

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

if __name__ == '__main__':
    filenames = [f"../Data/2025_06_30/data_{key}.txt" for key in ['h','w','c']]
    data = extract_file_data(filenames, 7e-2, cols_to_extract=['x','drift'])