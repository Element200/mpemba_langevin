LabVIEW code was mostly written by Avinash Kumar with some modifications by me for my experiments.

Python code contains methods to integrate the Langevin equation. 

To install this code, download the zip folder, unzip, etc etc. You're very smart, I'm sure you know how to download files. Make sure all the files are in the same directory 

Use `import mpemba` to import all of the methods you need to run Langevin simulations and such. For solving the FPE, use ```import fokker_planck```. 

Special libraries: you will need the latest versions (at least as of Dec 2025) of
1. SciPy (for a bunch of curve fitting things and stuff)
2. SymPy (Some methods use analytic derivative computations so that we have more generalisable methods)
3. Xarray (Pandas on steroids – this allows you to have pandas-like objects in more than two dimensions)
4. tqdm (for clean progress bars. You don't really need this but I like to have it. If you don't want it, just delete all references to ```tqdm``` in the code)

in addition to other basic libraries like NumPy, Pandas, and so on which you probably already have.

There are two ways to run this code: you either generate data by simulation or you load data from a datafile like that produced by labVIEW code like that in the folder. 
No matter what you do, you need a Potential child class to tell the code how to calculate distances and such. There are a couple of pre-baked ways to do this; these are defined in ```special_potentials```.
To generate your own potential, add it to the `special_potentials.py` file.

Here's some example code to generate a simulation:

```python
  import mpemba
  potential = mpemba.special_potentials.AsymmetricDoubleWellPotential(E_barrier=2, E_tilt=1.3, x_well=0.5) # Other necessary params are defined by default
  data = mpemba.simulation_methods.run_mpemba_simulations(k_BTs=[1000,12,1], N=10_000, potential=potential) # Specify initial temperatures and number of particles, plus the potential object
```

Here's some example code to load data from a file
```python
  import mpemba
  potential = mpemba.special_potentials.AsymmetricDoubleWellPotential(E_barrier=2, E_tilt=1.3, x_well=0.5) # Other necessary params are defined by default
  data = mpemba.file_processing.extract_file_data_v2(filename="example.txt", protocol_time=7e-2).x # Use mpemba.file_processing.extract_file_data instead of extract_file_data_v2 if the version of "Mpembe_exp_sane" is v6 or lower
  # extract_file_data_v2 returns an xarray.Dataset, not an xarray.DataArray, so you have to explicitly pull out the x data
```

`data` is going to be an `xarray.DataArray`. You need to pass this into an Ensemble object to do anything interesting with it

```python
  ensemble = mpemba.Ensemble(data, potential)
  ensemble.gut_checks() # Will plot a bunch of histograms so that you can check whether stuff is working properly
  ensemble.plot_distances() # Will plot the distance curves
```

Other methods are documented in the code itself. If you ask me super nicely, I can explain it to you as well.
