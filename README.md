Python code contains methods to integrate the Langevin equation. 

## Installation
To install this code, download the zip folder, unzip, etc. Or clone the repo using the standard way of cloning repos. Make sure all the files are in the same directory as whatever python file you use to import the code.  

## Requirements
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

## Basic philosophy and jargon
This code is developed with the experimental constraints listed in [Kumar and Bechhoefer, 2020](https://www.nature.com/articles/s41586-020-2560-x) in mind -- namely, the requirement that there is a certain maximum force $F_{\max}$ that the force profile may not exceed. The object `BoundedForcePotential` encapsulates this constraint. In order to define a special potential with these constraints yourself, you must make a child class of the `BoundedForcePotential` object. If you don't want to use bounded forces, you must use the `UnboundedForcePotential` object instead. Include any useful parameters in the `__init__` statement. Once you're done, use `super().__init__()` to initialise all of the parent methods. Define a function called `U_0(self, x)` (the name must be exact). This will contain the basic shape (without maximum slopes). For example, for a double-welled quartic, here is some sample code:

```python
  class SamplePotential(mpemba.potential_methods.BoundedForcePotential):
    def __init__(self, E_1, x_well):
      self.E_1 = E_1
      self.x_well = x_well
      super().__init__() # Initialise the parent
    def U_0(self, x):
      return self.E_1*(1-x**2/x_well**2)**2 
```

You may also want to define a `__str__` and `__repr__` for readability, but you don't have to. Once you've defined this class, the `potential_methods`  parent classes will define a whole bunch of methods. For bounded force potentials, this will define a new class method `SamplePotential.U(x)` which caps the maximum forces, as well as a set of new class methods `SamplePotential.F_0(x)` and `SamplePotential.F(x)` that can compute the unbounded and bounded forces respectively. For unbounded force potentials, `U(x)` and `U_0(x)` will trivially be equal. Additionally, a number of nice tools such as the eigenvalues and eigenfunctions associated with the Fokker-Planck equation are also generated automatically (automatic pregeneration is important so that the code runs efficiently). You pass the potential object to all simulations and data analysis code; this code will expect these automatically generated methods to run. Some special potentials I commonly use are defined in `mpemba.special_potentials`. 

Once the potential is well-defined, you can either run simulations or process data using the code documented above. This should be compiled into an `xarray.DataArray` with appropriate dimension names, that you can then send the data as well as the potential to an `Ensemble` object. The `Ensemble` object contains a number of useful methods for histogramming, computing PDFs, computing distances to equilibrium, and so on. The ensemble will store histogram and distance data that it generates so that you won't have to wait ~10 seconds every time you call the distances. Additionally, the `Ensemble` object uses a custom-built histogram function that's designed to very quickly histogram data in the required format. 

Other methods are documented in the code itself. If you ask me super nicely, I can explain it to you as well.
