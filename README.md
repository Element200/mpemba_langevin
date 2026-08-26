Python code contains methods to integrate the Langevin equation. This library will likely not be updated much anymore (unless you want to fork it).

## Installation
To install this code, download the zip folder, unzip, etc. Or clone the repo using the standard way of cloning repos. Make sure all the files are in the same directory as whatever python file you use to import the code, or use
```python
  import sys
  sys.path.append("path/to/directory/with/python/files")
```

## Requirements
Use `import mpemba` to import all of the methods you need to run Langevin simulations and such. For solving the FPE, use ```import fokker_planck```. 

Special libraries: you will need the latest versions (at least as of Dec 2025) of
1. SciPy (for a bunch of curve fitting things and stuff)
2. SymPy (Some methods use analytic derivative computations so that we have more generalisable methods)
3. Xarray (Pandas on steroids – this allows you to have pandas-like objects in more than two dimensions)
4. tqdm (for clean progress bars. You don't really need this but I like to have it. If you don't want it, just delete all references to ```tqdm``` in the code)
5. Polars (faster Rust-based data structures for lightning-fast loading of very large .txt files. This is more recent but I got annoyed about waiting 4+ minutes for pandas to load in a txt file; Polars cut it down to 30 seconds.)

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

Quench techniques are also objects, and precompute a bunch of useful functions if you're doing coordinate transformation stuff (which would be really awesome if you did; please email me if you do because that would be sick AF and I wanna know). `quench_methods.py` contains the base `QuenchProtocol` class and a number of sample quench functions. Here's an example of how to create a few.
```python
  import mpemba

  instantaneous_quench = mpemba.quench_methods.InstantaneousQuench()
  exponential_quench = mpemba.quench_methods.ExponentialQuench(tau=65) # I used tau instead of Lambda here. You can change this if you like -- think of it as a mini-assignment!
```

## For the new student
### Some prerequisiste knowledge
1. You need to know Python. In other news, water: wet. I mean, you probably already know some Python but I guess this is my version of a closure axiom.
2. You're going to need to have a decent understanding of object-oriented programming in Python. I know this isn't often taught in physics undergrad courses but it's valuable knowledge (and makes you marginally more employable) so you might as well. I find that MIT OCW 6.0001 has a bunch of really nice lectures (and more importantly, assignments) that you can use to teach yourself this stuff fairly quickly.
3. I had to run a lot of this code very repetitively so my focus when designing it was to minimise the number of lines I'd have to write while analysing data. That might have come at the cost of readability, unfortunately. Sorry!
4. Some knowledge of how to use JIT libraries like Numba would be pretty useful too. If your kernel crashes, 90% of the time it's going to be Numba's fault.
5. Very little AI was used to write this code. I find that it just doesn't work too well (and also have general political objections to AI), and it's best to just *git gud*, as we say in the Silksong community. If you think it might help you by all means use it but be aware that you are in territory where it may well fail. I dunno, maybe I'm just a Luddite.
6. I use Spyder to develop the .py code and VSCodium to run ipython sandbox stuff. You don't have to, of course, but maybe you find this info useful for some reason.
7. I didn't bother to make this an actual package you can download from PIP but maybe if you know how to do that you can give it a shot. For now just use `sys.path.append` at the start of your code or make sure whatever you're running is in the same working directory.
8. You're going to be going back and forth between your personal computer and the lab computer that runs the experiment. I found it way too annoying to keep physically transferring files with a pen drive, so I just use Git for file-syncing and version control. I highly recommend you learn Git if you don't know how to use it already.

### Some little assignments
Here's some little assignments you could try to make the code your own and familiarise yourself with its structure
1. Build a `Potential` object. You could try `AsymmetricDoubleWellPotential()` with default parameters as a jumping-off point.
2. Use the Potential either to run a simulation or load in a dataset.
3. Build an `Ensemble` object.
4. Plot the distance curves. Is there a Mpemba effect?
5. Try plotting the average energy (Look for the method that does this). Does it decay monotonically?
6. Try running another simulation with a non-instantaneous quench. Plot the distance curves again
7. How do these results stack up against the solution to the Fokker-Planck equation? Try plotting the distance curves and overlaying the distances you get from the analytic solution to the FPE.
8. Try making your own special potential and quench methods! Use the code in `special_potentials.py` as a template.

### TODO
Once you know what you're doing with the code, here's a few patch jobs you could try 
1. Make this whole mini-library a proper PIP package with actual version control.
2. I never could get Numba to play nicely with the central Langevin simulations because Numba absolutely hates custom objects and the code to make my potential objects Numba-friendly would've been super janky to implement. Still, it might be nice to have it; it'd cut down on simulation time by quite a bit (although it only takes like 30 seconds for a 10k particle sim so it's not *terrible*).
3. Implement techniques for more robust FPE solving. Right now I find that setting an error tolerance is always a bit of a tightrope act between ensuring the integrator doesn't fail and ensuring that it finishes running quickly enough. Maybe you can also think of something that'd make this guesswork automatic.
4. Implement techniques for 2D potentials. This shouldn't be *super* hard; in principle you may only need to change a few lines of code in `simulation_methods` and `potential_methods`. Still, it might create a bunch of downstream issues.
5. Make the file-processing techniques faster. I added a couple of techniques that use the Polars library for faster file loading but I find in practice that it doesn't *consistently* cut down on loading time; it only runs really fast if there's enough free RAM. (This is why restarting the IPython kernel can really speed up the bits of code where it loads in the dataset). The writeout libraries are particularly slow, and can sometimes take almost a minute to convert a 16 GB raw datafile into three ~400 MB csv files. Maybe also try to use faster file formats like Parquet or Pickle.
6. Run some more comprehensive tests on the virtual potential Langevin stuff. I did a few tests myself but I don't know if I fully trust it yet. If you define a potential with really high curvature in some areas you should see the `langevin_simulation_virtualPotential` and `langevin_simulation` results start to diverge.
7. Add your own functionality to the code I wrote!

Good luck!
