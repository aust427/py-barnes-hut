# N-body: the Barnes-Hut algorithm with quasi-peroidic boundary conditions

The Barnes-Hut algorithm ([Barnes & Hut 1986](https://ui.adsabs.harvard.edu/abs/1986Natur.324..446B/abstract)) was a novel algorithm that utilizes hierarchical tree structures to reduce particle-particle calculations from O(n^2) to O(n log(n)), vastly speeding up N-body simulations. This module represents a Python-based implementation of the algorithm via a self-implemented QuadTree data structure. ODE integration is conducted via the leapfrog method.

For more information regarding quasi-periodic boundary conditions with Barnes-Hut, we refer the reader to [Bouchet & Hernquist 1988](https://ui.adsabs.harvard.edu/abs/1988ApJS...68..521B), specifically Fig. 1 and Section 2c. 

In-depth documentation regarding the free-parameters of the simulation can be seen in the notebook `example.ipynb`. A cursory glance can be seen via the following command and output: 

```
python3 simulation.py --help
usage: simulation.py [-h] [-f FILE] [-L L] [-N N] [-rand_type RAND_TYPE [RAND_TYPE ...]] [-opening_angle OPENING_ANGLE] [-softening SOFTENING] [-mass_scale MASS_SCALE] [--store]

2D Barnes-Hut particle simulation code with quasi-periodic boundary conditions.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  File to particles containing positions and velocity, shape: (N, 4).
  -L L, --L L           Side length of the box, spanning [-L/2, L/2].
  -N N, --N N           Number of particles in the simulation.
  -rand_type RAND_TYPE [RAND_TYPE ...], --rand_type RAND_TYPE [RAND_TYPE ...]
                        type of random distribution to use for creating the particle data. 1 - uniform, 2 - normal
  -opening_angle OPENING_ANGLE, --opening_angle OPENING_ANGLE
                        the opening angle which determines if the algorithm looks at cluster-level or individual-level masses. Fiducial value is 0.5
  -softening SOFTENING, --softening SOFTENING
                        softening length of the simulation, limiting the max gravitational interaction. Recomended value is 1e-3
  -mass_scale MASS_SCALE, --mass_scale MASS_SCALE
                        path to particle data
  --store               boolean to store particle results results
```  

A presentation based summary can be viewed via the following [Google slides link](https://docs.google.com/presentation/d/1D0xXtax_BdveVjTFs1x-myLu-NqF3165VPx6DBuV6GU/edit?usp=sharing).

## Running the simulation 

The module `simulation.py` can be run via command line via the following example: 

```simulation.py -L 12 -opening_angle 0.2 -softening 1e-2 -N 14 -rand_type 2 0.2 0.3 --store```

Where inputs are processed using the `argparse` module. The following output should be produced: 

```
Particle data generated via normal distribution with the following parameters: 
sigma = 0.2; mu = 0.3.
Simulation running with the following parameters:
L = 12.0; N = 14; theta = 0.2; softening = 0.01; M_scale = 1000.0.
Particle data will be stored in ./data/L12.0n14

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1001/1001 [00:05<00:00, 175.51it/s]
Parameter file written to: ./data/L12.0n14
```

The notebook `example.ipynb` has a step-by-step walkthrough and explanation of the free parameters of the simulation, how particles are generated, and how the simulation operates. 

Example data can be seen and used from the following directory: `./data/example/particles.csv`. 

## Generating visualizations 

The module `visualizer.py` can be run via command line via the following example (provided simulation data exists to be modeled): 

```visualizer.py -param ./data/example/param.pkl --draw_trees```

