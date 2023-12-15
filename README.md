# N-body: The Barnes-Hut Algorithm with Quasi-Peroidic Boundary conditions

The Barnes-Hut algorithm ([Barnes & Hut 1986](https://ui.adsabs.harvard.edu/abs/1986Natur.324..446B/abstract)) was a novel algorithm that utilized hierarchical tree structures to reduce particle-particle calculations from O(n$^2$) to O(n log(n)), vastly speeding up N-body simulations. This module represents a Python-based implementation of the algorithm via a implemented QuadTree data structure.

For more information regarding quasi-periodic boundary conditions with Barnes-Hut, we refer the reader to [Bouchet & Hernquist 1988](https://ui.adsabs.harvard.edu/abs/1988ApJS...68..521B), specifically Fig. 1 and Section 2c. 

Documentation regarding the free-parameters of the simulation can be seen in the notebook `example.ipynb`. 

## Running the Simulation 

The module `simulation.py` can be run via command line via the following example: 

```simulation.py -L 3 -opening_angle 0.2 -softening 1e-2 -N 24 -rand_type 0 --store```

Where inputs are processed using the `argparse` module. 

The notebook `example.ipynb` also has a step-by-step walkthrough and explanation of the free parameters of the simulation, how particles are generated, and how the simulation operates. 