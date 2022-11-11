import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT

from gofee.candidates import CandidateGenerator, StartGenerator, RattleMutation
from gofee import GOFEE

### Define calculator ###
calc = EMT()

### Set up system ###
# make empty cell
template = Atoms('',
             cell=[20,20,20],
             pbc=[0, 0, 0])

# Stoichiometry of atoms to be placed
stoichiometry = 15*[29]

# Box in which to place atoms randomly
v = 5*np.eye(3)
p0 = np.array((7.5, 7.5, 7.5))
box = [p0, v]

# initialize startgenerator (used to generate initial structures)
sg = StartGenerator(template, stoichiometry, box)

### Set up candidate generation operations ###
# initialize rattle mutation
n_to_optimize = len(stoichiometry)
rattle = RattleMutation(n_to_optimize, Nrattle=3, rattle_range=4)

candidate_generator = CandidateGenerator(probabilities=[0.2, 0.8],
                                         operations=[sg, rattle])

### Initialize and run search ###
search = GOFEE(calc=calc,
               startgenerator=sg,
               candidate_generator=candidate_generator,
               max_steps=60,
               population_size=5)
search.run()