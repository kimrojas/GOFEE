import numpy as np

from gpaw import GPAW, FermiDirac, PoissonSolver, Mixer, PW
from gpaw import extra_parameters
extra_parameters['blacs'] = True
from gpaw.utilities import h2gpts

from surrogate.gpr import GPR

from ase.io import read, write
from candidate_operations.candidate_generation import CandidateGenerator, StartGenerator
from candidate_operations.basic_mutations import RattleMutation, RattleMutation2, PermutationMutation

import sys

from gofee import GOFEE

### Set up StartGenerator and mutations ###
# read slab
slab = read('slab.traj', index='0')

# Stoichiometry of atoms to be placed
stoichiometry = 6*[1]+6*[6]

# Box in which to place atoms
v = 3*np.eye(3)
p0 = np.array((8.5, 8.5, 8.5))
box = [p0, v]

# initialize startgenerator
sg = StartGenerator(slab, stoichiometry, box)

# initialize rattle mutation
n_to_optimize = len(stoichiometry)
mutationSelector = CandidateGenerator([0.3, 0.7],
                                      [sg,
                                       RattleMutation(n_to_optimize, Nrattle=3, rattle_range=2)])

### Define calculator ###
calc=GPAW(poissonsolver = PoissonSolver(relax = 'GS',eps = 1.0e-7),
          mode = 'lcao',
          basis = 'dzp',
          xc='PBE',
          gpts = h2gpts(0.2, slab.get_cell(), idiv = 8),
          occupations=FermiDirac(0.1),
          maxiter=99,
          mixer=Mixer(nmaxold=5, beta=0.05, weight=75),
          nbands=-50,
          kpts=(1,1,1),
          txt='eval.txt')

### Initialize and run search ###
search = GOFEE(structures=None,
               calc=calc,
               gpr=None,
               startgenerator=sg,
               candidate_generator=None,
               max_steps=10,
               dmax_cov=2.5,
               population_size=5,
               dualpoint=True,
               restart='restart')

search.run()
