# Creates: structures.traj
import numpy as np

from ase import Atoms
from ase.calculators.dftb import Dftb

from gofee.candidates import CandidateGenerator, StartGenerator, RattleMutation
from gofee import GOFEE

### Define calculator ###
calc = Dftb(label='TiO2_surface',
            Hamiltonian_SCC='No',
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_Ti='"d"',
            Hamiltonian_MaxAngularMomentum_O='"p"',
            Hamiltonian_Charge='0.000000',
            Hamiltonian_Filling ='Fermi {',
            Hamiltonian_Filling_empty= 'Temperature [Kelvin] = 0.000000',
            kpts=(1,1,1))

### Set up system ###
# make empty cell
template = Atoms('',
            cell=[20,20,20],
            pbc=[0, 0, 0])

# Stoichiometry of atoms to be placed
stoichiometry = 5*[22]+10*[8]

# Box in which to place atoms randomly
v = 4*np.eye(3)
p0 = np.array((8.0, 8.0, 8.0))
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
                max_steps=100,
                population_size=5)
search.run()