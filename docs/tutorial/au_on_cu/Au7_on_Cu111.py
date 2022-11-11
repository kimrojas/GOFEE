import numpy as np

from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT

from gofee.candidates import CandidateGenerator, StartGenerator, RattleMutation, PermutationMutation
from gofee.utils import OperationConstraint
from gofee import GOFEE

### Define calculator ###
calc = EMT()

### Set up system ###
# 1-layer fixed Cu(111) slab
template = fcc111('Cu', size=(5, 5, 1), vacuum=10.0)
c = FixAtoms(indices=np.arange(len(template)))
template.set_constraint(c)

# Stoichiometry of atoms to be placed
stoichiometry = 7*[79]

## Box for startgenerator and rattle-mutation
k = 0.2  # Shrinkage fraction from each side of the box in v[0] and v[1] directions.
cell = template.get_cell()
# Initialize box with cell
v = np.copy(cell)
# Set height of box
v[2][2] = 5
# Shrink box in v[0] and v[1] directions
v[0] *= (1-2*k)
v[1] *= (1-2*k)
# Chose anker point p0 so box in centered in v[0] and v[1] directions.
z_max_slab = np.max(template.get_positions()[:,2])
p0 = np.array((0, 0, z_max_slab+0.3)) + k*(cell[0]+cell[1])
# Make box
box = [p0, v]

# initialize startgenerator (used to generate initial structures)
sg = StartGenerator(template, stoichiometry, box)

### Set up candidate generation operations ###
# Set up constraint for rattle-mutation
box_constraint = OperationConstraint(box=box)

# initialize rattle-mutation
n_to_optimize = len(stoichiometry)
rattle = RattleMutation(n_to_optimize, Nrattle=2, rattle_range=4)

candidate_generator = CandidateGenerator([0.2, 0.8], [sg, rattle])

### Initialize and run search ###
search = GOFEE(calc=calc,
               startgenerator=sg,
               candidate_generator=candidate_generator,
               max_steps=150,
               population_size=5,
               position_constraint=box_constraint)
search.run()