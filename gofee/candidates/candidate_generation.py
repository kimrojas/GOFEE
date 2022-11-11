import numpy as np
from abc import ABC, abstractmethod
from ase.data import covalent_radii
from ase.geometry import get_distances
from ase import Atoms
from ase.visualize import view

from gofee.utils import check_valid_bondlengths, get_min_distances_as_fraction_of_covalent

import warnings

class OffspringOperation(ABC):
    """Baseclass for mutation and crossover operations as well
    as the startgenerator.

    Parameters:

    blmin: The minimum allowed distance between atoms in units of
    the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: The maximum allowed distance, in units of the covalent 
    distance, from a single isolated atom to the closest atom. If
    blmax=None, no constraint is enforced on isolated atoms.

    force_all_bonds_valid: If True all bondlengths are forced to
    be valid according to blmin/blmax. If False, only bondlengths 
    of atoms specified in bondlength checks during operations are
    tested. The specified atoms are typically the ones changed 
    during operations. Default is False, as True might cause
    problems with GOFEE, as GPR-relaxations and dual-steps might
    result in structures that does not obey blmin/blmax.
    """
    def __init__(self, blmin=0.7, blmax=1.2, constraints=None,
                 force_all_bonds_valid=False, *args, **kwargs):
        self.blmin = blmin
        self.blmax = blmax
        self.constraints = constraints
        self.force_all_bonds_valid = force_all_bonds_valid
        self.description = 'Unspecified'

    def check_bondlengths(self, a, indices=None, indices_placed=None,
                                check_too_close=True, check_isolated=True, slack=0):
        """ Method to check if bondlengths are valid according to blmin
        amd blmax.
        """
        if self.blmin is not None:
            blmin_use = self.blmin-slack
        else:
            blmin_use = None
        if self.blmax is not None:
            blmax_use = self.blmax+0.1+slack
        else:
            blmax_use = None
        if self.force_all_bonds_valid:
            # Check all bonds (mainly for testing)
            return check_valid_bondlengths(a, self.blmin-slack, blmax_use,
                                           check_too_close=check_too_close,
                                           check_isolated=check_isolated)
        else:
            # Check only specified ones
            # (typically only for the atoms changed during operation)
            return check_valid_bondlengths(a, self.blmin-slack, blmax_use,
                                           indices=indices,
                                           indices_placed=indices_placed,
                                           check_too_close=check_too_close,
                                           check_isolated=check_isolated)

    def get_new_candidate(self, parents=None):
        """Standardized candidate generation method for all mutation
        and crossover operations.
        """
        # Check bondlengths
        if parents is not None:
            for i, parent in enumerate(parents):
                self.check_all_bondlengths(parent, f'SHORT BONDS IN PARENT {i}')

        for _ in range(5): # Make five tries
            a = self.operation(parents)
            if a is not None:
                a = self.finalize(a)
                break
        else:
            return None
        return a

    def train(self):
        """ Method to be implemented for the operations that rely on
        a Machine-Learned model to perform more informed/guided 
        mutation and crossover operations.
        """
        pass

    @abstractmethod
    def operation(self):
        pass

    def finalize(self, a, a0=None, successfull=True):
        """ Method to finalize new candidates.
        """
        # Wrap positions
        a.wrap()

        # finalize description
        if successfull:
            description = self.description
        else:
            description = 'failed ' + self.description

        # Save description
        a.info['key_value_pairs'] = {'origin': description}
        
        self.check_all_bondlengths(a, 'SHORT BONDS AFTER OPPERATION')
        return a

    def check_all_bondlengths(self, a, warn_text, slack=0.2):
            if self.force_all_bonds_valid:
                # Check all bonds
                valid_bondlengths = self.check_bondlengths(a, slack=slack)
                assert valid_bondlengths, 'bondlengths are not valid'
            else:
                d_shortest_bond, index_shortest_bond = get_min_distances_as_fraction_of_covalent(a)
                if d_shortest_bond < self.blmin - slack:
                    text = f"""{warn_text}:
                               Atom {index_shortest_bond} has bond with d={d_shortest_bond}d_covalent"""
                    warnings.warn(text)

    def set_constraints(self, constraints):
        self.constraints = constraints

    def check_constraints(self, indices=None):
        if self.constraints is not None:
            valid = self.constraints.check_if_valid(indices)
            return valid
        else:
            return True

class CandidateGenerator():
    """Class to produce new candidates by applying one of the 
    candidate generation operations which is supplied in the
    'operations'-list. The operations are drawn randomly according
    to the 'probabilities'-list.
    
    operations : list or list of lists
        Defines the operations to generate new candidates in GOFEE.
        of mutations/crossovers. Either a list of mutations, e.g. the
        RattleMutation, or alternatively a list of lists of such mutations,
        in which case consecutive operations, one drawn from each list,
        are performed. 

    probabilities : list or list of lists
        probability for each of the mutations/crossovers
        in operations. Must have the same dimensions as operations.
    """
    def __init__(self, probabilities, operations):
        cond1 = isinstance(operations[0], list)
        cond2 = isinstance(probabilities[0], list)
        if not cond1 and not cond2:
            operations = [operations]
            probabilities = [probabilities]
        element_count_operations = [len(op_list) for op_list in operations]
        element_count_probabilities = [len(prob_list)
                                       for prob_list in probabilities]
        assert element_count_operations == element_count_probabilities, 'the two lists must have the same shape'
        self.operations = operations
        self.rho = [np.cumsum(prob_list) for prob_list in probabilities]

    def __get_index__(self, rho):
        """Draw from the cumulative probalility distribution, rho,
        to return the index of which operation to use"""
        v = np.random.random() * rho[-1]
        for i in range(len(rho)):
            if rho[i] > v:
                return i
        
    def get_new_candidate(self, parents):
        """Generate new candidate by applying a randomly drawn
        operation on the structures. This is done successively for
        each list of operations, if multiple are present.
        """
        for op_list, rho_list in zip(self.operations, self.rho):
            for i_trial in range(5): # Do five trials
                to_use = self.__get_index__(rho_list)
                anew = op_list[to_use].get_new_candidate(parents)
                if anew is not None:
                    parents[0] = anew
                    break
            else:
                print('failed completely')
                anew = parents[0]
                anew = op_list[to_use].finalize(anew, successfull=False)
        return anew

    def set_constraints(self, constraints):
        for op_list in self.operations:
            for op in op_list:
                op.set_constraints(constraints)
        self.constraints = constraints

    def train(self, data):
        """ Method to train all trainable operations in
        self.operations.
        """
        for op_list in self.operations:
            for operation in op_list:
                operation.train(data)


def random_pos(box):
    """ Returns a random position within the box
         described by the input box. """
    p0 = box[0].astype(float)
    vspan = box[1]
    r = np.random.random((1, len(vspan)))
    pos = p0.copy()
    for i in range(len(vspan)):
        pos += vspan[i] * r[0, i]
    return pos

class StartGenerator(OffspringOperation):
    """ Class used to generate random initial candidates.

    Generates new candidates by iteratively adding
    one atom at a time within a user-defined box.

    Parameters:

    slab: Atoms object
        The atoms object describing the super cell to
        optimize within. Can be an empty cell or a cell 
        containing the atoms of a slab.

    stoichiometry: list
        A list of atomic numbers for the atoms
        that are placed on top of the slab (if one is present).

    box_to_place_in: list
        The box within which atoms are placed. The box
        should be on the form [p0, vspan] where 'p0' is the position of
        the box corner and 'vspan' is a matrix containing the three
        spanning vectors.

    blmin: float
        The minimum allowed distance between atoms in units of
        the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: float
        The maximum allowed distance, in units of the covalent 
        distance, from a single isolated atom to the closest atom. If
        blmax=None, no constraint is enforced on isolated atoms.

    cluster: bool
        If True atoms are required to be placed within
        blmin*d_cov of one of the other atoms to be placed. If
        False the atoms in the slab are also included.
    """
    def __init__(self, slab, stoichiometry, box_to_place_in,
                 cluster=False, description='StartGenerator',
                 *args, **kwargs):
        OffspringOperation.__init__(self, *args, **kwargs)
        self.slab = slab
        self.stoichiometry = stoichiometry
        self.box = box_to_place_in
        self.cluster = cluster
        self.description = description

    def operation(self, parents=None):
        a = self.make_structure()
        return a

    def make_structure(self):
        """ Generates a new random structure """
        Nslab = len(self.slab)
        Ntop = len(self.stoichiometry)
        num = np.random.permutation(self.stoichiometry)

        for i_trials in range(1000):
            a = self.slab.copy()
            for i in range(Ntop):
                pos_found = False
                for _ in range(300):
                    # Place new atom
                    posi = random_pos(self.box)
                    a += Atoms([num[i]], posi.reshape(1,3))

                    # Check if position of new atom is valid
                    not_too_close = self.check_bondlengths(a, indices=[Nslab+i],
                                                          check_too_close=True,
                                                          check_isolated=False)
                    if i == 0:  # The first atom
                        not_isolated = True
                    else:
                        if self.cluster:  # Check isolation excluding slab atoms.
                            not_isolated = self.check_bondlengths(a, indices=[Nslab+i], indices_placed=list(np.arange(Nslab,Nslab+i+1)),
                                                                        check_too_close=False,
                                                                        check_isolated=True)
                        else:  # All atoms.
                            not_isolated = self.check_bondlengths(a, indices=[Nslab+i],
                                                                        check_too_close=False,
                                                                        check_isolated=True)
                    valid_bondlengths = not_too_close and not_isolated
                    if not valid_bondlengths:
                        del a[-1]
                    else:
                        pos_found = True
                        break
                if not pos_found:
                    break
            if pos_found:
                break
        if i_trials == 999 and not pos_found:
            raise RuntimeError('StartGenerator: No valid structure was produced in 1000 trials.')
        else:
            return a
                
    
if __name__ == '__main__':
    from ase.io import read
    from ase.visualize import view

    from candidate_operations.basic_mutations import RattleMutation, RattleMutation2, PermutationMutation

    print(0.7*2*covalent_radii[1], 1.3*2*covalent_radii[1])
    
    np.random.seed(7)
    
    #a = read('/home/mkb/DFT/gpLEA/Si3x3/ref/gm_unrelaxed_done.traj', index='0')
    #a = read('si3x3.traj', index='0')
    #a = read('c6h6.traj', index='0')
    traj = read('c6h6_init.traj', index=':')
    #a = read('sn2o3.traj', index='0')
    #slab = read('slab_sn2o3.traj', index='0')

    """
    stoichiometry = 6*[50] + 10*[8]
    c = slab.get_cell()
    c[2,2] = 3.3
    p0 = np.array([0,0,14])
    box = [p0, c]
    
    sg = StartGenerator(slab, stoichiometry, box)
    """

    a = traj[0]
    rattle = RattleMutation(n_top=len(a), Nrattle=3, rattle_range=2)
    rattle2 = RattleMutation2(n_top=16, Nrattle=0.1)
    permut = PermutationMutation(n_top=16, Npermute=2)

    candidategenerator = OperationSelector([1], [rattle])
    #candidategenerator = CandidateGenerator([0., 1., 0.], [rattle, rattle2, permut])
    #candidategenerator = CandidateGenerator([[1],[1]], [[rattle2], [permut]])

    """
    for a in traj:
        vb = rattle.check_bondlengths(a)
        print(vb)
    """

    traj_rattle = []
    for i in range(100):
        for j, a in enumerate(traj[13:14]):
            print('i =', i, 'j =', j)
            a0 = a.copy()
            anew = candidategenerator.get_new_candidate([a0,a0])
            traj_rattle += [a0, anew]

    view(traj_rattle)
    """
    a_mut = rattle.get_new_candidate([a])
    view([a,a_mut])
    """
