import numpy as np
from scipy.special import erf
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.writer import agox_writer
from agox.observer import Observer


class ChangingKappaLowerConfidenceBoundAcquisitor(LowerConfidenceBoundAcquisitor):
    name = "ChangingKappaLowerConfidenceBoundAcquisitor"

    def __init__(self, N_iterations=25, intervall=[5,1], functype='linear', step=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_iterations = N_iterations
        self.intervall = intervall
        self.intervall_len = intervall[0] - intervall[1]
        self.functype = functype
        self.kappa = intervall[0]
        self.step = step

    def acquisition_function(self, E, sigma):
        print('#' + str(self.kappa), end='')
        return E - self.kappa * sigma

    def acquisition_force(self, E, F, sigma, sigma_force):
        return F - self.kappa * sigma_force

    def linear_kappa_changer(self):
        self.kappa = self.intervall[0] - self.get_iteration_counter() * (self.intervall_len) / (self.N_iterations-1)

    def other_linear_kappa_changer(self):
        self.kappa = - self.get_iteration_counter() * (self.intervall_len) / (self.N_iterations-7) + self.intervall[1] + self.intervall_len / (self.N_iterations-7) * self.N_iterations

    def errorfunc_kappa_changer(self):
        x = 4 * self.get_iteration_counter() / (self.N_iterations-1) - 2
        self.kappa = - self.intervall_len / 2 * erf(x) + (self.intervall[0] + self.intervall[1]) / 2

    def step_kappa_changer(self):
        for i in range(self.N_iterations):
            if self.get_iteration_counter() < ((i+1) * self.N_iterations / self.step):
                self.kappa = self.intervall[0] - self.intervall_len * i / (self.step-1)
                break

    @agox_writer
    @Observer.observer_method
    def prioritize_candidates(self, state):
        """
        Method that is attached to the AGOX iteration loop as an observer - not intended for use outside of that loop. 

        The method does the following: 
        1. Gets candidates from the cache using 'get_key'.
        2. Removes 'None' from the candidate list. 
        3. Calculates and sorts according to acquisition function. 
        4. Adds the sorted candidates to cache with 'set_key'
        5. Prints information.         
        """

        # Get data from the iteration data dict. 
        candidate_list = state.get_from_cache(self, self.get_key)
        candidate_list = list(filter(None, candidate_list))

        # Calculate acquisition function values and sort:
        if self.do_check():
            candidate_list, acquisition_values = self.sort_according_to_acquisition_function(candidate_list)
        else:
            acquisition_values = np.zeros(len(candidate_list))
        # Add the prioritized candidates to the iteration data in append mode!
        state.add_to_cache(self, self.set_key, candidate_list, mode='a')
        
        if self.functype=='linear':
            self.linear_kappa_changer()
        elif self.functype=='error':
            self.errorfunc_kappa_changer()
        elif self.functype=='step':
            self.step_kappa_changer()
        #print(self.kappa)
        self.print_information(candidate_list, acquisition_values)
        
