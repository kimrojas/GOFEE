import numpy as np
from gofee import GOFEE
from scipy.special import erf
from datetime import timedelta
from time import time
from gofee.utils import array_to_string




class kappa_changing_GOFEE(GOFEE):
    '''
    GOFEE implementation for different kappa-changing functions.
    The given function override the internal class variable self.kappa within every iteration.

    Parameters:
        functype: str (default='error')
            Chooses the type of function for the kappa parameter to change on. The option are:
                'error': standart implementation of an error function
                'linear': linear decay
                'step': decay on a step function. Requires the kappastep parameter
                'alpha_error': decay on the error function with an additional parameter alpha that changes the steepness of the function
                'beta_error': decay on the error function with an additional parameter beta that shifts the function on the x-axis

        intervall: array-like (default=[5, 1])
            List or other iterable, where the first entry is the starting value for kappa and the second entry the ending value

        kappastep: int (default=4)
            Only used with the step function. Integer gives the number of steps for the decay

        alpha: int (default=2)
            Only used with the alpha_error implementation. Parameter controls the steepness of the function. alpha=0.5 is almost linear while alpha=50 is almost a two-step step function

        beta: int (default=0)!!!
            Only used with the beta_error implementation. Parameter ranges from zero to the number of iterations. ATTENTION: The default parameter is actually not a good parameter. A beta as big as the half of the number of iterations would equal the standard implementation of the error function. Recommended is something a little bit bigger than that
    '''
    def __init__(self, functype='error', intervall=[5, 1], alpha=2, beta=0, kappastep=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervall = intervall
        self.intervall_len = intervall[0] - intervall[1]
        self.functype = functype
        self.kappa = intervall[0]
        self.kappastep = kappastep
        self.alpha = alpha
        self.beta = beta





    def change_kappa(self):
        '''Function that is called in the run function to change the kappa parameter according to the given function'''
        if self.functype=='linear':
            self.linear_kappa_changer()
        elif self.functype=='error':
            self.errorfunc_kappa_changer()
        elif self.functype=='alpha_error':
            self.alpha_errorfunc_kappa_changer()
        elif self.functype=='beta_error':
            self.beta_errorfunc_kappa_changer()
        elif self.functype=='step':
            self.step_kappa_changer()




    def linear_kappa_changer(self):
        '''Function for the linear decay of kappa'''
        self.kappa = self.intervall[0] - self.steps * (self.intervall_len) / (self.max_steps-1)

    def errorfunc_kappa_changer(self):
        '''Function for the standard implementation of the error function'''
        x = 4 * self.steps / (self.max_steps-1) - 2
        self.kappa = - self.intervall_len / 2 * erf(x) / erf(2) + (self.intervall[0] + self.intervall[1]) / 2

    def alpha_errorfunc_kappa_changer(self):
        '''Function for the steepness controlling implementation of the error function'''
        x = 2 * self.alpha * self.steps / (self.max_steps-1) - self.alpha
        self.kappa = self.intervall_len / 2 * erf(x) / erf(-self.alpha) + (self.intervall[0] + self.intervall[1]) / 2

    def beta_errorfunc_kappa_changer(self):
        '''Function for the implementation of the error function that shift along the x-axis'''
        a = -erf(-4*self.beta/self.max_steps)
        b = -erf(4*(1-self.beta/self.max_steps))
        self.kappa = -self.intervall_len / (a - b) * erf(4 * (self.steps - self.beta) / self.max_steps) - self.intervall_len / (a - b) * erf(4 * (self.beta) / self.intervall_len) + self.intervall[0]

    def step_kappa_changer(self):
        '''Function for the decay on a step function'''
        for i in range(self.max_steps):
            if self.steps < ((i+1) * self.max_steps / self.kappastep):
                self.kappa = self.intervall[0] - self.intervall_len * i / (self.kappastep-1)
                break




    def run(self, max_steps=None, restart=None):
        """ Method to run the search. The existing minor changes have been marked by comments after the lines.
        """
        if restart is not None:
            self.read(restart)

        if max_steps is not None:
            self.max_steps = max_steps

        if self.steps == 0:
            self.evaluate_initial_structures()

        while self.steps < self.max_steps:
            self.log_msg += (f"\n##### STEPS: {self.steps} #####\n")
            self.log_msg += (f"Runtime: {timedelta(seconds=time()-self.t_start)}\n\n")
            t0 = time()
            self.train_surrogate()
            t1 = time()
            self.update_population()
            t2 = time()
            unrelaxed_candidates = self.generate_new_candidates()
            t3 = time()
            self.change_kappa() # new line for the key change in the implementation
            relaxed_candidates = self.relax_candidates_with_surrogate(unrelaxed_candidates)
            t4 = time()
            kappa = self.kappa
            self.log_msg += (f"kappa = {kappa} \n") #new line for logging the kappa value in the log file
            a_add = []
                
            for _ in range(3):
                try:
                    anew0 = self.select_with_acquisition(relaxed_candidates, kappa)
                    idx_relax = anew0.info['key_value_pairs'].get('index_closest_min')
                    if idx_relax is not None:
                        anew = self.make_relax_step_in_target_potential(idx_relax)
                    else:
                        anew = self.evaluate(anew0)
                    self.save_structures([anew])
                    a_add.append(anew)
                    if self.dualpoint:
                        if idx_relax is not None:
                            # Continue BFGS on from "anew".
                            adp = self.make_relax_step_in_target_potential(len(self.gpr.memory.energies)-1)
                        else:
                            adp = self.get_dualpoint(anew)
                            adp = self.evaluate(adp)
                        self.save_structures([adp])
                        a_add.append(adp)
                    
                    ###### testing ######
                    #self.write_BFGS_stats(anew0, idx_relax)  # for testing only
                    #####################
                    break
                except Exception as err:
                    kappa /=2
                    if self.master:
                        print(f'Exception :\n{err}', file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
            else:
                raise RuntimeError('Evaluation failed repeatedly - It might help to constrain the atomic positions during search.')
            self.comm.barrier()
            # Add evaluated structure(s) to GPR-memory
            #self.save_structures(a_add)

            # log timing
            self.log_msg += "Timing:\n"
            if self.verbose:
                self.log_msg += f"E_true:\n{array_to_string([a.info['key_value_pairs']['Epred'] for a in relaxed_candidates], unit='eV')}\n\n"
            self.log_msg += f"{'Training':12}{'Relax pop.':12}{'Make cands.':15}{'Relax cands.':16}{'Evaluate':12}\n"
            self.log_msg += f"{t1-t0:<12.2e}{t2-t1:<12.2e}{t3-t2:<15.2e}{t4-t3:<16.2e}{time()-t4:<12.2e}\n\n"

            # Print status on sufficiently optimized structures
            self.status_sufficiently_optimized_structures(a_add)

            # Add structure to population
            index_lowest = np.argmin([a.get_potential_energy() for a in a_add])
            self.population.add([a_add[index_lowest]])
            
            self.log_msg += (f"Prediction:\nenergy = {anew0.info['key_value_pairs']['Epred']:.5f}eV,  energy_std = {anew0.info['key_value_pairs']['Epred_std']:.5f}eV\n")
            self.log_msg += (f"E_true:\n{array_to_string([a.get_potential_energy() for a in a_add], unit='eV')}\n\n")
            self.log_msg += (f"Energy of population:\n{array_to_string([a.get_potential_energy() for a in self.population.pop], unit='eV')}\n")
            if self.verbose:
                self.log_msg += (f"Max force of ML-relaxed population:\n{array_to_string([(a.get_forces()**2).sum(axis=1).max()**0.5 for a in self.population.get_refined_pop()], unit='eV/A')}\n")
            
            if self.reference_structure is not None:
                self.print_reference_comparison(a_add)

            self.steps += 1
            self.log()

            # Save search state
            self.save_state()

