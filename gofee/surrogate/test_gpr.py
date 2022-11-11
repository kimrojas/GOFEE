import numpy as np
import unittest
from gpr import GPR as gpr
from bfgslinesearch_constrained import BFGSLineSearch_constrained

from ase.io import read

class test_gpr(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setupClass')

    @classmethod
    def tearDownClass(cls):
        print('teardownClass')

    def setUp(self):
        print('setUp')
        a = read('structures.traj', index='0')
        #self.gpr = gpr()
        #self.gpr = gpr(kernel='double', template_structure=a)
        self.gpr = gpr(kernel='single', template_structure=a)

    def tearDown(self):
        print('tearDown\n')

    def test_lml_gradient(self):
        traj = read('structures.traj', index=':50')
        traj_train = traj[:40]
        traj_predict = traj[40:]
        self.gpr.train(traj_train)

        _, lml_ddTheta = self.gpr.neg_log_marginal_likelihood(eval_gradient=True)
        lml_ddTheta_numeric = self.gpr.numerical_neg_lml()
        np.testing.assert_almost_equal(lml_ddTheta, lml_ddTheta_numeric, decimal=6)
    
    def test_forces(self):
        traj = read('structures.traj', index=':50')
        traj_train = traj[:40]
        traj_predict = traj[40:]

        self.gpr.train(traj_train)

        a = traj_predict[0]
        F = self.gpr.predict_forces(a)
        F_numeric = self.gpr.numerical_forces(a)
        np.testing.assert_almost_equal(F, F_numeric, decimal=6)

    def test_forces_std(self):
        traj = read('structures.traj', index=':50')
        traj_train = traj[:40]
        traj_predict = traj[40:]

        self.gpr.train(traj_train)

        a = traj_predict[0]
        _, Fstd = self.gpr.predict_forces(a, eval_std=True)
        _, Fstd_numeric = self.gpr.numerical_forces(a, eval_std=True)
        np.testing.assert_almost_equal(Fstd, Fstd_numeric, decimal=6)
        
if __name__ == '__main__':
    unittest.main()

    import matplotlib.pyplot as plt
    from ase import Atoms
    from ase.visualize import view
    from descriptor.fingerprint import Fingerprint
    from custom_calculators import doubleLJ_calculator
    from gpr import GPR
    
    def finite_diff(krr, a, dx=1e-5, eval_std=False):
        Natoms, dim = a.positions.shape
        F = np.zeros((Natoms, dim))
        Fstd = np.zeros((Natoms, dim))
        for ia in range(Natoms):
            for idim in range(dim):
                a_up = a.copy()
                a_down = a.copy()
                a_up.positions[ia,idim] += dx/2
                a_down.positions[ia,idim] -= dx/2


                if not eval_std:
                    E_up = krr.predict_energy(a_up, eval_std=False)
                    E_down = krr.predict_energy(a_down, eval_std=False)
                    F[ia,idim] = -(E_up - E_down)/dx
                else:
                    E_up, err_up = krr.predict_energy(a_up, eval_std=True)
                    E_down, err_down = krr.predict_energy(a_down, eval_std=True)
                    
                    F[ia,idim] = -(E_up - E_down)/dx
                    Fstd[ia,idim] = -(err_up - err_down)/dx
        if eval_std:
            return F[1,0], Fstd[1,0]
        else:
            return F
    
    def createData(r):
        positions = np.array([[0,0,0],[r,0,0]])
        a = Atoms('2H', positions, cell=[3,3,1], pbc=[0,0,0])
        calc = doubleLJ_calculator()
        a.set_calculator(calc)
        return a

    def test1():
        a_train = [createData(r) for r in [0.9,1,1.3,2,3]]

        view(a_train[3])
        
        E_train = np.array([a.get_potential_energy() for a in a_train])
        Natoms = a_train[0].get_number_of_atoms()
        
        Rc1 = 5
        binwidth1 = 0.2
        sigma1 = 0.2
        
        Rc2 = 4
        Nbins2 = 30
        sigma2 = 0.2
        
        gamma = 1
        eta = 30
        use_angular = False
        
        descriptor = Fingerprint(Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
        
        # Set up KRR-model
        gpr = GPR(kernel='single', descriptor=descriptor)
        
        gpr.train(atoms_list=a_train)
        
        Ntest = 500
        r_test = np.linspace(0.87, 3.5, Ntest)
        E_test = np.zeros(Ntest)
        err_test = np.zeros(Ntest)
        F_test = np.zeros(Ntest)
        Fstd_test = np.zeros(Ntest)
        
        E_true = np.zeros(Ntest)
        F_true = np.zeros(Ntest)
        F_num = np.zeros(Ntest)
        Fstd_num = np.zeros(Ntest)
        for i, r in enumerate(r_test):
            ai = createData(r)
            E, err = gpr.predict_energy(ai, eval_std=True)
            E_test[i] = E
            err_test[i] = err
            
            result_test = gpr.predict_forces(ai, eval_std=True)
            F_test[i] = result_test[0][1,0]
            Fstd_test[i] = result_test[1][1,0]
            F_num[i], Fstd_num[i] = finite_diff(gpr, ai, eval_std=True)
            
            E_true[i] = ai.get_potential_energy()
            F_true[i] = ai.get_forces()[1,0]
            
            
        plt.figure()
        plt.title('Energy')
        plt.xlabel('r')
        plt.ylabel('E')
        plt.plot(r_test, E_true, color='darkgreen', label='true')
        plt.plot(r_test, E_test, color='steelblue', label='model')
        plt.fill_between(r_test, E_test-err_test, E_test+err_test, color='steelblue', alpha=0.3, label='model')
        plt.legend()
        
        plt.figure()
        plt.title('Force')
        plt.xlabel('r')
        plt.ylabel('F')
        plt.plot(r_test, F_true, color='darkgreen', label='true')
        plt.plot(r_test, F_test, color='steelblue', label='model')
        plt.plot(r_test, F_num, 'k:', label='num')
        plt.plot(r_test, F_test-Fstd_test, color='crimson', label='model')
        plt.plot(r_test, F_num-Fstd_num, 'k:', label='num')
        plt.legend()

    test1()
    plt.show()
