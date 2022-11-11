import numpy as np
import matplotlib.pyplot as plt
import unittest

from ase.io import read

from descriptor.fingerprint import Fingerprint



def finite_diff(descriptor, a, dx=1e-5):
    Nf = descriptor.get_feature(a).shape[0]
    Natoms, dim = a.positions.shape
    f_ddr = np.zeros((Natoms, dim, Nf))
    for ia in range(Natoms):
        for idim in range(dim):
            a_up = a.copy()
            a_down = a.copy()
            a_up.positions[ia,idim] += dx/2
            a_down.positions[ia,idim] -= dx/2
            
            
            f_up = descriptor.get_feature(a_up)
            f_down = descriptor.get_feature(a_down)
            f_ddr[ia,idim,:] = (f_up - f_down)/dx
    return f_ddr.reshape((-1,Nf))

def get_E_with_std(traj, gpr):
    E = []
    F = []
    for a in traj:
        e = gpr.predict_energy(a, )

class test_gpr(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setupClass')

    @classmethod
    def tearDownClass(cls):
        print('teardownClass')

    def setUp(self):
        print('setUp')
        ### Set up feature ###

        # Initialize feature
        self.descriptor = Fingerprint()

    def tearDown(self):
        print('tearDown\n')
    
    def test_forces(self):
        a = read('structures.traj', index='0')

        f_ddr = self.descriptor.get_featureGradient(a)
        f_ddr_num = finite_diff(self.descriptor, a)
        Na, Nf = f_ddr.shape
        x = np.arange(Nf)
        fig, ax = plt.subplots(1,1)
        ax.plot(x, f_ddr.T)
        ax.plot(x, f_ddr_num.T, 'k:')
        plt.show()
        print(f_ddr - f_ddr_num)
        np.testing.assert_almost_equal(f_ddr, f_ddr_num)

if __name__ == '__main__':
    unittest.main()
