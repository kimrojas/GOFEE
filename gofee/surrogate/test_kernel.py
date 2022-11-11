import numpy as np
import unittest
from kernel import GaussKernel, DoubleGaussKernel

class test_kernel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setupClass')

    @classmethod
    def tearDownClass(cls):
        print('teardownClass')

    def setUp(self):
        print('setUp')
        #self.kernel = GaussKernel()
        #self.kernel_old = C(10, (1e1, 1e6)) * RBF(10, (1,1000))
        
        #self.kernel = DoubleGaussKernel(Nsplit_eta=2, eta=10)
        self.kernel = GaussKernel(Nsplit_eta=2, eta=10)

    def tearDown(self):
        print('tearDown\n')

    def test_jac(self):
        x1 = np.array([1.,2.,3.])
        x2 = np.array([1.5,2.,1.])
        x3 = np.array([3.,1.,1.])
        y = np.c_[x1,x2,x3].T
        np.testing.assert_almost_equal(self.kernel.kernel_jacobian(x1,x2),
                                       self.kernel.numerical_jacobian(x1,x2))
        np.testing.assert_almost_equal(self.kernel.kernel_jacobian(x1,y),
                                       self.kernel.numerical_jacobian(x1,y))
        np.testing.assert_almost_equal(self.kernel.kernel_jacobian(x2,y),
                                       self.kernel.numerical_jacobian(x2,y))
        
    def test_hyper_jac(self):
        x1 = np.array([1.,2.,3.])
        x2 = np.array([1.5,2.,1.])
        x3 = np.array([3.,1.,1.])
        X = np.c_[x1,x2].T
        K_ddtheta = self.kernel.kernel_hyperparameter_gradient(X)
        K_ddtheta_num = self.kernel.numerical_hyperparameter_gradient(X)
        #print(K_ddtheta, '\n\n')
        #print(K_ddtheta_num, '\n\n')
        #print(K_ddtheta-K_ddtheta_num, '\n')
        np.testing.assert_almost_equal(K_ddtheta, K_ddtheta_num)

if __name__ == '__main__':
    unittest.main()

