import unittest

class test_importing(unittest.TestCase):
    def test_all_imports(self):
        from gofee import GOFEE
        from gofee.surrogate import GPR, Fingerprint, RepulsivePrior
        from gofee.surrogate.kernel import GaussKernel, DoubleGaussKernel

        from gofee.candidates import CandidateGenerator, StartGenerator
        from gofee.candidates import RattleMutation, RattleMutation2, PermutationMutation

        from gofee.utils import OperationConstraint

if __name__ == '__main__':
    unittest.main()