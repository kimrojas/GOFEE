from ase.calculators.calculator import Calculator


class gpr_calculator(Calculator):
    implemented_properties = ['energy', 'forces']
    default_parameters = {}

    def __init__(self, gpr, kappa=None, **kwargs):
        self.gpr = gpr
        self.kappa = kappa
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties:
            if self.kappa is None:
                #E = self.gpr.predict_energy(atoms, eval_std=False)
                E, Estd = self.gpr.predict_energy(atoms, eval_std=True)
                atoms.info['key_value_pairs']['Epred'] = E
                atoms.info['key_value_pairs']['Epred_std'] = Estd
            else:
                E, Estd = self.gpr.predict_energy(atoms, eval_std=True)
                atoms.info['key_value_pairs']['Epred'] = E
                atoms.info['key_value_pairs']['Epred_std'] = Estd
                E = E - self.kappa*Estd
            self.results['energy'] = E

        if 'forces' in properties:
            if self.kappa is None:
                F = self.gpr.predict_forces(atoms)
            else:
                F, Fstd = self.gpr.predict_forces(atoms, eval_with_energy_std=True)
                F = F - self.kappa*Fstd
            self.results['forces'] = F
